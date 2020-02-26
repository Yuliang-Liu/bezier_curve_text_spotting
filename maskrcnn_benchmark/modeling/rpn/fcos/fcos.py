import math
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from maskrcnn_benchmark.modeling.backbone.fbnet_builder import ShuffleV2Block

from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator
from .predictors import make_offset_predictor

import time

def snv2_block(in_channels, out_channels, kernel_size, stride):
    return ShuffleV2Block(in_channels, out_channels, expansion=2, stride=stride, kernel=kernel_size)


class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        if cfg.MODEL.FCOS.USE_LIGHTWEIGHT:
            conv_block = snv2_block
        else:
            conv_block = conv_with_kaiming_uniform(
                cfg.MODEL.FCOS.USE_GN, cfg.MODEL.FCOS.USE_RELU,
                cfg.MODEL.FCOS.USE_DEFORMABLE, cfg.MODEL.FCOS.USE_BN)

        for head in ['bbox']:
            tower = []
            for i in range(cfg.MODEL.FCOS.NUM_CONVS):
                tower.append(
                    conv_block(in_channels, in_channels, 3, 1))
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )

        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        self.bezier_pred = nn.Conv2d(
            in_channels, 16, kernel_size=3, stride=1,
            padding=1)

        # initialization
        for modules in [self.cls_logits, self.bbox_pred,
                        self.centerness, self.bezier_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        logits = []
        bbox_reg = []
        bezier_reg = []
        centerness = []

        tt = 0.0
        for l, feature in enumerate(x):
            bbox_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(bbox_tower))
            centerness.append(self.centerness(bbox_tower))
            bbox_reg.append(F.relu(self.bbox_pred(bbox_tower)))
            
            bezier_reg.append(self.bezier_pred(bbox_tower))
        return logits, bbox_reg, centerness, bezier_reg


class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FCOSModule, self).__init__()

        self.cfg = cfg.clone()

        head = FCOSHead(cfg, in_channels)

        box_selector_train = make_fcos_postprocessor(cfg, is_train=True)
        box_selector_test = make_fcos_postprocessor(cfg)

        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.num_iters = 0

    def forward(self, images, features, targets=None, vis=False):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness, bezier_regression = self.head(features)
        locations = self.compute_locations(features)

        if self.training:
            return self._forward_train(
                locations, box_cls,
                box_regression,
                bezier_regression,
                centerness,
                targets, images.image_sizes
            )
        else:
            # scale regression targets
            box_regression = [r * s for r, s in zip(box_regression, self.fpn_strides)]
            bezier_regression = [r * s for r, s in zip(bezier_regression, self.fpn_strides)]
            return self._forward_test(
                locations, box_cls, box_regression, bezier_regression,
                centerness, images.image_sizes
            )

    def _forward_train(self, locations, box_cls, box_regression,
                       bezier_regression, centerness,
                       targets, image_sizes):
        loss_box_cls, loss_box_reg, loss_bezier_reg, loss_centerness = self.loss_evaluator(
            locations, box_cls, box_regression, bezier_regression, centerness, targets
        )
        """
        if self.cfg.MODEL.RPN_ONLY:
            boxes = None
        else:
            with torch.no_grad():
                boxes = self.box_selector_train(
                    locations, box_cls, box_regression,
                    centerness, image_sizes)
        """
        boxes = None
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_bezier": loss_bezier_reg,
            "loss_centerness": loss_centerness
        }
        return boxes, losses

    def _forward_test(
            self, locations, box_cls, box_regression, bezier_regression,
            centerness, image_sizes):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression, bezier_regression,
            centerness, image_sizes
        )
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


def build_fcos(cfg, in_channels):
    return FCOSModule(cfg, in_channels)
