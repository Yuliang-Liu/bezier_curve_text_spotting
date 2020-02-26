from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from maskrcnn_benchmark.layers import ContextBlock
from maskrcnn_benchmark.modeling import registry

from . import fpn as fpn_module


@registry.NECKS.register("none")
def build_empty_neck(cfg):
    return nn.Sequential()


@registry.NECKS.register("libra")
def build_libra_neck(cfg):
    in_channels = cfg.MODEL.NECK.IN_CHANNELS
    num_levels = cfg.MODEL.NECK.NUM_LEVELS
    refine_level = cfg.MODEL.NECK.REFINE_LEVEL
    refine_type = cfg.MODEL.NECK.REFINE_TYPE
    use_gn = cfg.MODEL.NECK.USE_GN
    use_deformable = cfg.MODEL.NECK.USE_DEFORMABLE
    return BFP(in_channels,
               num_levels,
               refine_level=refine_level,
               refine_type=refine_type,
               use_gn=use_gn,
               use_deformable=use_deformable)


@registry.NECKS.register("fpn-retinanet")
def build_retina_neck(cfg):
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    last_stride = cfg.MODEL.NECK.LAST_STRIDE
    num_levels = cfg.MODEL.NECK.NUM_LEVELS
    if num_levels == 5:
        top_blocks = fpn_module.LastLevelP6P7(in_channels_p6p7,
                                              out_channels,
                                              last_stride)
    else:
        top_blocks = fpn_module.LastLevelP6(in_channels_p6p7,
                                            out_channels)

    return fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU,
            cfg.MODEL.FPN.USE_DEFORMABLE
        ),
        top_blocks=top_blocks,
    )


@registry.NECKS.register("fpn-align")
def build_retina_neck(cfg):
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    last_stride = cfg.MODEL.NECK.LAST_STRIDE
    num_levels = cfg.MODEL.NECK.NUM_LEVELS
    if num_levels == 5:
        top_blocks = fpn_module.LastLevelP6P7(in_channels_p6p7,
                                              out_channels,
                                              last_stride)
    else:
        top_blocks = fpn_module.LastLevelP6(in_channels_p6p7,
                                            out_channels)

    return fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU,
            cfg.MODEL.FPN.USE_DEFORMABLE
        ),
        top_blocks=top_blocks,
    )


class BFP(nn.Module):
    """BFP (Balanced Feature Pyrmamids)
    BFP takes multi-level features as inputs and gather them into a single one,
    then refine the gathered feature and scatter the refined results to
    multi-level features. This module is used in Libra R-CNN (CVPR 2019), see
    https://arxiv.org/pdf/1904.02701.pdf for details.

    """

    def __init__(self,
                 in_channels,
                 num_levels,
                 refine_level=1,
                 refine_type=None,
                 use_gn=False,
                 use_deformable=False):
        """
        Arguments:
            in_channels (int): Number of input channels (feature maps of all levels
                should have the same channels).
            num_levels (int): Number of input feature levels.
            refine_level (int): Index of integration and refine level of BSF in
                multi-level features from bottom to top.
            refine_type (str): Type of the refine op, currently support
                [None, 'conv', 'non_local'].
        """
        super(BFP, self).__init__()
        assert refine_type in [None, 'conv', 'non_local', 'gc_block']

        self.in_channels = in_channels
        self.num_levels = num_levels

        self.refine_level = refine_level
        self.refine_type = refine_type
        assert 0 <= self.refine_level < self.num_levels

        if self.refine_type == 'conv':
            conv_block = conv_with_kaiming_uniform(
                use_gn=use_gn,
                use_deformable=use_deformable)
            self.refine = conv_block(
                self.in_channels,
                self.in_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        elif self.refine_type == 'gc_block':
            self.refine = ContextBlock(
                self.in_channels,
                ratio=1. / 16.)

    def forward(self, inputs):
        assert len(inputs) == self.num_levels

        # step 1: gather multi-level features by resize and average
        feats = []
        gather_size = inputs[self.refine_level].size()[2:]
        for i in range(self.num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(
                    inputs[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    inputs[i], size=gather_size, mode='nearest')
            feats.append(gathered)

        bsf = sum(feats) / len(feats)

        # step 2: refine gathered features
        if self.refine_type is not None:
            bsf = self.refine(bsf)

        # step 3: scatter refined features to multi-levels by a residual path
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]
            if i < self.refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
            outs.append(residual + inputs[i])

        return tuple(outs)


def build_neck(cfg):
    assert cfg.MODEL.NECK.CONV_BODY in registry.NECKS, \
        "cfg.MODEL.NECK.CONV_BODY: {} is not registered in registry".format(
            cfg.MODEL.NECK.CONV_BODY)
    return registry.NECKS[cfg.MODEL.NECK.CONV_BODY](cfg)
