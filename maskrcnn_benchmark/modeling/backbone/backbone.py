# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from .pan import PAN
from .msr import MSR
from . import resnet
from . import resnet_bn
# from . import resnet_lpf
from . import mobilenet


@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model


@registry.BACKBONES.register("R-50-BN")
def build_resnet_50_bn_backbone(cfg):
    model = resnet_bn.resnet50()
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model


@registry.BACKBONES.register("R-50-LPF")
def build_resnet_50_antialiased_backbone(cfg):
    filter_size = 3
    model = resnet_lpf.resnet50(cfg, filter_size=filter_size)
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model

@registry.BACKBONES.register("R-14")
@registry.BACKBONES.register("R-50")
@registry.BACKBONES.register("R-101")
def build_resnet_backbone(cfg):
    model = resnet.ResNet(cfg)
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model


@registry.BACKBONES.register("HNASNetSearch")
def build_hnasnet_backbone(cfg):
    in_channels_stage2 = (cfg.MODEL.HNASNET.FILTER_MULTIPLIER
                          * cfg.MODEL.HNASNET.NUM_BLOCKS)
    in_channels_list = [in_channels_stage2 * 2 ** i
                        for i in range(len(cfg.MODEL.HNASNET.STRIDE_MULTIPLIER))]
    body = HNASNetSearch(cfg)
    out_channels = cfg.MODEL.HNASNET.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU,
            cfg.MODEL.FPN.USE_DEFORMABLE
        ))
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("HNASNet")
def build_hnasnet_backbone(cfg):
    in_channels_stage2 = (cfg.MODEL.HNASNET.FILTER_MULTIPLIER
                          * cfg.MODEL.HNASNET.NUM_BLOCKS)
    in_channels_list = [in_channels_stage2 * 2 ** i
                        for i in range(1, len(cfg.MODEL.HNASNET.STRIDE_MULTIPLIER))]
    in_channels_list = [0] + in_channels_list
    body = HNASNet(cfg)
    out_channels = cfg.MODEL.HNASNET.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpn = fpn_module.FPN(
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU,
            cfg.MODEL.FPN.USE_DEFORMABLE
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("DetNASNet-RCNN")
def build_detnasnet_fpn_backbone(cfg):
    in_channels_stage2 = cfg.MODEL.HNASNET.FILTER_MULTIPLIER
    in_channels_list = [in_channels_stage2 * s
                        for s in cfg.MODEL.HNASNET.STRIDE_MULTIPLIER[1:]]
    in_channels_list = [0] + in_channels_list
    body = DetNASNet(cfg)
    out_channels = cfg.MODEL.HNASNET.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.Scaler(
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU,
            cfg.MODEL.FPN.USE_DEFORMABLE,
            cfg.MODEL.FPN.USE_BN
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("DetNASNet")
def build_detnasnet_backbone(cfg):
    in_channels_stage2 = cfg.MODEL.HNASNET.FILTER_MULTIPLIER
    in_channels_list = [in_channels_stage2 * s
                        for s in cfg.MODEL.HNASNET.STRIDE_MULTIPLIER[1:]]
    in_channels_list = [0] + in_channels_list
    body = DetNASNet(cfg)
    out_channels = cfg.MODEL.HNASNET.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = out_channels
    fpn = fpn_module.Scaler(
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU,
            cfg.MODEL.FPN.USE_DEFORMABLE,
            cfg.MODEL.FPN.USE_BN
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    body = resnet.ResNet(cfg)
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU,
            cfg.MODEL.FPN.USE_DEFORMABLE
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    if cfg.MODEL.MSR_ON:
        model = MSR(body, in_channels_list, fpn=fpn)
    else:
        model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-50-PAN")
@registry.BACKBONES.register("R-101-PAN")
@registry.BACKBONES.register("R-152-PAN")
def build_resnet_pan_backbone(cfg):
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    body = resnet.ResNet(cfg)
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU,
            cfg.MODEL.FPN.USE_DEFORMABLE
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    pan = PAN()
    if cfg.MODEL.MSR_ON:
        model = MSR(body, in_channels_list, fpn=fpn, pan=pan)
    else:
        model = nn.Sequential(OrderedDict([("body", body),
                                           ("pan", pan),
                                           ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpn = fpn_module.FPN(
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
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("MNV2-FPN-RETINANET")
def build_mnv2_fpn_backbone(cfg):
    body = mobilenet.MobileNetV2(cfg)
    in_channels_stage2 = body.return_features_num_channels
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2[1],
            in_channels_stage2[2],
            in_channels_stage2[3],
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(out_channels, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
