"""
Discrete structure of Auto-DeepLab

Includes utils to convert continous Auto-DeepLab to discrete ones
"""

import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.nas.cell import FixCell
from .hnas_common import conv3x3_bn, conv1x1_bn


class Scaler(nn.Module):
    """Reshape features"""
    def __init__(self, scale, inp, C, relu=True):
        """
        Arguments:
            scale (int) [-2, 2]: scale < 0 for downsample
            inp (int): input channel
            C (int): output channel
            relu (bool): set to False if the modules are pre-relu
        """
        super(Scaler, self).__init__()
        if scale == 0:
            self.scaler = conv1x1_bn(inp, C, 1, relu=relu)
        if scale == 1:
            self.scaler = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=False),
                conv1x1_bn(inp, C, 1, relu=relu))
        # official implementation used bilinear for all scalers
        if scale == -1:
            self.scaler = conv3x3_bn(inp, C, 2, relu=relu)

    def forward(self, hidden_state):
        return self.scaler(hidden_state)


class DeepLabScaler(nn.Module):
    """Official implementation
    https://github.com/tensorflow/models/blob/master/research/deeplab/core/nas_cell.py#L90
    """
    def __init__(self, scale, inp, C):
        super(DeepLabScaler, self).__init__()
        self.scale = 2 ** scale
        self.conv = conv1x1_bn(inp, C, 1, relu=False)

    def forward(self, hidden_state):
        if self.scale != 1:
            hidden_state = F.interpolate(hidden_state,
                                         scale_factor=self.scale,
                                         mode='bilinear',
                                         align_corners=False)
        return self.conv(F.relu(hidden_state))


class HNASNet(nn.Module):
    def __init__(self, cfg):
        super(HNASNet, self).__init__()

        # load genotype
        geno_file = cfg.MODEL.HNASNET.GENOTYPE
        print("Loading genotype from {}".format(geno_file))
        geno_cell, geno_path = torch.load(geno_file)
        self.geno_path = geno_path

        # basic configs
        self.f = cfg.MODEL.HNASNET.FILTER_MULTIPLIER
        self.num_layers = cfg.MODEL.HNASNET.NUM_LAYERS
        self.num_blocks = cfg.MODEL.HNASNET.NUM_BLOCKS
        BxF = self.f * self.num_blocks
        stride_mults = cfg.MODEL.HNASNET.STRIDE_MULTIPLIER
        self.num_strides = len(stride_mults)

        self.stem1 = nn.Sequential(
            conv3x3_bn(3, 64, 2),
            conv3x3_bn(64, 64, 1))
        self.stem2 = conv3x3_bn(64, BxF, 2)

        # feature pyramids
        self.bases = nn.ModuleList()
        in_channels = 64
        for s in range(self.num_strides):
            out_channels = BxF * stride_mults[s]
            self.bases.append(conv3x3_bn(in_channels, out_channels, 2))
            in_channels = out_channels

        # create cells
        self.cells = nn.ModuleList()
        self.scalers = nn.ModuleList()
        if cfg.MODEL.HNASNET.TIE_CELL:
            geno_cell = [geno_cell] * self.num_layers

        h_0 = 0  # prev prev hidden index
        for layer, (geno, h) in enumerate(zip(geno_cell, geno_path), 1):
            stride = stride_mults[h]
            self.cells.append(FixCell(geno, self.f * stride))
            # scalers
            inp0 = BxF * stride_mults[h_0]
            scaler0 = Scaler(h_0 - h, inp0, stride * self.f, relu=False)
            scaler1 = Scaler(0, BxF * stride, stride * self.f, relu=False)
            h_0 = h
            self.scalers.append(scaler0)
            self.scalers.append(scaler1)

    def forward(self, x, drop_prob=-1):
        h1 = self.stem1(x)
        h0 = self.stem2(h1)

        # get feature pyramids
        fps = []
        for base in self.bases:
            h1 = base(h1)
            fps.append(h1)

        s_1 = 0
        for i, (cell, s) in enumerate(zip(self.cells, self.geno_path)):
            input_0 = self.scalers[i * 2](h0)
            input_1 = self.scalers[i * 2 + 1](fps[s])
            # update feature pyramid at s_{-1}
            fps[s_1] = h0
            if s == s_1:
                h0 = cell(input_0, input_1, drop_prob) + h0
            else:
                h0 = cell(input_0, input_1, drop_prob) + fps[s]
            s_1 = s
        fps[s_1] = h0
        return fps
