import torch
from torch import nn
from torch.nn import functional as F


class ConcatUpConv(nn.Module):
    def __init__(self, inplanes, outplanes, upsample=True):
        super(ConcatUpConv, self).__init__()
        out_channels = outplanes
        self.upsample = upsample
        self.con_1x1 = nn.Conv2d(inplanes, outplanes, 1, bias=False)
        nn.init.kaiming_uniform_(self.con_1x1.weight, a=1)
        self.nor_1 = nn.BatchNorm2d(out_channels)
        self.leakyrelu_1 = nn.ReLU()
        if self.upsample:
            self.con_3x3 = nn.Conv2d(outplanes, out_channels // 2,
                                     kernel_size=3, stride=1, padding=1, bias=False)
            nn.init.kaiming_uniform_(self.con_3x3.weight, a=1)
            self.nor_3 = nn.BatchNorm2d(out_channels // 2)
            self.leakyrelu_3 = nn.ReLU()

    def forward(self, x1, x2):
        fusion = torch.cat([x1, x2], dim=1)
        out_1 = self.leakyrelu_1(self.nor_1(self.con_1x1(fusion)))
        out = None
        if self.upsample:
            out = self.leakyrelu_3(self.nor_3(self.con_3x3(out_1)))
            out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        return out, out_1


class MSR(nn.Module):
    def __init__(self, body, channels, fpn=None, pan=None):
        super(MSR, self).__init__()
        self.body = body
        cucs = nn.ModuleList()
        channel = channels[0]
        cucs.append(ConcatUpConv(channel * 2, channel, upsample=False))
        for i, channel in enumerate(channels[1:]):
            cucs.append(ConcatUpConv(channel * 2, channel))
        self.cucs = cucs
        if fpn is not None:
            self.fpn = fpn
        if pan is not None:
            self.pan = pan

    def forward(self, x):
        outputs = self.body(x)

        re_x = F.interpolate(x, scale_factor=0.5,
                             mode='bilinear', align_corners=False)
        output_re = self.body(re_x)[-1]
        low = F.interpolate(output_re,
                            size=outputs[-1].shape[2:],
                            mode='bilinear', align_corners=False)
        new_outputs = []
        for cuc, high in zip(self.cucs[::-1], outputs[::-1]):
            low, out = cuc(high, low)
            new_outputs.append(out)
        outs = new_outputs[::-1]
        if hasattr(self, 'pan'):
            outs = self.pan(outs)
        if hasattr(self, 'fpn'):
            outs = self.fpn(outs)
        return outs
