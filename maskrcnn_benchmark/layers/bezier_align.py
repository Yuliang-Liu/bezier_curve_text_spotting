# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from maskrcnn_benchmark import _C

from apex import amp


class _BezierAlign(Function):
    @staticmethod
    def forward(ctx, input, bezier, output_size, spatial_scale, sampling_ratio):
        ctx.save_for_backward(bezier)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        output = _C.bezier_align_forward(
            input, bezier, spatial_scale, output_size[0], output_size[1], sampling_ratio
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        beziers, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.bezier_align_backward(
            grad_output,
            beziers,
            spatial_scale,
            output_size[0],
            output_size[1],
            bs,
            ch,
            h,
            w,
            sampling_ratio,
        )
        return grad_input, None, None, None, None


bezier_align = _BezierAlign.apply


class BezierAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super(BezierAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    @amp.float_function
    def forward(self, input, beziers):
        return bezier_align(
            input, beziers, self.output_size, self.spatial_scale, self.sampling_ratio
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr
