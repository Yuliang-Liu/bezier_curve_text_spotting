from torch import nn


def batchnorm(in_planes, affine=True, eps=1e-5, momentum=0.1):
    """2D Batch Normalisation.
    Args:
      in_planes (int): number of input channels.
      affine (bool): whether to add learnable affine parameters.
      eps (float): stability constant in the denominator.
      momentum (float): running average decay coefficient.
    Returns:
      `nn.BatchNorm2d' instance.
    """
    return nn.BatchNorm2d(in_planes, affine=affine, eps=eps, momentum=momentum)


def conv3x3(in_planes, out_planes, stride=1, dilation=1, groups=1, bias=False):
    """2D 3x3 convolution.
    Args:
      in_planes (int): number of input channels.
      out_planes (int): number of output channels.
      stride (int): stride of the operation.
      dilation (int): dilation rate of the operation.
      groups (int): number of groups in the operation.
      bias (bool): whether to add learnable bias parameter.
    Returns:
      `nn.Conv2d' instance.
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        bias=bias)


def conv1x1(in_planes, out_planes, stride=1, groups=1, bias=False):
    """2D 1x1 convolution.
    Args:
      in_planes (int): number of input channels.
      out_planes (int): number of output channels.
      stride (int): stride of the operation.
      groups (int): number of groups in the operation.
      bias (bool): whether to add learnable bias parameter.
    Returns:
      `nn.Conv2d' instance.
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=0,
        groups=groups,
        bias=bias)


class BasicBlock(nn.Module):
    """Basic residual block.
    Conv-BN-ReLU => Conv-BN => Residual => ReLU.
    Args:
      inplanes (int): number of input channels.
      planes (int): number of intermediate and output channels.
      stride (int): stride of the first convolution.
      downsample (nn.Module or None): downsampling operation.
    Attributes:
      expansion (int): equals to the ratio between the numbers
                       of output and intermediate channels.
    """
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = batchnorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = batchnorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block.
    Conv-BN-ReLU => Conv-BN-ReLU => Conv-BN => Residual => ReLU.
    Args:
      inplanes (int): number of input channels.
      planes (int): number of intermediate and output channels.
      stride (int): stride of the first convolution.
      downsample (nn.Module or None): downsampling operation.
    Attributes:
      expansion (int): equals to the ratio between the numbers
                       of output and intermediate channels.
    """
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, bias=False)
        self.bn1 = batchnorm(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = batchnorm(planes)
        self.conv3 = conv1x1(planes, planes * 4, bias=False)
        self.bn3 = batchnorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
