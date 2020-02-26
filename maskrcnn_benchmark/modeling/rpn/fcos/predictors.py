import torch
from torch import nn

from maskrcnn_benchmark.layers import DeformConv


class PolarPredictor(nn.Module):
    """
    Use center point to predict all offsets
    """
    def __init__(self, in_channels, num_chars=32, voc_size=38, kernel_size=3):
        super(PolarPredictor, self).__init__()
        self.num_chars = num_chars
        self.locator = nn.Conv2d(
            in_channels, 3 * num_chars, kernel_size=3, stride=1, padding=1)
        self.clf = DeformConv(
            in_channels, voc_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.offset_repeat = kernel_size ** 2

    def forward(self, x, y, vis=False):
        """ Predict offsets with x and rec with y
        Offsets is relative starting from the center
        """
        N, _, H, W = x.size()
        features = self.locator(x)
        offsets, masks = features[:, :self.num_chars * 2], features[:, self.num_chars * 2:]
        location = offsets[:, :2]
        recs = [self.clf(y, location)]
        locations = [location]
        for i in range(1, self.num_chars):
            delta = offsets[:, i * 2:(i + 1) * 2]
            location = location + delta
            recs.append(self.clf(y, location))
            locations.append(location)
        return torch.stack(recs, dim=4), masks, torch.cat(locations, dim=1)


class SequentialPredictor(nn.Module):
    """
    Sequentially predict the offsets
    """
    def __init__(self, in_channels, num_chars=32, voc_size=38, kernel_size=3):
        super(SequentialPredictor, self).__init__()
        self.num_chars = num_chars
        self.voc_size = voc_size
        self.locator = nn.Conv2d(
            in_channels, num_chars + 2, kernel_size=3, stride=1, padding=1)
        self.clf = DeformConv(
            in_channels, voc_size + 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.offset_repeat = kernel_size ** 2

    def forward(self, x, y, max_len):
        """ Predict offsets with x and rec with y
        Offsets is relative starting from the center
        """
        N, _, H, W = x.size()
        init_features = self.locator(x)
        location, masks = init_features[:, :2], init_features[:, 2:]
        recs = torch.zeros(N, self.voc_size, H, W, max_len).cuda()
        locations = torch.zeros(N, max_len * 2, H, W).cuda()
        delta = 0
        for i in range(max_len):
            # parallel?
            # during training, early stopping with gt
            # TODO: early stopping for testing
            location = location + delta
            locations[:, i * 2: i * 2 + 2] = location
            local_features = self.clf(y, location)
            rec, delta = local_features[:, :-2], local_features[:, -2:]
            recs[:, :, :, :, i] = rec
        return recs, masks, locations


def make_offset_predictor(cfg, in_channels):
    # not using kernel_size now
    predictor = cfg.MODEL.OFFSET.PREDICTOR
    kwargs = {"num_chars": cfg.MODEL.OFFSET.NUM_CHARS,
              "voc_size": cfg.MODEL.OFFSET.VOC_SIZE,
              "kernel_size": cfg.MODEL.OFFSET.KERNEL_SIZE}
    if predictor == "polar":
        return PolarPredictor(in_channels, **kwargs)
    elif predictor == "sequential":
        return SequentialPredictor(in_channels, **kwargs)
    else:
        raise NotImplementedError("{} is not a valid predictor".format(predictor))
