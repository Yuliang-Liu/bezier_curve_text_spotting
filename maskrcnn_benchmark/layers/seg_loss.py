import torch
from torch import nn
from torch.nn import functional as F


class SegLoss(nn.Module):
    def __init__(self, other=-1, scale_factor=1):
        super(SegLoss, self).__init__()
        self.other = other
        self.scale_factor = scale_factor

    def prepare_target(self, targets, mask):
        labels = []

        for t in targets:
            t = t.get_field("seg_masks").get_mask_tensor().unsqueeze(0)
            if self.other > 0:
                t = torch.clamp(t, max=self.other)
            if self.scale_factor != 1:
                t = F.interpolate(
                    t.unsqueeze(0),
                    scale_factor=self.scale_factor,
                    mode='nearest').long().squeeze()
            labels.append(t)

        batched_labels = mask.new_full(
            (mask.size(0), mask.size(2), mask.size(3)),
            mask.size(1) - 1,
            dtype=torch.long)
        for label, pad_label in zip(labels, batched_labels):
            pad_label[: label.shape[0], : label.shape[1]].copy_(label)

        return batched_labels

    def forward(self, mask, target):
        '''
            mask : Tensor
            target : list[Boxlist]
        '''
        target = self.prepare_target(target, mask)

        loss = F.cross_entropy(mask, target)
        return loss
