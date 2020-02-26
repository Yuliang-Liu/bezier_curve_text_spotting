import torch

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class REC(object):
    def __init__(self, rec, size, mode=None):
        # FIXME remove check once we have better integration with device
        # in my version this would consistently return a CPU tensor
        device = rec.device if isinstance(rec, torch.Tensor) else torch.device('cpu')
        rec = torch.as_tensor(rec, dtype=torch.int64, device=device)

        self.rec = rec

        self.size = size
        self.mode = mode

    def crop(self, box):
        w, h = box[2] - box[0], box[3] - box[1]
        return type(self)(self.rec, (w, h), self.mode)

    def resize(self, size, *args, **kwargs):
        return type(self)(self.rec, size, self.mode)

    def transpose(self, method):
        raise NotImplementedError(
                    "Not implemented")

    def to(self, *args, **kwargs):
        return type(self)(self.rec.to(*args, **kwargs), self.size, self.mode)

    def __getitem__(self, item):
        return type(self)(self.rec[item], self.size, self.mode)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_instances={}, '.format(len(self.rec))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={})'.format(self.size[1])
        return s
