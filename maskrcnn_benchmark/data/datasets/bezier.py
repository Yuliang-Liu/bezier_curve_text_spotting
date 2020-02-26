import torch

# transpose

class BEZIER(object):
    def __init__(self, bezier, image_size):
        device = bezier.device if isinstance(bezier, torch.Tensor) else torch.device("cpu")
        bezier = torch.as_tensor(bezier, dtype=torch.float32, device=device)
        if bezier.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bezier.ndimension())
            )
        if bezier.size(-1) != 16:
            raise ValueError(
                "last dimension of bezier should have a "
                "size of 16, got {}".format(bezier.size(-1))
            )

        self.bbox = bezier
        self.size = image_size  # (image_width, image_height)

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box
        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """

        ratios = (float(size[0]) / float(self.size[0]), float(size[1]) / float(self.size[1]))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bezier = BEZIER(scaled_box, size)
            return bezier

        ratio_width, ratio_height = ratios
        bezier = self.bbox.view(-1, 8, 2)
        bezier_x = bezier[:, :, 1] * ratio_width
        bezier_y = bezier[:, :, 0] * ratio_height
        scaled_bezier = torch.stack((bezier_y, bezier_x), dim=2).view(-1, 16)
        return BEZIER(scaled_bezier, size)

    def pad(self, new_size):
        self.size = new_size
        return self

    def transpose(self, method):
        raise NotImplementedError(
                "Not implemented yet."
            )
    
    def crop(self, box, remove_empty=False):
        """
        Crops a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        w, h = box[2] - box[0], box[3] - box[1]
        k = self.bbox.clone()
        k[:, 0::2] -= box[1] # y
        k[:, 1::2] -= box[1] # x
        return type(self)(k, (w, h))
        # raise NotImplementedError(
        #         "Not implemented yet."
        #     )

    def to(self, device):
        bezier = BEZIER(self.bbox.to(device), self.size)
        return bezier

    def __getitem__(self, item):
        bezier = BEZIER(self.bbox[item], self.size)
        return bezier

    def __len__(self):
        return self.bbox.shape[0]

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_bezierBox={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s


if __name__ == "__main__":
    # print('TODO ...')
    bezier = BEZIER([[0, 0, 10, 10], [0, 0, 5, 5]], (10, 10))
    s_bezier = bezier.resize((5, 5))
    print(s_bezier)
    print(s_bezier.bezier)
