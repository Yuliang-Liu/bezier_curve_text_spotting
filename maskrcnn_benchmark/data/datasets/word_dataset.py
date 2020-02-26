import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList

from .rec import REC
from .bezier import BEZIER

BEZIER_IDX = [1, 0, 3, 2, 5, 4, 7, 6, 15, 14, 13, 12, 11, 10, 9, 8]


class WordDataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(WordDataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(WordDataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if anno and 'rec' in anno[0]:
            rec = [obj["rec"] for obj in anno]
            rec = REC(rec, img.size)
            target.add_field("rec", rec)

        if anno and 'bezier_pts' in anno[0]:
            bezier = [obj["bezier_pts"] for obj in anno]
            bezier = torch.as_tensor(bezier).reshape(-1, 16)  # guard against no boxes
            bezier = bezier[:, BEZIER_IDX]
            bezier = BEZIER(bezier, img.size)
            target.add_field("beziers", bezier)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
