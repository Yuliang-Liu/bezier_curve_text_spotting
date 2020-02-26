# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import cv2
from PIL import Image

import torch
import torchvision
from torch.nn import functional as F

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints


min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
            self, ann_file, root, remove_images_without_annotations, transforms=None,
            cfg=None,
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

        # use mask as fg for pointmask
        self.use_binary_mask = (cfg.MODEL.META_ARCHITECTURE == "OneStage"
                                and cfg.MODEL.ONE_STAGE_HEAD in ['pointmask', 'align'])
        self.use_polygon_det = cfg.MODEL.POLYGON_DET
        self.binary_mask_size = cfg.MODEL.POINTMASK.MASK_SIZE
        self.panoptic_on = cfg.MODEL.PANOPTIC.PANOPTIC_ON

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

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

        if anno and "segmentation" in anno[0]:
            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, img.size, mode='poly')
            target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self.panoptic_on:
            # add semantic masks to the boxlist for panoptic
            img_id = self.ids[idx]
            img_path = self.coco.loadImgs(img_id)[0]['file_name']

            seg_path = self.root.replace('coco', 'coco/annotations').replace('train2017', 'panoptic_train2017_semantic_trainid_stff').replace('val2017', 'panoptic_val2017_semantic_trainid_stff') + '/' + img_path
            seg_img = Image.open(seg_path.replace('jpg', 'png'))

            # seg_img.mode = 'L'
            seg_gt = torch.ByteTensor(torch.ByteStorage.from_buffer(seg_img.tobytes()))
            seg_gt = seg_gt.view(seg_img.size[1], seg_img.size[0], 1)
            seg_gt = seg_gt.transpose(0, 1).transpose(0, 2).contiguous().float()

            seg_gt = SegmentationMask(seg_gt, seg_img.size, "mask")
            target.add_field("seg_masks", seg_gt)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        if self.use_binary_mask and target.has_field("masks"):
            if self.use_polygon_det:
                # compute target maps
                masks = target.get_field("masks")
                w, h = target.size
                assert target.mode == "xyxy"
                targets_map = np.ones((h, w), dtype=np.uint8) * 255
                assert len(masks.instances) <= 255
                for target_id, polygons in enumerate(masks.instances):
                    targets_map = self.compute_target_maps(
                        targets_map, target_id, polygons
                    )
                    target.add_field("targets_map", torch.Tensor(targets_map))

            # compute binary masks
            MASK_SIZE = self.binary_mask_size
            binary_masks = torch.zeros(len(target), MASK_SIZE[0] * MASK_SIZE[1])
            masks = target.get_field("masks")
            # assert len(target) == len(masks.instances)
            for i, polygons in enumerate(masks.instances):
                mask = self.polygons_to_mask(polygons)
                mask = mask.to(binary_masks.device)
                mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), MASK_SIZE)
                binary_masks[i, :] = mask.view(-1)
            target.add_field("binary_masks", binary_masks)

        return img, target, idx

    def compute_target_maps(self, targets_map, target_id, polygons):
        '''
        :param label:
        :param polygons:
        :param box:
        :param im_sizes: (h, w)
        :return:
        '''
        polygons = polygons.polygons

        for single_polygon in polygons:
            single_polygon = single_polygon.view(-1, 2).to(torch.int32)
            targets_map = cv2.fillPoly(targets_map, single_polygon.numpy()[np.newaxis], target_id)

        return targets_map

    def polygons_to_mask(self, polygons):
        polygons = polygons.polygons

        all_polygons = torch.cat(polygons, dim=0).to(torch.int32).view(-1, 2)
        (min_x, min_y), _ = all_polygons.min(dim=0)
        (max_x, max_y), _ = all_polygons.max(dim=0)
        mask_w = int(max_x - min_x + 1)
        mask_h = int(max_y - min_y + 1)
        offsets = [int(min_x), int(min_y)]

        mask = np.zeros((mask_h, mask_w), dtype=np.float32)
        for single_polygon in polygons:
            single_polygon = single_polygon.view(-1, 2).to(torch.int32)
            single_polygon -= torch.IntTensor(offsets)
            mask = cv2.fillPoly(mask, single_polygon.numpy()[np.newaxis], 1.0)
        return torch.Tensor(mask)

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
