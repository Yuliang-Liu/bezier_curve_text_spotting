# coding: utf-8
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
from scipy.special import comb as n_over_k

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util

from maskrcnn_benchmark.data.datasets.bezier import BEZIER

CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']

class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image


class COCODemo(object):
    # COCO categories for pretty print
    CATEGORIES = [
        "__background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    def __init__(
        self,
        cfg,
        confidence_threshold=0.7,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=224,
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size
        self.predict_type = cfg.MODEL.ALIGN.PREDICTOR

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        transform = T.Compose(
            [
                T.ToPILImage(),
                Resize(min_size, max_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self, image):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)

        result = image.copy()
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)
        # result = self.overlay_boxes(result, top_predictions)
        #if self.cfg.MODEL.MASK_ON:
        #    result = self.overlay_mask(result, top_predictions)
        if self.cfg.MODEL.KEYPOINT_ON:
            result = self.overlay_keypoints(result, top_predictions)
        # result = self.overlay_offsets(result, top_predictions)
        result = self.overlay_bezier(result, top_predictions)
        result = self.overlay_class_names(result, top_predictions, self.predict_type)

        return result

    def compute_prediction(self, original_image, measure_model=False):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            if measure_model:
                predictions, _, _ = measure_model(self.model, image_list)
            else:
                predictions = self.model(image_list, vis=True)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        test_size = prediction.size
        prediction = prediction.resize((width, height))

        # resize beziers
        beziers = prediction.get_field("beziers")
        beziers = BEZIER(beziers, test_size)
        beziers = beziers.resize((width, height))
        prediction.add_field("beziers", beziers.bbox)

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )

        return image

    def overlay_mask(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")

        colors = self.compute_colors_for_labels(labels).tolist()

        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None]
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.drawContours(image, contours, -1, color, 3)

        composite = image

        return composite

    def overlay_keypoints(self, image, predictions):
        keypoints = predictions.get_field("keypoints")
        kps = keypoints.keypoints
        scores = keypoints.get_field("logits")
        kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).numpy()
        for region in kps:
            image = vis_keypoints(image, region.transpose((1, 0)))
        return image

    def overlay_offsets(self, image, predictions):
        if not predictions.has_field("offsets"):
            return image
        offsets = predictions.get_field("offsets")
        masks = predictions.get_field("rec_masks")
        locations = predictions.get_field("locations")
        N, K = offsets.size()
        if N < 1: return image
        num_offsets = K // 2
        offsets = offsets.reshape(N, num_offsets, 2)
        for offset, mask, location in zip(offsets, masks, locations):
            image = vis_offsets(image, offset, mask, location)
        return image

    def overlay_bezier(self, image, predictions):
        """
        Adds the predicted bezier on top of the image
        """
        labels = predictions.get_field("labels")
        # print('predictions', predictions.fields())
        beziers = predictions.get_field("beziers")

        colors = self.compute_colors_for_labels(labels).tolist()

        Mtk = lambda n, t, k: t**k * (1-t)**(n-k) * n_over_k(n,k)
        BezierCoeff = lambda ts: [[Mtk(3,t,k) for k in range(4)] for t in ts]

        for bez_pts, color in zip(beziers, colors):
            bez_pts = bez_pts.to(torch.int64)
            assert(len(bez_pts) == 16), 'The numbr of bezier control points must be 8, but got {}'.format(len(bez_pts))
            s1_bezier = bez_pts[:8].reshape((4,2))[:, [1, 0]]
            s2_bezier = bez_pts[8:].reshape((4,2))[:, [1, 0]]
            tpbtp = tuple(map(tuple, np.int32(s1_bezier)))
            tpbbm = tuple(map(tuple, np.int32(s2_bezier)))
            t_plot = np.linspace(0, 1, 81)
            Bezier_top = np.array(BezierCoeff(t_plot)).dot(s1_bezier)
            Bezier_bottom = np.array(BezierCoeff(t_plot)).dot(s2_bezier)
            # use np.int32 for cv2 draw func
            image = cv2.polylines(image, np.int32([Bezier_top]), False, (0,0,255), 3)
            image = cv2.polylines(image, np.int32([Bezier_bottom]), False, (0,0,255), 3)
            image = cv2.line(image, tpbtp[0], tpbbm[0], (0,0,255), 3)
            image = cv2.line(image, tpbtp[3], tpbbm[3], (0,0,255), 3)
        
        return image

    def create_mask_montage(self, image, predictions):
        """
        Create a montage showing the probability heatmaps for each one one of the
        detected objects

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask`.
        """
        masks = predictions.get_field("mask")
        masks_per_dim = self.masks_per_dim
        masks = L.interpolate(
            masks.float(), scale_factor=1 / masks_per_dim
        ).byte()
        height, width = masks.shape[-2:]
        max_masks = masks_per_dim ** 2
        masks = masks[:max_masks]
        # handle case where we have less detections than max_masks
        if len(masks) < max_masks:
            masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
            masks_padded[: len(masks)] = masks
            masks = masks_padded
        masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
        result = torch.zeros(
            (masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
        )
        for y in range(masks_per_dim):
            start_y = y * height
            end_y = (y + 1) * height
            for x in range(masks_per_dim):
                start_x = x * width
                end_x = (x + 1) * width
                result[start_y:end_y, start_x:end_x] = masks[y, x]
        return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)

    def overlay_class_names(self, image, predictions, predict_type):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        recs = predictions.get_field("recs").tolist()
        labels = []
        if predict_type == 'ctc':
            last_char = False
            for rec in recs:
                s = ''
                for c in rec:
                    c = int(c)
                    if c < 95:
                        if last_char != c:
                            s += CTLABELS[c]
                            last_char = c
                    elif c == 95:
                        s += u'口'
                    else:
                        last_char = False
                labels.append(s)
        elif predict_type == 'attention':
            for rec in recs:
                s = ''
                for c in rec:
                    c = int(c)
                    if c < 95:
                        s += CTLABELS[c]
                    elif c == 95:
                        s += u'口'
                labels.append(s)
        else:
            raise('Unknown predict type: {}'.format(predict_type))

        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(label, score)
            text = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)
            # print('text', text)
            cv2.rectangle(image, (x-1,y+2), (x+text[0][0]+2, y-text[0][1]-2), (255,255,255), cv2.FILLED)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1
            )

        return image

import numpy as np
import matplotlib.pyplot as plt


def vis_offsets(img, offset, masks, location, thresh=0.5, alpha=0.7):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    n_tiles = offset.size(0)
    colors = [cmap(i) for i in np.linspace(0, 1, n_tiles + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    os_mask = np.copy(img)

    # Draw lines.
    for i, (point, mask) in enumerate(zip(offset, masks)):
        if mask < thresh:
            continue
        cv2.line(
            os_mask, (point[1] + location[0], point[0] + location[1]),
            (location[0], location[1]),
            color=colors[i], thickness=1, lineType=cv2.LINE_AA
        )

    # Blend the offsets.
    return cv2.addWeighted(img, 1.0 - alpha, os_mask, alpha, 0)
