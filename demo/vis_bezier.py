# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2
import torch
import numpy as np

np.random.seed(2)
torch.manual_seed(3)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from predictor import COCODemo
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="configs/pointmask/R-50_def_2x2_c.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=800,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    data_loaders = make_data_loader(cfg, is_train=False, is_distributed=False)
    for data_loader in data_loaders:
        for i, batch in tqdm(enumerate(data_loader)):
            images = batch[0].tensors.permute(0, 2, 3, 1)
            gt = batch[1][0]
            assert images.size(0) == 1

            image = images[0] + torch.tensor(cfg.INPUT.PIXEL_MEAN)
            img = np.ascontiguousarray(np.uint8(image.numpy()))

            composite = coco_demo.run_on_opencv_image(img)

            if composite.shape[-1] != 28:
                cv2.imshow("COCO detections", composite)
                if cv2.waitKey(0) == 27:
                    break  # esc to quit

if __name__ == "__main__":
    main()