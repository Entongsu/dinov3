# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os
from enum import Enum
from typing import Any, Callable, Optional, Tuple, Union
import sys
from PIL import Image
from torchvision.datasets import VisionDataset
import numpy as np

sys.path.append('/home/ensu/Documents/weird/IsaacLab/submodule/dinov3')


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"

    @property
    def dirname(self) -> str:
        return {
            _Split.TRAIN: "training",
            _Split.VAL: "validation",
        }[self]


def list_image_files(root_dir,
                     extensions=(".png", ".jpg", ".jpeg", ".bmp", ".gif")):
    """Recursively collect image file paths."""
    image_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(extensions):
                image_files.append(os.path.join(root, file))
    return image_files


class IsaacDataset(VisionDataset):

    def __init__(
        self,
        split: _Split,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root=root,
                         transforms=transforms,
                         transform=transform,
                         target_transform=target_transform)

        self.split = split
        self.image_paths = list_image_files(root)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in: {root}")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # Load image
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")

        target = None  # No labels or targets currently available

        # Apply transforms if defined
        if self.transforms:
            image, target = self.transforms(image, target)
        elif self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        return len(self.image_paths)


from dinov3.data.datasets.isaac_dataset import IsaacDataset
from dinov3.data.augmentations import DataAugmentationDINO  # path to your augmentation script
import cv2
if __name__ == "__main__":
    # --- 1. Define augmentation ---
    dino_aug = DataAugmentationDINO(
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=4,
        global_crops_size=224,
        local_crops_size=96,
        gram_teacher_crops_size=None,
        share_color_jitter=False,
    )

    # --- 2. Create dataset (without torchvision transforms) ---
    dataset = IsaacDataset(
        split=None,
        root="/home/ensu/Documents/weird/IsaacLab/logs/trash/sort_image")

    print(f"Dataset length: {len(dataset)}")

    # --- 3. Load an image and apply DINOv3 augmentations manually ---
    pil_img, _ = dataset[0]
    aug_out = dino_aug(pil_img)

    # --- 4. Visualize all crops with OpenCV ---
    def show_tensor(t):
        """Convert tensor -> numpy -> show via cv2"""
        img = t.permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    # Global crops
    for i, crop in enumerate(aug_out["global_crops"]):
        cv2.imshow(f"Global Crop {i+1}", show_tensor(crop))
        cv2.waitKey(0)

    # Local crops
    for i, crop in enumerate(aug_out["local_crops"]):
        cv2.imshow(f"Local Crop {i+1}", show_tensor(crop))
        cv2.waitKey(0)

    cv2.destroyAllWindows()
