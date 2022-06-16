import glob
import os

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def get_transforms(
    img_size: int,
    mean: tuple[float, ...] = (0.5, 0.5, 0.5),
    std: tuple[float, ...] = (0.5, 0.5, 0.5),
    additional_targets: dict[str, str] = None,
) -> A.Compose:

    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
        ],
        # See: https://albumentations.ai/docs/examples/example_multi_target/
        additional_targets=additional_targets,
    )


def get_test_transforms(
    img_size: int,
    mean: tuple[float, ...] = (0.5, 0.5, 0.5),
    std: tuple[float, ...] = (0.5, 0.5, 0.5),
) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
        ],
    )


class MonetDataset(Dataset):
    def __init__(
        self,
        monet_dir: str,
        photo_dir: str,
        transform: A.Compose = None,
        output_keys: tuple[str, str] = ("monet_img", "image"),
    ) -> None:
        # Set the directory path for Monet images
        self.monet_dir = monet_dir
        # Set the directory path for normal images
        self.img_dir = photo_dir

        self.transform = transform

        # Get a list of paths of all Monet and normal images
        self.monet_paths = glob.glob(os.path.join(self.monet_dir, "*.jpg"))
        self.img_paths = glob.glob(os.path.join(self.img_dir, "*.jpg"))

        # Get the number of Monet and normal images
        self.n_monet = len(self.monet_paths)
        self.n_img = len(self.img_paths)

        # The length of the dataset is the maximum of the two
        self.data_len = max(self.n_monet, self.n_img)

        self.output_keys = output_keys

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        monet_key, image_key = self.output_keys

        # Modulus is required so that idx cycles back if idx > n_monet or n_imgs
        # Both are modulo just so that it can handle the case n_monet > n_imgs
        imgs = {
            monet_key: self.monet_paths[idx % self.n_monet],
            image_key: self.img_paths[idx % self.n_img],
        }

        # Load the images as NumPy arrays
        for k, img in imgs.items():
            img = np.array(Image.open(img))
            imgs[k] = img

        # Apply augmentations if required
        if self.transform is not None:
            imgs = self.transform(**imgs)

        # Move channel to front and convert to tensor
        for k, img in imgs.items():
            img = np.moveaxis(img, -1, 0)
            imgs[k] = torch.tensor(img, dtype=torch.float32)

        return imgs
