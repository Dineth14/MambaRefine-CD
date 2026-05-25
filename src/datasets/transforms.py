"""Synchronized transforms for image A, image B, and mask."""
from __future__ import annotations

import random

from PIL import ImageEnhance
from src.utils.torchvision_compat import patch_register_fake

patch_register_fake()
import torchvision.transforms.functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CDTrainTransform:
    def __init__(self, image_size: int) -> None:
        self.image_size = int(image_size)

    def _color_jitter(self, image):
        brightness = 1.0 + random.uniform(-0.2, 0.2)
        contrast = 1.0 + random.uniform(-0.2, 0.2)
        saturation = 1.0 + random.uniform(-0.2, 0.2)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Color(image).enhance(saturation)
        return image

    def __call__(self, image_a, image_b, mask):
        size = [self.image_size, self.image_size]
        image_a = TF.resize(image_a, size)
        image_b = TF.resize(image_b, size)
        mask = TF.resize(mask, size, interpolation=TF.InterpolationMode.NEAREST)

        if random.random() < 0.5:
            image_a = TF.hflip(image_a)
            image_b = TF.hflip(image_b)
            mask = TF.hflip(mask)
        if random.random() < 0.5:
            image_a = TF.vflip(image_a)
            image_b = TF.vflip(image_b)
            mask = TF.vflip(mask)
        k = random.randint(0, 3)
        if k:
            angle = 90 * k
            image_a = TF.rotate(image_a, angle)
            image_b = TF.rotate(image_b, angle)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        image_a = self._color_jitter(image_a)
        image_b = self._color_jitter(image_b)
        image_a = TF.normalize(TF.to_tensor(image_a), IMAGENET_MEAN, IMAGENET_STD)
        image_b = TF.normalize(TF.to_tensor(image_b), IMAGENET_MEAN, IMAGENET_STD)
        mask = TF.to_tensor(mask)
        return image_a, image_b, mask


class CDEvalTransform:
    def __init__(self, image_size: int) -> None:
        self.image_size = int(image_size)

    def __call__(self, image_a, image_b, mask):
        size = [self.image_size, self.image_size]
        image_a = TF.normalize(TF.to_tensor(TF.resize(image_a, size)), IMAGENET_MEAN, IMAGENET_STD)
        image_b = TF.normalize(TF.to_tensor(TF.resize(image_b, size)), IMAGENET_MEAN, IMAGENET_STD)
        mask = TF.to_tensor(TF.resize(mask, size, interpolation=TF.InterpolationMode.NEAREST))
        return image_a, image_b, mask


def get_train_transform(cfg):
    return CDTrainTransform(cfg.data.image_size)


def get_val_transform(cfg):
    return CDEvalTransform(cfg.data.image_size)
