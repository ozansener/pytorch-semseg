# Adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py

import math
import numbers
import random
import numpy as np

from PIL import Image, ImageOps

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask, ins=None):
        if ins is not None:
            img, mask, ins = Image.fromarray(img, mode='RGB'), Image.fromarray(mask, mode='L'), Image.fromarray(ins, mode='I')
            assert img.size == mask.size
            assert img.size == ins.size
        else:
            img, mask = Image.fromarray(img, mode='RGB'), Image.fromarray(mask, mode='L')            
            assert img.size == mask.size

        for a in self.augmentations:
            img, mask, ins = a(img, mask, ins)

        if ins is not None:
            return np.array(img), np.array(mask, dtype=np.uint8), np.array(ins, dtype=np.uint64)
        else:
            return np.array(img), np.array(mask, dtype=np.uint8), None


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask, ins=None):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)
            if ins:
                ins = ImageOps.expand(ins, border=self.padding, fill=0)

        assert img.size == mask.size
        if ins:
            assert img.size == ins.size

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            if ins is not None:
                return img, mask, ins
            else:
                return img, mask
        if w < tw or h < th:
            if ins is not None:
                return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST), ins.resize((tw,th), Image.NEAREST)
            else:
                return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST), None

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        if ins is not None:
            return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)), ins.crop((x1, y1, x1 + tw, y1 + th))
        else:
            return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)), None


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, ins=None):
        if random.random() < 0.5:
            if ins is not None:
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), ins.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), None
        return img, mask, ins


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask, ins=None):
        assert img.size == mask.size
        if ins:
            assert img.size == ins.size
        if ins:
            return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST), ins.resize(self.size, Image.NEAREST)
        else:
            return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST), None


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask, ins=None):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            if ins is not None:
                return img, mask, ins
            else:
                return img, mask, None
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            if ins is not None:
                return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST), ins.resize((ow, oh), Image.NEAREST)
            else:
                return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST), None
        else:
            oh = self.size
            ow = int(self.size * w / h)
            if ins:
                return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST), ins.resize((ow,oh), Image.NEAREST)
            else:
                return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST), None


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), Image.BILINEAR), mask.resize((self.size, self.size),
                                                                                       Image.NEAREST)

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask, ins = None):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        if ins:
            return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST), ins.rotate(rotate_degree, Image.NEAREST)
        else:
            return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST), None


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size

        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return self.crop(*self.scale(img, mask))