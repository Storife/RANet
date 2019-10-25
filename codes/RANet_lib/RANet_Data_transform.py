# ************************************
# Author: Ziqin Wang
# Email: ziqin.wang.edu@gmail.com
# Github: https://github.com/Storife
# ************************************
from os.path import exists, join, basename
from os import makedirs, remove
from PIL import Image, ImageOps
import random
from torchvision.transforms import CenterCrop, ToTensor, Scale, RandomCrop, RandomHorizontalFlip, RandomSizedCrop, Normalize
from torchvision.transforms import Compose as Compose_source
from torchvision.transforms import functional as F
import numpy as np
import cv2, math
import collections


class Compose(Compose_source):
    def __init__(self, transforms):
        super(Compose, self).__init__(transforms=transforms)


    def __call__(self, img, **kwargs):
        for t in self.transforms:
            img = t(img, **kwargs)
        return img

class Pad_to_size_list(object):

    def __init__(self, out_size):
        self.out_size = np.asarray([out_size[1], out_size[0]], dtype='float32')

    def __call__(self, sample, **kwargs):
        for idx, img in enumerate(sample):
            size = np.asarray(img.size, dtype='float32')
            p = self.out_size[1] / self.out_size[0]
            size = size / [1, p]
            ms = max(size)
            t_size = np.floor(size / ms * self.out_size[0])
            t_size = np.asanyarray(t_size * [1, p] + t_size % 2, dtype='int32')
            img = img.resize(t_size)
            sample[idx] =  img.crop(((t_size[0] - self.out_size[0])/2,
                             (t_size[1] - self.out_size[1])/2, (t_size[0] - self.out_size[0])/2 + self.out_size[0], (t_size[1] - self.out_size[1])/2 + self.out_size[1]))
        return sample

class ToTensor_list(object):
    def __init__(self):
        self.totennsor = ToTensor()
    def __call__(self, sample, **kwargs):
        for idx, tmp in enumerate(sample):
            sample[idx] = self.totennsor(tmp)
        return sample

class Rotate_list(object):
    def __init__(self, rots=(-30, 30), scales=False):
        assert (isinstance(rots, type(scales)))
        self.rots = rots
        self.scales = scales

    def __call__(self, sample, **kwargs):
        isrot = random.random() < 0.7
        rot = (self.rots[1] - self.rots[0]) * random.random() - \
              (self.rots[1] - self.rots[0])/2
        if isrot:
            for idx, tmp in enumerate(sample):
                sample[idx] = tmp.rotate(rot)
        return sample

class Resize_list(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample, **kwargs):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        # return F.resize(img, self.size, self.interpolation)
        for idx, tmp in enumerate(sample):
            sample[idx] = F.resize(tmp, self.size, self.interpolation)
        return sample

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class Normalize_adapt_list(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.norm = Normalize(mean, std)

    def __call__(self, sample, **kwargs):
        try:
            norm = kwargs['norm']
            norm.extend([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
        except:
            norm = None
        for idx, tmp in enumerate(sample):
            if tmp.size()[0] == 1:
                sample[idx] = tmp
            else:
                if norm == None or norm[idx]:
                    sample[idx] = self.norm(tmp)
        return sample

class Rand_num(object):
    def __call__(self, num=1, *args, **kwargs):
        return np.asarray([random.random() for _ in range(num)])




def PAD_transform(outSize, random=False):
    if random == False:
        return Compose([
            Pad_to_size_list(outSize),
            ToTensor_list(),
            Normalize_adapt_list(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])

