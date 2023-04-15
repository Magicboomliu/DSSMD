from __future__ import division
from genericpath import samefile
import torch
import numpy as np
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as F
import random
import matplotlib.pyplot as plt
import cv2

class DataAugmentation(object):
    def __init__(self):
        self.random_brightness = np.random.uniform(0.8,1.2)
        self.random_contrast = np.random.uniform(0.8,1.2)
        self.random_gamma = np.random.uniform(0.8,1.2)
        self.rng = np.random.RandomState(0)
        self.min_scale = 0.8
        self.max_scale = 1.5
    
    def chromatic_augmentation(self,img):
        
        img = Image.fromarray(np.uint8(img))
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(self.random_brightness)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.random_contrast)

        gamma_map = [
            255 * 1.0 * pow(ele / 255.0, self.random_gamma) for ele in range(256)
        ] * 3
        img = img.point(gamma_map)  # use PIL's point-function to accelerate this part
        img_ = np.array(img).astype(np.float32)
        
        return img_
        
    def __call__(self, sample):
        
        # (1)chromatic augmentation
        sample['img_left'] = self.chromatic_augmentation(sample['img_left'])

        
        # 2.2) random resize
        resize_scale = self.rng.uniform(self.min_scale, self.max_scale)
        
        sample['img_left'] = cv2.resize(
            sample['img_left'],
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        
        return sample
      
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

class ToTensor(object):
    """Convert numpy array to torch tensor"""

    def __call__(self, sample):
        left = np.transpose(sample['img_left'], (2, 0, 1))  # [3, H, W]
        sample['img_left'] = torch.from_numpy(left) / 255.
        sample['label'] = torch.from_numpy(sample['label'])
        return sample

class Normalize(object):
    """Normalize image, with type tensor"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        norm_keys = ['img_left']

        for key in norm_keys:
            # Images have converted to tensor, with shape [C, H, W]
            for t, m, s in zip(sample[key], self.mean, self.std):
                t.sub_(m).div_(s)

        return sample

class RandomCrop(object):
    def __init__(self, img_height, img_width, validate=False):
        self.img_height = img_height
        self.img_width = img_width
        self.validate = validate

    def __call__(self, sample):
        

        ori_height, ori_width = sample['img_left'].shape[:2]
        if self.img_height > ori_height or self.img_width > ori_width:
            top_pad = self.img_height - ori_height
            right_pad = self.img_width - ori_width

            assert top_pad >= 0 and right_pad >= 0

            sample['img_left'] = np.lib.pad(sample['img_left'],
                                        ((top_pad, 0), (0, right_pad), (0, 0)),
                                        mode='constant',
                                        constant_values=0)
            
        else:
            assert self.img_height <= ori_height and self.img_width <= ori_width

            # Training: random crop
            if not self.validate:
                self.offset_x = np.random.randint(ori_width - self.img_width + 1)
                start_height = 0
                assert ori_height - start_height >= self.img_height
                self.offset_y = np.random.randint(start_height, ori_height - self.img_height + 1)

            # Validatoin, center crop
            else:
                self.offset_x = (ori_width - self.img_width) // 2
                self.offset_y = (ori_height - self.img_height) // 2
            sample['img_left'] = self.crop_img(sample['img_left'])
        
        return sample

    def crop_img(self, img):
        return img[self.offset_y:self.offset_y + self.img_height,
               self.offset_x:self.offset_x + self.img_width]

class RandomVerticalFlip(object):
    """Randomly vertically filps"""

    def __call__(self, sample):
        if np.random.random() < 0.09:
            sample['img_left'] = np.copy(np.flipud(sample['img_left']))
        return sample

class ToPILImage(object):

    def __call__(self, sample):
        sample['img_left'] = Image.fromarray(sample['img_left'].astype('uint8'))
        return sample

class ToNumpyArray(object):

    def __call__(self, sample):
        sample['img_left'] = np.array(sample['img_left']).astype(np.float32)
        return sample

# Random coloring
class RandomContrast(object):
    """Random contrast"""
    def __call__(self, sample):
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)
            sample['img_left'] = F.adjust_contrast(sample['img_left'], contrast_factor)
        return sample

class RandomGamma(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            gamma = np.random.uniform(0.7, 1.5)  # adopted from FlowNet
            sample['img_left'] = F.adjust_gamma(sample['img_left'], gamma)
        return sample

class RandomBrightness(object):
    def __call__(self, sample):
        if np.random.random() < 0.5:
            brightness = np.random.uniform(0.5, 2.0)
            sample['img_left'] = F.adjust_brightness(sample['img_left'], brightness)
        return sample

class RandomHue(object):
    def __call__(self, sample):
        if np.random.random() < 0.5:
            hue = np.random.uniform(-0.1, 0.1)
            sample['img_left'] = F.adjust_hue(sample['img_left'], hue)
        return sample

class RandomSaturation(object):
    def __call__(self, sample):
        if np.random.random() < 0.5:
            saturation = np.random.uniform(0.8, 1.2)
            sample['img_left'] = F.adjust_saturation(sample['img_left'], saturation)
        return sample

class RandomColor(object):
    def __call__(self, sample):
        transforms = [RandomContrast(),
                      RandomGamma(),
                      RandomBrightness(),
                      RandomHue(),
                      RandomSaturation()]

        sample = ToPILImage()(sample)

        if np.random.random() < 0.5:
            t = random.choice(transforms)
            sample = t(sample)
        else:
            random.shuffle(transforms)
            for t in transforms:
                sample = t(sample)

        sample = ToNumpyArray()(sample)

        return sample