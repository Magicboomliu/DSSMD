from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import os
import sys
sys.path.append("..")
from utils.utils import read_text_lines
from utils.file_io import read_disp,read_img
from skimage import io, transform
import numpy as np

class StereoDataset(Dataset):
    def __init__(self, data_dir,
                 train_datalist,
                 test_datalist,
                 dataset_name='SceneFlow',
                 mode='train',
                 save_filename=False,
                 load_pseudo_gt=False,
                 transform=None):
        super(StereoDataset, self).__init__()

        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.mode = mode
        self.save_filename = save_filename
        self.transform = transform
        self.train_datalist = train_datalist
        self.test_datalist = test_datalist
        self.img_size=(540, 960)
        self.scale_size =(576,960)
        

        sceneflow_finalpass_dict = {
            'train':  self.train_datalist,
            'val':    self.test_datalist,
            'test':   self.test_datalist 
        }

        kitti_2012_dict = {
            'train': 'filenames/KITTI_2012_train.txt',
            'train_all': 'filenames/KITTI_2012_train_all.txt',
            'val': 'filenames/KITTI_2012_val.txt',
            'test': 'filenames/KITTI_2012_test.txt'
        }

        kitti_2015_dict = {
            'train': 'filenames/KITTI_2015_train.txt',
            'train_all': 'filenames/KITTI_2015_train_all.txt',
            'val': 'filenames/KITTI_2015_val.txt',
            'test': 'filenames/KITTI_2015_test.txt'
        }

        kitti_mix_dict = {
            'train': 'filenames/KITTI_mix.txt',
            'test': 'filenames/KITTI_2015_test.txt'
        }

        dataset_name_dict = {
            'SceneFlow': sceneflow_finalpass_dict,
            'KITTI2012': kitti_2012_dict,
            'KITTI2015': kitti_2015_dict,
            'KITTI_mix': kitti_mix_dict,
        }

        assert dataset_name in dataset_name_dict.keys()
        self.dataset_name = dataset_name

        self.samples = []

        data_filenames = dataset_name_dict[dataset_name][mode]

        lines = read_text_lines(data_filenames)

        for line in lines:
            splits = line.split()
            left_img = splits[0]
            label = splits[1]
            sample = dict()

            if self.save_filename:
                sample['left_name'] = left_img.split('/', 1)[1]

            sample['left'] = os.path.join(data_dir, left_img)
            sample['label'] = label

            self.samples.append(sample)

    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]

        if self.save_filename:
            sample['left_name'] = sample_path['left_name']

        sample['img_left'] = read_img(sample_path['left'])  # [H, W, 3]
        sample['label'] =  np.array(int(sample_path['label']))
    
        if self.mode=='test' or self.mode=='val':
            img_left = sample['img_left']
            img_left = img_left.astype(np.float32)
            sample['img_left'] = img_left

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)
    
    def get_img_size(self):
        return self.img_size

    def get_scale_size(self):
        return self.scale_size