# ************************************
# Author: Ziqin Wang
# Email: ziqin.wang.edu@gmail.com
# Github: https://github.com/Storife
# ************************************
import torch.utils.data as data
import torch
from os import listdir
from os.path import join
from PIL import Image
from glob import glob
import numpy as np
from PIL import Image, ImageChops, ImageOps
from torchvision.transforms import RandomCrop, CenterCrop
import math
import random

Dataset_trans = dict(train=(1, 0, 0),
                     val=(0, 1, 0),
                     test=(0, 0, 1),
                     predict=(0, 0, 1))
Mode_trans = dict(train='train',
                  valid='valid',
                  test='test',
                  predict='valid',
                  online_train='train',
                  online_val='train',
                  online_all='train')


def Data_combinePicNameList(data_list):
    dataset = data_list[0]
    for idx in range(1, len(data_list)):
        for xnum in range(len(data_list[idx]['X_train'])):
            dataset['X_train'][xnum] = dataset['X_train'][xnum] + data_list[idx]['X_train'][xnum]
        for xnum in range(len(data_list[idx]['X_valid'])):
            dataset['X_valid'][xnum] = dataset['X_valid'][xnum] + data_list[idx]['X_valid'][xnum]
        for xnum in range(len(data_list[idx]['X_test'])):
            dataset['X_test'][xnum] = dataset['X_test'][xnum] + data_list[idx]['X_test'][xnum]
        for xnum in range(len(data_list[idx]['y_train'])):
            dataset['y_train'][xnum] = dataset['y_train'][xnum] + data_list[idx]['y_train'][xnum]
        for xnum in range(len(data_list[idx]['y_valid'])):
            dataset['y_valid'][xnum] = dataset['y_valid'][xnum] + data_list[idx]['y_valid'][xnum]
        for xnum in range(min(len(dataset['y_test']), len(data_list[idx]['y_test']))):
            dataset['y_test'][xnum] = dataset['y_test'][xnum] + data_list[idx]['y_test'][xnum]
    return dataset

class DAVIS2017_loader(data.Dataset):
    def __init__(self, Datasets_params, mode,  transform=None, input_transform=None, target_transform=None, randstep=5, rand=None, in_type=None):
        super(DAVIS2017_loader, self).__init__()
        self.iter_mode = mode
        self.randstep = randstep
        self.Datasets_params = Datasets_params
        self.reading_type = Datasets_params[0]['reading_type']
        self.num_objects = []
        datasets = []
        X_train = []
        y_train = []
        X_val = []
        y_val = []
        X_test = []
        y_test = []

        for DP in Datasets_params:
            X = []
            Y = []
            self.root = DP['root']
            if DP['reading_type'] in ['SVOS', 'SVOS-YTB']:
                self.years = DP['year']
                if DP['mode'] in ['test', '16val', '17val', 'YTB18']:
                    Set = '/val.txt'
                elif DP['mode'] in ['16all']:
                    Set = '/trainval.txt'
                elif DP['mode'] in ['test_dev', '17test_dev']:
                    Set = '/test-dev.txt'
                with open(self.root + 'ImageSets/' + self.years + Set) as f:
                    SetsTxts = f.readlines()
                # if DP['mode'] in ['all', 'online_all']:
                #     with open(self.root + 'ImageSets/' + self.years + '/val.txt') as f:
                #         SetsTxts2 = f.readlines()
                #     SetsTxts = SetsTxts + SetsTxts2
                Dirs = [self.root + 'JPEGImages/480p/' + name[0:-1] for name in SetsTxts]
                Dirs.sort()
                for dir in Dirs:
                    files = glob(dir + '/*.*')
                    files.sort()
                    if self.iter_mode == 'test':
                        X.append(files)
                        if DP['tar_mode'] == 'find':
                            Y_files = glob(dir.replace('JPEGImages', 'Annotations') + '/*.*')
                            if len(Y_files) == 0:
                                print(dir + 'Not find')
                        else:
                            Y_files = [f.replace('.jpg', '.png').replace('JPEGImages', 'Annotations').replace('.bmp', '.png') for f
                            in files]
                        Y_files.sort()
                        Y.append(Y_files)
                    else:
                        assert('error')
                    if DP['reading_type'] != 'SVOS-YTB':
                        _mask = np.array(Image.open(Y_files[0]).convert("P"))
                        self.num_objects.append(np.max(_mask))
                if DP['mode'] == 'train':
                    X_train = X
                    y_train = Y
                elif DP['mode'] in ['test', 'all', 'test_dev', '17test_dev', '16val', '17val', '16all', 'YTB18']:
                    X_test = X
                    y_test = Y
                datasets.append(
                    dict(X_train=[X_train], y_train=[y_train], X_valid=[X_val], y_valid=[y_val], X_test=[X_test],
                         y_test=[y_test]))
        self.image_filenames = Data_combinePicNameList(datasets)
        self.transform = transform
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.centerCrop = CenterCrop((480, 864))
        self.random_crop = RandomCrop((512, 960))
        self.rand = rand
        self.in_type = in_type
        self.idx_0 = 0

    def P2msks(self, Img, objs_ids):
        img = np.array(Img)
        Imgs = []
        for idx in objs_ids:
            Imgs.append(Image.fromarray((img == idx) * 255.0).convert('L'))
        return Imgs

    def msks2P(self, msks, objs_ids):
        # if max_num == 1:
        #     return msks[0]
        if len(msks) != len(objs_ids):
            print('error, len(msks) != len(objs_ids)')
        P = torch.zeros(msks[0].size())
        for idx, msk in enumerate(msks):
            ids = torch.nonzero(msk)
            if len(ids) > 0:
                P[ids[:, 0], ids[:, 1], ids[:, 2]] = idx + 1
        return P

    def msks2P_gen(self, msks, bg=0):
        P = torch.zeros(msks[0].size()) + bg
        for idx, msk in enumerate(msks):
            ids = torch.nonzero(msk)
            if len(ids) > 0:
                P[ids[:, 0], ids[:, 1], ids[:, 2]] = idx + 1 + bg
        return P

    def get_len(self):
        return len(self.image_filenames['X_' + Mode_trans[self.iter_mode]][0])

    def __getitem__(self, index):
        image_names = self.image_filenames['X_' + Mode_trans[self.iter_mode]]
        inputs = []
        names = []
        targets = []
        Names = image_names[0]
        size = len(Names[index])
        image_names = image_names[0]
        image_names_y = self.image_filenames['y_' + Mode_trans[self.iter_mode]][0]
        Imgs = []
        Masks = []
        Img_sizes = []
        for idx in range(len(image_names[index])):
            img = Image.open(image_names[index][idx]).convert('RGB')
            Img_sizes.append(list(img.size))
            try:
                label = Image.open(image_names_y[index][idx])
            except:
                label = Image.open(image_names_y[index][0])
            objs_ids = list(set(np.asarray(label).reshape(-1)))
            if label.mode == 'P':
                images = self.transform[1]([img] + self.P2msks(label, objs_ids), norm=[1, 0])  # img_t1.cnp
                img = images[0]
                label = self.msks2P(images[1::], objs_ids)
            else:
                img, label = self.transform[0]([img, label], norm=[1, 0])
            Imgs.append(img)
            Masks.append(label)
        return Imgs, Masks, image_names[index], Img_sizes
        

    def __len__(self):
        return len(self.image_filenames['X_' + Mode_trans[self.iter_mode]][0])











