# coding: utf-8
# ************************************
# Author: Ziqin Wang
# Email: ziqin.wang.edu@gmail.com
# Github: https://github.com/Storife
# ************************************

import argparse
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import numpy as np
import os
from RANet_lib import *
from RANet_lib.RANet_lib import *
from RANet_model import RANet as Net
import os
import os.path as osp
from glob import glob

net_name = 'RANet'
parser = argparse.ArgumentParser(description='RANet')
parser.add_argument('--deviceID', default=[0], help='device IDs')
parser.add_argument('--threads', type=int, default=16, help='number of threads for data loader to use')
parser.add_argument('--workfolder', default='../models/')
parser.add_argument('--savePName', default=net_name)
parser.add_argument('--net_type', default='single_object')
parser.add_argument('--fp16', default=True)
print('===> Setting ......')
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

try:
    os.mkdir(opt.workfolder)
    print('build working folder: ' + opt.workfolder)
except:
    print(opt.workfolder + 'exists')

# print(opt)
print('using device ID: {}'.format(opt.deviceID))

print('===> Building model')

model = Net(pretrained=False, type=opt.net_type)
model_cuda = None


def predict_SVOS(model_cuda=None, params='', add_name='', dataset='16val', save_root='./test/'):
    inSize1 = 480
    inSize2 = 864
    print('save root = ' + save_root)
    if dataset in ['16val', '16trainval', '16all']:
        model.set_type('single_object')
        year = '2016'
    elif dataset in ['17val', '17test_dev']:
        model.set_type('multi_object')
        year = '2017'
    else:
        assert('dataset error')

    DAVIS = dict(reading_type='SVOS',
                     year=year,
                 root='../datasets/DAVIS/',
                 subfolder=['', '', ''],
                 mode=dataset,
                 tar_mode='rep',
                 train=0, val=0, test=0, predict=1,
                 length=None,
                 init_folder=None,
                 )
    dataset = DAVIS2017_loader(
        [DAVIS], mode='test',
        transform=[PAD_transform([inSize1, inSize2], random=False),
                   PAD_transform([inSize1, inSize2], random=False)],
        rand=Rand_num())
    checkpoint_load(opt.workfolder + params, model)

    if opt.deviceID==[0]:
        model_cuda = model.cuda()
    else:
        model_cuda = nn.DataParallel(model).cuda()
    if opt.fp16:
        model_cuda = model_cuda.half()
        model_cuda.fp16 = True

    fitpredict17(dataset, model_cuda, add_name=add_name, threads=1, batchSize=1, save_root=save_root)


if __name__ == '__main__':

    predict_SVOS(params='RANet_video_single.pth', dataset='16val', save_root='../predictions/RANet_Video_16val')

    # predict_SVOS(params='RANet_image_single.pth', dataset='16all', save_root='../predictions/RANet_Image_16all')

    # predict_SVOS(params='RANet_video_multi.pth', dataset='17val', save_root='../predictions/RANet_Video_17val')

    # predict_SVOS(params='RANet_video_multi.pth', dataset='17test_dev', save_root='../predictions/RANet_Video_17test_dev')




