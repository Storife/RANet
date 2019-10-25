# ************************************
# Author: Ziqin Wang
# Email: ziqin.wang.edu@gmail.com
# Github: https://github.com/Storife
# ************************************
import torch
import torch.nn as nn
import numpy as np
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt
from torch.nn import functional as f
from torch.autograd import Variable
import random
from torch.nn import DataParallel as DP
from RANet_lib.RANet_Model_imagenet import *
import time

def make_layer2(input_feature, out_feature, up_scale=1, ksize=3, d=1, groups=1):
    p = int((ksize - 1) / 2)
    if up_scale == 1:
        return nn.Sequential(
        nn.InstanceNorm2d(input_feature),
        nn.ReLU(),
        nn.Conv2d(input_feature, out_feature, ksize, padding=p, dilation=d, groups=groups),
    )
    return nn.Sequential(
        nn.InstanceNorm2d(input_feature),
        nn.ReLU(),
        nn.Conv2d(input_feature, out_feature, ksize, padding=p),
        nn.UpsamplingBilinear2d(scale_factor=up_scale),
    )

class ResBlock2(nn.Module):
    def __init__(self, input_feature, planes, dilated=1, group=1):
        super(ResBlock2, self).__init__()
        self.conv1 = nn.Conv2d(input_feature, planes, kernel_size=1, bias=False, groups=group)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1 * dilated, bias=False, dilation=dilated, groups=group)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, input_feature, kernel_size=1, bias=False, groups=group)
        self.bn3 = nn.InstanceNorm2d(input_feature)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

class ResBlock_f(nn.Module):
    def __init__(self, input_feature, planes, dilated=1, group=1):
        super(ResBlock2, self).__init__()
        self.dilated = dilated
        self.conv1 = nn.Conv2d(input_feature, planes, kernel_size=1, bias=False, groups=group)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False, groups=group)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, input_feature, kernel_size=1, bias=False, groups=group)
        self.bn3 = nn.InstanceNorm2d(input_feature)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = f.avg_pool2d(out, self.dilated)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class MS_Block(nn.Module):
    def __init__(self, input_feature, out_feature, d=[1, 2, 4], group=1):
        super(MS_Block, self).__init__()
        self.l1 = nn.Conv2d(input_feature, out_feature, 3, padding=d[0], dilation=d[0], bias=False, groups=group)
        self.l2 = nn.Conv2d(input_feature, out_feature, 3, padding=d[1], dilation=d[1], bias=False, groups=group)
        self.l3 = nn.Conv2d(input_feature, out_feature, 3, padding=d[2], dilation=d[2], bias=False, groups=group)
    def forward(self, x):
        out = self.l1(x) + self.l2(x) + self.l3(x)
        return out


class RANet(ResNet101):
    def __init__(self, with_relu=0, pretrained=True, type='single_object'):
        super(RANet, self).__init__(with_relu=with_relu, pretrained=pretrained)
        self.fp16 = False
        self.net_type = type
        self._init_net()
        self.p_1 = make_layer2(256, 256)
        self.res_1 = ResBlock2(256, 128, 1)
        self.p_2 = make_layer2(256, 128)

        self.p_1b = make_layer2(256, 256)
        self.res_1b = ResBlock2(256, 128, 1)
        self.p_2b = make_layer2(256, 128)
        self.ls13 = make_layer2(512, 32, up_scale=1, ksize=1)
        self.ls14 = make_layer2(1024, 16, up_scale=2, ksize=1)
        self.ls15 = make_layer2(2048, 16, up_scale=4, ksize=1)

        self.ls22 = make_layer2(256, 32, up_scale=1, ksize=1)
        self.ls23 = make_layer2(512, 16, up_scale=2, ksize=1)
        self.ls24 = make_layer2(1024, 16, up_scale=4, ksize=1)

        self.ls31 = make_layer2(64, 32, up_scale=1, ksize=1)
        self.ls32 = make_layer2(256, 16, up_scale=2, ksize=1)
        self.ls33 = make_layer2(512, 16, up_scale=4, ksize=1)

        self.R2 = nn.Sequential(make_layer2(128 + 64, 128),
                                make_layer2(128, 64),
                                MS_Block(64, 32, d=[1,3,6]),
                                ResBlock2(32, 16),
                                nn.UpsamplingBilinear2d(scale_factor=2))
        self.R3 = nn.Sequential(make_layer2(32 + 64, 64),
                                make_layer2(64, 32),
                                MS_Block(32, 16, d=[1,3,6]),
                                nn.UpsamplingBilinear2d(scale_factor=2),
                                ResBlock2(16, 8),
                                nn.Conv2d(16, 1, 3, padding=1)
                                )
        self.R1 = nn.Sequential(make_layer2(256 + 64 + 1, 256),
                                make_layer2(256, 256),
                                MS_Block(256, 128, d=[1,3,6]),
                                ResBlock2(128, 64),
                                nn.UpsamplingBilinear2d(scale_factor=2))
        self.L4 = make_layer2(1024, 256, ksize=3)
        self.L5 = make_layer2(2048, 512, ksize=3)
        self.L3 = make_layer2(512, 128, ksize=3)
        self.L_g = make_layer2(512 + 256 + 128, 512)
        self.rank_A = nn.Sequential(nn.Conv1d(2, 8, 1),
                                    nn.PReLU(),
                                    nn.Conv1d(8, 1, 1),
                                    nn.ReLU())
        self.Ranking = nn.Sequential(make_layer2(405, 128), ResBlock2(128, 32, 2), make_layer2(128, 1))
        # self.Ranking = nn.Sequential(nn.Conv2d(405, 16, 1),
        #                              nn.InstanceNorm2d(16), nn.ReLU(),
        #                              nn.Conv2d(16, 1, 1, bias=False), nn.ReLU())

    def Dtype(self, data):
        if self.fp16:
            return torch._C._TensorBase.half(data)
        else:
            return torch._C._TensorBase.float(data)

    def _init_net(self):
        if self.net_type == 'single_object':
            self.forward = self.RANet_Single_forward_eval
            print('Single-object mode')
        elif self.net_type == 'multi_object':
            self.forward = self.RANet_Multiple_forward_eval
            print('Multi-object mode')

    def set_type(self, type):
        if self.net_type != 'single_object' and type == 'single_object':
            self.forward = self.RANet_Single_forward_eval
            print('Change to single-object mode')
        elif self.net_type != 'multi_object' and type == 'multi_object':
            self.forward = self.RANet_Multiple_forward_eval
            print('Change to multi-object mode')
        else:
            pass

    def corr_fun(self, Kernel_tmp, Feature, KERs=None):
        size = Kernel_tmp.size()
        if len(Feature) == 1:
            Kernel = Kernel_tmp.view(size[1], size[2] * size[3]).transpose(0, 1)
            Kernel = Kernel.unsqueeze(2).unsqueeze(3)
            if not (type(KERs) == type(None)):
                Kernel = KERs[0]
            corr = torch.nn.functional.conv2d(Feature, Kernel.contiguous())
            Kernel = Kernel.unsqueeze(0)
        else:
            CORR = []
            Kernel = []
            for i in range(len(Feature)):
                ker = Kernel_tmp[i:i + 1]
                fea = Feature[i:i + 1]
                ker = ker.view(size[1], size[2] * size[3]).transpose(0, 1)
                ker = ker.unsqueeze(2).unsqueeze(3)
                if not (type(KERs) == type(None)):
                    ker = torch.cat([ker, KERs[i]], 0)
                co = f.conv2d(fea, ker.contiguous())
                CORR.append(co)
                ker = ker.unsqueeze(0)
                Kernel.append(ker)
            corr = torch.cat(CORR, 0)
            Kernel = torch.cat(Kernel, 0)
        return corr, Kernel

    def to_kernel(self, feature):
        size = feature.size()
        return feature.view(size[1], size[2] * size[3]).transpose(0, 1).unsqueeze(2).unsqueeze(3).contiguous()

    def correlate(self, Kernel, Feature):
        corr = torch.nn.functional.conv2d(Feature, Kernel,stride=1)
        return corr

    def P2masks(self, P, num):
        M = []
        M.append(self.Dtype((P == 0) + (P > int(num))))
        for idx in range(1, num + 1):
            M.append(self.Dtype(P == idx))
        return M

    def bbox_uncrop(img, bbox, size, crop_size):  # 4D input
        img = F.upsample_bilinear(img, size=crop_size[2::])
        msk = F.pad(img, (bbox[1], 864 - bbox[3], bbox[0], 480 - bbox[2],))
        return msk

    def RANet_Single_forward_eval(self, x1, Ker, msk2, msk_p, mode=''):  # vxd  feature * msk *2  _feature_Rf
        if mode in ['first', 'encoder']:
            # Exact template features
            x2 = Ker
            base_features2 = self.res_forward(x2)
            Kernel_3 = f.normalize(f.max_pool2d(self.L3(base_features2[2]), 2))
            Kernel_4 = f.normalize(self.L4(base_features2[3]))
            Kernel_5 = f.normalize(f.upsample(self.L5(base_features2[4]), scale_factor=2, mode='bilinear'))
            Kernel_tmp = f.normalize(self.L_g(torch.cat([Kernel_3, Kernel_4, Kernel_5], dim=1)))
            if mode == 'encoder':
                return [Kernel_tmp]
            Kernel_tmp = f.adaptive_avg_pool2d(Kernel_tmp, [15, 27])
            return [Kernel_tmp]
        if msk2.max() > 1:
            msk2 = self.Dtype(msk2.ge(1.6))
            msk_p = self.Dtype(msk_p.ge(1.6))
        # Current frame feature
        base_features1 = self.res_forward(x1)
        Feature_3 = f.normalize(f.max_pool2d(self.L3(base_features1[2]), 2))
        Feature_4 = f.normalize(self.L4(base_features1[3]))
        Feature_5 = f.normalize(f.upsample(self.L5(base_features1[4]), scale_factor=2, mode='bilinear'))
        Feature = f.normalize(self.L_g(torch.cat([Feature_3, Feature_4, Feature_5], dim=1)))

        '''
        Kernel_tmp = Ker
        m = f.adaptive_avg_pool2d(msk2.detach(), Kernel_tmp.size()[-2::])
        Kernel = Kernel_tmp * m.repeat(1, 512, 1, 1)
        mb = (1 - m).ge(0.9).float()
        Kernel_back = Kernel_tmp * mb.repeat(1, 512, 1, 1).float()
        corr, Kerner = self.corr_fun(Kernel, Feature)
        corr_b, Kerner_b = self.corr_fun(Kernel_back, Feature)
        '''
        # Correlation
        Kernel = Ker
        m = f.adaptive_avg_pool2d(msk2.detach(), Kernel.size()[-2::])
        mb = self.Dtype((1 - m).ge(0.9))
        h_size = 15
        w_size = 27
        c_size = h_size * w_size
        Correlation, a = self.corr_fun(Kernel, Feature)
        # Select FG / BG similarity maps
        corr = Correlation * m.view(-1, c_size, 1, 1)
        corr_b = Correlation * mb.view(-1, c_size, 1, 1)
        # Ranking attention scores

        T_corr = f.max_pool2d(corr, 2).permute(0, 2, 3, 1).view(-1, c_size, h_size, w_size)
        T_corr_b = f.max_pool2d(corr_b, 2).permute(0, 2, 3, 1).view(-1, c_size, h_size, w_size)
        R_map = (f.relu(self.Ranking(T_corr)) * self.Dtype(m != 0)).view(-1, 1, c_size) * 0.2
        R_map_b = (f.relu(self.Ranking(T_corr_b)) * mb).view(-1, 1, c_size) * 0.2

        # Rank & select
        co_size = corr.size()[2::]
        max_only, indices = f.max_pool2d(corr, co_size, return_indices=True)
        max_only = max_only.view(-1, 1, c_size) + R_map
        m_sorted, m_sorted_idx = max_only.sort(descending=True, dim=2)
        corr = torch.cat([co.index_select(0, m_sort[0, 0:256]).unsqueeze(0) for co, m_sort in zip(corr, m_sorted_idx)])
        # corr = corr[0].index_select(0, m_sorted_idx[0, 0, 0:256]).unsqueeze(0)
        max_only_b, indices = f.max_pool2d(corr_b, co_size, return_indices=True)
        max_only_b = max_only_b.view(-1, 1, c_size) + R_map_b
        m_sorted, m_sorted_idx = max_only_b.sort(descending=True, dim=2)
        corr_b = torch.cat([co.index_select(0, m_sort[0, 0:256]).unsqueeze(0) for co, m_sort in zip(corr_b, m_sorted_idx)])
        # corr_b = corr_b[0].index_select(0, m_sorted_idx[0, 0, 0:256]).unsqueeze(0)
        # Merge net
        fcorr = self.p_2(self.res_1(self.p_1(f.upsample(corr, scale_factor=2, mode='bilinear'))))
        fcorr_b = self.p_2(self.res_1(self.p_1(f.upsample(corr_b, scale_factor=2, mode='bilinear'))))

        # Decoder
        base1 = torch.cat([self.ls13(base_features1[2]),
                           self.ls14(base_features1[3]),
                           self.ls15(base_features1[4]),
                           fcorr,
                           fcorr_b,
                           f.adaptive_avg_pool2d(msk_p, fcorr.size()[-2::])], 1)
        fea1 = self.R1(base1)
        base2 = torch.cat([self.ls22(base_features1[1]),
                           self.ls23(base_features1[2]),
                           self.ls24(base_features1[3]),
                           fea1], 1)
        fea2 = self.R2(base2)
        base3 = torch.cat([self.ls31(base_features1[0]),
                           self.ls32(base_features1[1]),
                           self.ls33(base_features1[2]),
                           fea2], 1)
        fea3 = self.R3(base3)
        out_R = f.sigmoid(fea3)

        features = []
        out = [out_R]
        return out, features

    def RANet_Multiple_forward_eval(self, x1, Ker, msk2, msk_p, mode=''):  # vxd  feature * msk *2  _feature_Rf
        if mode == 'first':
            # Exact template features
            x2 = Ker
            base_features2 = self.res_forward(x2)
            Kernel_3 = f.normalize(f.max_pool2d(self.L3(base_features2[2]), 2))
            Kernel_4 = f.normalize(self.L4(base_features2[3]))
            Kernel_5 = f.normalize(f.upsample(self.L5(base_features2[4]), scale_factor=2, mode='bilinear'))
            Kernel_tmp = f.normalize(self.L_g(torch.cat([Kernel_3, Kernel_4, Kernel_5], dim=1)))
            Kernel_tmp = f.avg_pool2d(Kernel_tmp, 2)
            return [Kernel_tmp]
        # Current frame feature
        base_features1 = self.res_forward(x1)
        Feature_3 = f.normalize(f.max_pool2d(self.L3(base_features1[2]), 2))
        Feature_4 = f.normalize(self.L4(base_features1[3]))
        Feature_5 = f.normalize(f.upsample(self.L5(base_features1[4]), scale_factor=2, mode='bilinear'))
        Feature = f.normalize(self.L_g(torch.cat([Feature_3, Feature_4, Feature_5], dim=1)))

        Kernel_tmp = Ker
        Out_Rs = []

        basef1 = torch.cat([self.ls13(base_features1[2]),
                            self.ls14(base_features1[3]),
                            self.ls15(base_features1[4]), ], 1)
        basef2 = torch.cat([self.ls22(base_features1[1]),
                            self.ls23(base_features1[2]),
                            self.ls24(base_features1[3]), ], 1)
        basef3 = torch.cat([self.ls31(base_features1[0]),
                            self.ls32(base_features1[1]),
                            self.ls33(base_features1[2])], 1)

        for idx in range(len(Feature)):  # batch
            ker = Kernel_tmp[idx: idx + 1]
            feature = Feature[idx: idx + 1]
            m2 = msk2[idx: idx + 1]
            mp = msk_p[idx: idx + 1]
            max_obj = m2.max().int().data.cpu().numpy()
            if max_obj < 2:
                m2[0, 0, 0, 0] = 2
                max_obj = m2.max().int().data.cpu().numpy()
            M2s = self.P2masks(f.relu(m2 - 1), max_obj - 1)
            M2_all = m2.ge(1.5).float()
            Mps = self.P2masks(f.relu(mp - 1), max_obj - 1)
            Mp_all = mp.ge(1.5).float()

            # Correlation
            Corr_subs = []
            ker_R = self.to_kernel(ker)
            corr_R = self.correlate(ker_R, feature)

            # Ranking attention scores
            T_corr = f.max_pool2d(corr_R, 2).view(-1, 405, 405).transpose(1, 2).view(-1, 405, 15, 27)
            R_map = f.relu(self.Ranking(T_corr)) * 0.2
            Rmaps = []

            for idy in range(max_obj):  # make corrs (backgrounds(=1) and objs)
                m2_rep = f.adaptive_avg_pool2d(M2s[idy], ker.size()[-2::])
                corr_sub = m2_rep.view(m2_rep.size()[0], -1, 1, 1) * corr_R
                Corr_subs.append(corr_sub)
                Rmaps.append((R_map * m2_rep).view(-1, 1, 405))

            Outs = []
            for idy in range(1, max_obj):  # training:with_bg, testing: w/o BG
                corr = Corr_subs[idy]
                co_size = Corr_subs[idy].size()[2::]
                max_only, indices = f.max_pool2d(corr, co_size, return_indices=True)
                max_only = max_only.view(-1, 1, 405) + Rmaps[idy]
                # Rank & select FG
                m_sorted, m_sorted_idx = max_only.sort(descending=True, dim=2)
                corr = torch.cat([co.index_select(0, m_sort[0, 0:256]).unsqueeze(0) for co, m_sort in zip(corr, m_sorted_idx)])
                # Merge net FG
                corr_fores = self.p_2(self.res_1(self.p_1(f.upsample(corr, scale_factor=2, mode='bilinear'))))
                if max_obj == 1:  # only bg
                    print('missing obj')
                    corr_backs = torch.zeros(corr_fores.size()).cuda()
                else:
                    backs_idx = Corr_subs[0:idy] + Corr_subs[idy + 1::]
                    corr_b = torch.cat(backs_idx, 1)
                    R_map_b = Rmaps[0:idy] + Rmaps[idy + 1::]
                    R_map_b = torch.cat(R_map_b, 2)
                    max_only_b, indices = f.max_pool2d(corr_b, co_size, return_indices=True)
                    max_only_b = max_only_b.view(R_map_b.size()[0], 1, -1) + R_map_b
                    # Rank & select BG
                    m_sorted, m_sorted_idx = max_only_b.sort(descending=True, dim=2)
                    corr_b = torch.cat([co.index_select(0, m_sort[0, 0:256]).unsqueeze(0) for co, m_sort in zip(corr_b, m_sorted_idx)])
                    # Merge net BG
                    corr_backs = self.p_2(self.res_1(self.p_1(f.upsample(corr_b, scale_factor=2, mode='bilinear'))))
                if idy == 0:
                    tmp = corr_fores
                    corr_fores = corr_backs
                    corr_backs = tmp
                    m_p = f.adaptive_avg_pool2d(Mp_all, corr_fores.size()[-2::])
                else:
                    m_p = f.adaptive_avg_pool2d(Mps[idy], corr_fores.size()[-2::])
                # low level features
                base1 = torch.cat([basef1[idx: idx + 1], corr_fores, corr_backs, m_p], 1)
                fea1 = self.R1(base1)
                base2 = torch.cat([basef2[idx: idx + 1],
                                   fea1], 1)
                fea2 = self.R2(base2)
                base3 = torch.cat([basef3[idx: idx + 1],
                                   fea2], 1)
                fea3 = self.R3(base3)
                out = f.sigmoid(fea3)
                Outs.append(out)
            Out = torch.cat(Outs, 1)
            Out_Rs.append(Out)
        features = []
        out = [Out_Rs]
        return out, features
