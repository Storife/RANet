# ************************************
# Author: Ziqin Wang
# Email: ziqin.wang.edu@gmail.com
# Github: https://github.com/Storife
# ************************************
import os
import time
import numpy as np
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import functional as F



'''
Functions for Checkpoints
'''

def checkpoint_save(fpath, epoch, model, cost=None, is_best=False):
    if cost:
        model_out_path = fpath + "_{:.2f}_epoch{}.pth".format(cost, epoch)
    else:
        model_out_path = fpath + "_epoch{}.pth".format(epoch)
    try:
        torch.save(model.module.state_dict(), model_out_path)
    except:
        torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    if is_best:
        shutil.copy(model_out_path, fpath + '.pth')

def checkpoint_load(fpath, model=None):
    if os.path.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("=> Loaded checkpoint '{}'".format(fpath))
        if model == None:
            return checkpoint
        else:
            model.state_dict()
            model.load_state_dict(checkpoint)
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))

def msk2bbox(msk, k=1.5):
    input_size = [480.0, 864.0]
    if torch.max(msk) == 0:
        return torch.from_numpy(np.asarray([0, 0, 480, 864]))
    p = float(input_size[0]) / input_size[1]
    msk_x = torch.max(msk[0], 1)[0]
    msk_y = torch.max(msk[0], 0)[0]
    nzx = torch.nonzero(msk_x)
    nzy = torch.nonzero(msk_y)
    bbox_init = [(nzx[0] + nzx[-1]) / 2, (nzy[0] + nzy[-1]) / 2, (nzx[-1] - nzx[0]).float() * k / 2, (nzy[-1] - nzy[0]).float() * k / 2]
    tmp = torch.max(bbox_init[2], p * bbox_init[3])
    bbox_init = [bbox_init[0], bbox_init[1], tmp.long(), (tmp / p).long()]
    bbox = torch.cat([bbox_init[0] - bbox_init[2], bbox_init[1] - bbox_init[3], bbox_init[0] + bbox_init[2], bbox_init[1] + bbox_init[3]])
    bbox = torch.min(torch.max(bbox, torch.zeros(4).cuda().long()),
          torch.from_numpy(np.array([input_size[0], input_size[1], input_size[0], input_size[1]])).cuda().long())
    if bbox[2] - bbox[0] < 32 or bbox[3] - bbox[1] < 32:
        return torch.from_numpy(np.asarray([0, 0, 480, 864])).cuda()
    return bbox

def bbox_crop(img, bbox):
    img = img[:, bbox[0]:bbox[2], bbox[1]: bbox[3]]
    return img

def bbox_uncrop(img, bbox, size, crop_size): # 4D input
    img = F.upsample_bilinear(img, size=crop_size[2::])
    msk = F.pad(img, (bbox[1], 864 - bbox[3], bbox[0], 480 - bbox[2], ))
    return msk

def test_SVOS_Video_batch(data_loader, model, save_root, threshold=0.5, single_object=False, pre_first_frame=False, add_name=''):
    ms = [864, 480]
    palette_path = '../datasets/palette.txt'
    with open(palette_path) as f:
        palette = f.readlines()
    palette = list(np.asarray([[int(p) for p in pal[0:-1].split(' ')] for pal in palette]).reshape(768))
    def init_Frame(batchsize):
        Key_features = [[] for i in range(batchsize)]
        Key_masks = [[] for i in range(batchsize)]
        Init_Key_masks = [[] for i in range(batchsize)]
        Frames = [[] for i in range(batchsize)]
        Box = [[] for i in range(batchsize)]
        Image_names = [[] for i in range(batchsize)]
        Img_sizes = [[] for i in range(batchsize)]
        Frames_batch = dict(Frames=Frames, Key_features=Key_features, Key_masks=Key_masks, Box=Box, Img_sizes=Img_sizes, Init_Key_masks=Init_Key_masks,
                            Image_names=Image_names, Sizes=[0 for i in range(batchsize + 1)], batchsize=batchsize, Flags=[[] for i in range(batchsize)],
                            Img_flags=[[] for i in range(batchsize)])
        return Frames_batch
    batchsize = 4
    max_iter = 10
    torch.set_grad_enabled(False)
    _ = None
    Frames_batch = init_Frame(batchsize)
    print('Loading Data .........')
    for iteration, batch in enumerate(data_loader, 1):
        if model.fp16:
            batch[0] = [Variable(datas, volatile=True).cuda().half() for datas in batch[0]]
            batch[1] = [Variable(datas, volatile=True).cuda().half() for datas in batch[1]]
        else:
            batch[0] = [Variable(datas, volatile=True).cuda() for datas in batch[0]]
            batch[1] = [Variable(datas, volatile=True).cuda() for datas in batch[1]]
        frame_num = len(batch[0])
        Key_frame = batch[0][0]
        init_Key_mask = batch[1][0]
        size = Key_frame.size()[2::]
        # cc for key frame
        bbox = msk2bbox(init_Key_mask[0].ge(1.6), k=1.5)
        Key_frame = F.upsample(bbox_crop(Key_frame[0], bbox).unsqueeze(0), size, mode='bilinear')
        Key_mask = F.upsample(bbox_crop(init_Key_mask[0], bbox).unsqueeze(0), size)
        S_name = batch[2][0][0]
        Key_feature = model(_, Key_frame, _, _, mode='first')[0]
        Frames = batch[0]
        Img_sizes = batch[3]

        loc = np.argmin(Frames_batch['Sizes'][0:batchsize])
        Fsize = len(batch[2])
        # print(loc)
        # print(Fsize)
        Frames_batch['Frames'][loc].extend(Frames[1::])
        Frames_batch['Key_features'][loc].extend([Key_feature] + [None] * (Fsize - 2))
        Frames_batch['Key_masks'][loc].extend([Key_mask] * (Fsize - 1))
        Frames_batch['Init_Key_masks'][loc].extend([init_Key_mask] * (Fsize - 1))
        Frames_batch['Box'][loc].extend([bbox] + [None] * (Fsize - 2))
        Frames_batch['Flags'][loc].extend([1] + [2 for i in range(Fsize - 3)] + [3])
        Frames_batch['Sizes'][loc] += Fsize - 1

        Frames_batch['Image_names'][loc].extend([b[0] for b in batch[2]])
        Frames_batch['Img_sizes'][loc].extend(Img_sizes)
        Frames_batch['Img_flags'][loc].extend([1] + [2 for i in range(Fsize - 2)] + [3])

        if iteration % max_iter == 0 or iteration == len(data_loader):
            for idx in range(batchsize):
                Frames_batch['Flags'][idx].append(False)
            Frames_batch['Sizes'][batchsize] = min(Frames_batch['Sizes'][0:batchsize - 1])
            Out_Mask = process_SVOS_batch(Frames_batch, model, threshold, single_object, pre_first_frame)
            Image_names = Frames_batch['Image_names']
            Img_sizes = Frames_batch['Img_sizes']
            Img_flags = Frames_batch['Img_flags']
            for Masks, Names, Sizes, Flags in zip(Out_Mask, Image_names, Img_sizes, Img_flags):
                for mask, name, ss, flag in zip(Masks, Names, Sizes, Flags):
                    folder_name, fname = name.split('/')[-2::]
                    save_path = save_root + '/' + folder_name + '/'
                    if not (os.path.exists(save_path)):
                        os.mkdir(save_path)
                    if flag == 1:
                        print(folder_name)
                        I0 = Image.open(name)
                    if I0.size[0] > 864:
                        t = min(864.0 / I0.size[0], 480.0 / I0.size[1])
                        ms = [int(864.0 / t + 0.5), int(480.0 / t + 0.5)]
                    else:
                        ms = [864, 480]
                    ss = [float(ss[0].cpu().numpy()), float(ss[1].cpu().numpy())]
                    img = Image.fromarray((mask - 1).astype('float32')).convert('P').resize(ms)
                    img.putpalette(palette)
                    img.crop(((ms[0] - ss[0]) / 2, (ms[1] - ss[1]) / 2, (ms[0] - ss[0]) / 2 + ss[0],
                        (ms[1] - ss[1]) / 2 + ss[1])).resize(ss).save(save_path + fname.split('.')[0] + add_name + '.png', palette=palette)
                    # print(save_path + fname.split('.')[0])
            del(Frames_batch)
            Frames_batch = init_Frame(batchsize)
    return

def process_SVOS_batch(Frames_batch, model, threshold=0.5, single_object=False, pre_first_frame=False):
    def msks2P(msks, objs_ids_num, fp16=False):
        P = torch.zeros(msks[0].size()).cuda()
        if fp16:
            P = P.half()
        for idx, msk in enumerate(msks):
            ids = torch.nonzero(msk)
            if len(ids) > 0:
                P[ids[:, 0], ids[:, 1], ids[:, 2], ids[:, 3]] = idx + 1
        if len(msks) == objs_ids_num:
            return P + 1
        return P
    Frames = Frames_batch['Frames']
    Key_features = Frames_batch['Key_features']
    Init_masks = Frames_batch['Init_Key_masks']
    Key_masks = Frames_batch['Key_masks']
    Boxs = Frames_batch['Box']
    # Image_names = Frames_batch['Image_names']
    batchsize = Frames_batch['batchsize']
    Frame_Flags = Frames_batch['Flags']
    Out_Mask = [[] for i in range(batchsize)]
    size = Frames[0][0].size()[2::]
    torch.cuda.empty_cache()
    # msk_p = Key_mask
    KFea = [[] for i in range(batchsize)]
    KMsk = [[] for i in range(batchsize)]
    Img = [[] for i in range(batchsize)]
    PMsk = [[] for i in range(batchsize)]
    Crop_size = [[] for i in range(batchsize)]
    BBox = [0 for i in range(batchsize)]
    Flags = [True for i in range(batchsize)]
    for idx in range(max(Frames_batch['Sizes'])):
        for batch, (frame, key, key_mask, box, flag, init_mask) in enumerate(zip([i[min(idx, len(i)-1)] for i in Frames], [i[min(idx, len(i)-1)] for i in Key_features], [i[min(idx, len(i)-1)] for i in Key_masks], [i[min(idx, len(i)-1)] for i in Boxs], [i[min(idx, len(i)-1)] for i in Frame_Flags], [i[min(idx, len(i)-1)] for i in Init_masks])):
            if flag == False:
                Flags[batch] = 0
                continue
            Flags[batch] = 1
            if flag == 1:
                # update template frame
                KFea[batch] = key
                KMsk[batch] = key_mask
                PMsk[batch] = key_mask
                # bbox = box
                BBox[batch] = box
                if not (pre_first_frame):
                    out0 = init_mask[0][0]
                    if single_object:
                        if model.fp16:
                            out0 = out0.ge(1.6).half() + 1
                        else:
                            out0 = out0.ge(1.6).float() + 1
                    Out_Mask[batch].append(out0.data.cpu().numpy())
            # crop current frame
            frame = bbox_crop(frame[0], BBox[batch]).unsqueeze(0)
            # print(Crop_size)
            # print(batch)
            Crop_size[batch] = frame.size()
            frame = F.upsample(frame, size, mode='bilinear')
            Img[batch] = frame

        index_select = torch.nonzero(torch.tensor(Flags).cuda()).view(-1)
        # print(index_select)
        tmp = 0
        for i, g in enumerate(Flags):
            if g ==0:
                Flags[i] = -1
            else:
                Flags[i] = tmp
                tmp +=1
        inputs = [torch.cat(Img)[index_select], torch.cat(KFea)[index_select], torch.cat(KMsk)[index_select], torch.cat(PMsk)[index_select]]
        outputs, _ = model(*inputs)
        Msk2 = [[] for i in range(batchsize)]
        Output = [None if Flags[ids] == -1 else outputs[0][Flags[ids]] for ids in range(batchsize)]
        for batch, (out, crop_size) in enumerate(zip(Output, Crop_size)):
            if Flags[batch] == -1:
                continue
            if len(out.size()) == 3:
                out = bbox_uncrop(out.unsqueeze(0), BBox[batch], out.size(), crop_size)
            else:
                out = bbox_uncrop(out, BBox[batch], out[0].size(), crop_size)
            if single_object:
                if threshold:
                    if model.fp16:
                        out = (out > threshold).half()
                    else:
                        out = (out > threshold).float()
                Msk2[batch].append(out)
            else:
                if not (threshold):
                    assert ('Must set threshold for multi-object')
                for idm in range(out.size(1)):
                    outi = ((out[:, idm: idm + 1] > threshold) * (
                                out[:, idm: idm + 1] >= torch.max(out, 1, keepdim=True)[0])).float()
                    Msk2[batch].append(outi)
            if threshold:
                msk = msks2P(Msk2[batch], len(Msk2[batch]), fp16=model.fp16)
            else:  # single object & no th
                msk = Msk2[batch][0] + 1
            Out_Mask[batch].append(msk[0, 0].data.cpu().numpy())
            BBox[batch] = msk2bbox(msk[0].ge(1.6))
            PMsk[batch] = F.upsample(Variable(bbox_crop(msk[0], BBox[batch]).unsqueeze(0), volatile=True).cuda(), size)
    return Out_Mask

def fitpredict17(data_set, model, add_name='', threads=32, batchSize=1, save_root ='./test/'):
    if data_set.Datasets_params[0]['mode'] in ['16val', '16all']:
        threshold = 0.5
        single_object = True
    elif data_set.Datasets_params[0]['mode'] in ['17val', '17test_dev']:
        threshold = 0.5
        single_object = False
    else:
        threshold = 0.5
        single_object = False
    print('Start testing ...')
    data_set.iter_mode = 'test'
    data_loader = DataLoader(dataset=data_set, num_workers=threads, batch_size=batchSize, shuffle=False, pin_memory=True)
    model.eval()
    if not (os.path.exists(save_root)):
        os.mkdir(save_root)
    test_SVOS_Video_batch(data_loader, model, save_root, threshold, single_object)
    return


