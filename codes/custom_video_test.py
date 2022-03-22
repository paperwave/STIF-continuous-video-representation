import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
import utils.util as util
import data.util as data_util
import models.modules.Sakuya_arch as Sakuya_arch
import models.modules.Sakuya_arch_test as Sakuya_arch_test
import models.modules.STVSR as STVSR
import csv
import time
import itertools
import re
import options.options as option
from pdb import set_trace as bp
from PIL import Image
from data.util import imresize_np



# print('export CUDA_VISIBLE_DEVICES=' + "4,5,6,7")
device = 'cuda'

scale = 4
N_ot = 6
N_in = 1+ N_ot // 2
header_written = False

mode = "LIIF" # TMNet & LIIF
if mode == "LIIF":
    model = Sakuya_arch_test.LunaTokis(64, N_ot, 8, 5, 40)
    model.load_state_dict(torch.load('/home/users/zeyuan_chen/model_repo/latest_G.pth'), strict=True)
elif mode == "TMNet":
    model = STVSR.TMNet(64, N_ot, 8, 5, 40)
    model.load_state_dict(torch.load('/home/users/zeyuan_chen/model_repo/tmnet-pretrain.pth'), strict=True)

model.eval()
model = model.to(device)

def single_forward(model, imgs_in, use_time=True, N_ot=6):
    with torch.no_grad():
        # imgs_in.size(): [1,n,3,h,w]
        b,n,c,h,w = imgs_in.size()
        h_n = int(4*np.ceil(h/4))
        w_n = int(4*np.ceil(w/4))
        imgs_temp = imgs_in.new_zeros(b,n,c,h_n,w_n)
        imgs_temp[:,:,:,0:h,0:w] = imgs_in
        if mode == "LIIF":
            time_Tensors = [torch.tensor([i / 8])[None].to(device) for i in range(8)]
        elif mode == "TMNet":
            time_Tensors = torch.tensor([i / 8 for i in range(1, 8)])[None].to(device)
        # print(imgs_temp.shape)
        
        model_output = model(imgs_temp, time_Tensors)
        # model_output = model(imgs_temp, time_Tensors, test=True)
        return model_output


# folder_path = '/home/users/zeyuan_chen/DAVIS2017/DAVIS/JPEGImages/Full-Resolution/'
# folder_path = '/home/users/zeyuan_chen/DAVIS/JPEGImages/Full-Resolution/'
folder_path = '/home/users/zeyuan_chen/davis2016/DAVIS/JPEGImages/1080p/'
# folder_path = '/home/users/zeyuan_chen/DAVIS-2019-challenge/DAVIS/JPEGImages/Full-Resolution/'
folder_list = sorted(os.listdir(folder_path))
all_name_list = {}
for folder in folder_list:
    f_path = os.path.join(folder_path, folder)
    print(f_path)
    all_name_list.update({folder: [os.path.join(f_path, name) for name in sorted(os.listdir(f_path))]})
# out_path = '/home/users/zeyuan_chen/image_repo/DAVIS/dolphins-show/'

# folder_list = 

for folder in all_name_list.keys():
    # if not folder == 'kite-surf':
    #     continue
    # if not folder in ['hike', 'hockey', 'kite-walk', 'paragliding', 'rhino', 'soapbox']:
    # if not folder in ['bmx-rider', 'boxing', 'dog-control', 'dolphins', 'dolphins-show']:
    # if not folder in ['demolition', 'grass-chopper', 'helicopter-landing', 'inflatable', 'kart-turn', 'selfie', 'wings-turn']:
    # if not folder in ['grass-chopper', 'helicopter-landing', 'inflatable', 'kart-turn', 'selfie', 'wings-turn']:
    # if not folder in ['car-shadow', 'drift-chicane', 'flamingo', 'train']:
    # if not folder in ['bmx-rider']:
    # if not folder in ['paragliding', 'hockey', 'kite-surf']:
    if not folder in ['bmx-bumps']:
        continue
    name_list = all_name_list[folder]
    out_path = '/home/users/zeyuan_chen/image_repo/DAVIS2016/{}/HR/'.format(folder)
    out_path1 = '/home/users/zeyuan_chen/image_repo/DAVIS2016/{}/bicubic/'.format(folder)
    out_path2 = '/home/users/zeyuan_chen/image_repo/DAVIS2016/{}/LR/'.format(folder)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.exists(out_path1):
        os.makedirs(out_path1)
    if not os.path.exists(out_path2):
        os.makedirs(out_path2)

    index = 0
    index_0 = 0
    for i in range(len(name_list) - 1):
        # if i >= 50:
        #     break
        imgpath1 = os.path.join(name_list[i])
        imgpath2 = os.path.join(name_list[i + 1])
        img1 = cv2.imread(imgpath1, cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(imgpath2, cv2.IMREAD_UNCHANGED)
        # bp()
        img1 = imresize_np(img1, 1/10, True).astype(np.float32) / 255.
        img2 = imresize_np(img2, 1/10, True).astype(np.float32) / 255.
        print(img1.shape)
        # img1 = cv2.resize(img1, (320, 180))
        # img2 = cv2.resize(img2, (320, 180))
        Image.fromarray((np.clip(img1[:, :, [2,1,0]], 0, 1) * 255).astype(np.uint8)).save(os.path.join(out_path2, name_list[i].split('/')[-1]))
        print(os.path.join(out_path2, name_list[i].split('/')[-1]), "SAVED!")
        imgs = np.stack([img1, img2], axis=0)[:, :, :, [2,1,0]]
        imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()[None].to(device)

        time_Tensors = [torch.tensor([i / 8])[None].to(device) for i in range(2)]
        output = single_forward(model, imgs)
        # bp()
        if mode == "LIIF":
            for i in range(len(output)):
                img = output[i][0]
                img = Image.fromarray((img.clamp(0., 1.).detach().cpu().permute(1, 2, 0) * 255).numpy().astype(np.uint8))
                img.save(os.path.join(out_path, '{}.jpg'.format(index_0)))
                index_0 += 1
            for i in range(8):
                HH, WW = img1.shape[0] * 4, img1.shape[1] * 4
                img = Image.fromarray((np.clip(img1[:, :, [2,1,0]], 0, 1) * 255).astype(np.uint8)).resize((WW, HH), Image.BICUBIC)
                img.save(os.path.join(out_path1, '{}.jpg'.format(index)))
                print(os.path.join(out_path1, '{}.jpg'.format(index)), "SAVED!")
                index += 1
        elif mode == "TMNet":
            for i in range(output.shape[1] - 1):
                img = output[0][i]
                img = Image.fromarray((img.clamp(0., 1.).detach().cpu().permute(1, 2, 0) * 255).numpy().astype(np.uint8))
                img.save(os.path.join(out_path1, '{}.jpg'.format(index)))
                print(os.path.join(out_path1, '{}.jpg'.format(index)), "SAVED!")
                index += 1