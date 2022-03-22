# from https://github.com/myungsub/CAIN/blob/master/utils.py, 
# but removed the errenous normalization and quantization steps from computing the PSNR.

import math
import os
import torch
import shutil
from PIL import Image
import numpy as np
from torchvision import transforms
from pdb import set_trace as bp
import time
import argparse
from data.util import imresize_np, bgr2ycbcr

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8

import torch
import torch.nn.functional as F
from math import exp
import numpy as np
import cv2
import utils.util as util


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0).cuda()
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def create_window_3d(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _2D_window.unsqueeze(2) @ (_1D_window.t())
    window = _3D_window.expand(1, channel, window_size, window_size, window_size).contiguous().cuda()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    
    # mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    # mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1 = F.conv2d(F.pad(img1, (5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=channel)
    mu2 = F.conv2d(F.pad(img2, (5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    # sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    # sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    sigma1_sq = F.conv2d(F.pad(img1 * img1, (5, 5, 5, 5), 'replicate'), window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(F.pad(img2 * img2, (5, 5, 5, 5), 'replicate'), window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(F.pad(img1 * img2, (5, 5, 5, 5), 'replicate'), window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def ssim_matlab(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=1):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, _, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window_3d(real_size, channel=1).to(img1.device)
        # Channel is set to 1 since we consider color images as volumetric images

    img1 = img1.unsqueeze(1)
    img2 = img2.unsqueeze(1)

    mu1 = F.conv3d(F.pad(img1, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)
    mu2 = F.conv3d(F.pad(img2, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(F.pad(img1 * img1, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_sq
    sigma2_sq = F.conv3d(F.pad(img2 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu2_sq
    sigma12 = F.conv3d(F.pad(img1 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 3 channel for SSIM
        self.channel = 3
        self.window = create_window(window_size, channel=self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        _ssim = ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
        dssim = (1 - _ssim) / 2
        return dssim

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)


def init_meters(loss_str):
    losses = init_losses(loss_str)
    psnrs = AverageMeter()
    ssims = AverageMeter()
    return losses, psnrs, ssims

def eval_metrics(output, gt, psnrs, ssims):
    # PSNR should be calculated for each image, since sum(log) =/= log(sum).
    for b in range(gt.size(0)):
        psnr = calc_psnr(output[b], gt[b])
        psnrs.update(psnr)

        ssim = ssim_matlab(output[b].unsqueeze(0).clamp(0,1), gt[b].unsqueeze(0).clamp(0,1) , val_range=1.)
        ssims.update(ssim)

def init_losses(loss_str):
    loss_specifics = {}
    loss_list = loss_str.split('+')
    for l in loss_list:
        _, loss_type = l.split('*')
        loss_specifics[loss_type] = AverageMeter()
    loss_specifics['total'] = AverageMeter()
    return loss_specifics

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calc_psnr(pred, gt):
    diff = (pred - gt).pow(2).mean() + 1e-8
    return -10 * math.log10(diff)


def save_checkpoint(state, directory, is_best, exp_name, filename='checkpoint.pth'):
    """Saves checkpoint to disk"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory , filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(directory , 'model_best.pth'))

def log_tensorboard(writer, loss, psnr, ssim, lpips, lr, timestep, mode='train'):
    writer.add_scalar('Loss/%s/%s' % mode, loss, timestep)
    writer.add_scalar('PSNR/%s' % mode, psnr, timestep)
    writer.add_scalar('SSIM/%s' % mode, ssim, timestep)
    if mode == 'train':
        writer.add_scalar('lr', lr, timestep)
        

def make_coord_3d(shape, time, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
        bias = torch.tensor(time).repeat(ret.shape[0], 1)
        ret = torch.cat([ret, bias], dim=1)
    return ret

    
    
def test_images_outp(model, device, time, epoch=0, iter_id=0):
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        out_path = '/output/Image-222333444/Epoch_{}'.format(epoch)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            
        imgpath = '/data/nnice1216/vimeo_septuplet/DAVIS/JPEGImages/Full-Resolution/bmx-rider/'
        imgpaths = [imgpath + f'/0000{i}.jpg' for i in range(1,8)]
        pth_ = imgpaths
        
        images = [Image.open(pth) for pth in imgpaths]
        h, w = images[0].size
        images = [img.resize((720, 416)) for img in images]
        inputs = [int(e)-1 for e in list('2345')]
        inputs = inputs[:len(inputs)//2] + inputs[len(inputs)//2:]
        images = [images[i] for i in inputs]
        imgpaths = [imgpaths[i] for i in inputs]
        
        T = transforms.ToTensor()
        # images = [((T(img_) - 0.5) * 2)[None] for img_ in images]
        images = [T(img_)[None] for img_ in images]
        h, w = images[0].shape[2], images[0].shape[3]
        coord = make_coord_3d((h, w), time)
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w
        cell[:, 2] *= 0.5 
        coord, cell = coord.to(device), cell.to(device)
        
        images = [img_.to(device) for img_ in images]
        images = torch.stack(images, dim=2)
        # images = images[:, :, 1:3]
                  
        torch.cuda.synchronize()
        
        out, flows, warp1, warp2 = model(images, coord[None], False) 
        # out, loss_map = model(images, coord[None], False)
        # loss_map = loss_map.view(1, h, w, 576).permute(0, 1, 2, 3)[0].sum(2)
        # loss_map = (loss_map - loss_map.min()) / (loss_map.max() - loss_map.min())
        torch.cuda.synchronize()
        # Image.fromarray((loss_map[0, 0].cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(out_path, 'Iter{}_loss_map1.jpg'.format(iter_id)))
        # Image.fromarray((loss_map2[0, 0].cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(out_path, 'Iter{}_loss_map2.jpg'.format(iter_id)))
        Image.fromarray((out[0].clamp(0., 1.).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)).save(os.path.join(out_path, 'Iter{}.png'.format(iter_id)))
        Image.fromarray((warp1[0].clamp(0., 1.).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)).save(os.path.join(out_path, 'warp1_Iter{}.png'.format(iter_id)))
        Image.fromarray((warp2[0].clamp(0., 1.).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)).save(os.path.join(out_path, 'warp2_Iter{}.png'.format(iter_id)))
        # Image.fromarray((warp[0].clamp(0., 1.).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)).save(os.path.join(out_path, 'warp_Iter{}.png'.format(iter_id)))
        # Image.fromarray((mask[0][0].clamp(0., 1.).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(out_path, 'mask_Iter{}.jpg'.format(iter_id)))
        Image.fromarray(flow_to_image(flows.detach().cpu()[0, :2].numpy().transpose((1,2,0)))).save(os.path.join(out_path, 'T0_Iter{}.png'.format(iter_id)))
        # Image.fromarray(flow_to_image(flows.detach().cpu()[0, 2:].numpy().transpose((1,2,0)))).save(os.path.join(out_path, 'T1_Iter{}.jpg'.format(iter_id)))
        
        # del out, flows, warp1, warp2
        # torch.cuda.empty_cache()
        

def test_metric2(model, opt, epoch_id, iter_id, temp=False):
    torch.cuda.empty_cache()
    
    if temp == False:
        path = '/zychenpvc/vid4/LR/'
        path2 = '/zychenpvc/vid4/HR/'
        folder_list = os.listdir(path)
        name_list = []
        input_list = []
        for folder in folder_list:
            f_path = os.path.join(path, folder)
            f_path2 = os.path.join(path2, folder)
            input_list.extend([os.path.join(f_path, name) for name in sorted(os.listdir(f_path)[::2])])
            name_list.extend([os.path.join(f_path2, name) for name in sorted(os.listdir(f_path2))])
    elif temp == True:
        path = '/zychenpvc/vid4/LR/walk/'
        path1 = '/zychenpvc/vid4/HR/walk/'
        name_list = sorted(os.listdir(path))
        input_list = name_list[::2]
        name_list = [os.path.join(path1, name) for name in name_list]
        input_list = [os.path.join(path, name) for name in input_list]
    
    losses, psnrs, ssims = init_meters('1*L1')
    psnr_list = []
    avg_psnr_sum, avg_ssim_sum, num_sample = 0, 0, 0
    index = 0
    out_path = '/zychenpvc/zoomin-imgs/walkimgs1/Epoch_{}'.format(epoch_id)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    with torch.no_grad():
        start = time.time()
        for i in range(len(input_list) - 1):
            
            # if not input_list[i][:44] == input_list[i+1][:44]:
            #     # print(input_list[i][:44], name_list[2 * i + 1 - index][:44])
            #     if not input_list[i][40:44] == name_list[2 * i + 1 - index][40:44]:
            #         index += 1
            #     continue
            # if i == 22:
            #     bp()
            # print(input_list[i], name_list[2 * i + 1 - index])   
            imgs = [cv2.imread(input_list[i]).astype(np.float32) / 255.,
                    cv2.imread(input_list[i+1]).astype(np.float32) / 255.]

            gt1 = [cv2.imread(name_list[2 * i + 1 - index]).astype(np.float32) / 255.]
            # print(input_list[i], name_list[2 * i + 1 - index])
            gt = np.copy(gt1[0])

            imgs = np.stack(imgs, axis=0)
            imgs = imgs[:, :, :, [2, 1, 0]]
            imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()[None].cuda()
            
            data = {'LQs': imgs, 'time': [torch.tensor([1 / 2])[None]]}
            model.feed_data(data, False)
            output = model.test(True)
            # bp()
            # output = output / 255.
            if opt['network_G']['which_model_G'] == 'LIIF':
                for i in range(len(output)):
                    output_f = output[0].data.float().cpu().squeeze(0)   
                    output = util.tensor2img(output_f) / 255.
                    
                    crt_psnr = util.calculate_psnr(output * 255, gt[i] * 255)
                    crt_ssim = util.ssim(output * 255, gt[i] * 255)
                    avg_psnr_sum += crt_psnr
                    avg_ssim_sum += crt_ssim
                    num_sample += 1

            elif opt['network_G']['which_model_G'] == 'LunaTokis':
                outputs = output.data.float().cpu().squeeze(0)   
                output_f = outputs[1,:,:,:].squeeze(0)
            
                output = util.tensor2img(output_f) / 255.
                Image.fromarray((output[:,:,::-1]*255).astype(np.uint8)).save(os.path.join(out_path, 'iter{}.png'.format(iter_id)))
                
                crt_psnr = util.calculate_psnr(output * 255, gt * 255)
                crt_ssim = util.ssim(output * 255, gt * 255)
                avg_psnr_sum += crt_psnr
                avg_ssim_sum += crt_ssim
                num_sample += 1
    
    avg_psnr_sum = avg_psnr_sum / num_sample
    avg_ssim_sum = avg_ssim_sum / num_sample
    end = time.time()
    print('Epoch [{0}/{1}], Cost time: {2:.2f}s, Val_PSNR:{3:.4f}, Val_SSIM:{4:.4f}'
          .format(epoch_id, 30, end - start, avg_psnr_sum, avg_ssim_sum))
    
    with open('/zychenpvc/zoomin-imgs/logs1.txt', 'a') as f:
        print('Date: {0}s, Epoch: [{1}/{2}], Cost time: {3:.2f}s, Val_PSNR: {4:.2f}, Val_SSIM: {5:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch_id, iter_id, end - start, avg_psnr_sum, avg_ssim_sum), file=f)
        
    torch.cuda.empty_cache()
    return

def test_metric_full(model, opt, epoch_id, iter_id):
    torch.cuda.empty_cache()
    path = '/home/users/zeyuan_chen/VID4/LR/'
    path2 = '/home/users/zeyuan_chen/VID4/HR/'
    folder_list = os.listdir(path)
    all_name_list = {}
    all_input_list = {}
    for folder in folder_list:
        f_path = os.path.join(path, folder)
        f_path2 = os.path.join(path2, folder)
        all_input_list.update({folder: [os.path.join(f_path, name) for name in sorted(os.listdir(f_path))[::2]]})
        all_name_list.update({folder: [os.path.join(f_path2, name) for name in sorted(os.listdir(f_path2))]})
    
    avg_psnr_sum, avg_ssim_sum = 0, 0
    index = 0
    num_sample = 0

    with torch.no_grad():
        start = time.time()
        for folder in all_name_list.keys():
            print(folder)
            name_list = all_name_list[folder]
            input_list = all_input_list[folder]
            
            out_path = '/home/users/zeyuan_chen/zeyuan3/zoomin-imgs//vid4-zoomin-6x/{}/'.format(folder)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            for i in range(len(input_list) - 1):
                imgs = [cv2.imread(input_list[i]).astype(np.float32) / 255.,
                        cv2.imread(input_list[i+1]).astype(np.float32) / 255.]
                print(input_list[i], name_list[2 * i + 1])
                gt1 = [cv2.imread(name_list[2 * i + 1]).astype(np.float32) / 255.]
                gt2 = [cv2.imread(name_list[2 * i]).astype(np.float32) / 255.]
                gt = np.copy(gt1[0])

                imgs = np.stack(imgs, axis=0)
                imgs = imgs[:, :, :, [2, 1, 0]]
                imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()[None].cuda()
                
                data = {'LQs': imgs, 'time': [torch.tensor([1 / 2])[None]], 'shape': (int(imgs.shape[-2]*4),int(imgs.shape[-1]*4)), 'test': True}
                model.feed_data(data, False)
                output = model.test(True)
                if opt['network_G']['which_model_G'] == 'LIIF':
                    outputs = output[0].data.float().cpu().squeeze(0)   
                    output_f = outputs
                elif opt['network_G']['which_model_G'] == 'LunaTokis':
                    outputs = output.data.float().cpu().squeeze(0)   
                    output_f = outputs[1,:,:,:].squeeze(0)
                
                output = util.tensor2img(output_f) / 255.
                output_y = bgr2ycbcr(output, only_y=True)
                gt_y = bgr2ycbcr(gt, only_y=True)
                # Image.fromarray((output[:,:,::-1]*255).astype(np.uint8)).save(os.path.join(out_path, name_list[2 * i + 1].split('/')[-1]))
                # print(os.path.join(out_path, name_list[2 * i + 1].split('/')[-1]), "SAVED!")
                crt_psnr = util.calculate_psnr(output_y * 255, gt_y * 255)
                crt_ssim = util.ssim(output_y * 255, gt_y * 255)
                avg_psnr_sum += crt_psnr
                avg_ssim_sum += crt_ssim
                num_sample += 1
                # print(crt_psnr, crt_ssim)

                gt = np.copy(gt2[0])
                data = {'LQs': imgs, 'time': [torch.tensor([0.])[None]], 'shape': (int(imgs.shape[-2]*4),int(imgs.shape[-1]*4)), 'test': True}
                model.feed_data(data, False)
                output = model.test(True)
                if opt['network_G']['which_model_G'] == 'LIIF':
                    outputs = output[0].data.float().cpu().squeeze(0)   
                    output_f = outputs
                elif opt['network_G']['which_model_G'] == 'LunaTokis':
                    outputs = output.data.float().cpu().squeeze(0)   
                    output_f = outputs[0,:,:,:].squeeze(0)
                
                output = util.tensor2img(output_f) / 255.
                output_y = bgr2ycbcr(output, only_y=True)
                gt_y = bgr2ycbcr(gt, only_y=True)
                crt_psnr = util.calculate_psnr(output_y * 255, gt_y * 255)
                crt_ssim = util.ssim(output_y * 255, gt_y * 255)
                avg_psnr_sum += crt_psnr
                avg_ssim_sum += crt_ssim
                num_sample += 1
                # Image.fromarray((output[:,:,::-1]*255).astype(np.uint8)).save(os.path.join(out_path, name_list[2 * i].split('/')[-1]))
                # print(os.path.join(out_path, name_list[2 * i].split('/')[-1]), "SAVED!")
                # print(crt_psnr, crt_ssim)
                
    avg_psnr_sum = avg_psnr_sum / num_sample
    avg_ssim_sum = avg_ssim_sum / num_sample
    end = time.time()
    print('Epoch [{0}/{1}], Cost time: {2:.2f}s, Val_PSNR:{3:.4f}, Val_SSIM:{4:.4f}'
          .format(epoch_id, 30, end - start, avg_psnr_sum, avg_ssim_sum))
    
    with open('/home/users/zeyuan_chen/logs_repo/logs-shic0e6bbf.txt', 'a') as f:
        print('Date: {0}s, Epoch: [{1}/{2}], Cost time: {3:.2f}s, Val_PSNR: {4:.2f}, Val_SSIM: {5:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch_id, iter_id, end - start, avg_psnr_sum, avg_ssim_sum), file=f)
        
    torch.cuda.empty_cache()
    return

    
    

def convert_to_gray(img):
    gray_img = img[:, 0] * 0.299 + img[:, 1] * 0.587 + img[:, 2] * 0.114
    return gray_img.unsqueeze(1)


def adjust_learning_rate(epoch):
    if epoch <= 1:
        lr = 2e-5
    elif epoch <=3:
        lr = 1e-5
    elif epoch <= 6:
        lr = 1e-5
    elif epoch <= 15:
        lr = 1e-5
    else:
        lr = 1e-5
    
    return lr


def get_raft_args():
    parser_raft = argparse.ArgumentParser()
    parser_raft.add_argument('--model', default='/model/nnice1216/video/raft-small.pth', help="restore checkpoint")
    parser_raft.add_argument('--path', default='/data/nnice1216/vimeo_septuplet/DAVIS/JPEGImages/Full-Resolution/bmx-rider/', help="dataset for evaluation")
    parser_raft.add_argument('--small', default=True, help='use small model')
    parser_raft.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser_raft.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args1 = parser_raft.parse_known_args()[0]
    
    return args1


def flow_to_image(flow, display=False):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    if display:
        print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0
    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def test_metric_adobe(model, opt, epoch_id, iter_id, mode='gopro'):
    torch.cuda.empty_cache()
    if mode == 'adobe':
        path = '/home/users/zeyuan_chen/adobe240fps/test/'
    elif mode == 'gopro':
        path = '/home/users/zeyuan_chen/GOPRO/test/'
    folder_list = os.listdir(path)
    all_name_list = {}
    for folder in folder_list:
        f_path = os.path.join(path, folder)
        if mode == 'adobe':
            frames = os.listdir(f_path)
            frames = sorted([int(frame[:-4]) for frame in frames])
            frames = [str(frame) + '.png' for frame in frames]
        elif mode == 'gopro':
            frames = sorted(os.listdir(f_path))
        all_name_list.update({folder: [os.path.join(f_path, frame) for frame in frames]})
    
    avg_psnr_sum, avg_ssim_sum = 0, 0
    mid_psnr_sum, mid_ssim_sum = 0, 0
    index = 0
    out_path = '/home/users/zeyuan_chen/zoomin-imgs/walkimgs1/Epoch_{}'.format(epoch_id)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    num_sample = 0
    mid_sample = 0
    indexed_psnr_sum = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    indexed_ssim_sum = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    indexed_sample_num = 0
    scale0 = 12
    with torch.no_grad():
        start = time.time()
        for folder in sorted(all_name_list.keys()):
            print(folder)
            name_list = all_name_list[folder]
            
            out_path = '/home/users/zeyuan_chen/zoomin-imgs/TMNet-8x2/{}/'.format(folder)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            index = 0
            # bp()
            
            while index + 12 < len(name_list):
                # if index > 100:
                    # break
                print("{} / {}".format(index, len(name_list)))
                # imgs = [cv2.imread(name_list[index]), cv2.imread(name_list[index + 8]), cv2.imread(name_list[index + 16]), cv2.imread(name_list[index + 24])]
                imgs = [cv2.imread(name_list[index]), cv2.imread(name_list[index + 12])]
                imgs = [imresize_np(img_, 1 / scale0, True).astype(np.float32) / 255. for img_ in imgs]
                if not opt['network_G']['which_model_G'] == 'LunaTokis':
                    gts = [cv2.imread(name_list[index + i]) for i in range(12)]
                    gts = [img_.astype(np.float32) / 255. for img_ in gts]
                    # gts = [imresize_np(img_, 1 / 2, True).astype(np.float32) / 255. for img_ in gts]
                else:
                    gts = [cv2.imread(name_list[index + i]) for i in [0, 4, 8]]
                    gts = [img_.astype(np.float32) / 255. for img_ in gts]
                    # gts = [imresize_np(img_, 1 / 2, True).astype(np.float32) / 255. for img_ in gts]

                h_lr, w_lr, h_hr, w_hr = imgs[0].shape[0], imgs[0].shape[1], gts[0].shape[0], gts[0].shape[1]
                h_hr, w_hr = h_lr * scale0, w_lr * scale0
                if not imgs[0].shape[0] % 4 == 0:
                    h_lr = imgs[0].shape[0] + 4 - imgs[0].shape[0] % 4
                    h_hr = h_lr * scale0
                if not imgs[0].shape[1] % 4 == 0:
                    w_lr = imgs[0].shape[1] + 4 - imgs[0].shape[1] % 4
                    w_hr = w_lr * scale0
                
                if h_lr != imgs[0].shape[0] or w_lr != imgs[0].shape[1]:
                    imgs = [cv2.resize(img_, (w_lr, h_lr), interpolation=cv2.INTER_LANCZOS4) for img_ in imgs]
                    gts = [cv2.resize(img_, (w_hr, h_hr), interpolation=cv2.INTER_LANCZOS4) for img_ in gts]
 
                gt = np.copy(gts)

                imgs = np.stack(imgs, axis=0)
                imgs = imgs[:, :, :, [2, 1, 0]]
                imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()[None].cuda()
                
                if opt['network_G']['which_model_G'] == 'LIIF' or opt['network_G']['which_model_G'] == 'LIIF_test1':
                    
                    data = {'LQs': imgs, 'time': [torch.tensor([i / 12])[None] for i in range(12)], 'scale': (int(imgs.shape[-2] * scale0), int(imgs.shape[-1] * scale0)), 'test': True}
                    model.feed_data(data, False)
                    output = model.test(True)
                    indexed_sample_num += 1
                    for i in range(len(output)):
                        # print(i)
                        output_f = output[i].data.float().cpu().squeeze(0)   
                        output1 = util.tensor2img(output_f) / 255.
                        output_y = bgr2ycbcr(output1, only_y=True)
                        gt_y = bgr2ycbcr(gt[i], only_y=True)
                        crt_psnr = util.calculate_psnr(output_y * 255, gt_y * 255)
                        crt_ssim = util.ssim(output_y * 255, gt_y * 255)
                        # bp()
                        # print(crt_psnr, crt_ssim)
                        # Image.fromarray((output1[:,:,::-1]*255).astype(np.uint8)).save(os.path.join(out_path, name_list[index + i].split('/')[-1][:-4] + '.jpg'))
                        # print(os.path.join(out_path, name_list[index + i].split('/')[-1][:-4] + '.jpg'), "saved")
                        # Image.fromarray((gt[i][:,:,::-1]*255).astype(np.uint8)).save(os.path.join(out_path, 'GT_iter{}.jpg'.format(i)))
                        avg_psnr_sum += crt_psnr
                        avg_ssim_sum += crt_ssim
                        num_sample += 1
                        # indexed_psnr_sum[i] += crt_psnr
                        # indexed_ssim_sum[i] += crt_ssim
                        if i == 4 or i == 0 or i == 20:
                            mid_psnr_sum += crt_psnr
                            mid_ssim_sum += crt_ssim
                            mid_sample += 1

                elif opt['network_G']['which_model_G'] == 'TMNet':
                    data = {'LQs': imgs, 'time': torch.tensor([i / 16 for i in range(16)])[None], 'shape': (int(imgs.shape[-2]*4),int(imgs.shape[-1]*4))}
                    model.feed_data(data, False)
                    start_time = time.time()
                    output = model.test(True)
                    end_time = time.time()
                    print(end_time-start_time)
                    bp()
                    outputs = output.data.float().cpu().squeeze(0)  
                    indexed_sample_num += 1
                    for i in range(outputs.shape[0]-1):
                        # bp()
                        output_f = outputs[i,:,:,:].squeeze(0)
                        print(i)
                        output = util.tensor2img(output_f) / 255.
                        output_y = bgr2ycbcr(output, only_y=True)
                        gt_y = bgr2ycbcr(gt[i].copy(), only_y=True)
                        crt_psnr = util.calculate_psnr(output_y * 255, gt_y * 255)
                        crt_ssim = util.ssim(output_y * 255, gt_y * 255)
                        # Image.fromarray((output * 255).astype(np.uint8)).save(os.path.join(out_path, name_list[index + i].split('/')[-1][:-4] + '.jpg'))
                        # print(os.path.join(out_path, name_list[index + i].split('/')[-1][:-4] + '.jpg'), "saved")
                        avg_psnr_sum += crt_psnr
                        avg_ssim_sum += crt_ssim
                        num_sample += 1
                        # indexed_psnr_sum[i] += crt_psnr
                        # indexed_ssim_sum[i] += crt_ssim
                        if i == 4 or i == 0:
                            mid_psnr_sum += crt_psnr
                            mid_ssim_sum += crt_ssim
                            mid_sample += 1
                    data = {'LQs': imgs, 'time': torch.tensor([i / 16 for i in range(9, 16)])[None], 'shape': (int(imgs.shape[-2]*4),int(imgs.shape[-1]*4))}
                    model.feed_data(data, False)
                    output = model.test(True)
                    # bp()
                    outputs = output.data.float().cpu().squeeze(0)  
                    indexed_sample_num += 1
                    for i in range(outputs.shape[0]-1):
                        if i == 0:
                            continue
                        # bp()
                        output_f = outputs[i,:,:,:].squeeze(0)
                        print(i + 8)
                        output = util.tensor2img(output_f) / 255.
                        output_y = bgr2ycbcr(output, only_y=True)
                        gt_y = bgr2ycbcr(gt[i+8].copy(), only_y=True)
                        crt_psnr = util.calculate_psnr(output_y * 255, gt_y * 255)
                        crt_ssim = util.ssim(output_y * 255, gt_y * 255)
                        # Image.fromarray((output * 255).astype(np.uint8)).save(os.path.join(out_path, name_list[index + i].split('/')[-1][:-4] + '.jpg'))
                        # print(os.path.join(out_path, name_list[index + i].split('/')[-1][:-4] + '.jpg'), "saved")
                        avg_psnr_sum += crt_psnr
                        avg_ssim_sum += crt_ssim
                        num_sample += 1
                        # indexed_psnr_sum[i] += crt_psnr
                        # indexed_ssim_sum[i] += crt_ssim
                        if i == 4 or i == 0:
                            mid_psnr_sum += crt_psnr
                            mid_ssim_sum += crt_ssim
                            mid_sample += 1



                elif opt['network_G']['which_model_G'] == 'LunaTokis':
                    data = {'LQs': imgs}
                    model.feed_data(data, False)
                    output = model.test(True)
                    outputs = output.data.float().cpu().squeeze(0) 
                    for i in range(2):
                        output_f = outputs[i,:,:,:].squeeze(0)
                        output = util.tensor2img(output_f) / 255.
                        output_y = bgr2ycbcr(output, only_y=True)
                        gt_y = bgr2ycbcr(gt[i].copy(), only_y=True)
                        crt_psnr = util.calculate_psnr(output_y * 255, gt_y * 255)
                        crt_ssim = util.ssim(output_y * 255, gt_y * 255)
                        avg_psnr_sum += crt_psnr
                        avg_ssim_sum += crt_ssim
                        num_sample += 1
                        if i == 1:
                            mid_psnr_sum += crt_psnr
                            mid_ssim_sum += crt_ssim
                            mid_sample += 1
                if opt['network_G']['which_model_G'] == 'LunaTokis':
                    print("AVG PSNR: {:.4f}, AVG SSIM: {:.4f}".format(avg_psnr_sum / num_sample, avg_ssim_sum / num_sample))
                    print("MID PSNR: {:.4f}, MID SSIM: {:.4f}".format(mid_psnr_sum / mid_sample, mid_ssim_sum / mid_sample))
                else:
                    print("AVG PSNR: {:.4f}, AVG SSIM: {:.4f}".format(avg_psnr_sum / num_sample, avg_ssim_sum / num_sample))
                    print("MID PSNR: {:.4f}, MID SSIM: {:.4f}".format(mid_psnr_sum / mid_sample, mid_ssim_sum / mid_sample))
                    # print([psnr_ / indexed_sample_num for psnr_ in indexed_psnr_sum])
                    # print([ssim_ / indexed_sample_num for ssim_ in indexed_ssim_sum])
                index += 12
            # if index > 100:
            # break    
    avg_psnr_sum = avg_psnr_sum / num_sample
    avg_ssim_sum = avg_ssim_sum / num_sample
    end = time.time()
    print('Epoch [{0}/{1}], Cost time: {2:.2f}s, Val_PSNR:{3:.4f}, Val_SSIM:{4:.4f}'
          .format(epoch_id, 30, end - start, avg_psnr_sum, avg_ssim_sum))
    print("AVG PSNR: {:.4f}, AVG SSIM: {:.4f}".format(avg_psnr_sum / num_sample, avg_ssim_sum / num_sample))
    print("MID PSNR: {:.4f}, MID SSIM: {:.4f}".format(mid_psnr_sum / mid_sample, mid_ssim_sum / mid_sample))
    print([psnr_ / indexed_sample_num for psnr_ in indexed_psnr_sum])
    print([ssim_ / indexed_sample_num for ssim_ in indexed_ssim_sum])
    with open('/home/users/zeyuan_chen/logs_repo/logs_adobe-zoomcus.txt', 'a') as f:
        print('Date: {0}s, Epoch: [{1}/{2}], Cost time: {3:.2f}s, Val_PSNR: {4:.2f}, Val_SSIM: {5:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch_id, iter_id, end - start, avg_psnr_sum, avg_ssim_sum), file=f)
        
    torch.cuda.empty_cache()
    return


def test_metric_adobe_4x(model, opt, epoch_id, iter_id, mode='adobe'):
    torch.cuda.empty_cache()
    if mode == 'adobe':
        path = '/home/users/zeyuan_chen/adobe240fps/test/'
    elif mode == 'gopro':
        path = '/home/users/zeyuan_chen/GOPRO/test/'
    folder_list = os.listdir(path)
    all_name_list = {}
    for folder in folder_list:
        f_path = os.path.join(path, folder)
        if mode == 'adobe':
            frames = os.listdir(f_path)
            frames = sorted([int(frame[:-4]) for frame in frames])
            frames = [str(frame) + '.png' for frame in frames]
        elif mode == 'gopro':
            frames = sorted(os.listdir(f_path))
        all_name_list.update({folder: [os.path.join(f_path, frame) for frame in frames]})
    
    avg_psnr_sum, avg_ssim_sum = 0, 0
    mid_psnr_sum, mid_ssim_sum = 0, 0
    index = 0
    out_path = '/home/users/zeyuan_chen/zoomin-imgs/walkimgs1/Epoch_{}'.format(epoch_id)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    num_sample = 0
    mid_sample = 0

    with torch.no_grad():
        start = time.time()
        for folder in sorted(all_name_list.keys()):
            print(folder)
            # if not folder == 'IMG_0056':
            #     continue
            name_list = all_name_list[folder]
            
            out_path = '/home/users/zeyuan_chen/zoomin-imgs//GOPRO/{}/'.format(folder)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            index = 0
            while index + 4 < len(name_list):
                # if index > 400:
                #     break
                print("{} / {}".format(index, len(name_list)))
                imgs = [cv2.imread(name_list[index]), cv2.imread(name_list[index + 4])]
                imgs = [imresize_np(img_, 1 / 16, True).astype(np.float32) / 255. for img_ in imgs]
                if not opt['network_G']['which_model_G'] == 'LunaTokis':
                    gts = [cv2.imread(name_list[index + i]) for i in range(5)]
                    # gts = [imresize_np(img_, 1 / 2, True).astype(np.float32) / 255. for img_ in gts]
                    gts = [img_.astype(np.float32) / 255. for img_ in gts]
                else:
                    gts = [cv2.imread(name_list[index + i]) for i in range(5)]
                    # gts = [imresize_np(img_, 1 / 2, True).astype(np.float32) / 255. for img_ in gts]
                    gts = [img_.astype(np.float32) / 255. for img_ in gts]

                h_lr, w_lr, h_hr, w_hr = imgs[0].shape[0], imgs[0].shape[1], gts[0].shape[0], gts[0].shape[1]
                if not imgs[0].shape[0] % 4 == 0:
                    h_lr = imgs[0].shape[0] + 4 - imgs[0].shape[0] % 4
                    h_hr = h_lr * 16
                if not imgs[0].shape[1] % 4 == 0:
                    w_lr = imgs[0].shape[1] + 4 - imgs[0].shape[1] % 4
                    w_hr = w_lr * 16
                
                if h_lr != imgs[0].shape[0] or w_lr != imgs[0].shape[1]:
                    imgs = [cv2.resize(img_, (w_lr, h_lr), interpolation=cv2.INTER_LANCZOS4) for img_ in imgs]
                    gts = [cv2.resize(img_, (w_hr, h_hr), interpolation=cv2.INTER_LANCZOS4) for img_ in gts]
                
                gt = np.copy(gts)

                imgs = np.stack(imgs, axis=0)
                imgs = imgs[:, :, :, [2, 1, 0]]
                imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()[None].cuda()
                
                if opt['network_G']['which_model_G'] == 'LIIF':
                    data = {'LQs': imgs, 'time': [torch.tensor([i / 4])[None] for i in range(5)], 'scale': 16}
                    model.feed_data(data, False)
                    output = model.test(True)
                    for i in range(len(output)):
                        output_f = output[i].data.float().cpu().squeeze(0)   
                        output1 = util.tensor2img(output_f) / 255.
                        crt_psnr = util.calculate_psnr(output1 * 255, gt[i] * 255)
                        crt_ssim = util.ssim(output1 * 255, gt[i] * 255)
                        # Image.fromarray((output1[:,:,::-1]*255).astype(np.uint8)).save(os.path.join(out_path, name_list[index + i].split('/')[-1][:-4] + '.jpg'))
                        # print(os.path.join(out_path, name_list[index + i].split('/')[-1][:-4] + '.jpg'), "saved")
                        # Image.fromarray((gt[i][:,:,::-1]*255).astype(np.uint8)).save(os.path.join(out_path, 'GT_iter{}.jpg'.format(i)))
                        avg_psnr_sum += crt_psnr
                        avg_ssim_sum += crt_ssim
                        num_sample += 1
                        if i == 4:
                            mid_psnr_sum += crt_psnr
                            mid_ssim_sum += crt_ssim
                            mid_sample += 1

                elif opt['network_G']['which_model_G'] == 'LunaTokis':
                    data = {'LQs': imgs}
                    model.feed_data(data, False)
                    output0 = model.test(True)
                    for j in range(2):
                        data = {'LQs': output0[:, [j, j+1]]}
                        model.feed_data(data, False)
                        output = model.test(True)
                        outputs = output.data.float().cpu().squeeze(0) 
                        for i in range(outputs.shape[0] - 1):
                            # if j * 2 + i == 0 or j * 2 + i == 4
                            output_f = outputs[i,:,:,:].squeeze(0)
                            output = util.tensor2img(output_f) / 255.
                            output_y = bgr2ycbcr(output, only_y=True)
                            gt_y = bgr2ycbcr(gt[j * 2 + i].copy(), only_y=True)
                            # print(j * 2 + i)
                            crt_psnr = util.calculate_psnr(output_y * 255, gt_y * 255)
                            crt_ssim = util.ssim(output_y * 255, gt_y * 255)
                            # crt_psnr = util.calculate_psnr(output * 255, gt[j * 2 + i] * 255)
                            # crt_ssim = util.ssim(output * 255, gt[j * 2 + i] * 255)
                            avg_psnr_sum += crt_psnr
                            avg_ssim_sum += crt_ssim
                            num_sample += 1
                            mid_psnr_sum += crt_psnr
                            mid_ssim_sum += crt_ssim
                            mid_sample += 1
                

                print("AVG PSNR: {:.4f}, AVG SSIM: {:.4f}".format(avg_psnr_sum / num_sample, avg_ssim_sum / num_sample))
                print("MID PSNR: {:.4f}, MID SSIM: {:.4f}".format(mid_psnr_sum / mid_sample, mid_ssim_sum / mid_sample))
                index += 4
            # if index > 400:
            #     break
    avg_psnr_sum = avg_psnr_sum / num_sample
    avg_ssim_sum = avg_ssim_sum / num_sample
    end = time.time()
    print('Epoch [{0}/{1}], Cost time: {2:.2f}s, Val_PSNR:{3:.4f}, Val_SSIM:{4:.4f}'
          .format(epoch_id, 30, end - start, avg_psnr_sum, avg_ssim_sum))
    
    with open('/home/users/zeyuan_chen/logs_repo/logs_temp.txt', 'a') as f:
        print('Date: {0}s, Epoch: [{1}/{2}], Cost time: {3:.2f}s, Val_PSNR: {4:.2f}, Val_SSIM: {5:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch_id, iter_id, end - start, avg_psnr_sum, avg_ssim_sum), file=f)
        
    torch.cuda.empty_cache()
    return

def test_metric_adobe_liif4x(model, opt, epoch_id, iter_id, mode='gopro'):
    torch.cuda.empty_cache()
    if mode == 'adobe':
        path = '/home/users/zeyuan_chen/adobe240fps/test/'
    elif mode == 'gopro':
        path = '/home/users/zeyuan_chen/GOPRO/test/'
    folder_list = os.listdir(path)
    all_name_list = {}
    for folder in folder_list:
        f_path = os.path.join(path, folder)
        if mode == 'adobe':
            frames = os.listdir(f_path)
            frames = sorted([int(frame[:-4]) for frame in frames])
            frames = [str(frame) + '.png' for frame in frames]
        elif mode == 'gopro':
            frames = sorted(os.listdir(f_path))
        all_name_list.update({folder: [os.path.join(f_path, frame) for frame in frames]})
    
    avg_psnr_sum, avg_ssim_sum = 0, 0
    mid_psnr_sum, mid_ssim_sum = 0, 0
    index = 0
    out_path = '/home/users/zeyuan_chen/zoomin-imgs/walkimgs1/Epoch_{}'.format(epoch_id)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    num_sample = 0
    mid_sample = 0

    with torch.no_grad():
        start = time.time()
        for folder in sorted(all_name_list.keys()):
            print(folder)
            # if not folder == 'IMG_0056':
            #     continue
            name_list = all_name_list[folder]
            
            out_path = '/home/users/zeyuan_chen/zoomin-imgs/TMNet-temp/{}/'.format(folder)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            index = 0
            # bp()
            while index + 4 < len(name_list):
                # if index > 400:
                #     break
                print("{} / {}".format(index, len(name_list)))
                # imgs = [cv2.imread(name_list[index]), cv2.imread(name_list[index + 8]), cv2.imread(name_list[index + 16]), cv2.imread(name_list[index + 24])]
                imgs = [cv2.imread(name_list[index]), cv2.imread(name_list[index + 4])]
                imgs = [imresize_np(img_, 1 / 16, True).astype(np.float32) / 255. for img_ in imgs]

                gts = [cv2.imread(name_list[index + i]).astype(np.float32) / 255. for i in range(1,4)]
                # gts = [img_.astype(np.float32) / 255. for img_ in gts]
                # gts = [imresize_np(img_, 1 / 2, True).astype(np.float32) / 255. for img_ in gts]

                h_lr, w_lr, h_hr, w_hr = imgs[0].shape[0], imgs[0].shape[1], gts[0].shape[0], gts[0].shape[1]
                h_hr, w_hr = h_lr * 16, w_lr * 16
                if not imgs[0].shape[0] % 4 == 0:
                    h_lr = imgs[0].shape[0] + 4 - imgs[0].shape[0] % 4
                    h_hr = h_lr * 16
                if not imgs[0].shape[1] % 4 == 0:
                    w_lr = imgs[0].shape[1] + 4 - imgs[0].shape[1] % 4
                    w_hr = w_lr * 16
                
                if h_lr != imgs[0].shape[0] or w_lr != imgs[0].shape[1]:
                    imgs = [cv2.resize(img_, (w_lr, h_lr), interpolation=cv2.INTER_LANCZOS4) for img_ in imgs]
                    gts = [cv2.resize(img_, (w_hr, h_hr), interpolation=cv2.INTER_LANCZOS4) for img_ in gts]
 
                gt = np.copy(gts)

                imgs = np.stack(imgs, axis=0)
                imgs = imgs[:, :, :, [2, 1, 0]]
                imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()[None].cuda()
                
                if opt['network_G']['which_model_G'] == 'LIIF' or opt['network_G']['which_model_G'] == 'LIIF_test1':
                    data = {'LQs': imgs, 'time': [torch.tensor([i / 4])[None] for i in range(1,4)], 'scale': (int(imgs.shape[-2] * 16), int(imgs.shape[-1] * 16)), 'test': True}
                    model.feed_data(data, False)
                    output = model.test(True)
                    # bp()
                    for i in range(len(output)):
                        output_f = output[i].data.float().cpu().squeeze(0)   
                        output1 = util.tensor2img(output_f) / 255.
                        output_y = bgr2ycbcr(output1, only_y=True)
                        gt_y = bgr2ycbcr(gt[i].copy(), only_y=True)
                        crt_psnr = util.calculate_psnr(output_y * 255, gt_y * 255)
                        crt_ssim = util.ssim(output_y * 255, gt_y * 255)
                        # crt_psnr = util.calculate_psnr(output1 * 255, gt[i] * 255)
                        # crt_ssim = util.ssim(output1 * 255, gt[i] * 255)
                        # print(crt_psnr, crt_ssim)
                        # Image.fromarray((output1[:,:,::-1]*255).astype(np.uint8)).save(os.path.join(out_path, name_list[index + i].split('/')[-1][:-4] + '.jpg'))
                        # print(os.path.join(out_path, name_list[index + i].split('/')[-1][:-4] + '.jpg'), "saved")
                        # Image.fromarray((gt[i][:,:,::-1]*255).astype(np.uint8)).save(os.path.join(out_path, 'GT_iter{}.jpg'.format(i)))
                        avg_psnr_sum += crt_psnr
                        avg_ssim_sum += crt_ssim
                        num_sample += 1
                        # if i == 4 or i == 12 or i == 20:
                        mid_psnr_sum += crt_psnr
                        mid_ssim_sum += crt_ssim
                        mid_sample += 1

                elif opt['network_G']['which_model_G'] == 'TMNet':
                    data = {'LQs': imgs, 'time': torch.tensor([i / 8 for i in range(1, 8)])[None], 'shape': (int(imgs.shape[-2]*4),int(imgs.shape[-1]*4))}
                    model.feed_data(data, False)
                    output = model.test(True)
                    # bp()
                    outputs = output.data.float().cpu().squeeze(0)  
                    for i in range(outputs.shape[0] - 1):
                        output_f = outputs[i,:,:,:].squeeze(0)
                    
                        output = util.tensor2img(output_f) / 255.
                        crt_psnr = util.calculate_psnr(output * 255, gt[i] * 255)
                        crt_ssim = util.ssim(output * 255, gt[i] * 255)
                        Image.fromarray((output * 255).astype(np.uint8)).save(os.path.join(out_path, name_list[index + i].split('/')[-1][:-4] + '.jpg'))
                        print(os.path.join(out_path, name_list[index + i].split('/')[-1][:-4] + '.jpg'), "saved")
                        avg_psnr_sum += crt_psnr
                        avg_ssim_sum += crt_ssim
                        num_sample += 1
                        if i == 4:
                            mid_psnr_sum += crt_psnr
                            mid_ssim_sum += crt_ssim
                            mid_sample += 1
                elif opt['network_G']['which_model_G'] == 'LunaTokis':
                    data = {'LQs': imgs}
                    model.feed_data(data, False)
                    output = model.test(True)
                    outputs = output.data.float().cpu().squeeze(0) 
                    for i in range(3):
                        output_f = outputs[i,:,:,:].squeeze(0)
                        output = util.tensor2img(output_f) / 255.
                        crt_psnr = util.calculate_psnr(output * 255, gt[i] * 255)
                        crt_ssim = util.ssim(output * 255, gt[i] * 255)
                        avg_psnr_sum += crt_psnr
                        avg_ssim_sum += crt_ssim
                        num_sample += 1
                        if i == 1:
                            mid_psnr_sum += crt_psnr
                            mid_ssim_sum += crt_ssim
                            mid_sample += 1

                print("AVG PSNR: {:.4f}, AVG SSIM: {:.4f}".format(avg_psnr_sum / num_sample, avg_ssim_sum / num_sample))
                print("MID PSNR: {:.4f}, MID SSIM: {:.4f}".format(mid_psnr_sum / mid_sample, mid_ssim_sum / mid_sample))
                index += 4
                
    avg_psnr_sum = avg_psnr_sum / num_sample
    avg_ssim_sum = avg_ssim_sum / num_sample
    end = time.time()
    print('Epoch [{0}/{1}], Cost time: {2:.2f}s, Val_PSNR:{3:.4f}, Val_SSIM:{4:.4f}'
          .format(epoch_id, 30, end - start, avg_psnr_sum, avg_ssim_sum))
    
    with open('/home/users/zeyuan_chen/logs_repo/logs_adobe-zoomcus.txt', 'a') as f:
        print('Date: {0}s, Epoch: [{1}/{2}], Cost time: {3:.2f}s, Val_PSNR: {4:.2f}, Val_SSIM: {5:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch_id, iter_id, end - start, avg_psnr_sum, avg_ssim_sum), file=f)
        
    torch.cuda.empty_cache()
    return


def test_metric_adobe_tmnet(model, opt, epoch_id, iter_id, mode='gopro'):
    torch.cuda.empty_cache()
    if mode == 'adobe':
        path = '/home/users/zeyuan_chen/adobe240fps/test/'
    elif mode == 'gopro':
        path = '/home/users/zeyuan_chen/GOPRO/test/'
    folder_list = os.listdir(path)
    all_name_list = {}
    for folder in folder_list:
        f_path = os.path.join(path, folder)
        if mode == 'adobe':
            frames = os.listdir(f_path)
            frames = sorted([int(frame[:-4]) for frame in frames])
            frames = [str(frame) + '.png' for frame in frames]
        elif mode == 'gopro':
            frames = sorted(os.listdir(f_path))
        all_name_list.update({folder: [os.path.join(f_path, frame) for frame in frames]})
    
    avg_psnr_sum, avg_ssim_sum = 0, 0
    mid_psnr_sum, mid_ssim_sum = 0, 0
    index = 0
    out_path = '/home/users/zeyuan_chen/zoomin-imgs/walkimgs1/Epoch_{}'.format(epoch_id)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    num_sample = 0
    mid_sample = 0

    with torch.no_grad():
        start = time.time()
        for folder in sorted(all_name_list.keys()):
            print(folder)
            # if not folder == 'GOPR0384_11_00':
            #     continue
            name_list = all_name_list[folder]
            
            out_path = '/home/users/zeyuan_chen/zoomin-imgs/TMNet-{}x/{}/'.format(24, folder)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            index = 0
            # bp()
            while index + 18 < len(name_list):
                # if index > 400:
                #     break
                print("{} / {}".format(index, len(name_list)))
                imgs = [cv2.imread(name_list[index]), cv2.imread(name_list[index + 6]), cv2.imread(name_list[index + 12]), cv2.imread(name_list[index + 18])]
                # imgs = [cv2.imread(name_list[index]), cv2.imread(name_list[index + 6])]
                imgs = [imresize_np(img_, 1 / 8, True).astype(np.float32) / 255. for img_ in imgs]

                gts = [cv2.imread(name_list[index + i]) for i in range(18)]
                # gts = [img_.astype(np.float32) / 255. for img_ in gts]
                gts = [imresize_np(img_, 1 / 2, True).astype(np.float32) / 255. for img_ in gts]

                h_lr, w_lr, h_hr, w_hr = imgs[0].shape[0], imgs[0].shape[1], gts[0].shape[0], gts[0].shape[1]
                if not imgs[0].shape[0] % 4 == 0:
                    h_lr = imgs[0].shape[0] + 4 - imgs[0].shape[0] % 4
                    h_hr = h_lr * 4
                if not imgs[0].shape[1] % 4 == 0:
                    w_lr = imgs[0].shape[1] + 4 - imgs[0].shape[1] % 4
                    w_hr = w_lr * 4
                
                if h_lr != imgs[0].shape[0] or w_lr != imgs[0].shape[1]:
                    imgs = [cv2.resize(img_, (w_lr, h_lr), interpolation=cv2.INTER_LANCZOS4) for img_ in imgs]
                    gts = [cv2.resize(img_, (w_hr, h_hr), interpolation=cv2.INTER_LANCZOS4) for img_ in gts]
 
                gt = np.copy(gts)

                imgs = np.stack(imgs, axis=0)
                imgs = imgs[:, :, :, [2, 1, 0]]
                imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()[None].cuda()
        
                data = {'LQs': imgs, 'time': torch.tensor([i / 6 for i in range(1, 6)])[None]}
                model.feed_data(data, False)
                output = model.test(True).clamp(0., 1.)
                
                outputs = output.data.float().cpu().squeeze(0)  
                for i in range(outputs.shape[0] - 1):
                    output_f = outputs[i,:,:,:].squeeze(0)
                
                    output = util.tensor2img(output_f) / 255.
                    # output_y = bgr2ycbcr(output, only_y=True)
                    # gt_y = bgr2ycbcr(gt[i], only_y=True)
                    crt_psnr = util.calculate_psnr(output * 255, gt[i] * 255)
                    crt_ssim = util.ssim(output * 255, gt[i] * 255)
                    # Image.fromarray((output[:, :, [2,1,0]] * 255).astype(np.uint8)).save(os.path.join(out_path, name_list[index + i].split('/')[-1][:-4] + '.jpg'))
                    # print(os.path.join(out_path, name_list[index + i].split('/')[-1][:-4] + '.jpg'), "saved")
                    avg_psnr_sum += crt_psnr
                    avg_ssim_sum += crt_ssim
                    num_sample += 1
                    if i == 4:
                        mid_psnr_sum += crt_psnr
                        mid_ssim_sum += crt_ssim
                        mid_sample += 1

                print("AVG PSNR: {:.4f}, AVG SSIM: {:.4f}".format(avg_psnr_sum / num_sample, avg_ssim_sum / num_sample))
                print("MID PSNR: {:.4f}, MID SSIM: {:.4f}".format(mid_psnr_sum / mid_sample, mid_ssim_sum / mid_sample))
                index += 18
                
    avg_psnr_sum = avg_psnr_sum / num_sample
    avg_ssim_sum = avg_ssim_sum / num_sample
    end = time.time()
    print('Epoch [{0}/{1}], Cost time: {2:.2f}s, Val_PSNR:{3:.4f}, Val_SSIM:{4:.4f}'
          .format(epoch_id, 30, end - start, avg_psnr_sum, avg_ssim_sum))
    
    with open('/home/users/zeyuan_chen/logs_repo/logs_adobe-zoomcus.txt', 'a') as f:
        print('Date: {0}s, Epoch: [{1}/{2}], Cost time: {3:.2f}s, Val_PSNR: {4:.2f}, Val_SSIM: {5:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch_id, iter_id, end - start, avg_psnr_sum, avg_ssim_sum), file=f)
        
    torch.cuda.empty_cache()
    return

def test_metric_full_tmnet(model, opt, epoch_id, iter_id):
    torch.cuda.empty_cache()
    path = '/home/users/zeyuan_chen/VID4/LR/'
    path2 = '/home/users/zeyuan_chen/VID4/HR/'
    folder_list = os.listdir(path)
    all_name_list = {}
    all_input_list = {}
    for folder in folder_list:
        f_path = os.path.join(path, folder)
        f_path2 = os.path.join(path2, folder)
        all_input_list.update({folder: [os.path.join(f_path, name) for name in sorted(os.listdir(f_path))[::2]]})
        all_name_list.update({folder: [os.path.join(f_path2, name) for name in sorted(os.listdir(f_path2))]})
    
    avg_psnr_sum, avg_ssim_sum = 0, 0
    index = 0
    num_sample = 0

    with torch.no_grad():
        start = time.time()
        for folder in all_name_list.keys():
            print(folder)
            name_list = all_name_list[folder]
            input_list = all_input_list[folder]
            
            out_path = '/home/users/zeyuan_chen/zoomin-imgs/vid4/TMNet-{}x2/{}/'.format(24, folder)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out_path2 = '/home/users/zeyuan_chen/zoomin-imgs/vid4/TMNet-multi{}/{}/'.format(24, folder)
            if not os.path.exists(out_path2):
                os.makedirs(out_path2)
            for i in range(len(input_list) - 3):
                imgs = [cv2.imread(input_list[i]).astype(np.float32) / 255.,
                        cv2.imread(input_list[i+1]).astype(np.float32) / 255.,
                        cv2.imread(input_list[i+2]).astype(np.float32) / 255.,
                        cv2.imread(input_list[i+3]).astype(np.float32) / 255.,]
                # imgs = [cv2.imread(input_list[i]).astype(np.float32) / 255.,
                #         cv2.imread(input_list[i+1]).astype(np.float32) / 255.]
                print(input_list[i], name_list[2 * i + 1])
                gt1 = [cv2.imread(name_list[2 * i + j]).astype(np.float32) / 255. for j in range(6)]
                gt = np.copy(gt1)

                imgs = np.stack(imgs, axis=0)
                imgs = imgs[:, :, :, [2, 1, 0]]
                imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()[None].cuda()
                [torch.tensor([1 / 2])[None]]
                data = {'LQs': imgs, 'time': torch.tensor([i / 8 for i in range(1, 8)])[None], 'shape': (int(imgs.shape[-2]*4),int(imgs.shape[-1]*4)), 'test': True}
                model.feed_data(data, False)
                output = model.test(True)
                outputs = output.data.float().cpu().squeeze(0)   
                print(outputs.shape[0])
                bp()
                for j in range(outputs.shape[0] - 1):
                    output_f = outputs[j, :, :, :].squeeze(0)
                    output = util.tensor2img(output_f) / 255.
                    if not j in [0, 4, 8, 12, 16, 20]:
                        Image.fromarray((output[:, :, [2,1,0]] * 255).astype(np.uint8)).save(os.path.join(out_path2, str(index) + '.jpg'))
                        index += 1
                        print(os.path.join(out_path2, str(index) + '.jpg'), "saved")
                        continue
                    Image.fromarray((output[:, :, [2,1,0]] * 255).astype(np.uint8)).save(os.path.join(out_path, name_list[2 * i + j].split('/')[-1][:-4] + '.jpg'))
                    print(os.path.join(out_path, name_list[2 * i + j].split('/')[-1][:-4] + '.jpg'), "saved")
                    output_y = bgr2ycbcr(output, only_y=True)
                    print(j, j // 4)
                    gt_y = bgr2ycbcr(gt[j//4], only_y=True)
                    crt_psnr = util.calculate_psnr(output_y * 255, gt_y * 255)
                    crt_ssim = util.ssim(output_y * 255, gt_y * 255)
                    avg_psnr_sum += crt_psnr
                    avg_ssim_sum += crt_ssim
                    num_sample += 1
                print("PSNR: {:4f}, SSIM: {:.4f}".format(avg_psnr_sum / num_sample, avg_ssim_sum / num_sample))
                
    avg_psnr_sum = avg_psnr_sum / num_sample
    avg_ssim_sum = avg_ssim_sum / num_sample
    end = time.time()
    print('Epoch [{0}/{1}], Cost time: {2:.2f}s, Val_PSNR:{3:.4f}, Val_SSIM:{4:.4f}'
          .format(epoch_id, 30, end - start, avg_psnr_sum, avg_ssim_sum))
    
    with open('/home/users/zeyuan_chen/logs_repo/logs-shi145e986.txt', 'a') as f:
        print('Date: {0}s, Epoch: [{1}/{2}], Cost time: {3:.2f}s, Val_PSNR: {4:.2f}, Val_SSIM: {5:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch_id, iter_id, end - start, avg_psnr_sum, avg_ssim_sum), file=f)
        
    torch.cuda.empty_cache()
    return


def test_metric_time(model, opt, epoch_id, iter_id, mode='gopro'):
    torch.cuda.empty_cache()
    if mode == 'adobe':
        path = '/home/users/zeyuan_chen/adobe240fps/test/'
    elif mode == 'gopro':
        path = '/home/users/zeyuan_chen/GOPRO/test/'
    folder_list = os.listdir(path)
    all_name_list = {}
    for folder in folder_list:
        f_path = os.path.join(path, folder)
        if mode == 'adobe':
            frames = os.listdir(f_path)
            frames = sorted([int(frame[:-4]) for frame in frames])
            frames = [str(frame) + '.png' for frame in frames]
        elif mode == 'gopro':
            frames = sorted(os.listdir(f_path))
        all_name_list.update({folder: [os.path.join(f_path, frame) for frame in frames]})
    
    avg_psnr_sum, avg_ssim_sum = 0, 0
    mid_psnr_sum, mid_ssim_sum = 0, 0
    index = 0
    out_path = '/home/users/zeyuan_chen/zoomin-imgs/walkimgs1/Epoch_{}'.format(epoch_id)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    num_sample = 0
    mid_sample = 0
    indexed_psnr_sum = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    indexed_ssim_sum = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    indexed_sample_num = 0
    scale0 = 12
    total_time = 0
    with torch.no_grad():
        start = time.time()
        for folder in sorted(all_name_list.keys()):
            print(folder)
            name_list = all_name_list[folder]
            
            out_path = '/home/users/zeyuan_chen/zoomin-imgs/TMNet-8x2/{}/'.format(folder)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            index = 0
            # bp()
            
            while index + 8 < len(name_list):
                # if index > 100:
                    # break
                print("{} / {}".format(index, len(name_list)))
                # imgs = [cv2.imread(name_list[index]), cv2.imread(name_list[index + 8]), cv2.imread(name_list[index + 16]), cv2.imread(name_list[index + 24])]
                imgs = [cv2.imread(name_list[index]), cv2.imread(name_list[index + 8])]
                imgs = [imresize_np(img_, 1 / scale0, True).astype(np.float32) / 255. for img_ in imgs]
                if not opt['network_G']['which_model_G'] == 'LunaTokis':
                    gts = [cv2.imread(name_list[index + i]) for i in range(8)]
                    # gts = [img_.astype(np.float32) / 255. for img_ in gts]
                    gts = [imresize_np(img_, 1 / 4, True).astype(np.float32) / 255. for img_ in gts]
                else:
                    gts = [cv2.imread(name_list[index + i]) for i in [0, 4, 8]]
                    gts = [img_.astype(np.float32) / 255. for img_ in gts]
                    # gts = [imresize_np(img_, 1 / 2, True).astype(np.float32) / 255. for img_ in gts]

                h_lr, w_lr, h_hr, w_hr = imgs[0].shape[0], imgs[0].shape[1], gts[0].shape[0], gts[0].shape[1]
                h_hr, w_hr = h_lr * 4, w_lr * 4
                if not imgs[0].shape[0] % 4 == 0:
                    h_lr = imgs[0].shape[0] + 4 - imgs[0].shape[0] % 4
                    h_hr = h_lr * 4
                if not imgs[0].shape[1] % 4 == 0:
                    w_lr = imgs[0].shape[1] + 4 - imgs[0].shape[1] % 4
                    w_hr = w_lr * 4
                
                if h_lr != imgs[0].shape[0] or w_lr != imgs[0].shape[1]:
                    imgs = [cv2.resize(img_, (w_lr, h_lr), interpolation=cv2.INTER_LANCZOS4) for img_ in imgs]
                    gts = [cv2.resize(img_, (w_hr, h_hr), interpolation=cv2.INTER_LANCZOS4) for img_ in gts]
                imgs = [cv2.resize(img_, (32, 32), interpolation=cv2.INTER_LANCZOS4) for img_ in imgs]
                gts = [cv2.resize(img_, (240, 240), interpolation=cv2.INTER_LANCZOS4) for img_ in gts]
                
                gt = np.copy(gts)

                imgs = np.stack(imgs, axis=0)
                imgs = imgs[:, :, :, [2, 1, 0]]
                imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()[None].cuda()
                
                if opt['network_G']['which_model_G'] == 'LIIF' or opt['network_G']['which_model_G'] == 'LIIF_test1':
                    
                    data = {'LQs': imgs, 'time': [torch.tensor([i / 8])[None] for i in range(8)], 'scale': (int(imgs.shape[-2] * 4), int(imgs.shape[-1] * 4)), 'test': True}
                    model.feed_data(data, False)
                    print("BEGIN")
                    start_time = time.time()
                    output = model.test(True)
                    end_time = time.time()
                    del output, imgs, gt, gts
                    torch.cuda.empty_cache() 
                    print(index)
                    if index == 0:
                        index += 4
                        continue
                    total_time += (end_time-start_time)
                    num_sample += 1
                    print((end_time-start_time))
                    print("AVG TIME: ", total_time / num_sample)

                elif opt['network_G']['which_model_G'] == 'TMNet':
                    data = {'LQs': imgs, 'time': torch.tensor([i / 15 for i in range(1, 15)])[None], 'shape': (int(imgs.shape[-2]*6),int(imgs.shape[-1]*6))}
                    # print(data['time'])
                    model.feed_data(data, False)
                    start_time = time.time()
                    output = model.test(True)
                    end_time = time.time()
                    # bp()
                    print(index)
                    if index == 0:
                        index += 4
                        continue
                    total_time += (end_time-start_time)
                    num_sample += 1
                    print((end_time-start_time))
                    print("AVG TIME: ", total_time / num_sample)


                elif opt['network_G']['which_model_G'] == 'LunaTokis':
                    data = {'LQs': imgs}
                    model.feed_data(data, False)
                    start_time = time.time()
                    output = model.test(True)
                    end_time = time.time()
                    print(index)
                    if index == 0:
                        index += 8
                        continue
                    total_time += (end_time-start_time)
                    num_sample += 1
                    print((end_time-start_time))
                    print("AVG TIME: ", total_time / num_sample)
                index += 4
            break    
    print("AVG TIME: ", total_time / num_sample)
    avg_psnr_sum = avg_psnr_sum / num_sample
    avg_ssim_sum = avg_ssim_sum / num_sample
    end = time.time()
    print('Epoch [{0}/{1}], Cost time: {2:.2f}s, Val_PSNR:{3:.4f}, Val_SSIM:{4:.4f}'
          .format(epoch_id, 30, end - start, avg_psnr_sum, avg_ssim_sum))
    print("AVG PSNR: {:.4f}, AVG SSIM: {:.4f}".format(avg_psnr_sum / num_sample, avg_ssim_sum / num_sample))
    print("MID PSNR: {:.4f}, MID SSIM: {:.4f}".format(mid_psnr_sum / mid_sample, mid_ssim_sum / mid_sample))
    print([psnr_ / indexed_sample_num for psnr_ in indexed_psnr_sum])
    print([ssim_ / indexed_sample_num for ssim_ in indexed_ssim_sum])
    with open('/home/users/zeyuan_chen/logs_repo/logs_adobe-zoomcus.txt', 'a') as f:
        print('Date: {0}s, Epoch: [{1}/{2}], Cost time: {3:.2f}s, Val_PSNR: {4:.2f}, Val_SSIM: {5:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch_id, iter_id, end - start, avg_psnr_sum, avg_ssim_sum), file=f)
        
    torch.cuda.empty_cache()
    return