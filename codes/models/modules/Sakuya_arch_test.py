''' network architecture for Sakuya '''
import functools
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from models.modules.convlstm import ConvLSTM, ConvLSTMCell
try:
    from models.modules.DCNv2.dcn_v2 import DCN_sep
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')
import models.modules.liif_models as liif_models
from pdb import set_trace as bp
from models.modules.SIREN import Siren
from models.modules.warplayer import warp, warpgrid, warpgrid2
import numpy as np
from PIL import Image

class PCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''
    def __init__(self, nf=64, groups=8):
        super(PCD_Align, self).__init__()

        # fea1
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack_1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        self.L2_fea_conv_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_1 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        self.L1_fea_conv_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # fea2 
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack_2 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_2 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        self.L2_fea_conv_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_2 = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        self.L1_fea_conv_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, fea1, fea2):
        '''align other neighboring frames to the reference frame in the feature level
        fea1, fea2: [L1, L2, L3], each with [B,C,H,W] features
        estimate offset bidirectionally
        '''
        y = []
        # param. of fea1
        # L3
        L3_offset = torch.cat([fea1[2], fea2[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2_1(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack_1(fea1[2], L3_offset))
        # L2
        L2_offset = torch.cat([fea1[1], fea2[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2_1(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3_1(L2_offset))
        L2_fea = self.L2_dcnpack_1(fea1[1], L2_offset)
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv_1(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([fea1[0], fea2[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2_1(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3_1(L1_offset))
        L1_fea = self.L1_dcnpack_1(fea1[0], L1_offset)
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv_1(torch.cat([L1_fea, L2_fea], dim=1))
        y.append(L1_fea)

        # param. of fea2
        # L3
        L3_offset = torch.cat([fea2[2], fea1[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_2(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2_2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack_2(fea2[2], L3_offset))
        # L2
        L2_offset = torch.cat([fea2[1], fea1[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_2(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2_2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3_2(L2_offset))
        L2_fea = self.L2_dcnpack_2(fea2[1], L2_offset)
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv_2(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([fea2[0], fea1[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_2(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2_2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3_2(L1_offset))
        L1_fea = self.L1_dcnpack_2(fea2[0], L1_offset)
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv_2(torch.cat([L1_fea, L2_fea], dim=1))
        y.append(L1_fea)
        
        y = torch.cat(y, dim=1)
        return y

class Easy_PCD(nn.Module):
    def __init__(self, nf=64, groups=8):
        super(Easy_PCD, self).__init__()

        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.pcd_align = PCD_Align(nf=nf, groups=groups)
        self.fusion = nn.Conv2d(2 * nf, nf, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, f1, f2):
        # input: extracted features
        # feature size: f1 = f2 = [B, N, C, H, W]
        # print(f1.size())
        L1_fea = torch.stack([f1, f2], dim=1)
        B, N, C, H, W = L1_fea.size()
        L1_fea = L1_fea.view(-1, C, H, W)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        fea1 = [L1_fea[:, 0, :, :, :].clone(), L2_fea[:, 0, :, :, :].clone(), L3_fea[:, 0, :, :, :].clone()]
        fea2 = [L1_fea[:, 1, :, :, :].clone(), L2_fea[:, 1, :, :, :].clone(), L3_fea[:, 1, :, :, :].clone()]
        aligned_fea = self.pcd_align(fea1, fea2)
        fusion_fea = self.fusion(aligned_fea) # [B, N, C, H, W]
        return fusion_fea

class DeformableConvLSTM(ConvLSTM):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, front_RBs, groups,
                 batch_first=False, bias=True, return_all_layers=False):
        ConvLSTM.__init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, 
              batch_first=batch_first, bias=bias, return_all_layers=return_all_layers)
        #### extract features (for each frame)
        nf = input_dim

        self.pcd_h = Easy_PCD(nf=nf, groups=groups)
        self.pcd_c = Easy_PCD(nf=nf, groups=groups)        

        cell_list = []
        for i in range(0, num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, input_tensor, hidden_state = None):
        '''        
        Parameters
        ----------
        input_tensor: 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: 
            None. 
            
        Returns
        -------
        last_state_list, layer_output
        '''
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            tensor_size = (input_tensor.size(3),input_tensor.size(4))
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0),tensor_size=tensor_size)
        
        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                in_tensor = cur_layer_input[:, t, :, :, :] 
                h_temp = self.pcd_h(in_tensor, h)
                c_temp = self.pcd_c(in_tensor, c)
                h, c = self.cell_list[layer_idx](input_tensor=in_tensor,
                                                 cur_state=[h_temp, c_temp])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]

        return layer_output_list, last_state_list
    
    def _init_hidden(self, batch_size, tensor_size):
        return super()._init_hidden(batch_size, tensor_size)

class BiDeformableConvLSTM(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, front_RBs, groups,
                 batch_first=False, bias=True, return_all_layers=False):
        super(BiDeformableConvLSTM, self).__init__()
        self.forward_net = DeformableConvLSTM(input_size=input_size, input_dim=input_dim, hidden_dim=hidden_dim,
                                           kernel_size=kernel_size, num_layers=num_layers, front_RBs=front_RBs,
                                           groups=groups, batch_first=batch_first, bias=bias, return_all_layers=return_all_layers)
        self.conv_1x1 = nn.Conv2d(2*input_dim, input_dim, 1, 1, bias=True)

    def forward(self, x):
        reversed_idx = list(reversed(range(x.shape[1])))
        x_rev = x[:, reversed_idx, ...]
        out_fwd, _ = self.forward_net(x)
        out_rev, _ = self.forward_net(x_rev)
        rev_rev = out_rev[0][:, reversed_idx, ...]
        B, N, C, H, W = out_fwd[0].size()
        result = torch.cat((out_fwd[0], rev_rev), dim=2)
        result = result.view(B*N,-1,H,W)
        result = self.conv_1x1(result)
        return result.view(B, -1, C, H, W) 

class LunaTokis(nn.Module):
    def __init__(self, nf=64, nframes=3, groups=8, front_RBs=5, back_RBs=10):
        super(LunaTokis, self).__init__()
        self.nf = nf
        self.in_frames = 1 + nframes // 2
        self.ot_frames = nframes
        p_size = 48 # a place holder, not so useful
        patch_size = (p_size, p_size) 
        n_layers = 1
        hidden_dim = []
        for i in range(n_layers):
            hidden_dim.append(nf)

        ResidualBlock_noBN_f = functools.partial(mutil.ResidualBlock_noBN, nf=nf)
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction = mutil.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.pcd_align = PCD_Align(nf=nf, groups=groups)
        self.fusion = nn.Conv2d(2 * nf, nf, 1, 1, bias=True)
        self.ConvBLSTM = BiDeformableConvLSTM(input_size=patch_size, input_dim=nf, hidden_dim=hidden_dim, \
            kernel_size=(3,3), num_layers=1, batch_first=True, front_RBs=front_RBs, groups=groups)
        #### reconstruction
        self.recon_trunk = mutil.make_layer(ResidualBlock_noBN_f, back_RBs)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        imnet_spec2 = {'name': 'mlp', 'args': {'out_dim': 3, 'hidden_list': [64, 64, 256, 256]}}
        # self.encode_imnet = liif_models.make(imnet_spec2, args={'in_dim': 194})
        self.feat_imnet = Siren(in_features=201, out_features=64, hidden_features=[64, 64, 256], 
                  hidden_layers=2, outermost_linear=True)
        self.flow_imnet = Siren(in_features=65+192+6, out_features=4, hidden_features=[64, 64, 256], 
                  hidden_layers=2, outermost_linear=True)
        self.encode_imnet = Siren(in_features=141+192*2, out_features=3, hidden_features=[64, 64, 256, 256], 
                  hidden_layers=3, outermost_linear=True)

    def gen_feat(self, x):
        self.inp = x
        B, N, C, H, W = x.size()  # N input video frames
        #### extract LR features
        # L1
        L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
        L1_fea = self.feature_extraction(L1_fea)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))
        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        #### align using pcd
        to_lstm_fea = []
        '''
        0: + fea1, fusion_fea, fea2
        1: + ...    ...        ...  fusion_fea, fea2
        2: + ...    ...        ...    ...       ...   fusion_fea, fea2
        '''
        for idx in range(N-1):
            fea1 = [
                L1_fea[:, idx, :, :, :].clone(), L2_fea[:, idx, :, :, :].clone(), L3_fea[:, idx, :, :, :].clone()
            ]
            fea2 = [
                L1_fea[:, idx+1, :, :, :].clone(), L2_fea[:, idx+1, :, :, :].clone(), L3_fea[:, idx+1, :, :, :].clone()
            ]
            aligned_fea = self.pcd_align(fea1, fea2)

            fusion_fea = self.fusion(aligned_fea) # [B, N, C, H, W]
            if idx == 0:
                to_lstm_fea.append(fea1[0])
            to_lstm_fea.append(fusion_fea)
            to_lstm_fea.append(fea2[0])
        lstm_feats = torch.stack(to_lstm_fea, dim = 1)
        #### align using bidirectional deformable conv-lstm
        feats = self.ConvBLSTM(lstm_feats) 
        B, T, C, H, W = feats.size()

        feats = feats.view(B*T, C, H, W)
        out = self.recon_trunk(feats)
        
        ###############################################
        out = out.view(B, T, 64, H, W)
        self.feat = out
        return 
    
    def decoding(self, times=None, scale=None):
        feat = torch.cat([self.feat[:, 0], self.feat[:, 1], self.feat[:, 2]], dim=1)
        
        bs, C, H, W = feat.shape
        if scale is None:
            HH, WW = H * 4, W * 4
        else:
            HH, WW = scale[0], scale[1]
        coord_lowres = make_coord((H, W)).repeat(bs, 1, 1).clamp(-1 + 1e-6, 1 - 1e-6).cuda()
        coord_highres = make_coord((HH, WW)).repeat(bs, 1, 1).clamp(-1 + 1e-6, 1 - 1e-6).cuda()
        
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])        

        preds = []
        for c in range(len(times)):
            qs = coord_highres.shape[1]
            q_feat = F.grid_sample(
                feat, coord_highres.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_inp = F.grid_sample(
                self.inp.view(feat.shape[0], -1, feat.shape[2], feat.shape[3]), coord_highres.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_coord = F.grid_sample(
                feat_coord, coord_highres.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            rel_coord = coord_highres - q_coord
            rel_coord[:, :, 0] *= feat.shape[-2]
            rel_coord[:, :, 1] *= feat.shape[-1]
            pe_coord = torch.ones_like(coord_highres[:, :, 0].unsqueeze(2)) * times[c].unsqueeze(2)
            
            inp = torch.cat([q_feat, q_inp, rel_coord, pe_coord], dim=-1)
            HRfeat = self.feat_imnet(inp.view(bs * qs, -1)).view(bs, qs, -1)
            HRfeat = HRfeat.permute(0, 2, 1).view(bs, 64, HH, WW)
            HRinp = self.inp.view(feat.shape[0], -1, feat.shape[2], feat.shape[3])
            # HRinp = F.upsample(HRinp, scale_factor=4, mode='bilinear')
            del q_coord, rel_coord, inp
            torch.cuda.empty_cache()
            q_feat = F.grid_sample(
                HRfeat, coord_highres.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_inp = F.grid_sample(
                HRinp, coord_highres.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_feat0 = F.grid_sample(
                feat, coord_highres.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            flow_inp = torch.cat([q_feat, q_feat0, q_inp, pe_coord], dim=-1)
            flow_pred = self.flow_imnet(flow_inp.view(bs * qs, -1)).view(bs, qs, -1)
            del q_feat, q_inp, q_feat0, flow_inp
            torch.cuda.empty_cache()
            flow_pred = flow_pred.permute(0, 2, 1).view(bs, 4, HH, WW)
            
            grid1, _ = warpgrid(self.inp[:, 0], flow_pred[:, :2])
            grid2, _ = warpgrid(self.inp[:, 1], flow_pred[:, 2:])
            del flow_pred
            torch.cuda.empty_cache()
            grid = grid1.view(grid1.shape[0], -1, grid1.shape[-1]).flip(-1).clamp(-1 + 1e-6, 1 - 1e-6)
            q_feat1 = F.grid_sample(
                HRfeat, grid.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_img1 = F.grid_sample(
                HRinp, grid.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_feat3 = F.grid_sample(
                feat, grid.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            grid = grid2.view(grid2.shape[0], -1, grid2.shape[-1]).flip(-1).clamp(-1 + 1e-6, 1 - 1e-6)
            q_feat2 = F.grid_sample(
                HRfeat, grid.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_img2 = F.grid_sample(
                HRinp, grid.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_feat4 = F.grid_sample(
                feat, grid.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
        
            inp = torch.cat([q_feat1, q_feat2, q_feat3, q_feat4, q_img1, q_img2, pe_coord], dim=-1)
            pred = self.encode_imnet(inp.view(bs * qs, -1)).view(bs, qs, -1)
            pred = pred.permute(0, 2, 1).view(bs, 3, HH, WW)
            preds.append(pred)
        return preds
    
    def decoding_test(self, times=None, scale=None):
        feat = torch.cat([self.feat[:, 0], self.feat[:, 1], self.feat[:, 2]], dim=1)
        
        bs, C, H, W = feat.shape
        if scale is None:
            HH, WW = H * 4, W * 4
        else:
            HH, WW = H * scale, W * scale
        
        coord_lowres = make_coord((H, W)).repeat(bs, 1, 1).clamp(-1 + 1e-6, 1 - 1e-6).cuda() / 4
        coord_highres = make_coord((HH, WW)).repeat(bs, 1, 1).clamp(-1 + 1e-6, 1 - 1e-6).cuda()
    
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])        

        preds = []
        for c in range(len(times)):
            qs = coord_highres.shape[1]
            qs1 = qs // 3
            qs2 = qs // 3
            qs3 = qs - qs1 - qs2
            q_feat = F.grid_sample(
                feat, coord_highres.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_inp = F.grid_sample(
                self.inp.view(feat.shape[0], -1, feat.shape[2], feat.shape[3]), coord_highres.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_coord = F.grid_sample(
                feat_coord, coord_highres.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            rel_coord = coord_highres - q_coord
            rel_coord[:, :, 0] *= feat.shape[-2]
            rel_coord[:, :, 1] *= feat.shape[-1]
            pe_coord = torch.ones_like(coord_highres[:, :, 0].unsqueeze(2)) * times[c].unsqueeze(2)
            
            inp = torch.cat([q_feat, q_inp, rel_coord, pe_coord], dim=-1)
            inp_p1 = inp[:, :qs1]
            inp_p2 = inp[:, qs1:qs1+qs2]
            inp_p3 = inp[:, qs1+qs2:]
            pred_p1 = self.feat_imnet(inp_p1.view(bs * qs1, -1)).view(bs, qs1, -1)
            pred_p2 = self.feat_imnet(inp_p2.view(bs * qs2, -1)).view(bs, qs2, -1)
            pred_p3 = self.feat_imnet(inp_p3.view(bs * qs3, -1)).view(bs, qs3, -1)
            
            HRfeat = torch.cat([pred_p1, pred_p2, pred_p3], dim=1)
            del q_coord, rel_coord, inp, inp_p1, inp_p2, inp_p3, pred_p1, pred_p2, pred_p3
            torch.cuda.empty_cache()    
            
            HRfeat = HRfeat.permute(0, 2, 1).view(bs, 64, HH, WW)
            HRinp = self.inp.view(feat.shape[0], -1, feat.shape[2], feat.shape[3])
            HRinp = F.upsample(HRinp, scale_factor=4, mode='bilinear')
            
            q_feat = F.grid_sample(
                HRfeat, coord_highres.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_inp = F.grid_sample(
                HRinp, coord_highres.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_feat0 = F.grid_sample(
                feat, coord_highres.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            flow_inp = torch.cat([q_feat, q_feat0, q_inp, pe_coord], dim=-1)
            flow_inp1 = flow_inp[:, :qs1]
            flow_inp2 = flow_inp[:, qs1:qs1+qs2]
            flow_inp3 = flow_inp[:, qs1+qs2:]
            flow_pred1 = self.flow_imnet(flow_inp1.view(bs * qs1, -1)).view(bs, qs1, -1)
            flow_pred2 = self.flow_imnet(flow_inp2.view(bs * qs2, -1)).view(bs, qs2, -1)
            flow_pred3 = self.flow_imnet(flow_inp3.view(bs * qs3, -1)).view(bs, qs3, -1)
            flow_pred = torch.cat([flow_pred1, flow_pred2, flow_pred3], dim=1)
            del q_feat, q_inp, q_feat0, flow_inp, flow_inp1, flow_inp2, flow_inp3, flow_pred1, flow_pred2, flow_pred3
            torch.cuda.empty_cache()
            flow_pred = flow_pred.permute(0, 2, 1).view(bs, 4, HH, WW)
            
            grid1, _ = warpgrid(self.inp[:, 0], flow_pred[:, :2])
            grid2, _ = warpgrid(self.inp[:, 1], flow_pred[:, 2:])
            # grid1 = warpgrid2(coord_lowres, flow_pred[:, :2], HH, WW)
            # grid2 = warpgrid2(coord_lowres, flow_pred[:, 2:], HH, WW)
            del flow_pred
            torch.cuda.empty_cache()

            grid = grid1.view(grid1.shape[0], -1, grid1.shape[-1]).flip(-1).clamp(-1 + 1e-6, 1 - 1e-6)
            q_feat1 = F.grid_sample(
                HRfeat, grid.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_img1 = F.grid_sample(
                HRinp, grid.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_feat3 = F.grid_sample(
                feat, grid.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            grid = grid2.view(grid2.shape[0], -1, grid2.shape[-1]).flip(-1).clamp(-1 + 1e-6, 1 - 1e-6)
            q_feat2 = F.grid_sample(
                HRfeat, grid.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_img2 = F.grid_sample(
                HRinp, grid.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_feat4 = F.grid_sample(
                feat, grid.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            # bp()
            inp_p1 = torch.cat([q_feat1[:, :qs1], q_feat2[:, :qs1], q_feat3[:, :qs1], q_feat4[:, :qs1], q_img1[:, :qs1], q_img2[:, :qs1], pe_coord[:, :qs1]], dim=-1)
            pred_p1 = self.encode_imnet(inp_p1.view(bs * qs1, -1)).view(bs, qs1, -1)
            del inp_p1
            torch.cuda.empty_cache()
            inp_p2 = torch.cat([q_feat1[:, qs1:qs1+qs2], q_feat2[:, qs1:qs1+qs2], q_feat3[:, qs1:qs1+qs2], q_feat4[:, qs1:qs1+qs2], q_img1[:, qs1:qs1+qs2], q_img2[:, qs1:qs1+qs2], pe_coord[:, qs1:qs1+qs2]], dim=-1)
            pred_p2 = self.encode_imnet(inp_p2.view(bs * qs2, -1)).view(bs, qs2, -1)
            del inp_p2
            torch.cuda.empty_cache()
            inp_p3 = torch.cat([q_feat1[:, qs1+qs2:], q_feat2[:, qs1+qs2:], q_feat3[:, qs1+qs2:], q_feat4[:, qs1+qs2:], q_img1[:, qs1+qs2:], q_img2[:, qs1+qs2:], pe_coord[:, qs1+qs2:]], dim=-1)
            pred_p3 = self.encode_imnet(inp_p3.view(bs * qs3, -1)).view(bs, qs3, -1)
            del inp_p3
            torch.cuda.empty_cache()
            # inp_p1 = inp[:, :qs1]
            # inp_p2 = inp[:, qs1:qs1+qs2]
            # inp_p3 = inp[:, qs1+qs2:]
            # pred_p1 = self.encode_imnet(inp_p1.view(bs * qs1, -1)).view(bs, qs1, -1)
            # pred_p2 = self.encode_imnet(inp_p2.view(bs * qs2, -1)).view(bs, qs2, -1)
            # pred_p3 = self.encode_imnet(inp_p3.view(bs * qs3, -1)).view(bs, qs3, -1)
            pred = torch.cat([pred_p1, pred_p2, pred_p3], dim=1)
            del pred_p1, pred_p2, pred_p3, q_feat1, q_feat2, q_feat3, q_feat4, q_img1, q_img2, pe_coord
            torch.cuda.empty_cache()
            
            pred = pred.permute(0, 2, 1).view(bs, 3, HH, WW)
            preds.append(pred)
        return preds
    
    def decoding_memory(self, times=None, scale=None, center=None, input_img=None, index=0, save=0):
        feat = torch.cat([self.feat[:, 0], self.feat[:, 1], self.feat[:, 2]], dim=1)
        
        bs, C, H, W = feat.shape
        if scale is None:
            HH, WW = H * 4, W * 4
        else:
            HH, WW = int(scale[0]), int(scale[1])
        # bp()
        folder = 'bmx-rider'
        if save == 0:
            out_path2 = '/home/users/zeyuan_chen/image_repo3/test/{}zoomtest39/LR/'.format(folder)
            out_path3 = '/home/users/zeyuan_chen/image_repo3/test/{}zoomtest39/bicubic/'.format(folder)
            out_path4 = '/home/users/zeyuan_chen/image_repo3/test/{}zoomtest39/nearest/'.format(folder)
        elif save == 1:
            out_path2 = '/home/users/zeyuan_chen/image_repo3/test/{}zoomtest39/LR2/'.format(folder)
            out_path3 = '/home/users/zeyuan_chen/image_repo3/test/{}zoomtest39/bicubic2/'.format(folder)
            out_path4 = '/home/users/zeyuan_chen/image_repo3/test/{}zoomtest39/nearest2/'.format(folder)
        if not os.path.exists(out_path2):
            os.makedirs(out_path2)
        if not os.path.exists(out_path3):
            os.makedirs(out_path3)
        if not os.path.exists(out_path4):
            os.makedirs(out_path4)
        bicubic = np.array(Image.fromarray((input_img[0, 0].permute(1,2,0).clamp(0., 1.) * 255).detach().cpu().numpy().astype(np.uint8)).resize((WW, HH), Image.BICUBIC))
        nearest = np.array(Image.fromarray((input_img[0, 0].permute(1,2,0).clamp(0., 1.) * 255).detach().cpu().numpy().astype(np.uint8)).resize((WW, HH), Image.NEAREST))
        # lowres = Image.fromarray((input_img[0, 0].permute(1,2,0) * 255).detach().cpu().numpy().astype(np.uint8))
        # lowres.save(os.path.join(out_path2, '{}.jpg'.format(index)))

        # bicubic = Image.fromarray((input_img[0, 0].permute(1,2,0) * 255).detach().cpu().numpy().astype(np.uint8)).resize((WW, HH), Image.BICUBIC)
        # nearest = Image.fromarray((input_img[0, 0].permute(1,2,0) * 255).detach().cpu().numpy().astype(np.uint8)).resize((WW, HH), Image.NEAREST)
        # bicubic.save(os.path.join(out_path3, '{}.jpg'.format(index)))
        # nearest.save(os.path.join(out_path4, '{}.jpg'.format(index)))

        

        H0, W0 = H * 4, W * 4
        coord_lowres = make_coord_demo((HH, WW), (H0, W0), center)
        coord_highres = make_coord((HH, WW), flatten=False)
        center_coord = ((center + 1) / 2) * np.array((HH, WW))
        x0, x1, y0, y1 = int(center_coord[0]) - H0 // 2, int(center_coord[0]) + H0 - H0 // 2, int(center_coord[1]) - W0 // 2, int(center_coord[1]) + W0 - W0 // 2
        if x0 < 0:
            x1 -= x0
            x0 = 0
        elif x1 > HH:
            x0 -= (x1 - HH)
            x1 = HH
        if y0 < 0:
            y1 -= y0
            y0 = 0
        elif y1 > WW:
            y0 -= (y1 - WW)
            y1 = WW
        # bp()
        coord_lowres = coord_highres.clone()[x0:x1, y0:y1]
        coord_highres = coord_highres.reshape(-1, coord_highres.shape[-1]).repeat(bs, 1, 1).clamp(-1 + 1e-6, 1 - 1e-6).cuda()
        coord_lowres = coord_lowres.reshape(-1, coord_lowres.shape[-1]).repeat(bs, 1, 1).clamp(-1 + 1e-6, 1 - 1e-6).cuda()
        Image.fromarray(bicubic[x0:x1, y0:y1]).save(os.path.join(out_path3, '{}.jpg'.format(index)))
        Image.fromarray(nearest[x0:x1, y0:y1]).save(os.path.join(out_path4, '{}.jpg'.format(index)))


        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])        

        preds = []
        for c in range(len(times)):
            # print(c)
            qs = coord_highres.shape[1]
            qs1 = qs // 12
            qs2 = qs // 12
            qs3 = qs // 12
            qs4 = qs // 12
            qs5 = qs // 12
            qs6 = qs // 12
            qs7 = qs // 12
            qs8 = qs // 12
            qs9 = qs // 12
            qs10 = qs // 12
            qs11 = qs // 12
            qs12 = qs - qs1 - qs2 - qs3 - qs4 - qs5 - qs6 - qs7 - qs8 - qs9 - qs10 - qs11
            q_feat = F.grid_sample(
                feat, coord_highres.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_inp = F.grid_sample(
                self.inp.view(feat.shape[0], -1, feat.shape[2], feat.shape[3]), coord_highres.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_coord = F.grid_sample(
                feat_coord, coord_highres.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            rel_coord = coord_highres - q_coord
            rel_coord[:, :, 0] *= feat.shape[-2]
            rel_coord[:, :, 1] *= feat.shape[-1]
            pe_coord = torch.ones_like(coord_highres[:, :, 0].unsqueeze(2)) * times[c].unsqueeze(2)
            
            inp_p1 = torch.cat([q_feat[:, :qs1], q_inp[:, :qs1], rel_coord[:, :qs1], pe_coord[:, :qs1]], dim=-1)
            pred_p1 = self.feat_imnet(inp_p1.view(bs * qs1, -1)).view(bs, qs1, -1)
            del inp_p1
            torch.cuda.empty_cache()   
            inp_p2 = torch.cat([q_feat[:, qs1:qs1+qs2], q_inp[:, qs1:qs1+qs2], rel_coord[:, qs1:qs1+qs2], pe_coord[:, qs1:qs1+qs2]], dim=-1)
            pred_p2 = self.feat_imnet(inp_p2.view(bs * qs2, -1)).view(bs, qs2, -1)
            del inp_p2
            torch.cuda.empty_cache()
            inp_p3 = torch.cat([q_feat[:, qs1+qs2:qs1+qs2+qs3], q_inp[:, qs1+qs2:qs1+qs2+qs3], rel_coord[:, qs1+qs2:qs1+qs2+qs3], pe_coord[:, qs1+qs2:qs1+qs2+qs3]], dim=-1)
            pred_p3 = self.feat_imnet(inp_p3.view(bs * qs3, -1)).view(bs, qs3, -1)
            del inp_p3
            torch.cuda.empty_cache()
            inp_p4 = torch.cat([q_feat[:, qs1+qs2+qs3:qs1+qs2+qs3+qs4], q_inp[:, qs1+qs2+qs3:qs1+qs2+qs3+qs4], rel_coord[:, qs1+qs2+qs3:qs1+qs2+qs3+qs4], pe_coord[:, qs1+qs2+qs3:qs1+qs2+qs3+qs4]], dim=-1)
            pred_p4 = self.feat_imnet(inp_p4.view(bs * qs4, -1)).view(bs, qs4, -1)
            del inp_p4
            torch.cuda.empty_cache()
            inp_p5 = torch.cat([q_feat[:, qs1+qs2+qs3+qs4:qs1+qs2+qs3+qs4+qs5], q_inp[:, qs1+qs2+qs3+qs4:qs1+qs2+qs3+qs4+qs5], rel_coord[:, qs1+qs2+qs3+qs4:qs1+qs2+qs3+qs4+qs5], pe_coord[:, qs1+qs2+qs3+qs4:qs1+qs2+qs3+qs4+qs5]], dim=-1)
            pred_p5 = self.feat_imnet(inp_p5.view(bs * qs5, -1)).view(bs, qs5, -1)
            del inp_p5
            torch.cuda.empty_cache()   
            inp_p6 = torch.cat([q_feat[:, qs1+qs2+qs3+qs4+qs5:qs1+qs2+qs3+qs4+qs5+qs6], q_inp[:, qs1+qs2+qs3+qs4+qs5:qs1+qs2+qs3+qs4+qs5+qs6], rel_coord[:, qs1+qs2+qs3+qs4+qs5:qs1+qs2+qs3+qs4+qs5+qs6], pe_coord[:, qs1+qs2+qs3+qs4+qs5:qs1+qs2+qs3+qs4+qs5+qs6]], dim=-1)
            pred_p6 = self.feat_imnet(inp_p6.view(bs * qs6, -1)).view(bs, qs6, -1)
            del inp_p6
            torch.cuda.empty_cache()   
            inp_p7 = torch.cat([q_feat[:, qs1+qs2+qs3+qs4+qs5+qs6:qs1+qs2+qs3+qs4+qs5+qs6+qs7], q_inp[:, qs1+qs2+qs3+qs4+qs5+qs6:qs1+qs2+qs3+qs4+qs5+qs6+qs7], rel_coord[:, qs1+qs2+qs3+qs4+qs5+qs6:qs1+qs2+qs3+qs4+qs5+qs6+qs7], pe_coord[:, qs1+qs2+qs3+qs4+qs5+qs6:qs1+qs2+qs3+qs4+qs5+qs6+qs7]], dim=-1)
            pred_p7 = self.feat_imnet(inp_p7.view(bs * qs7, -1)).view(bs, qs7, -1)
            del inp_p7
            inp_p8 = torch.cat([q_feat[:, qs1+qs2+qs3+qs4+qs5+qs6+qs7:qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8], q_inp[:, qs1+qs2+qs3+qs4+qs5+qs6+qs7:qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8], rel_coord[:, qs1+qs2+qs3+qs4+qs5+qs6+qs7:qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8], pe_coord[:, qs1+qs2+qs3+qs4+qs5+qs6+qs7:qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8]], dim=-1)
            pred_p8 = self.feat_imnet(inp_p8.view(bs * qs8, -1)).view(bs, qs8, -1)
            del inp_p8
            inp_p9 = torch.cat([q_feat[:, qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8:qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8+qs9], q_inp[:, qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8:qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8+qs9], rel_coord[:, qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8:qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8+qs9], pe_coord[:, qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8:qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8+qs9]], dim=-1)
            pred_p9 = self.feat_imnet(inp_p9.view(bs * qs9, -1)).view(bs, qs9, -1)
            del inp_p9
            inp_p10 = torch.cat([q_feat[:, qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8+qs9:qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8+qs9+qs10], q_inp[:, qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8+qs9:qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8+qs9+qs10], rel_coord[:, qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8+qs9:qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8+qs9+qs10], pe_coord[:, qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8+qs9:qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8+qs9+qs10]], dim=-1)
            pred_p10 = self.feat_imnet(inp_p10.view(bs * qs10, -1)).view(bs, qs10, -1)
            del inp_p10
            inp_p11 = torch.cat([q_feat[:, qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8+qs9+qs10:qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8+qs9+qs10+qs11], q_inp[:, qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8+qs9+qs10:qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8+qs9+qs10+qs11], rel_coord[:, qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8+qs9+qs10:qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8+qs9+qs10+qs11], pe_coord[:, qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8+qs9+qs10:qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8+qs9+qs10+qs11]], dim=-1)
            pred_p11 = self.feat_imnet(inp_p11.view(bs * qs11, -1)).view(bs, qs11, -1)
            del inp_p11
            inp_p12 = torch.cat([q_feat[:, qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8+qs9+qs10+qs11:], q_inp[:, qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8+qs9+qs10+qs11:], rel_coord[:, qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8+qs9+qs10+qs11:], pe_coord[:, qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8+qs9+qs10+qs11:]], dim=-1)
            pred_p12 = self.feat_imnet(inp_p12.view(bs * qs12, -1)).view(bs, qs12, -1)
            del inp_p12, q_coord, rel_coord
            # [:, qs1+qs2+qs3+qs4+qs5+qs6+qs7+qs8:]
            # bp()
            # inp = torch.cat([q_feat, q_inp, rel_coord, pe_coord], dim=-1)
            # inp_p1 = inp[:, :qs1]
            # inp_p2 = inp[:, qs1:qs1+qs2]
            # inp_p3 = inp[:, qs1+qs2:qs1+qs2+qs3]
            # inp_p4 = inp[:, qs1+qs2+qs3:qs1+qs2+qs3+qs4]
            # inp_p5 = inp[:, qs1+qs2+qs3+qs4:qs1+qs2+qs3+qs4+qs5]
            # inp_p6 = inp[:, qs1+qs2+qs3+qs4+qs5:]
            # pred_p1 = self.feat_imnet(inp_p1.view(bs * qs1, -1)).view(bs, qs1, -1)
            # pred_p2 = self.feat_imnet(inp_p2.view(bs * qs2, -1)).view(bs, qs2, -1)
            # pred_p3 = self.feat_imnet(inp_p3.view(bs * qs3, -1)).view(bs, qs3, -1)
            # pred_p4 = self.feat_imnet(inp_p4.view(bs * qs4, -1)).view(bs, qs4, -1)
            # pred_p5 = self.feat_imnet(inp_p5.view(bs * qs5, -1)).view(bs, qs5, -1)
            # pred_p6 = self.feat_imnet(inp_p6.view(bs * qs6, -1)).view(bs, qs6, -1)
            # del q_coord, rel_coord, inp, inp_p1, inp_p2, inp_p3, inp_p4, inp_p5, inp_p6
            # torch.cuda.empty_cache()    
            HRfeat = torch.cat([pred_p1, pred_p2, pred_p3, pred_p4, pred_p5, pred_p6, pred_p7, pred_p8, pred_p9, pred_p10, pred_p11, pred_p12], dim=1)
            del pred_p1, pred_p2, pred_p3, pred_p4, pred_p5, pred_p6, pred_p7, pred_p8, pred_p9, pred_p10, pred_p11, pred_p12
            torch.cuda.empty_cache()    
            
            HRfeat = HRfeat.permute(0, 2, 1).view(bs, 64, HH, WW)
            HRinp = self.inp.view(feat.shape[0], -1, feat.shape[2], feat.shape[3])
            HRinp = F.upsample(HRinp, scale_factor=4, mode='bilinear')
            
            q_feat = F.grid_sample(
                HRfeat, coord_lowres.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_inp = F.grid_sample(
                HRinp, coord_lowres.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_feat0 = F.grid_sample(
                feat, coord_lowres.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            pe_coord = torch.ones_like(coord_lowres[:, :, 0].unsqueeze(2)) * times[c].unsqueeze(2)
            flow_inp = torch.cat([q_feat, q_feat0, q_inp, pe_coord], dim=-1)
            qs0 = coord_lowres.shape[1]
            flow_pred = self.flow_imnet(flow_inp.view(bs * qs0, -1)).view(bs, qs0, -1)
            del flow_inp
            torch.cuda.empty_cache()
            # flow_pred = flow_pred.permute(0, 2, 1).view(bs, 4, H, W)
            # flow_inp1 = flow_inp[:, :qs1]
            # flow_inp2 = flow_inp[:, qs1:qs1+qs2]
            # flow_inp3 = flow_inp[:, qs1+qs2:]
            # flow_pred1 = self.flow_imnet(flow_inp1.view(bs * qs1, -1)).view(bs, qs1, -1)
            # flow_pred2 = self.flow_imnet(flow_inp2.view(bs * qs2, -1)).view(bs, qs2, -1)
            # flow_pred3 = self.flow_imnet(flow_inp3.view(bs * qs3, -1)).view(bs, qs3, -1)
            # flow_pred = torch.cat([flow_pred1, flow_pred2, flow_pred3], dim=1)
            # del q_feat, q_inp, q_feat0, flow_inp, flow_inp1, flow_inp2, flow_inp3, flow_pred1, flow_pred2, flow_pred3
            # torch.cuda.empty_cache()
            # flow_pred = flow_pred.permute(0, 2, 1).view(bs, 4, HH, WW)
            
            # grid1, _ = warpgrid(self.inp[:, 0], flow_pred[:, :2])
            # grid2, _ = warpgrid(self.inp[:, 1], flow_pred[:, 2:])
            # bp()
            grid1 = warpgrid2(coord_lowres, flow_pred[:, :, :2], HH, WW)
            grid2 = warpgrid2(coord_lowres, flow_pred[:, :, 2:], HH, WW)
            del flow_pred
            torch.cuda.empty_cache()
            # bp()
            # grid = grid1.view(grid1.shape[0], -1, grid1.shape[-1]).flip(-1).clamp(-1 + 1e-6, 1 - 1e-6)
            q_feat1 = F.grid_sample(
                HRfeat, grid1.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_img1 = F.grid_sample(
                HRinp, grid1.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_feat3 = F.grid_sample(
                feat, grid1.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            # grid = grid2.view(grid2.shape[0], -1, grid2.shape[-1]).flip(-1).clamp(-1 + 1e-6, 1 - 1e-6)
            q_feat2 = F.grid_sample(
                HRfeat, grid2.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_img2 = F.grid_sample(
                HRinp, grid2.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_feat4 = F.grid_sample(
                feat, grid2.flip(-1).unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            # bp()
            inp = torch.cat([q_feat1, q_feat2, q_feat3, q_feat4, q_img1, q_img2, pe_coord], dim=-1)
            pred = self.encode_imnet(inp.view(bs * qs0, -1)).view(bs, qs0, -1)
            pred = pred.permute(0, 2, 1).view(bs, 3, H0, W0)
            preds.append(pred)
            del inp, q_feat1, q_feat2, q_feat3, q_feat4, q_img1, q_img2, pe_coord
            torch.cuda.empty_cache()

            # inp_p1 = torch.cat([q_feat1[:, :qs1], q_feat2[:, :qs1], q_feat3[:, :qs1], q_feat4[:, :qs1], q_img1[:, :qs1], q_img2[:, :qs1], pe_coord[:, :qs1]], dim=-1)
            # pred_p1 = self.encode_imnet(inp_p1.view(bs * qs1, -1)).view(bs, qs1, -1)
            # del inp_p1
            # torch.cuda.empty_cache()
            # inp_p2 = torch.cat([q_feat1[:, qs1:qs1+qs2], q_feat2[:, qs1:qs1+qs2], q_feat3[:, qs1:qs1+qs2], q_feat4[:, qs1:qs1+qs2], q_img1[:, qs1:qs1+qs2], q_img2[:, qs1:qs1+qs2], pe_coord[:, qs1:qs1+qs2]], dim=-1)
            # pred_p2 = self.encode_imnet(inp_p2.view(bs * qs2, -1)).view(bs, qs2, -1)
            # del inp_p2
            # torch.cuda.empty_cache()
            # inp_p3 = torch.cat([q_feat1[:, qs1+qs2:], q_feat2[:, qs1+qs2:], q_feat3[:, qs1+qs2:], q_feat4[:, qs1+qs2:], q_img1[:, qs1+qs2:], q_img2[:, qs1+qs2:], pe_coord[:, qs1+qs2:]], dim=-1)
            # pred_p3 = self.encode_imnet(inp_p3.view(bs * qs3, -1)).view(bs, qs3, -1)
            # del inp_p3
            # torch.cuda.empty_cache()
            # # inp_p1 = inp[:, :qs1]
            # # inp_p2 = inp[:, qs1:qs1+qs2]
            # # inp_p3 = inp[:, qs1+qs2:]
            # # pred_p1 = self.encode_imnet(inp_p1.view(bs * qs1, -1)).view(bs, qs1, -1)
            # # pred_p2 = self.encode_imnet(inp_p2.view(bs * qs2, -1)).view(bs, qs2, -1)
            # # pred_p3 = self.encode_imnet(inp_p3.view(bs * qs3, -1)).view(bs, qs3, -1)
            # pred = torch.cat([pred_p1, pred_p2, pred_p3], dim=1)
            # del pred_p1, pred_p2, pred_p3, q_feat1, q_feat2, q_feat3, q_feat4, q_img1, q_img2, pe_coord
            # torch.cuda.empty_cache()
            
            # pred = pred.permute(0, 2, 1).view(bs, 3, HH, WW)
            # preds.append(pred)
        return preds
    
    def decoding_fasttest(self, times=None, scale=None):
        feat = torch.cat([self.feat[:, 0], self.feat[:, 1], self.feat[:, 2]], dim=1)
        
        bs, C, H, W = feat.shape
        if scale is None:
            HH, WW = H * 4, W * 4
        else:
            HH, WW = scale[0], scale[1]
        coord_highres = make_coord((HH, WW)).repeat(bs, 1, 1).clamp(-1 + 1e-6, 1 - 1e-6).cuda()
        
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])        
        # bp()
        new_times = torch.tensor(times).unsqueeze(1).cuda()
        qs = coord_highres.shape[1]
        bs = new_times.shape[0]
        q_feat = F.grid_sample(
            feat, coord_highres.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1).repeat(bs, 1, 1)
        q_inp = F.grid_sample(
            self.inp.view(feat.shape[0], -1, feat.shape[2], feat.shape[3]), coord_highres.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1).repeat(bs, 1, 1)
        q_coord = F.grid_sample(
            feat_coord, coord_highres.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        rel_coord = (coord_highres - q_coord).repeat(bs, 1, 1)
        rel_coord[:, :, 0] *= feat.shape[-2]
        rel_coord[:, :, 1] *= feat.shape[-1]
        new_pe_coord = torch.ones_like(coord_highres[:, :, 0].unsqueeze(2)) * new_times.unsqueeze(2)
        
        inp = torch.cat([q_feat, q_inp, rel_coord, new_pe_coord], dim=-1)
        HRfeat = self.feat_imnet(inp.view(bs * qs, -1)).view(bs, qs, -1)
        HRfeat = HRfeat.permute(0, 2, 1).view(bs, 64, HH, WW)
        
        HRinp = self.inp.view(feat.shape[0], -1, feat.shape[2], feat.shape[3]).repeat(bs, 1, 1, 1)
        
        coord_highres = coord_highres.repeat(bs, 1, 1)
        feat = feat.repeat(bs, 1, 1, 1)
        # HRinp = F.upsample(HRinp, scale_factor=4, mode='bilinear')
        # del q_coord, rel_coord, inp
        # torch.cuda.empty_cache()
        q_feat = F.grid_sample(
            HRfeat, coord_highres.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_inp = F.grid_sample(
            HRinp, coord_highres.flip(-1).unsqueeze(1),
            mode='bilinear', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_feat0 = F.grid_sample(
            feat, coord_highres.flip(-1).unsqueeze(1),
            mode='bilinear', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        flow_inp = torch.cat([q_feat, q_feat0, q_inp, new_pe_coord], dim=-1)
        flow_pred = self.flow_imnet(flow_inp.view(bs * qs, -1)).view(bs, qs, -1)
        # del q_feat, q_inp, q_feat0, flow_inp
        # torch.cuda.empty_cache()
        flow_pred = flow_pred.permute(0, 2, 1).view(bs, 4, HH, WW)
        
        grid1, _ = warpgrid(self.inp[:, 0].repeat(bs, 1, 1, 1), flow_pred[:, :2])
        grid2, _ = warpgrid(self.inp[:, 1].repeat(bs, 1, 1, 1), flow_pred[:, 2:])
        # del flow_pred
        # torch.cuda.empty_cache()
        grid = grid1.view(grid1.shape[0], -1, grid1.shape[-1]).flip(-1).clamp(-1 + 1e-6, 1 - 1e-6)
        q_feat1 = F.grid_sample(
            HRfeat, grid.flip(-1).unsqueeze(1),
            mode='bilinear', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_img1 = F.grid_sample(
            HRinp, grid.flip(-1).unsqueeze(1),
            mode='bilinear', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_feat3 = F.grid_sample(
            feat, grid.flip(-1).unsqueeze(1),
            mode='bilinear', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        grid = grid2.view(grid2.shape[0], -1, grid2.shape[-1]).flip(-1).clamp(-1 + 1e-6, 1 - 1e-6)
        q_feat2 = F.grid_sample(
            HRfeat, grid.flip(-1).unsqueeze(1),
            mode='bilinear', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_img2 = F.grid_sample(
            HRinp, grid.flip(-1).unsqueeze(1),
            mode='bilinear', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_feat4 = F.grid_sample(
            feat, grid.flip(-1).unsqueeze(1),
            mode='bilinear', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        
        inp = torch.cat([q_feat1, q_feat2, q_feat3, q_feat4, q_img1, q_img2, new_pe_coord], dim=-1)
        pred = self.encode_imnet(inp.view(bs * qs, -1)).view(bs, qs, -1)
        preds = pred.permute(0, 2, 1).view(bs, 3, HH, WW)
        return preds
    
    def decoding_localensemble(self, times=None, scale=None):
        feat = torch.cat([self.feat[:, 0], self.feat[:, 1], self.feat[:, 2]], dim=1)
        
        bs, C, H, W = feat.shape
        if scale is None:
            HH, WW = H * 4, W * 4
        else:
            HH, WW = scale[0], scale[1]
        coord_highres = make_coord((HH, WW)).repeat(bs, 1, 1).clamp(-1 + 1e-6, 1 - 1e-6).cuda()
        
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])        
        # bp()

        preds = []
        predss = []
        rets = []
        areas = []
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        for vx in vx_lst:
            for vy in vy_lst:
                new_times = torch.tensor(times).unsqueeze(1).cuda()
                qs = coord_highres.shape[1]
                bs = new_times.shape[0]
                coord_ = coord_highres.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1).repeat(bs, 1, 1)
                q_inp = F.grid_sample(
                    self.inp.view(feat.shape[0], -1, feat.shape[2], feat.shape[3]), coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1).repeat(bs, 1, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = (coord_highres - q_coord).repeat(bs, 1, 1)
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)
                new_pe_coord = torch.ones_like(coord_highres[:, :, 0].unsqueeze(2)) * new_times.unsqueeze(2)
                
                inp = torch.cat([q_feat, q_inp, rel_coord, new_pe_coord], dim=-1)
                HRfeat = self.feat_imnet(inp.view(bs * qs, -1)).view(bs, qs, -1)
                HRfeat = HRfeat.permute(0, 2, 1).view(bs, 64, HH, WW)
                
                HRinp = self.inp.view(feat.shape[0], -1, feat.shape[2], feat.shape[3]).repeat(bs, 1, 1, 1)
                
                coord_ = coord_.repeat(bs, 1, 1)
                feat1 = feat.repeat(bs, 1, 1, 1)
                # HRinp = F.upsample(HRinp, scale_factor=4, mode='bilinear')
                # del q_coord, rel_coord, inp
                # torch.cuda.empty_cache()
                q_feat = F.grid_sample(
                    HRfeat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_inp = F.grid_sample(
                    HRinp, coord_.flip(-1).unsqueeze(1),
                    mode='bilinear', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_feat0 = F.grid_sample(
                    feat1, coord_.flip(-1).unsqueeze(1),
                    mode='bilinear', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                flow_inp = torch.cat([q_feat, q_feat0, q_inp, new_pe_coord], dim=-1)
                flow_pred = self.flow_imnet(flow_inp.view(bs * qs, -1)).view(bs, qs, -1)
                # del q_feat, q_inp, q_feat0, flow_inp
                # torch.cuda.empty_cache()
                flow_pred = flow_pred.permute(0, 2, 1).view(bs, 4, HH, WW)
                
                grid1, _ = warpgrid(self.inp[:, 0].repeat(bs, 1, 1, 1), flow_pred[:, :2])
                grid2, _ = warpgrid(self.inp[:, 1].repeat(bs, 1, 1, 1), flow_pred[:, 2:])
                # del flow_pred
                # torch.cuda.empty_cache()
                grid = grid1.view(grid1.shape[0], -1, grid1.shape[-1]).flip(-1).clamp(-1 + 1e-6, 1 - 1e-6)
                q_feat1 = F.grid_sample(
                    HRfeat, grid.flip(-1).unsqueeze(1),
                    mode='bilinear', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_img1 = F.grid_sample(
                    HRinp, grid.flip(-1).unsqueeze(1),
                    mode='bilinear', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_feat3 = F.grid_sample(
                    feat1, grid.flip(-1).unsqueeze(1),
                    mode='bilinear', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                grid = grid2.view(grid2.shape[0], -1, grid2.shape[-1]).flip(-1).clamp(-1 + 1e-6, 1 - 1e-6)
                q_feat2 = F.grid_sample(
                    HRfeat, grid.flip(-1).unsqueeze(1),
                    mode='bilinear', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_img2 = F.grid_sample(
                    HRinp, grid.flip(-1).unsqueeze(1),
                    mode='bilinear', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_feat4 = F.grid_sample(
                    feat1, grid.flip(-1).unsqueeze(1),
                    mode='bilinear', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                
                inp = torch.cat([q_feat1, q_feat2, q_feat3, q_feat4, q_img1, q_img2, new_pe_coord], dim=-1)
                pred = self.encode_imnet(inp.view(bs * qs, -1)).view(bs, qs, -1)
                preds = pred.permute(0, 2, 1).view(bs, 3, HH, WW)
                predss.append(preds)
        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(predss, areas):
            ret = ret + pred * (area / tot_area).reshape(preds.shape[0], 1, preds.shape[2], preds.shape[3])
        return ret

    def decoding_fasttest_memory(self, times=None, scale=None):
        feat = torch.cat([self.feat[:, 0], self.feat[:, 1], self.feat[:, 2]], dim=1)
        
        bs, C, H, W = feat.shape
        if scale is None:
            HH, WW = H * 4, W * 4
        else:
            HH, WW = scale[0], scale[1]
        coord_highres = make_coord((HH, WW)).repeat(bs, 1, 1).clamp(-1 + 1e-6, 1 - 1e-6).cuda()
        
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])        
        # bp()
        new_times = torch.tensor(times).unsqueeze(1).cuda()
        qs = coord_highres.shape[1]
        qs1 = qs // 3
        qs2 = qs // 3
        qs3 = qs - qs1 - qs2
        bs = new_times.shape[0]
        q_feat = F.grid_sample(
            feat, coord_highres.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1).repeat(bs, 1, 1)
        q_inp = F.grid_sample(
            self.inp.view(feat.shape[0], -1, feat.shape[2], feat.shape[3]), coord_highres.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1).repeat(bs, 1, 1)
        q_coord = F.grid_sample(
            feat_coord, coord_highres.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        rel_coord = (coord_highres - q_coord).repeat(bs, 1, 1)
        rel_coord[:, :, 0] *= feat.shape[-2]
        rel_coord[:, :, 1] *= feat.shape[-1]
        new_pe_coord = torch.ones_like(coord_highres[:, :, 0].unsqueeze(2)) * new_times.unsqueeze(2)
        
        inp = torch.cat([q_feat, q_inp, rel_coord, new_pe_coord], dim=-1)
        inp_p1 = inp[:, :qs1]
        inp_p2 = inp[:, qs1:qs1+qs2]
        inp_p3 = inp[:, qs1+qs2:]
        pred_p1 = self.feat_imnet(inp_p1.reshape(bs * qs1, -1)).view(bs, qs1, -1)
        pred_p2 = self.feat_imnet(inp_p2.reshape(bs * qs2, -1)).view(bs, qs2, -1)
        pred_p3 = self.feat_imnet(inp_p3.reshape(bs * qs3, -1)).view(bs, qs3, -1)

        HRfeat = torch.cat([pred_p1, pred_p2, pred_p3], dim=1)
        del q_coord, rel_coord, inp, inp_p1, inp_p2, inp_p3, pred_p1, pred_p2, pred_p3
        torch.cuda.empty_cache()    

        HRfeat = HRfeat.permute(0, 2, 1).view(bs, 64, HH, WW)
        
        HRinp = self.inp.view(feat.shape[0], -1, feat.shape[2], feat.shape[3]).repeat(bs, 1, 1, 1)
        
        coord_highres = coord_highres.repeat(bs, 1, 1)
        feat = feat.repeat(bs, 1, 1, 1)
        # HRinp = F.upsample(HRinp, scale_factor=4, mode='bilinear')
        # del q_coord, rel_coord, inp
        # torch.cuda.empty_cache()
        q_feat = F.grid_sample(
            HRfeat, coord_highres.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_inp = F.grid_sample(
            HRinp, coord_highres.flip(-1).unsqueeze(1),
            mode='bilinear', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_feat0 = F.grid_sample(
            feat, coord_highres.flip(-1).unsqueeze(1),
            mode='bilinear', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        flow_inp = torch.cat([q_feat, q_feat0, q_inp, new_pe_coord], dim=-1)
        flow_inp1 = flow_inp[:, :qs1]
        flow_inp2 = flow_inp[:, qs1:qs1+qs2]
        flow_inp3 = flow_inp[:, qs1+qs2:]
        flow_pred1 = self.flow_imnet(flow_inp1.reshape(bs * qs1, -1)).view(bs, qs1, -1)
        flow_pred2 = self.flow_imnet(flow_inp2.reshape(bs * qs2, -1)).view(bs, qs2, -1)
        flow_pred3 = self.flow_imnet(flow_inp3.reshape(bs * qs3, -1)).view(bs, qs3, -1)
        flow_pred = torch.cat([flow_pred1, flow_pred2, flow_pred3], dim=1)
        del q_feat, q_inp, q_feat0, flow_inp, flow_inp1, flow_inp2, flow_inp3, flow_pred1, flow_pred2, flow_pred3
        torch.cuda.empty_cache()
        flow_pred = flow_pred.permute(0, 2, 1).view(bs, 4, HH, WW)

        # flow_pred = self.flow_imnet(flow_inp.view(bs * qs, -1)).view(bs, qs, -1)
        # del q_feat, q_inp, q_feat0, flow_inp
        # torch.cuda.empty_cache()
        # flow_pred = flow_pred.permute(0, 2, 1).view(bs, 4, HH, WW)
        
        grid1, _ = warpgrid(self.inp[:, 0].repeat(bs, 1, 1, 1), flow_pred[:, :2])
        grid2, _ = warpgrid(self.inp[:, 1].repeat(bs, 1, 1, 1), flow_pred[:, 2:])
        del flow_pred
        torch.cuda.empty_cache()

        grid = grid1.view(grid1.shape[0], -1, grid1.shape[-1]).flip(-1).clamp(-1 + 1e-6, 1 - 1e-6)
        q_feat1 = F.grid_sample(
            HRfeat, grid.flip(-1).unsqueeze(1),
            mode='bilinear', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_img1 = F.grid_sample(
            HRinp, grid.flip(-1).unsqueeze(1),
            mode='bilinear', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_feat3 = F.grid_sample(
            feat, grid.flip(-1).unsqueeze(1),
            mode='bilinear', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        grid = grid2.view(grid2.shape[0], -1, grid2.shape[-1]).flip(-1).clamp(-1 + 1e-6, 1 - 1e-6)
        q_feat2 = F.grid_sample(
            HRfeat, grid.flip(-1).unsqueeze(1),
            mode='bilinear', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_img2 = F.grid_sample(
            HRinp, grid.flip(-1).unsqueeze(1),
            mode='bilinear', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_feat4 = F.grid_sample(
            feat, grid.flip(-1).unsqueeze(1),
            mode='bilinear', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        
        inp_p1 = torch.cat([q_feat1[:, :qs1], q_feat2[:, :qs1], q_feat3[:, :qs1], q_feat4[:, :qs1], q_img1[:, :qs1], q_img2[:, :qs1], new_pe_coord[:, :qs1]], dim=-1)
        pred_p1 = self.encode_imnet(inp_p1.view(bs * qs1, -1)).view(bs, qs1, -1)
        del inp_p1
        torch.cuda.empty_cache()
        inp_p2 = torch.cat([q_feat1[:, qs1:qs1+qs2], q_feat2[:, qs1:qs1+qs2], q_feat3[:, qs1:qs1+qs2], q_feat4[:, qs1:qs1+qs2], q_img1[:, qs1:qs1+qs2], q_img2[:, qs1:qs1+qs2], new_pe_coord[:, qs1:qs1+qs2]], dim=-1)
        pred_p2 = self.encode_imnet(inp_p2.view(bs * qs2, -1)).view(bs, qs2, -1)
        del inp_p2
        torch.cuda.empty_cache()
        inp_p3 = torch.cat([q_feat1[:, qs1+qs2:], q_feat2[:, qs1+qs2:], q_feat3[:, qs1+qs2:], q_feat4[:, qs1+qs2:], q_img1[:, qs1+qs2:], q_img2[:, qs1+qs2:], new_pe_coord[:, qs1+qs2:]], dim=-1)
        pred_p3 = self.encode_imnet(inp_p3.view(bs * qs3, -1)).view(bs, qs3, -1)
        del inp_p3
        torch.cuda.empty_cache()
        pred = torch.cat([pred_p1, pred_p2, pred_p3], dim=1)
        preds = pred.permute(0, 2, 1).view(bs, 3, HH, WW)
        return preds
            
    def forward(self, x, times=None, scale=None, test=False, center=None, index=0):
        self.gen_feat(x)
        self.inp = x
        if test == True:
            return self.decoding_test(times, scale)
            # return self.decoding_memory(times, scale, center, x, index, save)
            # return self.decoding_fasttest(times, scale)
            # return self.decoding_localensemble(times, scale)
        else:
            return self.decoding(times, scale)

def make_coord(shape, ranges=None, flatten=True):
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
    return ret

def make_coord_demo(shape, new_shape, center):

    # bp()
    interval = (2 / shape[0], 2 / shape[1])
    h, w = new_shape
    ranges = []
    if h % 2 == 0:
        ranges.append((- interval[0] * (h / 2) +  interval[0] / 2 + center[0],  interval[0] * (h / 2) -  interval[0] / 2 + center[0]))
    else:
        ranges.append((-  interval[0] * (h // 2) + center[0],  interval[0] * (h // 2) + center[0]))
    if w % 2 == 0:
        ranges.append((-  interval[1] * (w / 2) + interval[1] / 2 + center[1], interval[1] * (w / 2) - interval[1] / 2 + center[1]))
    else:
        ranges.append((- interval[1] * (w // 2) + center[1], interval[1] * (w // 2) + center[1]))

    coord_seqs = []
    for i, n in enumerate(new_shape):
        v0, v1 = ranges[i]
        seq = v0 + interval[i] * torch.arange(n).float()
        # bp()
        if seq.min() < -1:
            seq += (-1 - seq.min())
        # bp()
        coord_seqs.append(seq)
        
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    ret = ret.view(-1, ret.shape[-1])
    return ret