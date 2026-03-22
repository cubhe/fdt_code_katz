# Created by Renzhi He, UC Davis, 2023

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
#import skimage
#from skimage.metrics import peak_signal_noise_ratio
import cv2
import math
import time
import gc
#from absl import flags
import logging

from ssim import SSIM
#from .dncnn import DnCNN

# get total number of visible gpus
NUM_GPUS = torch.cuda.device_count()


########################################
###       Tensorboard & Helper       ###
########################################
import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.transforms import functional as F

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features[:10]  # 使用VGG的前23层
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(3, 1, 1, 1).transpose(1,0)[20]
            target = target.repeat(3, 1, 1, 1).transpose(1,0)[20]
        input_features = self.vgg(input)
        target_features = self.vgg(target)
        loss = nn.functional.mse_loss(input_features, target_features)
        return loss

# 假设input_stack和target_stack是你的两个图像堆栈，形状为(N, H, W)
# 需要对它们进行适当的预处理（缩放、归一化等）



def record_summary(writer, name, value, step):
    writer.add_scalar(name, value, step)
    writer.flush()


def reshape_image(image):
    if len(image.shape) == 2:
        image_reshaped = image.unsqueeze(0).unsqueeze(-1)
    elif len(image.shape) == 3:
        image_reshaped = image.unsqueeze(-1)
    else:
        image_reshaped = image
    return image_reshaped


def reshape_image_2(image):
    image_reshaped = image.unsqueeze(0).unsqueeze(-1)
    return image_reshaped


def reshape_image_3(image):
    image_reshaped = image.unsqueeze(-1)
    return image_reshaped


def reshape_image_5(image):
    shape = image.shape
    image_reshaped = image.view(-1, shape[2], shape[3], 1)
    return image_reshaped


#################################################
# ***      CLASS OF NEURAL REPRESENTATION     ****
#################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
def diversity_loss(tensor):
    """
    Compute a diversity loss for a tensor of shape (H, W, C) to encourage
    different features in each channel/layer.

    Parameters:
    tensor (torch.Tensor): Input tensor of shape (H, W, C).

    Returns:
    torch.Tensor: Computed diversity loss.
    """
    H, W, C = tensor.shape

    # Reshape tensor to (C, H*W) and normalize
    tensor_flat = tensor.permute(2, 0, 1).reshape(C, H*W)
    tensor_norm = F.normalize(tensor_flat, p=2, dim=1)

    # Compute the cosine similarity matrix
    similarity_matrix = torch.mm(tensor_norm, tensor_norm.t())

    # Since we want diversity, we are interested in minimizing the similarity.
    # Diagonal elements are self-similarities, so we exclude them.
    loss = (similarity_matrix.sum() - similarity_matrix.trace()) / (C * (C - 1))
    return loss
def tv_loss(x):
    """
    Compute the Total Variation Loss for a 3D stack (H, W, N).

    Parameters:
    x (torch.Tensor): Input tensor of shape (H, W, N).

    Returns:
    torch.Tensor: Total Variation Loss.
    """
    # Calculate the difference in the horizontal direction
    horizontal_diff = torch.abs(x[:, :-1, :] - x[:, 1:, :])

    # Calculate the difference in the vertical direction
    vertical_diff = torch.abs(x[:-1, :, :] - x[1:, :, :])

    # Sum up the differences
    loss = horizontal_diff.sum() + vertical_diff.sum()
    return loss

if __name__ == '__main__':
    pass
    #main()


class Loss(nn.Module):
    def __init__(self, DnCNNN_channels=1, tower_idx=None, Hreal=None, Himag=None):
        super(Loss, self).__init__()
        self.tower_idx = tower_idx
        self.Hreal = Hreal
        self.Himag = Himag
        self.SSIM = SSIM()
        self.TVLoss = TVLoss()

        # model_path = os.path.join('./dncnn/model_zoo', 'dncnn_15' + '.pth')
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # from dncnn.models.network_dncnn import DnCNN as net
        # model = net(in_nc=1, out_nc=1, nc=64, nb=17, act_mode='R')
        # model.load_state_dict(torch.load(model_path), strict=True)
        # model.eval()
        # for k, v in model.named_parameters():
        #     v.requires_grad = False
        # self.DnCNN = model.to(device)
        # # dncnn
        # num_of_layers = 17
        # logdir = "./model/dncnn_logs/DnCNN-S-25"
        # net = DnCNN(channels=DnCNNN_channels, num_of_layers=num_of_layers)
        # device_ids = [0]
        # model = nn.DataParallel(net, device_ids=device_ids).cuda()
        # model.load_state_dict(torch.load(os.path.join(logdir, 'net.pth')), strict=False)
        # # print('load DnCNN Done...')
        # # lock the parameters of DnCNN
        # for name, parameter in model.named_parameters():
        #     parameter.requires_grad = False
        # self.dncnn = model

        # Setup parameters

    ##############################
    ###     Loss Functions     ###
    ##############################

    def forward(self, args, Hxhat, xhat, y, steps, xhat_gt=None,tower_idx=0, reuse=False):

        args.loss = 'l12'
        if args.loss == "l1":
            mse = torch.mean(torch.abs(Hxhat - y)) / 20
        elif args.loss == "l2":
            mse = torch.mean(torch.square(Hxhat - y)) / 20
        elif args.loss == "l12":
            decay_steps=500
            alpha=max((decay_steps-steps)/decay_steps,0)
            beta=max(steps/decay_steps,1)
            beta = min(beta, 1)
            mse1=torch.mean(torch.abs(Hxhat - y)) / 2
            mse2=torch.mean(torch.square(Hxhat - y)) / 2
            mse = alpha*mse2 + (1-alpha)*mse1
        else:
            raise NotImplementedError
        y.mean()
        Hxhat.mean()
        # 示例用法
        # perceptual_loss = PerceptualLoss()
        # pc_loss = perceptual_loss(Hxhat, y)
        pc_loss=0
        # regularizer
        # RI = np.load('/home/renzhihe/Desktop/phase_non_neural_real/RI_pred/RI.npy')
        # RI = torch.tensor(RI).cuda()
        # RI = RI.unsqueeze(0).permute(3, 0, 1, 2)
        # RI = RI / torch.max(RI)

        # RI_pred = self.DnCNN(xhat[:,:,:3].unsqueeze(0).permute(3, 0, 1, 2))

        args.regularize_type = ''
        if args.regularize_type == "dncnn2d":
            # print(xhat.shape)
            # print(xhat.grad_fn)
            xhat_trans = torch.transpose(torch.squeeze(xhat), 3, 0)
            xhat_concat = torch.cat([xhat_trans[0, ...], xhat_trans[1, ...]], 2)
            xhat_concat = torch.transpose(xhat_concat, 2, 0)
            xhat_expand = xhat_concat.unsqueeze(1)
            with torch.no_grad():
                dncnn_loss = self.dncnn(xhat_expand)
            phase_regularize_value = (dncnn_loss.mean().squeeze()) * 1
            # phase_regularize_value = dncnn_loss(args, xhat_expand.to('cpu'), reuse=reuse)

            # 记得打开
            # phase_regularize_value = torch.tensor(0.0)
            absorption_regularize_value = torch.tensor(0.0)

        if 1:
            decay_steps = 200
            alpha_xy = max((decay_steps - steps) / decay_steps, 0.001)
            tv_z = alpha_xy * self.__total_variation_z(xhat)
            tv_xy = alpha_xy * tv_loss(xhat)

        Hxhat = Hxhat.unsqueeze(1)
        y = y.unsqueeze(1)
        adaptive_ssim_ratio=1#+min(1*(steps)/3000,1)
        ssim = adaptive_ssim_ratio*(1 - self.SSIM(Hxhat, y)) / 2
        # print(ssim)


        if xhat_gt is not None and 0:
            mse_ri = torch.mean(torch.square(xhat - xhat_gt)) / 20

            div=diversity_loss(xhat)
            div_gt= diversity_loss(xhat_gt)
        else:
            mse_ri,div,div_gt=torch.tensor(0),torch.tensor(0),torch.tensor(0)
        # print(steps)
        ratio_mse = 80
        ratio_ssim = 3
        ratio_tv_z = args.tv_z
        ratio_tv_xy = args.tv_xy
        ratio_div=0.4
        ratio_pc= 0.0001
        mse = mse * ratio_mse
        ssim = ssim * ratio_ssim
        tv_z = tv_z * ratio_tv_z
        tv_xy = tv_xy * ratio_tv_xy
        div=abs(div-div_gt)*ratio_div*1e3
        pc_loss = pc_loss * ratio_pc
        #tv_xy = tv_xy * ratio_tv_xy
        #phase_regularize_value = phase_regularize_value * ratio_reg
        #dark_xy = dark_xy*ratio_dark

        #final loss
        loss = (
                mse
                + ssim
                #+ (absorption_regularize_value + phase_regularize_value)
                + tv_z
                + tv_xy
                #+ pc_loss
                #+ div
                #+ dark_xy
        )

        return (
            loss,
            mse,
            tv_z,
            ssim,
            tv_xy,
            div,
            pc_loss,
            mse_ri,
            mse1,
            mse2
        )

    def __total_variation_2d(self, images):
        pixel_dif2 = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :])
        pixel_dif3 = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1])
        total_var = torch.sum(pixel_dif2) + torch.sum(pixel_dif3)
        return total_var

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

    def __total_variation_z(self, images):
        """
        Normalized total variation 3d
        :param images: Images should have 4 dims: batch_size, z, x, y
        :return:
        """
        pixel_dif1 = torch.abs(images[:, :, 1:] - images[ :,:, :-1])
        total_var = torch.sum(pixel_dif1)
        return total_var
    # def __dncnn_inference(
    #     self,
    #     input,
    #     reuse,
    #     output_channel=1,
    #     layer_num=10,
    #     filter_size=3,
    #     feature_root=64,
    # ):
    #     # input layer
    #     with torch.no_grad():
    #         in_node = nn.Conv2d(input.size(1), feature_root, filter_size, padding=filter_size//2)
    #         in_node = F.relu(in_node)
    #         # composite convolutional layers
    #         for layer in range(2, layer_num):
    #             in_node = nn.Conv2d(feature_root, feature_root, filter_size, padding=filter_size//2, bias=False)
    #             in_node = F.relu(nn.BatchNorm2d(feature_root)(in_node))
    #         # output layer and residual learning
    #         in_node = nn.Conv2d(feature_root, output_channel, filter_size, padding=filter_size//2)
    #         output = input - in_node
    #     return output

    # def __dncnn_2d(self, args, images,reuse=True):  # [N, H, W, C]
    #     """
    #     DnCNN as 2.5 dimensional denoiser based on l-2 norm
    #     """
    #     a_min = args.DnCNN_normalization_min
    #     a_max = args.DnCNN_normalization_max
    #     normalized = (images - a_min) / (a_max - a_min)
    #     denoised = self.__dncnn_inference(torch.clamp(normalized, 0, 1),reuse)
    #     denormalized = denoised * (a_max - a_min) + a_min
    #     dncnn_res = torch.sum(denormalized**2)
    #     return dncnn_res


class dncnn_2d(nn.Module):
    def __init__(self, args, input_channel, output_channel=1, layer_num=10, filter_size=3, feature_root=64):
        super(dncnn_2d, self).__init__()
        self.input_conv = nn.Conv2d(input_channel, feature_root, filter_size, padding=filter_size // 2)
        self.convs = nn.ModuleList([
            nn.Conv2d(feature_root, feature_root, filter_size, padding=filter_size // 2, bias=False) for i in
            range(layer_num)
        ])
        self.output_conv = nn.Conv2d(feature_root, output_channel, filter_size, padding=filter_size // 2)
        self.relu = nn.ReLU()

        # in_node = nn.Conv2d(input.size(1), feature_root, filter_size, padding=filter_size // 2)
        # in_node = F.relu(in_node)
        # # composite convolutional layers
        # for layer in range(2, layer_num):
        #     in_node = nn.Conv2d(feature_root, feature_root, filter_size, padding=filter_size // 2, bias=False)
        #     in_node = F.relu(nn.BatchNorm2d(feature_root)(in_node))
        # # output layer and residual learning
        # in_node = nn.Conv2d(feature_root, output_channel, filter_size, padding=filter_size // 2)

    def forward(self, args, images, reuse=True):
        a_min = args.DnCNN_normalization_min
        a_max = args.DnCNN_normalization_max
        normalized = (images - a_min) / (a_max - a_min)
        denoised = self.__dncnn_inference(torch.clamp(normalized, 0, 1), reuse)
        denormalized = denoised * (a_max - a_min) + a_min
        dncnn_res = torch.sum(denormalized ** 2)
        return dncnn_res

        return 0

    def __dncnn_inference(self, input, reuse=True):
        x = self.input_conv(input)
        for f in self.convs:
            x = f(x)
            x = self.relu(x)
        output = self.output_conv(x)

        return output