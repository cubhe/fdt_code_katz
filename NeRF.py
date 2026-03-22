# Created by Renzhi He, COBI, UCDavis, 2023

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.tensorboard import SummaryWriter
#import skimage
#from skimage.metrics import peak_signal_noise_ratio
import cv2
import math
import time
import gc
from absl import flags
import logging


from optics import PhaseObject3D, TomographySolver

#import contexttimer

FLAGS = flags.FLAGS

NUM_Z = "nz"
INPUT_CHANNEL = "ic"
OUTPUT_CHANNEL = "oc"
MODEL_SCOPE = "infer_y"
NET_SCOPE = "MLP"
DNCNN_SCOPE = "DnCNN"



# get total number of visible gpus
NUM_GPUS = torch.cuda.device_count()

def smart_load_state_dict(model: nn.Module, state_dict: dict):
    if "network_fn_state_dict" in state_dict.keys():
        state_dict_fn = {k.lstrip("module."): v for k, v in state_dict["network_fn_state_dict"].items()}
        state_dict_fn = {"mlp_coarse." + k: v for k, v in state_dict_fn.items()}

        state_dict_fine = {k.lstrip("module."): v for k, v in state_dict["network_fine_state_dict"].items()}
        state_dict_fine = {"mlp_fine." + k: v for k, v in state_dict_fine.items()}
        state_dict_fn.update(state_dict_fine)
        state_dict = state_dict_fn
    elif "network_state_dict" in state_dict.keys():
        state_dict = {k[7:]: v for k, v in state_dict["network_state_dict"].items()}
    else:
        state_dict = state_dict

    if isinstance(model, nn.DataParallel):
        state_dict = {"module." + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

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

def remove_data(array, proportion_to_remove):
    N = array.shape[0]
    num_to_remove = int(np.round(N * proportion_to_remove))
    shuffle_index=np.arange(N)
    np.random.shuffle(shuffle_index)
    indices_to_remove = shuffle_index[0:num_to_remove]
    indices_to_remove = np.sort(indices_to_remove)
    remaining_data = np.delete(array, indices_to_remove, axis=0)
    org_index=np.arange(N)
    indices_to_remain = np.delete(org_index, indices_to_remove, axis=0)
    return remaining_data, indices_to_remove,indices_to_remain

def insert_data_torch(original_tensor, data, indices_to_remove,indices_to_remain):
    new_tensor = torch.zeros(data.shape).cuda().float()
    new_tensor[indices_to_remain] = original_tensor.float()
    new_tensor[indices_to_remove] = torch.tensor(data[indices_to_remove]).float()
    return new_tensor

#################################################
# ***      CLASS OF NEURAL REPRESENTATION     ****
#################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class InterpolateParameter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param, target_size):
        ctx.shape = param.size()
        ctx.target_size = target_size
        # 使用插值，但保持param的形状不变
        return F.interpolate(param, size=(target_size, target_size), mode='bilinear', align_corners=False)

    @staticmethod
    def backward(ctx, grad_output):
        # 反插值到原始大小
        return F.interpolate(grad_output, size=ctx.shape[-2:], mode='bilinear', align_corners=False), None


class NPRF(nn.Module):
    def __init__(self, FLAGS,RI=None, locations=None, name="model_summary"):
        super(NPRF, self).__init__()
        args=FLAGS
        # Setup parameters
        self.args = args
        self.name = name
        self.wavelength = args.wavelength
        self.NA = args.NA
        self.n_measure=args.n_measure
        self.n_b=args.n_b
        self.factor = args.factor

        self.dz = args.dz / self.factor
        self.layer = args.layers#int((layer+0.1) * 1)

        self.dx = args.dx / self.factor
        #
        grid_x = args.H / self.dx
        self.grid_x_org = int(grid_x * 1)
        self.grid_x = int(grid_x * 1)
        #self.dx = 2

        self.dy = args.dy / self.factor
        grid_y = args.W / self.dy #/ self.factor
        self.grid_y_org = int(grid_y * 1)
        self.grid_y = int(grid_y * 1)


        self.grid_x_org = args.grid_x
        self.grid_y_org = args.grid_y
        self.grid_x = args.grid_x
        self.grid_y = args.grid_y
        self.max_ri = args.max_ri
        if RI is None:
            self.refractive_update=np.random.rand(self.grid_x*self.grid_x*self.layer,1)*0.1
        # else:
        #     RI_pre=RI[(self.grid_x-self.shape_x)//2:-(self.grid_x-self.shape_x)//2, (self.grid_y-self.shape_y)//2:-(self.grid_y-self.shape_y)//2,1:]
        #     #RI_pre=
        #     self.refractive_update = RI_pre.reshape(self.shape_x*self.shape_y*self.layer,1)-self.n_b
        self.patch_ratio=args.patch_ratio

        ####selfcalibration
        self.locations=nn.Parameter(torch.tensor(locations))
        init_dx = getattr(args, "dx_init", self.dx)
        init_dy = getattr(args, "dy_init", self.dy)
        init_dz = getattr(args, "dz_init", self.dz)
        dxyz=torch.tensor([init_dx, init_dy, init_dz], dtype=torch.float32)
        self.dxyz=nn.Parameter(dxyz)
        # self.zz=nn.Parameter(torch.tensor([args.zz*100]))
        #print(self.locations.mean)
        #self.test=nn.Parameter(torch.tensor((1.)))
        
        self.model=args.model
        if self.model=='exp':
            self.sampling=4
            if getattr(args, "c2f_enable", 0):
                init_block = int(args.c2f_stage_resolutions[0])
            else:
                init_block = int(self.grid_x)
            self.RI_init = nn.Parameter(torch.zeros((init_block, init_block, self.layer)))
        elif self.model=='imp':
            self.xy_encoding_num=3
            self.z_encoding_num=3

            #save the coordiantes of the points
            self.coordinates=nn.Parameter(torch.tensor(self.generate_coordinates(self.grid_x, self.grid_x, self.layer)))
            self.feature_dim=24
            self.inputlayer = nn.Linear(self.feature_dim, args.mlp_kernel_size)
            self.lineares = nn.ModuleList(
                [nn.Linear(args.mlp_kernel_size, args.mlp_kernel_size) for i in range(args.mlp_layer_num)])
            self.outputlayer = nn.Linear(args.mlp_kernel_size, 1)
            #reshape the coordinates to the input of the 3D ri
            # self.coordinates=self.coordinates.view(-1,self.feature_dim)

        else:
            self.feature_dim = args.feature_dim
            print(self.feature_dim)
            input_dim = self.feature_dim
            output_dim = 1 ##RI
            args.mlp_layer_num=6
            args.mlp_kernel_size=32
            self.inputlayer = nn.Linear(input_dim, args.mlp_kernel_size)
            self.lineares = nn.ModuleList(
                [nn.Linear(args.mlp_kernel_size, args.mlp_kernel_size) for i in range(args.mlp_layer_num)])
            self.outputlayer = nn.Linear(args.mlp_kernel_size, output_dim)
            
            if self.model=='tri':
                self.xyplane = nn.Parameter(torch.ones((args.grid_x, args.grid_x, self.feature_dim))*1,requires_grad=True)
                self.yzplane = nn.Parameter(torch.ones((args.grid_x, args.layers, self.feature_dim))*1,requires_grad=True)
                self.xzplane = nn.Parameter(torch.ones((args.grid_x, args.layers, self.feature_dim))*1,requires_grad=True)
            elif self.model=='nvp':
                self.feature_points=nn.Parameter(torch.empty((args.grid_x* args.grid_x* self.layer,self.feature_dim)),requires_grad=True)
            else:
                raise ValueError("model must be imp, exp, tri or nvp")

        self.per_img_gain = nn.Parameter(torch.ones(self.n_measure))
        self.le_relu = nn.LeakyReLU(negative_slope=args.relu_slope, inplace=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    ###########################
    ###     Neural Nets     ###
    ###########################
    def generate_coordinates(self, grid_x, grid_y, grid_z):
        # 使用mgrid生成三维网格的坐标
        x, y, z = np.mgrid[-grid_x / 2:grid_x / 2, -grid_y / 2:grid_y / 2, 0:grid_z]

        # 将这些坐标合并并转换为(N, 3)的数组
        coordinates = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
        coordinates[:, 0]=coordinates[:,0] / grid_x
        coordinates[:, 1]=coordinates[:,1] / grid_y
        coordinates[:, 2]=coordinates[:,2] / grid_z*5 #* (grid_z/3)

        return coordinates

    def forward(self, light_loc_ids, training=True, steps=0, mask=None, ri_path=None, steps_c2f=None, block_sizes=None):
        self.dx = self.dxyz[0]
        self.dy = self.dxyz[1]
        self.dz = self.dxyz[2]
        # free_space=self.zz-self.dz*self.layer
        free_space = torch.tensor(self.args.fs)  # self.zz - self.dz * self.layer
        
        # Get light source locations
        light_loc = self.locations[light_loc_ids]
        
        # Choose RI computation method based on training mode and ri_path
        if not training and ri_path is not None:
            # Render mode: use provided ground truth RI
            print(f"Using provided RI for rendering, shape: {ri_path.shape}")
            RI = ri_path
            
            # # Ensure RI has the correct shape and device
            # if len(RI.shape) == 3:
            #     # RI shape should be (H, W, Z)
            #     if RI.shape != (self.grid_x, self.grid_y, self.layer):
            #         print(f"Resizing RI from {RI.shape} to ({self.grid_x}, {self.grid_y}, {self.layer})")
            #         # Resize RI to match expected dimensions
            #         RI = F.interpolate(
            #             RI.permute(2, 0, 1).unsqueeze(0), 
            #             size=(self.grid_x, self.grid_y), 
            #             mode='bilinear', 
            #             align_corners=False
            #         ).squeeze(0).permute(1, 2, 0)
                    
            #         # Adjust number of layers if needed
            #         if RI.shape[2] != self.layer:
            #             current_layers = RI.shape[2]
            #             if current_layers < self.layer:
            #                 # Pad with zeros
            #                 padding = torch.zeros(self.grid_x, self.grid_y, self.layer - current_layers, 
            #                                     dtype=RI.dtype, device=RI.device)
            #                 RI = torch.cat([RI, padding], dim=2)
            #             else:
            #                 # Truncate or interpolate
            #                 RI = F.interpolate(
            #                     RI.permute(2, 0, 1).unsqueeze(0),
            #                     size=(self.layer, self.grid_x, self.grid_y),
            #                     mode='trilinear',
            #                     align_corners=False
            #                 ).squeeze(0).permute(1, 2, 0)
            # else:
            #     raise ValueError(f"Expected RI to have 3 dimensions (H, W, Z), got {len(RI.shape)}")
            
            # Make sure RI is on the correct device
            if RI.device != next(self.parameters()).device:
                RI = RI.to(next(self.parameters()).device)
                
            # # Normalize RI values to expected range
            # # Assuming the input RI is in the range [0, 255] or [0, 1]
            # if RI.max() > 1.0:
            #     RI = RI / 255.0  # Normalize from [0, 255] to [0, 1]
            
            # # Convert to refractive index values
            # # Assuming we want RI values around 1.33 (water) + some variation
            # RI = RI * self.max_ri + self.n_b  # Scale and offset to proper RI range
            
            print(f"Final RI range: [{RI.min().item():.4f}, {RI.max().item():.4f}]")
            
        else:
            # Training mode: use neural network to compute RI
            if self.model == 'imp':
                coordinates = self.coordinates  # .view(-1, self.feature_dim)
                # coordinates=coordinates.view(-1, self.feature_dim)
                output = self.__neural_repres(coordinates)
                # output=self.inputlayer(coordinates)
                # for i in range(self.args.mlp_layer_num):
                #     output=self.lineares[i](output)
                # print(output.shape)
                # output=self.outputlayer(output)
                RI = output.view(self.grid_x, self.grid_x, self.layer)
                RI = F.interpolate(RI[:, :, :].permute(2, 0, 1).unsqueeze(0), 
                                size=(self.grid_x, self.grid_x), mode='bilinear').squeeze(0).permute(1, 2, 0)

            else:
                # Other model types (exp, tri, nvp)
                if self.model == 'exp':
                    if steps_c2f is not None and steps in steps_c2f:
                        index = np.where(steps == np.array(steps_c2f))[0][0]
                        block_size = block_sizes[index]
                        self.block_size = block_size
                        if self.RI_init.shape[0] != block_size or self.RI_init.shape[1] != block_size:
                            RI = F.interpolate(
                                self.RI_init.permute(2, 0, 1).unsqueeze(0),
                                size=(block_size, block_size),
                                mode='bilinear',
                            ).squeeze(0).permute(1, 2, 0)
                            self.RI_init = nn.Parameter(RI, requires_grad=True)
                    RI = F.interpolate(self.RI_init[:, :, :].permute(2, 0, 1).unsqueeze(0), 
                                    size=(self.grid_x, self.grid_x), mode='bilinear').squeeze(0).permute(1, 2, 0)
                elif self.model == 'tri':
                    feature_points = self.xyplane.unsqueeze(2) * self.xzplane.unsqueeze(1) * self.yzplane.unsqueeze(0)
                    RI = self.neural_repres(feature_points.view(-1, self.feature_dim)).view(self.grid_x, self.grid_x, self.layer)
                elif self.model == 'nvp':
                    RI = self.neural_repres(self.feature_points.view(-1, self.feature_dim)).view(self.grid_x, self.grid_x, self.layer)
                else:
                    raise ValueError("model must be exp, tri or nvp")
            
            # Apply neural network transformations (only in training mode)
            # last_layer=torch.zeros(1024,1024,1)
            # RI = torch.cat((RI, last_layer), dim=2)
            RI = (self.sigmoid(RI) - 0.5) / 0.5 * self.max_ri  # -0.1~0.1
            RI = self.le_relu(RI)  # 0-0.1
            RI = RI + self.n_b

        # print(torch.mean(RI))
        # patch=torch.ones((RI.shape[0],RI.shape[1],1))*1.33
        # RI=torch.cat((RI,patch),dim=2)
        
        # Render intensity using the computed or provided RI
        intensity = self.rendering(light_loc, RI, free_space)
        
        # ---------- 应用 per-image gain（关键新增） ----------
        # per_img_gain 的形状是 [n_measure]，在这里根据 light_loc_ids_t 取出当前 batch 的 gain
        gain = self.per_img_gain[light_loc_ids]                  # [B]
        gain = gain.view(-1, 1, 1)                               # [B, 1, 1]
        intensity = intensity * gain                             # [B, H, W]

        # Return results
        # In render mode, we return the actual RI values rather than RI-1.33
        if not training:
            return RI, intensity, ..., self.locations
        else:
            return RI - 1.33, intensity, ..., self.locations


    def rendering(self, light_source, refractive_index,free_space=75):
        # print("refractive index shape:", refractive_index.shape)
        # print("light source shape:", light_source.shape)
        # print("input:",self.input[0])
        self.wavelength = 0.6  # fluorescence wavelength
        # objective immersion media

        # background refractive index, PDMS
        self.n_b = 1.33
        #fx_illu_list = (light_source[ :, 0] - self.grid_x//2)*self.dx/self.grid_x
        #fy_illu_list = (light_source[ :, 1] - self.grid_y // 2) * self.dy / self.grid_y
        fx_illu_list = light_source[ :, 0]*self.dx
        fy_illu_list = light_source[ :, 1]*self.dy
        fz_illu_list = torch.zeros_like(fx_illu_list)#light_source[:,2]
        intensityfield=self.multislice(refractive_index,  fx_illu_list=fx_illu_list, fy_illu_list=fy_illu_list, fz_illu_list=fz_illu_list, dx=self.dx,
                   dy=self.dy, dz=self.dz,free_space=free_space)

        return intensityfield

    def multislice(self, refractive_index, fx_illu_list, fy_illu_list, fz_illu_list, dx=0.2, dy=0.2, dz=0.2,free_space=75):
        # Setup solver objects
        solver_params = dict(wavelength=self.wavelength, na=self.NA, \
                             RI_measure=self.n_measure, sigma=2 * np.pi * dz / self.wavelength, \
                             fx_illu_list=fx_illu_list, fy_illu_list=fy_illu_list, fz_illu_list=fz_illu_list, \
                             pad=False, pad_size=(50, 50))
        ## add value to the phantom
        phase_obj_3d = PhaseObject3D(shape=refractive_index.shape, RI_obj=refractive_index,voxel_size=(dy, dx, dz), RI=self.n_b,free_space=free_space,args=self.args)
        #phase_obj_3d.RI_obj[grid_x//2-50:grid_x//2+50, grid_y//2-50:grid_y//2+50,:] = phase_obj_3d.RI_obj[grid_x//2-50:grid_x//2+50, grid_y//2-50:grid_y//2+50,:] + refractive_index
        #phase_obj_3d.RI_obj=refractive_index
        solver_obj = TomographySolver(phase_obj_3d, **solver_params)
        solver_obj.setScatteringMethod(model="MultiPhaseContrast")
        forward_field_mb, fields = solver_obj.forwardPredict(field=True)

        forward_field_mb = torch.squeeze(torch.stack(forward_field_mb))
        intensityfield = torch.abs(forward_field_mb * torch.conj(forward_field_mb))
        return intensityfield
    def neural_repres(self,x):
        x = self.inputlayer(x)
        x = self.le_relu(x)
        for f in self.lineares:
            
            
            x = f(x)
            x = self.le_relu(x)
        x = self.outputlayer(x)
        output = self.le_relu(x)
        return output


    
    def __neural_repres(self, x, skip_layers=[]):
        # positional encoding
        x=x.float()
        if 1:
            s = torch.sin(torch.arange(0, 180, self.args.dia_digree) * np.pi / 180)[:, None]
            c = torch.cos(torch.arange(0, 180, self.args.dia_digree) * np.pi / 180)[:, None]
            fourier_mapping = torch.cat((s, c), dim=1).T
            if(torch.cuda.is_available()):
                fourier_mapping = fourier_mapping.to('cuda').cuda()
            xy_freq = torch.matmul(x[:, :2], fourier_mapping)
        
            for l in range(self.args.xy_encoding_num):
                cur_freq = torch.cat(
                    [
                        torch.sin(2 ** l * np.pi * xy_freq),
                        torch.cos(2 ** l * np.pi * xy_freq),
                    ],
                    dim=-1,
                )
                if l == 0:
                    tot_freq = cur_freq
                else:
                    tot_freq = torch.cat([tot_freq, cur_freq], dim=-1)
                # print('z', in_node[:, 2].max(), in_node[:, 2].min(), in_node[:, 2].mean())
            for l in range(self.args.z_encoding_num):
                cur_freq = torch.cat(
                    [
                        torch.sin(2 ** l * np.pi * x[:, 2].unsqueeze(-1)),
                        torch.cos(2 ** l * np.pi * x[:, 2].unsqueeze(-1)),
                    ],
                    dim=-1,
                )
        
                tot_freq = torch.cat([tot_freq, cur_freq], dim=-1)
        
        # else:
        #     raise NotImplementedError(FLAGS.positional_encoding_type)
        
        # input to MLP
        x = tot_freq

        x = self.inputlayer(x)
        x = self.le_relu(x)

        layer_cout = 1
        for f in self.lineares:
            layer_cout += 1
            if layer_cout in skip_layers:
                x = torch.cat([x, tot_freq], -1)
                x = self.skiplayer(x)
                x = self.le_relu(x)
            x = f(x)
            # print('x min a:',x.min())
            x = self.le_relu(x)
            # print('x min b:',x.min())
        x = self.outputlayer(x)
        output = self.le_relu(x)
        #output = 2*torch.sigmoid(x)
        #output = output / self.output_scale


        return output

    def save(self, directory, epoch=None, train_provider=None):
        if epoch is not None:
            directory = os.path.join(directory, "{}_model/".format(epoch))
        else:
            directory = os.path.join(directory, "latest/".format(epoch))
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, "model")
        if train_provider is not None:
            train_provider.save(directory)
        torch.save(self.state_dict(), path)
        print("saved to {}".format(path))
        return path

    def restore(self, model_path):

        param = torch.load(model_path)
        # param_model=self.state_dict()
        # new_dict={}
        # for k,v in param.items():
        #     if k in param_model:
        #         print(k)
        #         print(v)
        self.load_state_dict(param, strict=False)


    def load_ri(self,RI):
        self.refractive_update=RI





