# Created by Renzhi He, COBI, UCDavis, 2023
# 11/22 test self calibration
import os
import time
import glob

import cv2
import torch
import torch.nn as nn
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from NeRF import *
from load_data import *
# from run_nerf_helpers import *
#from metrics import compute_img_metric
from loss import Loss
# np.random.seed(0)
# import matplotlib.pyplot as plt
import torch.nn.functional as F
from pathlib import Path

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    from unet import Unet
    from diffusion import GaussianDiffusion
    from utilsdiff import get_named_beta_schedule
except ImportError:
    Unet = None
    GaussianDiffusion = None
    get_named_beta_schedule = None
import itertools

def create_video_with_stats(imgs, video_path, fps=10):
    """
    创建视频，每帧显示归一化图像和统计信息
    
    参数:
    - imgs: 图像数组，形状为 (N, H, W)
    - video_path: 输出视频路径
    - fps: 帧率
    """
    height, width = imgs.shape[1], imgs.shape[2]
    
    # 设置视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    print(f"Creating video with {imgs.shape[0]} frames...")
    
    for i in range(imgs.shape[0]):
        img = imgs[i]
        
        # 计算统计信息
        img_min = np.min(img)
        img_max = np.max(img)
        img_mean = np.mean(img)
        
        # 归一化图像到0-255
        if img_max > img_min:
            img_norm = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_norm = np.zeros_like(img, dtype=np.uint8)
        
        # 转换为3通道图像以便添加彩色文字
        img_color = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
        
        # 添加文字信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (0, 255, 0)  # 绿色
        thickness = 2
        
        # 文字位置
        y_offset = 30
        
        # 添加帧号
        cv2.putText(img_color, f'Frame: {i:04d}', (10, y_offset), 
                   font, font_scale, color, thickness)
        
        # 添加最小值
        cv2.putText(img_color, f'Min: {img_min:.4f}', (10, y_offset + 30), 
                   font, font_scale, color, thickness)
        
        # 添加最大值
        cv2.putText(img_color, f'Max: {img_max:.4f}', (10, y_offset + 60), 
                   font, font_scale, color, thickness)
        
        # 添加平均值
        cv2.putText(img_color, f'Mean: {img_mean:.4f}', (10, y_offset + 90), 
                   font, font_scale, color, thickness)
        
        # 写入视频帧
        video_writer.write(img_color)
        
        # 显示进度
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{imgs.shape[0]} frames")
    # 释放视频写入器
    video_writer.release()
    print(f"Video saved to: {video_path}")

def apply_custom_colormap(gray_image, palette):
    # Normalize the grayscale image to have values between 0 and len(palette)-1
    normalized_gray = cv2.normalize(gray_image, None, 0, len(palette) - 1, cv2.NORM_MINMAX)

    # Create an empty color image
    colored_image = np.zeros((*gray_image.shape, 3), dtype=np.uint8)

    # Apply the palette
    for i in range(len(palette)):
        colored_image[normalized_gray == i] = palette[i]

    return colored_image
if os.path.exists("colormap0627.npy"):
    palette = np.load("colormap0627.npy")
    palette = palette[:, [2, 1, 0]]
else:
    palette = np.stack(
        [
            np.arange(256, dtype=np.uint8),
            np.arange(256, dtype=np.uint8),
            np.arange(256, dtype=np.uint8),
        ],
        axis=1,
    )


def _bool_flag(value):
    return bool(int(value))


def _format_tag(value):
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "")


def build_experiment_name(args):
    ri_tag = _format_tag(round(args.max_ri, 3))
    return (
        f"{args.data_name}_fs{_format_tag(args.fs)}_layer{args.layers}_ri{ri_tag}"
        f"_ln{int(_bool_flag(args.location_noise_enable))}"
        f"_dn{int(_bool_flag(getattr(args, 'dxyz_noise_enable', 0)))}"
        f"_scp{int(_bool_flag(getattr(args, 'position_calibration_enable', 0)))}"
        f"_scd{int(_bool_flag(getattr(args, 'dxyz_calibration_enable', 0)))}"
        f"_c2f{int(_bool_flag(args.c2f_enable))}"
        f"_b2c{args.b2c}"
        f"_b2b{args.b2b}"
        f"_NB{_format_tag(args.n_b)}"
    )


def get_stage_schedule(args):
    if args.test_self_calibration:
        args.location_noise_enable = 1
    if args.self_calibration:
        args.self_calibration_enable = 1
    if args.c2f:
        args.c2f_enable = 1

    apply_training_policy(args)

    if not _bool_flag(args.c2f_enable):
        return [0], [args.grid_x]

    stage_steps = [int(step) for step in args.c2f_stage_steps]
    stage_resolutions = [int(res) for res in args.c2f_stage_resolutions]
    if len(stage_steps) != len(stage_resolutions):
        raise ValueError("c2f_stage_steps and c2f_stage_resolutions must have the same length")
    if not stage_steps or stage_steps[0] != 0:
        raise ValueError("c2f_stage_steps must start with 0")
    if stage_steps != sorted(stage_steps):
        raise ValueError("c2f_stage_steps must be sorted in ascending order")
    return stage_steps, stage_resolutions


def get_stage_resolution(step, stage_steps, stage_resolutions):
    stage_idx = 0
    for idx, stage_start in enumerate(stage_steps):
        if step >= stage_start:
            stage_idx = idx
        else:
            break
    return stage_resolutions[stage_idx]


def build_location_initializer(light_loc_gt, args):
    if not _bool_flag(args.location_noise_enable):
        print("No adding location noise")
        return light_loc_gt

    rng = np.random.default_rng(args.location_noise_seed)
    noise = rng.normal(0.0, args.location_noise_std, light_loc_gt.shape).astype(np.float32)
    max_abs = np.max(np.abs(noise))
    if max_abs > 0 and args.location_noise_scale > 0:
        noise = noise / max_abs * args.location_noise_scale
    else:
        noise[:] = 0.0

    light_loc = light_loc_gt + noise
    light_loc[:, 2] = 0
    print(
        "Added location noise",
        f"std={args.location_noise_std}",
        f"scale={args.location_noise_scale}",
        f"seed={args.location_noise_seed}",
    )
    return light_loc


def build_dxyz_initializer(args):
    dxyz_gt = np.array([args.dx, args.dy, args.dz], dtype=np.float32)
    if not _bool_flag(getattr(args, "dxyz_noise_enable", 0)):
        print("No adding dxyz noise")
        return dxyz_gt

    rng = np.random.default_rng(args.dxyz_noise_seed)
    noise = rng.normal(0.0, args.dxyz_noise_std, dxyz_gt.shape).astype(np.float32)
    max_abs = np.max(np.abs(noise))
    if max_abs > 0 and args.dxyz_noise_scale > 0:
        noise = noise / max_abs * args.dxyz_noise_scale
    else:
        noise[:] = 0.0

    dxyz = dxyz_gt * (1.0 + noise)
    dxyz = np.maximum(dxyz, dxyz_gt * 0.05)
    print(
        "Added dxyz noise",
        f"std={args.dxyz_noise_std}",
        f"scale={args.dxyz_noise_scale}",
        f"seed={args.dxyz_noise_seed}",
        f"init={dxyz.tolist()}",
    )
    return dxyz


def build_optimizer(args, nerf, diffusion=None):
    location_params = []
    main_params = []
    for name, param in nerf.named_parameters():
        if name.endswith("locations"):
            if not param.requires_grad and not _bool_flag(getattr(args, "position_calibration_enable", 0)):
                continue
            location_params.append(param)
        else:
            if not param.requires_grad:
                if name.endswith("dxyz") and _bool_flag(getattr(args, "dxyz_calibration_enable", 0)):
                    main_params.append(param)
                    continue
                continue
            main_params.append(param)

    if diffusion is not None:
        main_params.extend(param for param in diffusion.model.parameters() if param.requires_grad)

    param_groups = []
    if main_params:
        param_groups.append(
            {
                "params": main_params,
                "lr": args.lrate,
                "group_name": "main",
            }
        )
    if location_params:
        param_groups.append(
            {
                "params": location_params,
                "lr": args.position_lrate,
                "group_name": "locations",
            }
        )

    return torch.optim.Adam(param_groups, betas=(0.9, 0.999))


def set_calibration_trainability(nerf, args, global_step):
    calibration_active = global_step >= int(getattr(args, "self_calibration_step", 0))
    position_active = _bool_flag(getattr(args, "position_calibration_enable", 0)) and calibration_active
    dxyz_active = _bool_flag(getattr(args, "dxyz_calibration_enable", 0)) and calibration_active

    nerf.module.locations.requires_grad_(position_active)
    nerf.module.dxyz.requires_grad_(dxyz_active)
    return calibration_active


def apply_training_policy(args):
    legacy_self_cal = (
        _bool_flag(args.self_calibration_enable)
        or _bool_flag(getattr(args, "self_calibration", 0))
        or _bool_flag(getattr(args, "test_self_calibration", 0))
    )
    if legacy_self_cal and not (
        _bool_flag(getattr(args, "position_calibration_enable", 0))
        or _bool_flag(getattr(args, "dxyz_calibration_enable", 0))
    ):
        args.position_calibration_enable = 1
        args.dxyz_calibration_enable = 1

    args.self_calibration_enable = int(
        _bool_flag(getattr(args, "position_calibration_enable", 0))
        or _bool_flag(getattr(args, "dxyz_calibration_enable", 0))
    )


def get_lr_schedule(args):
    stage_steps = list(getattr(args, "lr_stage_steps", [150, 500, 750]))
    stage_values = list(getattr(args, "lr_stage_values", [args.lrate]))
    if stage_values:
        args.lrate = float(stage_values[0])
    return stage_steps, stage_values


def get_main_learning_rate(args, global_step):
    stage_steps, stage_values = get_lr_schedule(args)
    if not stage_values:
        return float(args.lrate)

    for idx, step in enumerate(stage_steps):
        if global_step < step:
            return float(stage_values[min(idx, len(stage_values) - 1)])
    return float(stage_values[-1])


def normalize_pred_volume_for_metrics(volume):
    volume = volume.astype(np.float32).copy()
    volume[volume < 0] = 0
    vmax = np.max(volume)
    if vmax > 0:
        volume = volume / vmax
    return volume


def normalize_gt_volume_for_metrics(volume):
    volume = volume.astype(np.float32).copy()
    vmin = np.min(volume)
    vmax = np.max(volume)
    if vmax > vmin:
        volume = (volume - vmin) / (vmax - vmin)
    else:
        volume = np.zeros_like(volume)
    volume[volume < 0] = 0
    return volume


def evaluate_ri_metrics(ri_pred, ri_gt):
    ri_pred = normalize_pred_volume_for_metrics(ri_pred)
    ri_gt = normalize_gt_volume_for_metrics(ri_gt)
    pred_t = torch.tensor(ri_pred).permute(2, 0, 1).unsqueeze(1).float()
    gt_t = torch.tensor(ri_gt).permute(2, 0, 1).unsqueeze(1).float()
    mse, ssim, lpi, psnr, pcc, pc = metrics()(pred_t, gt_t)
    return {
        "mse": float(mse.item()),
        "ssim": float(ssim.item()),
        "lpips": float(lpi.mean().item()),
        "psnr": float(psnr.item()),
        "pcc": float(pcc.item()),
        "pc": float(pc.item()),
    }


def evaluate_image_metrics(snapshot_dir):
    pred_paths = sorted(glob.glob(os.path.join(snapshot_dir, "img_*_pred.png")))
    pairs = []
    for pred_path in pred_paths:
        gt_path = pred_path.replace("_pred.png", "_gt.png")
        if os.path.exists(gt_path):
            pairs.append((pred_path, gt_path))
    if not pairs:
        return None

    pred_imgs = []
    gt_imgs = []
    for pred_path, gt_path in pairs:
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            continue
        pred_imgs.append(pred.astype(np.float32) / 255.0)
        gt_imgs.append(gt.astype(np.float32) / 255.0)
    if not pred_imgs:
        return None

    pred_t = torch.tensor(np.stack(pred_imgs)).unsqueeze(1).float()
    gt_t = torch.tensor(np.stack(gt_imgs)).unsqueeze(1).float()
    mse, ssim, lpi, psnr, pcc, pc = metrics()(pred_t, gt_t)
    return {
        "count": len(pred_imgs),
        "mse": float(mse.item()),
        "ssim": float(ssim.item()),
        "lpips": float(lpi.mean().item()),
        "psnr": float(psnr.item()),
        "pcc": float(pcc.item()),
        "pc": float(pc.item()),
    }


def find_latest_snapshot_dir(exp_dir):
    step_dirs = [
        os.path.join(exp_dir, name)
        for name in os.listdir(exp_dir)
        if name.isdigit() and os.path.isdir(os.path.join(exp_dir, name))
    ]
    if not step_dirs:
        return None
    return max(step_dirs, key=lambda path: int(os.path.basename(path)))


def run_final_evaluation(exp_dir, data_path):
    latest_dir = find_latest_snapshot_dir(exp_dir)
    if latest_dir is None:
        print("Skip final evaluation: no snapshot directory found")
        return

    lines = [f"latest_dir={latest_dir}"]
    ri_path = os.path.join(latest_dir, "RI.npy")
    ri_gt_path = os.path.join(data_path, "RI_gt.npy")
    if os.path.exists(ri_path) and os.path.exists(ri_gt_path):
        ri_metrics = evaluate_ri_metrics(np.load(ri_path), np.load(ri_gt_path))
        lines.append(
            "RI: "
            + ",".join(
                [
                    f"mse:{ri_metrics['mse']:.4e}",
                    f"ssim:{ri_metrics['ssim']:.4f}",
                    f"lpips:{ri_metrics['lpips']:.4f}",
                    f"psnr:{ri_metrics['psnr']:.4f}",
                    f"pcc:{ri_metrics['pcc']:.4f}",
                    f"pc:{ri_metrics['pc']:.4f}",
                ]
            )
        )
    else:
        lines.append("RI: skipped")

    image_metrics = evaluate_image_metrics(latest_dir)
    if image_metrics is not None:
        lines.append(
            "IMAGE: "
            + ",".join(
                [
                    f"count:{image_metrics['count']}",
                    f"mse:{image_metrics['mse']:.4e}",
                    f"ssim:{image_metrics['ssim']:.4f}",
                    f"lpips:{image_metrics['lpips']:.4f}",
                    f"psnr:{image_metrics['psnr']:.4f}",
                    f"pcc:{image_metrics['pcc']:.4f}",
                    f"pc:{image_metrics['pc']:.4f}",
                ]
            )
        )
    else:
        lines.append("IMAGE: skipped")

    eval_path = os.path.join(exp_dir, "final_evaluation.txt")
    with open(eval_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Final evaluation saved to {eval_path}")


def get_experiment_dir(args, expname=None):
    if expname is None:
        expname = args.object_category_ori
    return os.path.join(args.basedir, expname)


def get_tensorboard_dir(exp_dir):
    return os.path.join(exp_dir, "tensorboard")


def get_preview_dir(exp_dir):
    return os.path.join(exp_dir, "image_pred")


def get_ri_dir(exp_dir):
    return os.path.join(exp_dir, "RI_pred")


def get_summary_metrics_path(exp_dir):
    return os.path.join(exp_dir, "metricsRI.txt")

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = None
        try:
            from torchvision.models import vgg16

            vgg = vgg16(pretrained=True).features[:10]
            self.vgg = vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        except Exception:
            self.vgg = None

    def forward(self, input, target):
        if self.vgg is None:
            return torch.tensor(0.0, device=input.device, dtype=input.dtype)
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)#.transpose(1,0)#[20]
            target = target.repeat(1, 3, 1, 1)#.transpose(1,0)#[20]
        input_features = self.vgg(input)
        target_features = self.vgg(target)
        loss = nn.functional.mse_loss(input_features, target_features)
        return loss

class metrics(nn.Module):
    def __init__(self, DnCNNN_channels=1, tower_idx=None, Hreal=None, Himag=None):
        super(metrics, self).__init__()
        self.tower_idx = tower_idx
        self.Hreal = Hreal
        self.Himag = Himag
        from ssim import SSIM
        self.SSIM = SSIM()
        self.pc=PerceptualLoss()
        self.lpips_fn = None
        self.lpips_import_error = None
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net='alex').eval()
            for param in self.lpips_fn.parameters():
                param.requires_grad = False
        except Exception as exc:
            self.lpips_import_error = exc


    ##############################
    ###     Loss Functions     ###
    ##############################
    def calculate_lpips(self,img1, img2):
        """
        Calculate the LPIPS metric between two images.

        Parameters:
        - img1, img2: tensors representing the two images to compare.
                      They should have a shape of (B, C, H, W) and be normalized to the range [-1, 1].
        - net_type: the type of pretrained network to use ('alex', 'vgg', etc.). 'alex' is commonly used.
        - use_gpu: a boolean indicating whether to use a GPU for computation.

        Returns:
        - lpips_distance: the LPIPS distance between the two images.
        """
        if self.lpips_fn is None:
            return torch.zeros((img1.shape[0], 1), device=img1.device, dtype=img1.dtype)

        if img1.shape[1] == 1:
            img1 = img1.repeat(1, 3, 1, 1)
            img2 = img2.repeat(1, 3, 1, 1)

        img1 = img1.float().clamp(0, 1) * 2 - 1
        img2 = img2.float().clamp(0, 1) * 2 - 1

        lpips_fn = self.lpips_fn.to(img1.device)

        # Calculate LPIPS distance
        with torch.no_grad():
            lpips_distance = lpips_fn(img1, img2)

        return lpips_distance
    def forward(self, x, gt_x,tower_idx=0, reuse=False):
        mse = torch.mean(torch.square(gt_x - x)) / 1
        # mse = torch.mean(torch.abs(gt_x - x)) / 20
        ssim = self.SSIM(gt_x, x)
        lpi=self.calculate_lpips(gt_x,x)
        max_pixel=1
        psnr_value = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        pcc=self.pcc_loss(x,gt_x)
        pc= self.pc(x,gt_x)#perceptual_loss(Hxhat, y)


        return (
            mse,
            ssim,
            lpi,
            psnr_value,
            pcc,
            pc
            # tv_xy,
            # div,
            # pc_loss,
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
    import torch

    def pcc_loss(self,output, target):
        """
        Compute the Pearson Correlation Coefficient (PCC) loss.

        Parameters:
        - output: tensor of predictions from the model.
        - target: tensor of ground truth values.

        Returns:
        - loss: 1 - PCC, where a lower loss indicates a higher correlation between output and target.
        """
        x = output - output.mean()
        y = target - target.mean()
        loss = 1 - (x * y).sum() / (torch.sqrt((x ** 2).sum()) * torch.sqrt((y ** 2).sum()))
        return loss

def render(args):
    """
    渲染函数：基于真实的RI_gt和locations生成相应的图片
    """
    print("Starting render function...")
    
    # Create output directory
    basedir = args.basedir
    expname = args.object_category_ori+'_render'
    exp_dir = get_experiment_dir(args, expname)
    preview_dir = get_preview_dir(exp_dir)
    ri_dir = get_ri_dir(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)
    os.makedirs(ri_dir, exist_ok=True)
    
    # Save args
    f = os.path.join(exp_dir, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    
    # Load ground truth data
    print("Loading ground truth data...")
    data_path = args.dataset_path
    
    # Load RI_gt
    RI_gt_path = Path(data_path + args.data_name + f'/RI_gt.npy')
    if RI_gt_path.exists():
        RI_gt = np.load(RI_gt_path)
        print(np.max(RI_gt), np.min(RI_gt))
        print(f"Loaded RI_gt from {RI_gt_path}, shape: {RI_gt.shape}")
        # Normalize RI_gt similar to training code
        print('min', np.min(RI_gt), 'max', np.max(RI_gt))
        RI_gt = (RI_gt- np.min(RI_gt)) / (np.max(RI_gt) - np.min(RI_gt)) * args.max_ri +args.n_b
        # RI_gt = RI_gt +args.n_b
        print('min', np.min(RI_gt), 'max', np.max(RI_gt))

        # for i in range(RI_gt.shape[2]):
        #     ri_tmp=RI_gt[:,:,i]
        #     ri_tmp=(ri_tmp-np.min(ri_tmp)) / (np.max(ri_tmp) - np.min(ri_tmp)) * 255
        #     cv2.imshow('ri_tmp',ri_tmp.astype('uint8'))
        #     cv2.waitKey(1)
    else:
        print(f"RI_gt file not found at {RI_gt_path}")
        return
    
    # Load locations
    locations_path = Path(data_path + args.data_name + '/new_location1024org.npy')
    if locations_path.exists():
        light_loc_gt = np.load(locations_path)
        print(f"Loaded locations from {locations_path}, shape: {light_loc_gt.shape}")
    else:
        print(f"Locations file not found at {locations_path}")
        return
    
    ids = light_loc_gt.shape[0]
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create NPRF model with ground truth data
    nerf = NPRF(args, locations=light_loc_gt)
    nerf = nn.DataParallel(nerf, list(range(args.num_gpu)))
    nerf = nerf.to(device)
    nerf.eval()  # Set to evaluation mode
    
    # Convert RI_gt to tensor
    RI_gt_tensor = torch.tensor(RI_gt).to(device)
    print('min', torch.min(RI_gt_tensor), 'max', torch.max(RI_gt_tensor))
    # Set batch size
    batch = min(args.batch if hasattr(args, 'batch') else 20, 60)
    print(f"Using batch size: {batch}")
    
    # Initialize output array
    imgs = np.zeros((ids, args.grid_x, args.grid_y), dtype=np.float32)
    
    print("Starting rendering process...")
    
    # Render images in batches
    with torch.no_grad():  # No gradient computation needed for rendering
        for iter in range(0, (ids + batch - 1) // batch):  # Ceiling division
            start_idx = iter * batch
            end_idx = min(start_idx + batch, ids)
            light_loc_ids = np.array(range(start_idx, end_idx))
            
            print(f"Rendering batch {iter + 1}/{(ids + batch - 1) // batch}, "
                  f"indices {start_idx} to {end_idx - 1}")
            
            # Call NPRF forward with ground truth RI
            with torch.autograd.set_detect_anomaly(True):
                # Use the RI_gt directly for rendering
                RI, intensity_pred, index_pred, locations_calibration = nerf(
                    light_loc_ids, 
                    training=False, 
                    steps=0,
                    mask=None,
                    ri_path=RI_gt_tensor  # Pass ground truth RI
                )
            
            # Convert to numpy
            intensity_pred = intensity_pred.cpu().detach().numpy().astype(np.float32)
            # print(intensity_pred.shape)
            # print(imgs.shape)
            # Store rendered images
            imgs[light_loc_ids] = intensity_pred
            
            # Optional display during rendering
            if args.show_img or 1:
                for i, global_idx in enumerate(light_loc_ids):
                    img_pred = intensity_pred[i]
                    img_min = np.min(img_pred)
                    img_max = np.max(img_pred)
                    img_mean = np.mean(img_pred)
                    
                    # Normalize for display
                    if img_max > img_min:
                        img_display = ((img_pred - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                    else:
                        img_display = np.zeros_like(img_pred, dtype=np.uint8)
                    
                    # Convert to color for text overlay
                    img_color = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
                    
                    # Add statistics text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img_color, f'Frame: {global_idx:04d}', (10, 30), 
                                font, 0.6, (0, 255, 0), 2)
                    cv2.putText(img_color, f'Min: {img_min:.4f}', (10, 60), 
                                font, 0.6, (0, 255, 0), 2)
                    cv2.putText(img_color, f'Max: {img_max:.4f}', (10, 90), 
                                font, 0.6, (0, 255, 0), 2)
                    cv2.putText(img_color, f'Mean: {img_mean:.4f}', (10, 120), 
                                font, 0.6, (0, 255, 0), 2)
                    
                    # cv2.imshow('rendering_progress', cv2.resize(img_color, None, fx=0.5, fy=0.5))
                    cv2.imwrite(os.path.join(preview_dir, f'img_render{global_idx}.png'), img_color)
                    # cv2.waitKey(1)
                
            
    
    # Save complete image dataset
    output_path = Path(data_path + args.data_name + '/new_img1024org.npy')
    np.save(output_path, imgs)
    print(f"Saved rendered images to {output_path}")
    
    # Create video from rendered images
    print("Creating video from rendered images...")
    video_path = Path(data_path + args.data_name + '/new_img1024org.mp4')
    create_video_with_stats(imgs, str(video_path), fps=10)
    
    # Close any open windows
    if args.show_img:
        cv2.destroyAllWindows()
    
    print(f"Rendering complete! Generated {ids} images")
    print(f"Images saved to: {output_path}")
    print(f"Video saved to: {video_path}")

def train(args):

    # Create log dir and copy the config file
    expname = args.object_category_ori
    exp_dir = get_experiment_dir(args, expname)
    preview_dir = get_preview_dir(exp_dir)
    ri_dir = get_ri_dir(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)
    os.makedirs(ri_dir, exist_ok=True)

    #tensorboard
    tensorboard_dir = get_tensorboard_dir(exp_dir)
    args.tbdir = tensorboard_dir
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(Path(tensorboard_dir))
    #writer.add_scalar('Loss/train', 11, 1)
    #writer.close() log_dir
    #metrics_np=np.zeros((8,args.N_iters))

    # save args
    f = os.path.join(exp_dir, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    test_metric_file = os.path.join(exp_dir, 'test_metrics.txt')

    # Load data
    print("Start to load data.", end='  ')
    if 1:
        # load data including images and illuminate source location
        args.data_path=args.dataset_path+args.data_name
        images, light_loc_gt = load_phase_data(args.data_path,calib=0)
        #if we use subimage number, we need to select the subimage
        if args.sub_num>images.shape[0]:
            args.sub_num=images.shape[0]
        args.batch = args.sub_num if args.sub_num<20 else 20
        selected_item=np.random.choice(images.shape[0],args.sub_num,replace=False)
        images=images[selected_item]
        light_loc_gt=light_loc_gt[selected_item]
        ids = images.shape[0]
        #if we use the silumation data, we need to load the RI_gt
        if args.simulation:
            RI_gt=np.load(Path(args.data_path+f'/RI_gt.npy'))
            RI_gt =RI_gt-1.33
            print(np.max(RI_gt), np.min(RI_gt))
            RI_gt=RI_gt/np.max(RI_gt)*args.max_ri
        else:
            RI_gt=0
        # RI_gt=0
        if args.simulation:
            print('Loaded phase_data', images.shape, light_loc_gt.shape, RI_gt.shape)
        else:
            print('Loaded phase_data', images.shape, light_loc_gt.shape, 'RI_gt=0 (real data)')

        # if args.show_img:
        #     for i in range(images.shape[0]):
        #         img=images[i]
        #         # print(sampled_items[i], np.mean(img))
        #         #norm
        #         img=img/np.max(img)*255
        #         img=cv2.resize(img,(512,512))
        #         img=img.astype('uint8')
        #         cv2.imshow('img',img)
        #         cv2.waitKey(10)
    else:
        print('Unknown camera dataset type', args.camera_dataset_type, 'exiting')
        return
    shuffle_idx = np.random.permutation(images.shape[0])

    if args.add_noise:
        print('Adding img noise')
        for i in range(images.shape[0]):
            img = images[i, :, :]
            img = np.random.poisson(img.astype(np.float32) * args.add_noise) / args.add_noise
            images[i, :, :] = img
            if args.show_img:   
                img=img/np.max(img)*255
                img=cv2.resize(img,(512,512))
                img=img.astype('uint8')
                cv2.imshow('img',img)
                cv2.waitKey(10)
    else:
        print('No adding img noise')
            

    light_loc = build_location_initializer(light_loc_gt, args)
    dxyz_init = build_dxyz_initializer(args)
    args.dx_init = float(dxyz_init[0])
    args.dy_init = float(dxyz_init[1])
    args.dz_init = float(dxyz_init[2])
    print("Done")
    #add noise to the light location
    

    global_step = 0
    stage_steps, stage_resolutions = get_stage_schedule(args)
    args.init_block = stage_resolutions[0]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Diffusion Net
    if args.activate_diffusion:
        print('Diffusion')
        net = Unet(
                    in_ch = args.inch,
                    mod_ch = args.modch,
                    out_ch = args.outch,
                    ch_mul = args.chmul,
                    num_res_blocks = args.numres,
                    cdim = args.cdim + 128,
                    use_conv = args.useconv,
                    droprate = args.droprate,
                    dtype = args.dtype
                )
        args.T=10
        betas = get_named_beta_schedule(num_diffusion_timesteps = args.T)
        diffusion = GaussianDiffusion(
                    dtype = args.dtype,
                    model = net,
                    betas = betas,
                    w = args.w,
                    v = args.v,
                    device = device
                )
    else:
        print('No diffusion')
    


    # Load Checkpoints
    ckpts = [os.path.join(exp_dir, f) for f in sorted(os.listdir(exp_dir)) if '.tar' in f]
    print('Found ckpts', ckpts)

    # Make the optimizer start with the dataset-specific base LR from step 0.
    get_lr_schedule(args)
    args.block_size = get_stage_resolution(global_step, stage_steps, stage_resolutions)
    nerf = NPRF(args,locations=light_loc)
    nerf = nn.DataParallel(nerf, list(range(args.num_gpu)))
    calibration_active = set_calibration_trainability(nerf, args, global_step)

    optimizer = build_optimizer(
        args,
        nerf,
        diffusion=diffusion if args.activate_diffusion else None,
    )

    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        global_step = ckpt['global_step']
        netwrok_dict=ckpt['network_state_dict']
        netwrok_dict['module.dxyz']
        # ckpt['network_state_dict']['module.dxyz'][0]=0.25
        # ckpt['network_state_dict']['module.zz'][0]=73

        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # Load model
        smart_load_state_dict(nerf, ckpt)
        if args.activate_diffusion:
            diffusion.load_state_dict(ckpt['diffusion_state_dict'])
        ri_pred_path = os.path.join(exp_dir, 'RI_pred.npy')
        if not os.path.exists(ri_pred_path):
            ri_pred_path = os.path.join(args.data_path, 'RI_pred.npy')
        RI_pre = np.load(ri_pred_path)
        light_loc = light_loc_gt


    loss=Loss()

    # Move data to GPU
    nerf = nerf.to(device)
    if args.activate_diffusion:
        diffusion = diffusion.to(device)
    loss = loss.to(device)
    RI_gt=torch.tensor(RI_gt).to(device)


    #begin training
    print('Begin')  
    batch = args.batch
    nerf.train()
    if args.activate_diffusion:
        diffusion.model.train()
    i_batch = 0    
    for iter in range(0, args.N_iters):

        print(global_step, ':', end=' ')
        wants_calibration = _bool_flag(getattr(args, "position_calibration_enable", 0)) or _bool_flag(getattr(args, "dxyz_calibration_enable", 0))
        if wants_calibration and (not calibration_active) and global_step >= int(getattr(args, "self_calibration_step", 0)):
            calibration_active = set_calibration_trainability(nerf, args, global_step)
            print(f"Enable self calibration at step {global_step}", end=' ')
        if (i_batch+1) * batch >= ids:
            #print("Shuffle data after an epoch!")
            shuffle_idx = np.random.permutation(ids)
            i_batch = 0
        #shuffle_idx = np.array(range(ids))
        light_loc_training, intensity, light_loc_ids = process_traning_data_simu(images, light_loc_gt,shuffle_idx, i_batch, batch)
        temp_a=intensity.mean()
        intensity = torch.tensor(intensity).cuda().float()


        #####  Core optimization loop  #####
        with (torch.autograd.set_detect_anomaly(True)):


            light_loc_ids_t = torch.tensor(light_loc_ids).long().cuda()
            RI, intensity_pred, index_pred, locations_calibration = nerf(
                light_loc_ids_t,
                steps=global_step,
                steps_c2f=stage_steps,
                block_sizes=stage_resolutions,
            )
            # DataParallel gather concatenates non-batch outputs along dim 0;
            # take only the first replica for RI and locations.
            if args.num_gpu > 1:
                ri_size = int(RI.shape[0] // args.num_gpu)
                RI = RI[:ri_size]
                loc_size = int(locations_calibration.shape[0] // args.num_gpu)
                locations_calibration = locations_calibration[:loc_size]


            # intensity_pred = 1 * intensity_pred.permute(1, 2, 0) / (
            #     torch.max(intensity.view(intensity.size(0), -1), dim=1).values)
            # intensity_pred = intensity_pred.permute(2, 0, 1)
            
            # intensity = 1 * intensity.permute(1, 2, 0) / (
            #     torch.max(intensity.view(intensity.size(0), -1), dim=1).values)
            # intensity = intensity.permute(2, 0, 1)

            ####diffusion loss
            if args.activate_diffusion:
                print(RI.shape)#512,512,40
                # RI_input=RI
                #interpolate to 64,64
                RI_input=F.interpolate(RI.permute(2,0,1).unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0).unsqueeze(1)
                print(RI_input.shape)

                img_pred=F.interpolate(intensity_pred.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0)
                input_img=img_pred.unsqueeze(0).unsqueeze(0)
                print(input_img.shape)
                loss_dif, generated = diffusion.trainloss(input_img)
                if loss_dif<20 and loss_dif>0:
                    loss_dif*=0.5
                else:
                    loss_dif=torch.tensor((1e-8)).cuda()
            else:
                loss_dif=torch.tensor((1e-8)).cuda()




            radius = args.radius
            mask_batch = torch.zeros(images[light_loc_ids].shape).float()
            max_min_intensity_pred = torch.zeros(intensity_pred.shape[0],2)
            max_min_intensity = torch.zeros(intensity.shape[0],2)
            mean_std_intensity_pred = torch.zeros(intensity_pred.shape[0],2)
            mean_std_intensity = torch.zeros(intensity.shape[0],2)
            for i in range(intensity_pred.shape[0]):


                loc_x = int((light_loc_training[i][0] + 0.5) * mask_batch.shape[1])
                loc_y = int((light_loc_training[i][1] + 0.5) * mask_batch.shape[2])
                mask_cpu = (255 * mask_batch[i]/torch.max(mask_batch[i])).cpu().detach().numpy().astype('uint8')
                mask_cpu = cv2.circle(mask_cpu, (int(loc_x), int(loc_y)), radius, (255, 255, 155), -1)
                mask_batch[i] = torch.tensor(mask_cpu / 255).float().cuda()

                min_value=torch.min(intensity_pred[i])
                max_value=torch.max(intensity_pred[i])
                max_min_intensity_pred[i] = torch.tensor([min_value,max_value])
                mean_std_intensity_pred[i] = torch.tensor([intensity_pred[i].mean(),intensity_pred[i].std()])

                min_value=torch.min(intensity[i])
                max_value=torch.max(intensity[i])
                max_min_intensity[i] = torch.tensor([min_value,max_value])
                mean_std_intensity[i] = torch.tensor([intensity[i].mean(),intensity[i].std()])

            if getattr(args, "camera_dataset_type", "") != "ucdavis":
                # Legacy datasets use an additional fixed spatial crop.
                spatial_mask = torch.zeros(mask_batch.shape[1], mask_batch.shape[2]).float().cuda()
                spatial_mask[492:692, 412:612] = 1.0
                mask_batch = mask_batch * spatial_mask.unsqueeze(0)

            # Apply center mask if enabled
            if _bool_flag(getattr(args, "center_mask_enable", 0)):
                center_mask_size = getattr(args, "center_mask_size", 200)
                h, w = mask_batch.shape[1], mask_batch.shape[2]
                center_mask_cpu = np.zeros((h, w), dtype=np.uint8)
                cx, cy = w // 2, h // 2
                r = center_mask_size // 2
                cv2.circle(center_mask_cpu, (cx, cy), r, 255, -1)
                center_mask_t = torch.tensor(center_mask_cpu / 255.0).float().cuda()
                mask_batch = mask_batch * center_mask_t.unsqueeze(0)

            #normalize by std and mean
            # intensity_pred = ((intensity_pred.permute(1, 2, 0) - mean_std_intensity_pred[:,0]) / ( mean_std_intensity_pred[:,0])).permute(2, 0, 1)
            # intensity = ((intensity.permute(1, 2, 0) - mean_std_intensity[:,0]) / ( mean_std_intensity[:,0])).permute(2, 0, 1)



            



            # mean = intensity_pred[mask_batch == 1].mean()
            # std = intensity_pred[mask_batch == 1].std()
            # intensity_pred[mask_batch == 1] = (intensity_pred[mask_batch == 1] - mean) / (std + 1e-5)
            # intensity_pred = intensity_pred * mask_batch

            #normalize the intensity_pred
            # min_value=torch.min(intensity_pred.view(intensity_pred.size(0), -1), dim=1)
            # max_value=torch.max(intensity_pred.view(intensity_pred.size(0), -1), dim=1)
            # intensity_pred = (intensity_pred.permute(1, 2, 0) - min_value.values) / (max_value.values - min_value.values)
            # intensity_pred = intensity_pred.permute(2, 0, 1)
            #calculate the mean of intensity_pred for each channel
            # mean = intensity_pred.mean(dim=(1,2))
            # print(mean)
            # print(intensity_pred.shape,mask_batch.shape)
            # intensity_pred = intensity_pred * mask_batch
            #set the first pixel to 1
            # intensity_pred[:,0,0]=1e-6


            # mean = intensity[mask_batch == 1].mean()
            # std = intensity[mask_batch == 1].std()
            # intensity[mask_batch == 1] = (intensity[mask_batch == 1] - mean) / (std + 1e-5)
            # intensity = intensity * mask_batch
            
            # min_value=torch.min(intensity.view(intensity.size(0), -1), dim=1)
            # max_value=torch.max(intensity.view(intensity.size(0), -1), dim=1)
            # intensity = (intensity.permute(1, 2, 0) - min_value.values) / (max_value.values - min_value.values)
            # intensity = intensity.permute(2, 0, 1)
            # intensity = intensity * mask_batch

            # normalization
            norm_mode = getattr(args, 'norm_mode', 'minmax')
            if norm_mode == 'mean_std':
                intensity_pred = ((intensity_pred.permute(1, 2, 0) - mean_std_intensity[:,0]) / (mean_std_intensity[:,1] + 1e-5)).permute(2, 0, 1)
                intensity_pred = intensity_pred * mask_batch
                intensity = ((intensity.permute(1, 2, 0) - mean_std_intensity[:,0]) / (mean_std_intensity[:,1] + 1e-5)).permute(2, 0, 1)
                intensity = intensity * mask_batch
            elif norm_mode == 'std_minmax':
                # step 1: std normalization (using gt stats)
                intensity_pred = ((intensity_pred.permute(1, 2, 0) - mean_std_intensity[:,0]) / (mean_std_intensity[:,1] + 1e-5)).permute(2, 0, 1)
                intensity = ((intensity.permute(1, 2, 0) - mean_std_intensity[:,0]) / (mean_std_intensity[:,1] + 1e-5)).permute(2, 0, 1)
                # step 2: minmax normalization (per-image, using gt min/max after std)
                gt_min = intensity.view(intensity.size(0), -1).min(dim=1).values
                gt_max = intensity.view(intensity.size(0), -1).max(dim=1).values
                intensity_pred = ((intensity_pred.permute(1, 2, 0) - gt_min) / (gt_max - gt_min + 1e-5)).permute(2, 0, 1)
                intensity_pred = intensity_pred * mask_batch
                intensity = ((intensity.permute(1, 2, 0) - gt_min) / (gt_max - gt_min + 1e-5)).permute(2, 0, 1)
                intensity = intensity * mask_batch
            else:  # minmax
                intensity_pred = ((intensity_pred.permute(1, 2, 0) - max_min_intensity[:,0]) / (max_min_intensity[:,1] - max_min_intensity[:,0])).permute(2, 0, 1)
                intensity_pred = intensity_pred * mask_batch
                intensity = ((intensity.permute(1, 2, 0) - max_min_intensity[:,0]) / (max_min_intensity[:,1] - max_min_intensity[:,0])).permute(2, 0, 1)
                intensity = intensity * mask_batch

            # intensity_pred = intensity_pred / torch.sum(intensity_pred, (1,2)).unsqueeze(1).unsqueeze(1)*torch.sum(intensity, (1,2)).unsqueeze(1).unsqueeze(1)
            # intensity = intensity
            # print('total energy pred', torch.sum(intensity_pred, (1,2)))
            # print('total energy gt', torch.sum(intensity, (1,2)))

            for i in range(intensity_pred.shape[0]):
                # print(light_loc_ids[i])
                # tifffile.imwrite('./RI_pred/RI_3D.tif', RI.detach().cpu().numpy().transpose(2,1,0), photometric='minisblack')

                mask=mask_batch[i].cpu().detach().numpy()
                img_pred = (intensity_pred[i]).cpu().detach().numpy()
                img_pred=(255*(img_pred - np.min(img_pred))/(np.max(img_pred)-np.min(img_pred)))
                img_pred=(img_pred*1).astype('uint8')
                img_gt = (intensity[i]).cpu().detach().numpy()
                img_gt = (255*(img_gt - np.min(img_gt))/(np.max(img_gt)-np.min(img_gt)))
                img_gt = (img_gt * 1).astype('uint8')
                img_pred=cv2.resize(img_pred,img_gt.shape)
                cv2.imwrite(os.path.join(preview_dir, f'img_pred{shuffle_idx[i+batch*i_batch]}.png'), img_pred)
                loc_x = int((light_loc_training[i][0] + 0.5) * img_gt.shape[0])
                loc_y = int((light_loc_training[i][1] + 0.5) * img_gt.shape[1])
                img_gt_temp = cv2.circle(img_gt,(int(loc_x),int(loc_y)), radius, (255, 255, 155), 7)
                img_pred_temp = cv2.circle(img_pred,(int(loc_x),int(loc_y)), radius, (255, 255, 155), 7)
                if args.show_img:
                    #cv2.imshow('mask', cv2.resize(mask_cpu, None, fx=1, fy=1).astype('uint8'))
                    img3 = np.hstack((img_gt_temp, img_pred_temp))
                    # img3=255*(img3-np.min(img3))/(np.max(img3)-np.min(img3)).astype('uint8')
                    cv2.imshow('img_gt&pred', cv2.resize(img3, None, fx=0.5, fy=0.5))
                    cv2.waitKey(10)


            if _bool_flag(args.c2f_enable) and global_step in stage_steps[1:]:
                optimizer = build_optimizer(
                    args,
                    nerf,
                    diffusion=diffusion if args.activate_diffusion else None,
                )


            block_size = get_stage_resolution(global_step, stage_steps, stage_resolutions)
            intensity = intensity.unsqueeze(0)
            downsampled = F.interpolate(intensity, size=(block_size*1,block_size*1),mode='bilinear', align_corners=False)
            intensity = downsampled.squeeze(0)

            mask_batch = mask_batch.unsqueeze(0)
            downsampled = F.interpolate(mask_batch, size=(block_size*1,block_size*1),mode='bilinear', align_corners=False)
            mask_batch = downsampled.squeeze(0)

            intensity_pred = intensity_pred.unsqueeze(0)
            downsampled = F.interpolate(intensity_pred, size=(block_size*1,block_size*1),mode='bilinear', align_corners=False)
            intensity_pred = downsampled.squeeze(0)




            #mse = img2mse(intensity_pred, intensity) * 3
            #ssim_loss = 1 - compute_img_metric(intensity_pred * 100, intensity * 100, 'ssim')
            l1_reg=torch.mean(torch.norm(RI-1.33,p=1,dim=2))/50
            locations_mse=torch.mean(abs(locations_calibration-torch.tensor(light_loc_gt)))
            ri_mse=torch.mean(abs(RI-RI_gt)**2)
            #print(locations_mse)
            losses, mse, tv_z, ssim , tv_xy,div,pc,MSE_ri,mse1,mse2= loss(args, intensity_pred, RI, intensity, global_step,xhat_gt=RI_gt)
            if args.activate_diffusion:
                losses+=loss_dif
            #losses+=l1_reg
            #ri_loss=0#img2mse(RI[:,:,1:],RI_gt)*10
            # print(intensity_pred.dtype, intensity.dtype, torch.max(intensity_pred[0]), torch.max(intensity[0]))
            #loss = mse + ssim_loss+ri_loss

            print(f'loss mse,ssim,MSE_ri,mse_location,mse_ri, {losses.item():05f},{mse.item():05f}, {ssim.item():05f}, {locations_mse:05f},{ri_mse:05f}',end=' ')
            print(f'loss_dif,tv_xy,tv_z: {loss_dif.item():05f}, {tv_xy.item():05f},{tv_z.item():05f}')
            #re print these values

            with open(test_metric_file, 'a') as file:
                 file.write(f'{global_step:04d}:loss mse,ssim,MSE_ri {losses.item():5f}, {mse.item():05f}, {ssim.item():05f},;'
                            f'loc_mse,loss_dif,tv_xy,tv_z,pc: ,{loss_dif.item():05f}, {tv_xy.item():05f},{tv_z.item():05f}\n')

            optimizer.zero_grad()
            losses.backward()
            weight_loc = 1 if _bool_flag(args.self_calibration_enable) else 0

            # NOTE: IMPORTANT!
            ##   update learning rate   ###
            # decay_rate = 0.1
            # decay_steps = args.lrate_decay * 1000
            # new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            new_lrate = get_main_learning_rate(args, global_step)
            lr_scale = new_lrate / args.lrate if args.lrate > 0 else 1.0
            for param_group in optimizer.param_groups:
                if param_group.get('group_name') == 'locations':
                    param_group['lr'] = args.position_lrate * lr_scale
                else:
                    param_group['lr'] = new_lrate
            # param=optimizer.param_groups[0]['params'][0]
            # param.grad=param.grad*100
            # grad=param.grad/1000
            # grad.sum()
            optimizer.param_groups[0]['capturable'] = True
            optimizer.step()
            global_step += 1


        ########################################################
        if getattr(args, "i_save_override", 0) > 0:
            args.i_save = args.i_save_override
        elif global_step<100:
            args.i_save=10
        elif global_step<1000:
            args.i_save=100
        else:
            args.i_save=200

        if getattr(args, "i_weights_override", 0) > 0:
            args.i_weights = args.i_weights_override
        elif global_step<100:
            args.i_weights=20
        elif global_step<1000:
            args.i_weights=100
        else:
            args.i_weights = 200
        if global_step % args.i_weights == 0:
            path = os.path.join(exp_dir, '{:06d}.tar'.format(global_step))
            if args.activate_diffusion:
                torch.save({
                    'global_step': global_step,
                    'network_state_dict': nerf.state_dict(),
                    'diffusion_state_dict': diffusion.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            else:
                torch.save({
                    'global_step': global_step,
                    'network_state_dict': nerf.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            print('Saved checkpoints at', path)

        should_save_snapshot = (global_step % args.i_save == 0 and global_step > 0) or (iter == args.N_iters - 1)
        if should_save_snapshot:
            print('Start save', end=' ')
            # print('dxyz=', optimizer.param_groups[0]['params'][1].data)
            path = os.path.join(exp_dir, str(global_step))
            if not os.path.exists(path):
                os.makedirs(path)
            RI_cpu = RI.detach().cpu().numpy()
            locations=locations_calibration.detach().cpu().numpy()
            dxyz = nerf.module.dxyz.detach().cpu().numpy()
            if tifffile is not None:
                ri_tif = np.transpose(RI_cpu.astype(np.float32), (2, 0, 1))
                tifffile.imwrite(os.path.join(path, 'ri.tif'), ri_tif)
                tifffile.imwrite(os.path.join(ri_dir, 'ri.tif'), ri_tif)
            np.save(os.path.join(exp_dir, 'locations_calib.npy'), locations)
            np.save(os.path.join(exp_dir, 'dxyz_calib.npy'), dxyz)
            np.save(os.path.join(exp_dir, 'RI_pred.npy'), RI_cpu)
            np.save(path + '/locations.npy', locations)
            np.save(path + '/dxyz.npy', dxyz)
            np.save(path + '/RI.npy', RI_cpu)
            np.save(os.path.join(ri_dir, 'RI.npy'), RI_cpu)

            #########evaluate the RI#########
            if os.path.exists(args.data_path+'/RI_gt.npy'):
                # from run_nerf_evaluate import metrics
                print('Evaluate the RI')
                ri_gt=np.load(args.data_path+'/RI_gt.npy')
                ri_gt=(ri_gt-np.min(ri_gt))/(np.max(ri_gt)-np.min(ri_gt))
                # ri_gt=ri_gt.transpose(2,0,1)
                # ri_gt=ri_gt/np.max(ri_gt)
                ri_pred=RI_cpu
                ri_pred[ri_pred<0]=0
                ri_gt[ri_gt<0]=0
                ri_pred=ri_pred/np.max(ri_pred)

                print(ri_pred.shape, np.max(ri_pred), np.min(ri_pred))
                print(ri_gt.shape, np.max(ri_gt), np.min(ri_gt))

                ri_pred = torch.tensor(ri_pred).permute(2, 0, 1).unsqueeze(1).float()
                ri_gt = torch.tensor(ri_gt).permute(2, 0, 1).unsqueeze(1).float()

                try:
                    mse,ssim,lpi,psnr,pcc,pc=metrics()(ri_pred,ri_gt)
                    mse=mse
                    print(f'mse:{mse:.4e},ssim:{ssim:.4f},lpips:{lpi.mean().item():.4f},psnr:{psnr.item():.4f},pcc:{pcc.item():.4f},pc:{pc.item():.4f}')
                    with open(path+'/metrics.txt','a') as f:
                        f.write(f'RI: mse:{mse:.4e},ssim:{ssim:.4f},lpips:{lpi.mean().item():.4f},psnr:{psnr.item():.4f},pcc:{pcc.item():.4f},pc:{pc.item():.4f}\n')
                    with open(get_summary_metrics_path(exp_dir), 'a') as f:
                        f.write(f'{args.object_category_ori}:')
                        f.write(f'RI: mse:{mse:.4e},ssim:{ssim:.4f},lpips:{lpi.mean().item():.4f},psnr:{psnr.item():.4f},pcc:{pcc.item():.4f},pc:{pc.item():.4f}\n')
                except Exception as exc:
                    print(f'Skip RI metric evaluation: {exc}')
                # ri_gt=ri_gt.permute(1,2,3,0).cpu().numpy().squeeze()-1.33
                # ri_pred=ri_pred.permute(1,2,3,0).cpu().numpy().squeeze()-1.33
                # generate_video_ri(ri_gt,ri_pred,path)


            for i in range(RI_cpu.shape[2]):
                img = RI_cpu[:, :, i]
                img_norm = (img-np.min(img)) / (np.max(img)-np.min(img)) * 255
                cv2.imwrite(os.path.join(path, 'RI_pred_' + str(i)) + '.png', img_norm.astype('uint8'))
                if args.show_img:
                    cv2.imshow('ri',img_norm.astype('uint8'))
                    cv2.waitKey(10)
            for i in range(intensity_pred.shape[0]):
                img_pred = (255 * intensity_pred[i]/torch.max(intensity_pred[i])).cpu().detach().numpy().astype('uint8')
                img_gt = (255 * intensity[i]/torch.max(intensity[i])).cpu().detach().numpy().astype('uint8')
                img3 = np.hstack((img_gt, img_pred))
                cv2.imwrite(os.path.join(path, f'img_{str(shuffle_idx[i+batch*i_batch])}_pred.png'), img_pred)
                cv2.imwrite(os.path.join(path, f'img_{str(shuffle_idx[i+batch*i_batch])}_gt.png'), img_gt)
                # img_pred = (255 * intensity_pred[i]/torch.max(intensity_pred[i])).cpu().detach().numpy().astype('uint8')
                # img_gt = (255 * intensity[i]/torch.max(intensity[i])).cpu().detach().numpy().astype('uint8')
                # img3 = np.hstack((img_gt, img_pred))
                # cv2.imwrite(os.path.join(path, str(shuffle_idx[i+batch*i_batch])) + '.png', img_pred)
                if args.show_img:
                    cv2.imshow('img_gt&pred', cv2.resize(img3, None, fx=0.3, fy=0.3))
                    cv2.waitKey(1)
            print('start save video')
            img_cpu=intensity.detach().cpu().numpy()
            img_pred_cpu=intensity_pred.detach().cpu().numpy()
            #stack in the third dimension
            img_comp=np.concatenate((img_cpu,img_pred_cpu),axis=2)
            video_generate(img_comp,path,data_type='img')
            #for ri
            RI_cpu=RI_cpu.transpose(2,0,1)
            RI_cpu=RI_cpu/np.max(RI_cpu)
            if args.simulation:
                RI_GT=RI_gt.detach().cpu().numpy().transpose(2,0,1)
                RI_GT=RI_GT/np.max(RI_GT)
                ri_comp=np.concatenate((RI_GT,RI_cpu),axis=2)
            else:
                ri_comp=RI_cpu
            video_generate(ri_comp,path,data_type='ri')   
        if global_step % args.i_testset == 0 and i > 0:
            print('Start test')
            ########## to be completed ##########
            

        if global_step % args.i_tensorboard == 0:
            writer.add_scalar("all/Loss", losses.item(), global_step)
            writer.add_scalar("all/loss_dif", loss_dif.item(), global_step)
            writer.add_scalar("all/ssim", ssim.item(), global_step)
            #writer.add_scalar("all/MSE_ri*1e6", (MSE_ri * 1e6).item(), global_step)
            writer.add_scalar("MSE/mse", mse.item(), global_step)
            writer.add_scalar("MSE/mse1", mse1.item(), global_step)
            writer.add_scalar("MSE/mse2", mse2.item(), global_step)
            writer.add_scalar("MSE/locations_mse", locations_mse.item(), global_step)
            writer.add_scalar("MSE/ri_mse", ri_mse.item(), global_step)
            #writer.add_scalar("other/pc", pc.item(), global_step)
            writer.add_scalar("rothereg/tv_z", tv_z.item(), global_step)
            writer.add_scalar("rothereg/tv_xy", tv_xy.item(), global_step)
            writer.add_scalar("other/l1_reg", l1_reg.item(), global_step)
            writer.add_scalar("other/div", div.item(), global_step)
            # writer.add_scalar("other/zz", optimizer.param_groups[0]['params'][2].data.item(), global_step)
            writer.add_scalar("other/dx", nerf.module.dxyz.data[0].item(), global_step)
            writer.add_scalar("other/dz", nerf.module.dxyz.data[2].item(), global_step)
        i_batch += 1

    if _bool_flag(getattr(args, "final_eval_enable", 0)):
        run_final_evaluation(exp_dir, args.data_path)


if __name__ == '__main__':
    from datetime import datetime
    from args_real_beads import config_parser

    if (torch.cuda.is_available()):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    parser = config_parser()
    args = parser.parse_args()
    # if args.render:
        
    #     exit()
    # else:
    #     print("Running in training mode...")

    args.activate_diffusion=False
    seed = 1121
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    apply_training_policy(args)

    if args.object_category_ori == "auto":
        args.object_category_ori = build_experiment_name(args)

    if args.render:
        print("Running render stage before training...")
        render(args)

    time1 = datetime.now()
    train(args)
    time2 = datetime.now()
    time_diff = time2 - time1
    print(f"time_total(s): {time_diff.total_seconds()}")#verify

    if (torch.cuda.is_available()):
        torch.cuda.empty_cache()
