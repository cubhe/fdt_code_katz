# FDT

这个仓库用于 FDT 相关的 RI 数据生成、render 和训练。常用入口是 `run_nerf.py`、`test_ri_ucdavis_gen.py` 和 `test_ri_ucdavis_eval.py`。

## 项目组织

当前约定是：

- 代码仓库：`/global/home/users/cubhe/FDT`
- scratch 项目根目录：`/global/scratch/users/cubhe/FDT`
- 数据目录：`/global/scratch/users/cubhe/FDT/dataset`
- 训练日志与实验输出：`/global/scratch/users/cubhe/FDT/log`

说明：

- `home` 目录主要放代码、脚本和文档
- `scratch` 目录主要放数据、checkpoint、可视化结果等较重文件
- 仓库里的 `dataset/` 是指向 scratch 数据目录的软链接

## 数据位置

项目统一使用：

`/global/scratch/users/cubhe/FDT/dataset`

仓库里的 `dataset/` 是指向该目录的软链接。当前主要使用的数据集是：

`/global/scratch/users/cubhe/FDT/dataset/ucdavis_dx0.33`

其中常用文件包括：

- `RI_gt.npy`
- `new_location1024org.npy`
- `new_img1024org.npy`
- `new_img1024org.mp4`

## 环境

训练和 render 默认使用：

`/global/scratch/users/cubhe/conda_envs/lim/bin/python`

## 数据生成

生成 UCDavis 数据：

```bash
python test_ri_ucdavis_gen.py --save-root dataset
```

## 常用训练参数

- `--fs`：free-space 距离
- `--max_ri`：RI 振幅上限
- `--location_noise_enable`：是否给初始 location 加噪声
- `--self_calibration_enable`：是否开启 self calibration
- `--c2f_enable`：是否开启 coarse-to-fine
- `--c2f_stage_steps`：不同阶段的切换 step
- `--c2f_stage_resolutions`：不同阶段使用的 resolution
- `--render`：是否重新生成 `new_img1024org.npy`

## 训练示例

单次训练：

```bash
CUDA_VISIBLE_DEVICES=0 /global/scratch/users/cubhe/conda_envs/lim/bin/python -u run_nerf.py \
  --N_iters 500 \
  --render 0 \
  --fs 50 \
  --max_ri 0.03 \
  --location_noise_enable 1 \
  --self_calibration_enable 1 \
  --c2f_enable 1 \
  --c2f_stage_steps 0 10 30 50 \
  --c2f_stage_resolutions 128 256 512 512
```

## 日志和结果

训练日志统一放在：

`/global/scratch/users/cubhe/FDT/log`

每次训练的所有产物都放在同一个实验目录：

`/global/scratch/users/cubhe/FDT/log/<实验名>/`

该目录里会保留原来的保存内容和命名方式，包括：

- 启动参数：`args.txt`
- 训练指标：`test_metrics.txt`
- checkpoint：`000020.tar`、`000040.tar` 等
- 周期性保存目录：`11/`、`21/`、`31/` 等
- TensorBoard：`tensorboard/`
- 训练过程预览：`image_pred/`
- RI 中间结果：`RI_pred/`
- 最新保存结果：`RI_pred.npy`、`locations_calib.npy`
- 汇总指标：`metricsRI.txt`

数据集目录 `/global/scratch/users/cubhe/FDT/dataset/ucdavis_dx0.33` 只保留数据本身，以及 render 生成的 `new_img1024org.npy/mp4`。

## 给后续 agent 的说明

面向自动化 agent 的详细操作说明见：

`README_FOR_AI.md`
