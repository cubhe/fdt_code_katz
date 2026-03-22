# FDT Agent Notes

This file is for future agents working in `/global/home/users/cubhe/FDT`.

## Project intent

- Main training entry: `run_nerf.py`
- Main dataset generator: `test_ri_ucdavis_gen.py`
- Current active dataset: `/global/scratch/users/cubhe/FDT/dataset/ucdavis_dx0.33`
- Workspace symlink: `dataset -> /global/scratch/users/cubhe/FDT/dataset`

## Python environment

- Use `/global/scratch/users/cubhe/conda_envs/lim/bin/python`
- Typical launch pattern:

```bash
CUDA_VISIBLE_DEVICES=0 /global/scratch/users/cubhe/conda_envs/lim/bin/python -u run_nerf.py ...
```

## Dataset paths

- Ground truth RI: `/global/scratch/users/cubhe/FDT/dataset/ucdavis_dx0.33/RI_gt.npy`
- Light locations: `/global/scratch/users/cubhe/FDT/dataset/ucdavis_dx0.33/new_location1024org.npy`
- Rendered intensity stack: `/global/scratch/users/cubhe/FDT/dataset/ucdavis_dx0.33/new_img1024org.npy`
- Rendered preview video: `/global/scratch/users/cubhe/FDT/dataset/ucdavis_dx0.33/new_img1024org.mp4`

## Important args

- `--fs`: free-space distance
- `--max_ri`: RI delta scale used by render/training normalization
- `--location_noise_enable`: whether to perturb initial light locations
- `--location_noise_std`: Gaussian noise std before normalization
- `--location_noise_scale`: final max absolute scale of location perturbation
- `--location_noise_seed`: RNG seed for location perturbation
- `--self_calibration_enable`: whether `locations` and `dxyz` are trainable
- `--c2f_enable`: coarse-to-fine training switch
- `--c2f_stage_steps`: stage boundaries, must start with `0`
- `--c2f_stage_resolutions`: per-stage resolution, same length as `c2f_stage_steps`
- `--render`: `1` means regenerate `new_img1024org.npy/mp4`, `0` means reuse existing rendered data
- `--object_category_ori auto`: `run_nerf.py` auto-builds experiment names

## Current naming convention

Experiments are auto-named like:

```text
ucdavis_dx0.33_fs50_layer14_ri003_ln1_sc1_c2f0
```

Meaning:

- `fs50`: free-space distance 50
- `ri003`: `max_ri=0.03`
- `ln1`: location noise enabled
- `sc1`: self calibration enabled
- `c2f0`: coarse-to-fine disabled

## Output layout

- Human logs live under `/global/scratch/users/cubhe/FDT/log`
- One training run maps to one folder: `/global/scratch/users/cubhe/FDT/log/<exp>`
- Keep prior save contents and filenames, but place them inside that experiment folder
- TensorBoard lives inside the same folder: `/global/scratch/users/cubhe/FDT/log/<exp>/tensorboard`
- Preview images live inside the same folder: `/global/scratch/users/cubhe/FDT/log/<exp>/image_pred`
- RI side outputs live inside the same folder: `/global/scratch/users/cubhe/FDT/log/<exp>/RI_pred`

## Parallel experiment matrix

The 2x2 matrix currently used is:

- `sc0 c2f0`
- `sc0 c2f1`
- `sc1 c2f0`
- `sc1 c2f1`

Typical shared settings:

- `--N_iters 500`
- `--fs 50`
- `--max_ri 0.03`
- `--location_noise_enable 1`
- `--c2f_stage_steps 0 10 30 50`
- `--c2f_stage_resolutions 128 256 512 512`

## Operational notes

- If `new_img1024org.npy` already exists for the same `fs/max_ri` condition, launch training with `--render 0`.
- Only one process should render into `new_img1024org.npy` at a time.
- When running multiple trainings in parallel on the same GPU, expect slower throughput and higher memory pressure.
- `run_nerf.py` still keeps backward-compatible legacy args (`test_self_calibration`, `self_calibration`, `c2f`), but new work should prefer the explicit `*_enable` and `c2f_stage_*` args.
