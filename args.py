import argparse


def config_parser():
    parser = argparse.ArgumentParser(description="FDT training config for UCDavis dataset.")

    # Dataset
    parser.add_argument("--dataset_path", default="/global/scratch/users/cubhe/FDT/dataset/")
    parser.add_argument("--data_name", default="ucdavis_dx0.33")
    parser.add_argument("--object_category_ori", default="auto")
    parser.add_argument("--simulation", action="store_true", default=True)
    parser.add_argument("--sub_num", type=int, default=1500)
    parser.add_argument("--camera_dataset_type", default="ucdavis")

    # Logging / outputs
    parser.add_argument("--basedir", default="/global/scratch/users/cubhe/FDT/log")
    parser.add_argument("--tbdir", default="/global/scratch/users/cubhe/FDT/log/tensorboard")
    parser.add_argument("--txt", default="")
    parser.add_argument("--no_reload", action="store_true", default=False)

    # Training schedule
    parser.add_argument("--N_iters", type=int, default=1000)
    parser.add_argument("--lrate", type=float, default=5e-3)
    parser.add_argument("--position_lrate", type=float, default=1e-3)
    parser.add_argument("--lrate_decay", type=int, default=250)
    parser.add_argument("--lr_stage_steps", nargs="+", type=int, default=[150, 500, 750])
    parser.add_argument("--lr_stage_values", nargs="+", type=float, default=[5e-3, 5e-3, 5e-3, 5e-3])
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--i_save", type=int, default=100)
    parser.add_argument("--i_weights", type=int, default=100)
    parser.add_argument("--i_save_override", type=int, default=0)
    parser.add_argument("--i_weights_override", type=int, default=0)
    parser.add_argument("--i_testset", type=int, default=1000)
    parser.add_argument("--i_tensorboard", type=int, default=10)

    # Runtime toggles
    parser.add_argument("--render", type=int, default=1)
    parser.add_argument("--show_img", action="store_true", default=False)
    parser.add_argument("--add_noise", type=float, default=0.0)
    parser.add_argument("--location_noise_enable", type=int, default=0)
    parser.add_argument("--location_noise_std", type=float, default=0.1)
    parser.add_argument("--location_noise_scale", type=float, default=0.1)
    parser.add_argument("--location_noise_seed", type=int, default=1121)
    parser.add_argument("--dxyz_noise_enable", type=int, default=0)
    parser.add_argument("--dxyz_noise_std", type=float, default=0.1)
    parser.add_argument("--dxyz_noise_scale", type=float, default=0.1)
    parser.add_argument("--dxyz_noise_seed", type=int, default=2203)
    parser.add_argument("--disable_train_location_noise", type=int, default=1)
    parser.add_argument("--self_calibration_enable", type=int, default=0)
    parser.add_argument("--position_calibration_enable", type=int, default=0)
    parser.add_argument("--dxyz_calibration_enable", type=int, default=0)
    parser.add_argument("--self_calibration_step", type=int, default=100)
    parser.add_argument("--c2f_enable", type=int, default=0)
    parser.add_argument("--c2f_stage_steps", nargs="+", type=int, default=[0, 200, 400, 600])
    parser.add_argument("--c2f_stage_resolutions", nargs="+", type=int, default=[128, 256, 512, 512])
    parser.add_argument("--test_self_calibration", action="store_true", default=False)
    parser.add_argument("--self_calibration", type=int, default=0)
    parser.add_argument("--c2f", action="store_true", default=False)
    parser.add_argument("--activate_diffusion", action="store_true", default=False)
    parser.add_argument("--final_eval_enable", type=int, default=1)

    # Dataset / geometry matched to ucdavis_dx0.33
    parser.add_argument("--grid_x", type=int, default=512)
    parser.add_argument("--grid_y", type=int, default=512)
    parser.add_argument("--layers", type=int, default=14)
    parser.add_argument("--dx", type=float, default=0.33)
    parser.add_argument("--dy", type=float, default=0.33)
    parser.add_argument("--dz", type=float, default=1.5)
    parser.add_argument("--fs", type=float, default=50.0)
    parser.add_argument("--n_measure", type=int, default=1500)
    parser.add_argument("--radius", type=int, default=2400)

    # Physical model
    parser.add_argument("--wavelength", type=float, default=0.6)
    parser.add_argument("--NA", type=float, default=0.65)
    parser.add_argument("--n_b", type=float, default=1.33)
    parser.add_argument("--max_ri", type=float, default=0.03)
    parser.add_argument("--factor", type=float, default=1.0)
    parser.add_argument("--H", type=float, default=168.96)
    parser.add_argument("--W", type=float, default=168.96)
    parser.add_argument("--b2b", type=float, default=0.0)
    parser.add_argument("--b2c", type=float, default=0.0)

    # Representation
    parser.add_argument("--model", default="exp")
    parser.add_argument("--init_block", type=int, default=128)
    parser.add_argument("--feature_dim", type=int, default=24)
    parser.add_argument("--patch_ratio", type=float, default=1.0)
    parser.add_argument("--relu_slope", type=float, default=0.2)
    parser.add_argument("--mlp_kernel_size", type=int, default=32)
    parser.add_argument("--mlp_layer_num", type=int, default=6)
    parser.add_argument("--xy_encoding_num", type=int, default=3)
    parser.add_argument("--z_encoding_num", type=int, default=3)
    parser.add_argument("--dia_digree", type=int, default=10)
    parser.add_argument("--zz", type=float, default=0.0)

    # Loss / regularization
    parser.add_argument("--loss", default="l12")
    parser.add_argument("--regularize_type", default="")
    parser.add_argument("--tv_xy", type=float, default=0.0)
    parser.add_argument("--tv_z", type=float, default=0.0)
    parser.add_argument("--DnCNN_normalization_min", type=float, default=0.0)
    parser.add_argument("--DnCNN_normalization_max", type=float, default=1.0)

    # Diffusion placeholders
    parser.add_argument("--inch", type=int, default=1)
    parser.add_argument("--modch", type=int, default=64)
    parser.add_argument("--outch", type=int, default=1)
    parser.add_argument("--chmul", nargs="+", type=int, default=[1, 2, 2, 4])
    parser.add_argument("--numres", type=int, default=2)
    parser.add_argument("--cdim", type=int, default=64)
    parser.add_argument("--useconv", action="store_true", default=True)
    parser.add_argument("--droprate", type=float, default=0.0)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--w", type=float, default=0.0)
    parser.add_argument("--v", type=float, default=0.0)
    parser.add_argument("--T", type=int, default=10)

    return parser
