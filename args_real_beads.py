def config_parser():
    import argparse
    import torch
    parser = argparse.ArgumentParser()

    ## for current server (savio)
    parser.add_argument("--show_img", type=int, default=0, help="for ssh")
    parser.add_argument("--basedir", type=str, default='/global/scratch/users/cubhe/FDT/log/', required=False, help='where to store ckpts and logs')
    parser.add_argument("--tbdir", type=str, default='/global/scratch/users/cubhe/FDT/log/tensorboard/', required=False, help="tensorboard log directory")
    parser.add_argument("--dataset_path", type=str, default='/global/scratch/users/cubhe/FDT/dataset/', required=False, help="beads_real2 MDCK75v3 kindey100 kindey75 neuron14 neuron20 onion1 beads_multi2 neuron0412 neuron0422 musc50")
    parser.add_argument("--data_name", type=str, default='beads_real/beads13a22', required=False, help="bs101  beads_simu beads_new neuron0412 onion1 neuron0422")

    parser.add_argument("--render", type=int, default=0, help="render only")
    parser.add_argument("--object_category_ori", type=str,default='auto', required=False,
                        help='the object category in the data set used this time')

    # dataset
    parser.add_argument('--simulation', type=bool, default=False, help='is it simulation?')
    parser.add_argument("--camera_dataset_type", default="ucdavis")
    parser.add_argument('--c2f', type=bool, default=0, help='is it c2f?')
    parser.add_argument('--sub_num', type=int, default=600, help='subimage number')
    parser.add_argument("--txt", default="")

    parser.add_argument('--dz', type=float, default=1.5, help='dz value')
    parser.add_argument('--layers', type=int, default=20, help='Number of layers')
    parser.add_argument("--max_ri", type=float, default=0.05, help="ratio of patch points")
    parser.add_argument("--fs", type=int, default=190, help="free space")
    parser.add_argument("--b2b", type=int, default=0, help="back_to_back")
    parser.add_argument("--b2c", type=int, default=0, help="back_to_center")
    parser.add_argument("--radius", type=int, default=500, required=False, help="tensorboard log directory")
    parser.add_argument('--grid_x', type=float, default=1024, help='dy value')
    parser.add_argument('--grid_y', type=float, default=1024, help='dy value')
    parser.add_argument('--dx', type=float, default=0.33, help='dx the bigger the smaller')
    parser.add_argument('--dy', type=float, default=0.33, help='not used dy value')

    parser.add_argument('--add_noise', type=int, default=0.0, help='add noise')

    ####hyper parameters
    parser.add_argument("--test_self_calibration", type=int, default=0, help="None")
    parser.add_argument("--N_iters", type=int, default=350,help='number of iteration')
    parser.add_argument("--batch", type=int, default=20, help="number of generated images")
    parser.add_argument("--patch_ratio", type=float, default=0.7, help="ratio of ptch points")
    parser.add_argument("--loss", type=str, default='l12', help="ratio of ptch points")
    parser.add_argument("--norm_mode", type=str, default='std_minmax', help="normalization: mean_std, minmax, std_minmax")

    parser.add_argument("--self_calibration", type=int, default=0, help="ratio of patch points")
    parser.add_argument("--back_to_center_ratio", type=float, default=0.2, help="not used")
    parser.add_argument("--tv_xy", type=float, default=1e-5, help="tv_xy value")
    parser.add_argument("--tv_z", type=float, default=1e-5, help="tv_z value")
    parser.add_argument("--model", type=str, default='exp', help="nvp, tri, exp")

    ####brief
    parser.add_argument('--illu_x', type=int, default=20, help='Wavelength value')
    parser.add_argument('--illu_y', type=int, default=20, help='Wavelength value')
    parser.add_argument('--wavelength', type=float, default=0.6, help='Wavelength value')
    parser.add_argument('--NA', type=float, default=1.0, help='Numerical aperture')
    parser.add_argument('--n_measure', type=int, default=600, help='number of measurements (images)')
    parser.add_argument('--n_b', type=float, default=1.43, help='Base index')
    parser.add_argument('--factor', type=float, default=1.0, help='Factor value')
    parser.add_argument('--W', type=int, default=64, help='Width value')#64-->320 128-->640 160-->800 256-->1280
    parser.add_argument('--H', type=int, default=64, help='Height value')

    #nerf
    parser.add_argument('--feature_dim', type=int, default=64, help='input dimension')
    parser.add_argument('--dia_digree', type=int, default=60, help='input dimension')
    parser.add_argument('--xy_encoding_num', type=int, default=3, help='input dimension')
    parser.add_argument('--z_encoding_num', type=int, default=3, help='input dimension')
    parser.add_argument('--mlp_layer_num', type=int, default=3, help='input dimension')
    parser.add_argument('--mlp_kernel_size', type=int, default=16, help='input dimension')
    parser.add_argument("--init_block", type=int, default=128)
    parser.add_argument("--zz", type=float, default=0.0)

    ###########diffusion########################
    #activate diffusion
    parser.add_argument("--activate_diffusion", type=int, default=0,help='open diffusion')
    parser.add_argument("--lrate", type=float, default=2e-4,help='learning rate')
    parser.add_argument("--position_lrate", type=float, default=1e-3, help='position learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250)
    parser.add_argument('--inch',type=int,default=1, help='input channels for Unet model')
    parser.add_argument('--modch',type=int,default=64,help='model channels for Unet model')
    parser.add_argument('--outch',type=int,default=1,help='output channels for Unet model')
    parser.add_argument('--chmul',type=list,default=[1,2,2,2],help='architecture parameters training Unet model')
    parser.add_argument('--numres',type=int,default=2,help='number of resblocks for each block in Unet model')
    parser.add_argument('--cdim',type=int,default=32,help='dimension of conditional embedding')
    parser.add_argument('--useconv',type=bool,default=True,help='whether use convlution in downsample')
    parser.add_argument('--droprate',type=float,default=0.1,help='dropout rate for model')
    parser.add_argument('--dtype',default=torch.float32)
    parser.add_argument('--w',type=float,default=1.8,help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--v',type=float,default=0.3,help='hyperparameters for the variance of posterior distribution')
    parser.add_argument("--relu_slope", type=float, default=0.2,help='learning rate')
    parser.add_argument("--num_gpu", type=int, default=1,help='number of iteration')
    parser.add_argument("--no_reload", type=bool, default=False,help='frequency of tensorboard image logging')

    #### noise / calibration options (needed by run_nerf.py) ####
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
    parser.add_argument("--lr_stage_steps", nargs="+", type=int, default=[150, 500, 750])
    parser.add_argument("--lr_stage_values", nargs="+", type=float, default=[5e-3, 5e-3, 5e-3, 5e-3])
    parser.add_argument("--final_eval_enable", type=int, default=1)
    parser.add_argument("--i_save_override", type=int, default=0)
    parser.add_argument("--i_weights_override", type=int, default=0)
    parser.add_argument("--regularize_type", default="")
    parser.add_argument("--DnCNN_normalization_min", type=float, default=0.0)
    parser.add_argument("--DnCNN_normalization_max", type=float, default=1.0)
    parser.add_argument("--T", type=int, default=10)

    ################# logging/saving options ##################
    parser.add_argument("--i_tensorboard", type=int, default=2,help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=20,help='frequency of weight ckpt saving')
    parser.add_argument("--i_save", type=int, default=20,help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=200,help='frequency of weight ckpt saving')
    parser.add_argument("--i_show_img", type=int, default=True,help='frequency of weight ckpt saving')

    return parser
