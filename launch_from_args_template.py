import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import run_nerf


def parse_value(text: str):
    text = text.strip()
    try:
        return ast.literal_eval(text)
    except Exception:
        if text == "True":
            return True
        if text == "False":
            return False
        return text


def load_args_txt(path: Path):
    args_dict = {}
    with path.open() as f:
        for line in f:
            if " = " not in line:
                continue
            key, value = line.rstrip("\n").split(" = ", 1)
            args_dict[key] = parse_value(value)
    return args_dict


def main():
    parser = argparse.ArgumentParser(description="Launch FDT run from an existing args.txt template.")
    parser.add_argument("--template", required=True)
    parser.add_argument("--fs", type=float, required=True)
    parser.add_argument("--data-name", required=True)
    parser.add_argument("--expname", required=True)
    parser.add_argument("--n-iters", type=int, default=None)
    parser.add_argument("--render", type=int, default=None)
    parser.add_argument("--num-gpu", type=int, default=1)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override template args with key=value. May be provided multiple times.",
    )
    args_cli = parser.parse_args()

    args_dict = load_args_txt(Path(args_cli.template))
    args_dict.update(
        {
            "fs": args_cli.fs,
            "data_name": args_cli.data_name,
            "object_category_ori": args_cli.expname,
            "tbdir": f"/global/scratch/users/cubhe/FDT/log/{args_cli.expname}/tensorboard",
            "num_gpu": args_cli.num_gpu,
            "basedir": "/global/scratch/users/cubhe/FDT/log",
            "dataset_path": "/global/scratch/users/cubhe/FDT/dataset/",
        }
    )
    if args_cli.n_iters is not None:
        args_dict["N_iters"] = args_cli.n_iters
    if args_cli.render is not None:
        args_dict["render"] = args_cli.render
    for override in args_cli.overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}. Expected key=value.")
        key, value = override.split("=", 1)
        args_dict[key] = parse_value(value)

    args = SimpleNamespace(**args_dict)

    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    seed = 1121
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    run_nerf.apply_training_policy(args)
    if args.object_category_ori == "auto":
        args.object_category_ori = run_nerf.build_experiment_name(args)

    if getattr(args, "render", 0):
        run_nerf.render(args)
    run_nerf.train(args)


if __name__ == "__main__":
    main()
