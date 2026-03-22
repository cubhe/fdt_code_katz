import argparse
import json
import os

import cv2
import numpy as np
try:
    import tifffile as tiff
except ImportError:
    tiff = None


def auto_align_layer(bead_diameter_um, dz, margin_radius_count=1):
    """Choose an odd layer count so the bead center aligns to a slice with z margin."""
    radius_voxels = int(np.ceil((bead_diameter_um * 0.5) / dz))
    occupied_layers = radius_voxels * 2 + 1
    margin_layers = radius_voxels * int(margin_radius_count) * 2
    layer = occupied_layers + margin_layers
    if layer % 2 == 0:
        layer += 1
    return max(layer, 3)


def generate_single_center_bead(
    shape_x=512,
    shape_y=512,
    layer=50,
    bead_diameter_um=1.0,
    dx=0.1,
    dy=0.1,
    dz=0.1,
):
    """Generate one bead centered in the volume using voxel-center rasterization."""
    radius_um = bead_diameter_um / 2.0
    center_x = shape_x // 2
    center_y = shape_y // 2
    center_z = layer // 2

    stack = np.zeros((shape_x, shape_y, layer), dtype=np.uint8)
    xx, yy, zz = np.ogrid[:shape_x, :shape_y, :layer]
    dist2_um = ((xx - center_x) * dx) ** 2 + ((yy - center_y) * dy) ** 2 + ((zz - center_z) * dz) ** 2
    stack[dist2_um <= radius_um**2] = 255

    bead_info = {
        "center_xyz_pixel": [float(center_x), float(center_y), float(center_z)],
        "volume_center_xyz_pixel": [float(center_x), float(center_y), float(center_z)],
    }
    return stack, bead_info


def generate_single_center_gaussian(
    shape_x=512,
    shape_y=512,
    layer=21,
    sigma_um=0.1,
    dx=0.1,
    dy=0.1,
    dz=0.1,
    peak_value=255.0,
):
    """Generate one centered 3D Gaussian target."""
    center_x = shape_x // 2
    center_y = shape_y // 2
    center_z = layer // 2

    xx, yy, zz = np.ogrid[:shape_x, :shape_y, :layer]
    dist2_um = ((xx - center_x) * dx) ** 2 + ((yy - center_y) * dy) ** 2 + ((zz - center_z) * dz) ** 2
    stack = np.exp(-dist2_um / (2.0 * sigma_um**2)).astype(np.float32) * float(peak_value)

    bead_info = {
        "center_xyz_pixel": [float(center_x), float(center_y), float(center_z)],
        "volume_center_xyz_pixel": [float(center_x), float(center_y), float(center_z)],
        "sigma_um": float(sigma_um),
    }
    return stack, bead_info


def generate_light_positions(num_positions=900):
    light_loc = np.zeros((num_positions, 3), dtype=np.float32)

    grid_size = int(np.sqrt(num_positions))
    x_values = np.linspace(0.25, 0.75, grid_size)
    y_values = np.linspace(0.25, 0.75, grid_size)
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    light_loc[:, 0] = x_grid.ravel()[:num_positions] - 0.5
    light_loc[:, 1] = y_grid.ravel()[:num_positions] - 0.5
    light_loc[:, 2] = 0.0
    return light_loc


def create_bead_video(beads, video_path, fps=5, resize_factor=0.5):
    """Create layer-by-layer preview video."""
    height, width, num_layers = beads.shape
    display_height = int(height * resize_factor)
    display_width = int(width * resize_factor)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (display_width, display_height))

    print(f"Creating single-bead video with {num_layers} layers...")

    for z in range(num_layers):
        img_slice = beads[:, :, z]
        img_min = np.min(img_slice)
        img_max = np.max(img_slice)
        img_mean = np.mean(img_slice)
        img_nonzero = np.count_nonzero(img_slice)

        if img_max > img_min:
            img_norm = ((img_slice - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_norm = img_slice.astype(np.uint8)

        img_resized = cv2.resize(img_norm, (display_width, display_height))
        img_color = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 0)
        thickness = 2
        y_offset = 25

        cv2.putText(img_color, f"Z-Layer: {z:02d}/{num_layers - 1:02d}", (10, y_offset), font, font_scale, color, thickness)
        cv2.putText(img_color, f"Min: {img_min:.0f}", (10, y_offset + 25), font, font_scale, color, thickness)
        cv2.putText(img_color, f"Max: {img_max:.0f}", (10, y_offset + 50), font, font_scale, color, thickness)
        cv2.putText(img_color, f"Mean: {img_mean:.1f}", (10, y_offset + 75), font, font_scale, color, thickness)
        cv2.putText(img_color, f"NonZero: {img_nonzero}", (10, y_offset + 100), font, font_scale, color, thickness)

        if img_nonzero > 0:
            cv2.putText(img_color, "SINGLE BEAD", (10, y_offset + 125), font, font_scale, (0, 255, 255), thickness)

        video_writer.write(img_color)
        if (z + 1) % 5 == 0:
            print(f"Processed layer {z + 1}/{num_layers}")

    video_writer.release()
    print(f"Single-bead video saved to: {video_path}")


def create_bead_comparison_video(beads, video_path, fps=3):
    """Create XY/XZ/YZ comparison video."""
    height, width, num_layers = beads.shape
    mip_xz = np.max(beads, axis=0)
    mip_yz = np.max(beads, axis=1)

    output_width = width + num_layers + 50
    output_height = height + width + 50

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (output_width, output_height))

    print(f"Creating single-bead comparison video with {num_layers} layers...")

    for z in range(num_layers):
        canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)

        xy_slice = beads[:, :, z]
        xy_norm = ((xy_slice / 255.0) * 255).astype(np.uint8)
        xy_color = cv2.cvtColor(xy_norm, cv2.COLOR_GRAY2BGR)

        center_x, center_y = width // 2, height // 2
        cv2.line(xy_color, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 255), 2)
        cv2.line(xy_color, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 255), 2)

        xz_norm = ((mip_xz / 255.0) * 255).astype(np.uint8)
        xz_color = cv2.cvtColor(xz_norm, cv2.COLOR_GRAY2BGR)

        yz_norm = ((mip_yz / 255.0) * 255).astype(np.uint8)
        yz_color = cv2.cvtColor(yz_norm, cv2.COLOR_GRAY2BGR)

        if z < xz_color.shape[1]:
            cv2.line(xz_color, (z, 0), (z, xz_color.shape[0] - 1), (0, 0, 255), 2)

        canvas[0:height, 0:width] = xy_color
        canvas[0:xz_color.shape[0], width + 25 : width + 25 + xz_color.shape[1]] = xz_color
        canvas[height + 25 : height + 25 + yz_color.shape[0], 0 : yz_color.shape[1]] = yz_color

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (255, 255, 255)
        thickness = 2

        cv2.putText(canvas, f"XY Slice (Z={z})", (10, 20), font, font_scale, color, thickness)
        cv2.putText(canvas, "XZ Projection", (width + 30, 20), font, font_scale, color, thickness)
        cv2.putText(canvas, "YZ Projection", (10, height + 40), font, font_scale, color, thickness)
        cv2.putText(canvas, f"Z-Layer: {z:02d}/{num_layers - 1:02d}", (width + 30, height + 40), font, font_scale, (0, 255, 0), thickness)
        cv2.putText(canvas, "Single Bead", (width + 30, height + 70), font, 0.5, (0, 255, 255), 1)

        video_writer.write(canvas)
        if (z + 1) % 5 == 0:
            print(f"Processed comparison layer {z + 1}/{num_layers}")

    video_writer.release()
    print(f"Single-bead comparison video saved to: {video_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate a single-bead RI dataset.")
    parser.add_argument("--target-type", choices=["bead", "gaussian"], default="bead")
    parser.add_argument("--dx", type=float, default=0.1)
    parser.add_argument("--dy", type=float, default=0.1)
    parser.add_argument("--dz", type=float, default=0.1)
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--layer", type=int, default=0, help="If <= 0, auto align layer to bead diameter and dz.")
    parser.add_argument("--bead-diameter-um", type=float, default=1.0)
    parser.add_argument("--gaussian-sigma-um", type=float, default=0.1)
    parser.add_argument("--num-light-positions", type=int, default=900)
    parser.add_argument("--save-root", default="dataset")
    args = parser.parse_args()

    dx = args.dx
    dy = args.dy
    dz = args.dz
    grid_size = args.grid_size
    bead_diameter_um = args.bead_diameter_um
    if args.layer > 0:
        layer = args.layer
    elif args.target_type == "gaussian":
        layer = 21
    else:
        layer = auto_align_layer(bead_diameter_um, dz)

    if args.target_type == "gaussian":
        beads, bead_info = generate_single_center_gaussian(
            shape_x=grid_size,
            shape_y=grid_size,
            layer=layer,
            sigma_um=args.gaussian_sigma_um,
            dx=dx,
            dy=dy,
            dz=dz,
        )
    else:
        beads, bead_info = generate_single_center_bead(
            shape_x=grid_size,
            shape_y=grid_size,
            layer=layer,
            bead_diameter_um=bead_diameter_um,
            dx=dx,
            dy=dy,
            dz=dz,
        )

    volume_x_um = grid_size * dx
    volume_y_um = grid_size * dy
    volume_z_um = layer * dz

    print(f"beads shape: {beads.shape}")
    print(f"nonzero voxels: {np.count_nonzero(beads)}")
    print(f"dx, dy, dz = {dx}, {dy}, {dz} um")
    if args.target_type == "gaussian":
        print(f"gaussian sigma = {args.gaussian_sigma_um} um")
    else:
        print(f"bead diameter = {bead_diameter_um} um")
    print(f"volume size (um): {volume_x_um} x {volume_y_um} x {volume_z_um}")
    print(f"center: {bead_info}")

    if args.target_type == "gaussian":
        name = f"single_gaussian_sigma{args.gaussian_sigma_um:.1f}um_dxyz{dx:.1f}_layer{layer}"
    else:
        name = f"single_bead_d{bead_diameter_um:.1f}um_dxyz{dx:.1f}_layer{layer}"
    save_dir = os.path.join(args.save_root, name)
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "RI_gt.npy"), beads)
    np.save(os.path.join(save_dir, "dxyz.npy"), np.array([dx, dy, dz], dtype=np.float32))
    if tiff is not None:
        tiff.imwrite(
            os.path.join(save_dir, f"{name}.tif"),
            beads.transpose(2, 0, 1).astype(np.uint8),
        )
    else:
        print("tifffile is not installed, skip tif export.")

    light_loc = generate_light_positions(num_positions=args.num_light_positions)
    np.save(os.path.join(save_dir, "new_location1024org.npy"), light_loc)

    params_info = {
        "dx_um": dx,
        "dy_um": dy,
        "dz_um": dz,
        "dxyz_um": [dx, dy, dz],
        "target_type": args.target_type,
        "bead_diameter_um": bead_diameter_um if args.target_type == "bead" else None,
        "bead_radius_um": bead_diameter_um / 2.0 if args.target_type == "bead" else None,
        "gaussian_sigma_um": args.gaussian_sigma_um if args.target_type == "gaussian" else None,
        "layer": layer,
        "grid_size_xyz": [grid_size, grid_size, layer],
        "volume_size_um": [volume_x_um, volume_y_um, volume_z_um],
        "bead_arrangement": f"single_center_{args.target_type}",
        "centers": bead_info,
    }
    with open(os.path.join(save_dir, "parameters.json"), "w", encoding="utf-8") as f:
        json.dump(params_info, f, indent=2)

    print("Generating single-bead layer video...")
    video_path = os.path.join(save_dir, f"{name}_layers.mp4")
    create_bead_video(beads, video_path, fps=5, resize_factor=0.5)

    print("Generating single-bead comparison video...")
    comparison_video_path = os.path.join(save_dir, f"{name}_comparison.mp4")
    create_bead_comparison_video(beads, comparison_video_path, fps=3)

    print(f"saved to: {save_dir}")


if __name__ == "__main__":
    main()
