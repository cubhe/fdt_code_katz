import argparse
import json
import os

import cv2
import numpy as np
import tifffile as tiff


def normalize_to_uint8(img, vmin, vmax):
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.uint8)
    img_clip = np.clip(img, vmin, vmax)
    img_norm = (img_clip - vmin) / (vmax - vmin)
    return (img_norm * 255.0).astype(np.uint8)


def generate_light_positions(num_positions=1500, grid_size=40):
    """Generate uniformly distributed light source positions in normalized coordinates."""
    light_loc = np.zeros((num_positions, 3), dtype=np.float32)

    x_values = np.linspace(0.25, 0.75, grid_size)
    y_values = np.linspace(0.25, 0.75, grid_size)
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    light_loc[:, 0] = x_grid.ravel()[:num_positions] - 0.5
    light_loc[:, 1] = y_grid.ravel()[:num_positions] - 0.5
    light_loc[:, 2] = 0.0
    return light_loc


def generate_ucdavis_volume(
    shape_x=512,
    shape_y=512,
    layer=14,
    letters="UCDavis",
    layer_step=2,
    font_scale=10,
    thickness=10,
    target_width=100,
    ri_base=1.33,
    ri_delta=0.03,
):
    """Generate the same UCDavis letter stack as the original script."""
    image_stack = np.zeros((shape_x, shape_y, layer), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    saved_letter_slices = []
    for letter_index, letter in enumerate(letters):
        z_idx = layer_step * letter_index
        if z_idx >= layer:
            break

        img = np.zeros((shape_x, shape_y, 3), dtype=np.uint8)
        text_size = cv2.getTextSize(letter, font, font_scale, thickness)[0]
        font_scale_adjusted = font_scale * target_width / text_size[1]
        text_size_adjusted = cv2.getTextSize(letter, font, font_scale_adjusted, thickness)[0]
        text_x = (img.shape[1] - text_size_adjusted[0]) // 2
        text_y = (img.shape[0] + text_size_adjusted[1]) // 2

        cv2.putText(
            img,
            letter,
            (text_x, text_y),
            font,
            font_scale_adjusted,
            (255, 255, 255),
            thickness,
        )

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_stack[:, :, z_idx] = img_gray
        saved_letter_slices.append(z_idx)

    image_stack = image_stack.astype(np.float32)
    image_stack = image_stack / np.max(image_stack) * ri_delta + ri_base
    return image_stack, saved_letter_slices


def create_ucdavis_video(volume, video_path, fps=3, resize_factor=0.5, ri_base=1.33, ri_delta=0.03):
    """Create a layer-by-layer preview video for the generated UCDavis volume."""
    height, width, num_layers = volume.shape
    display_height = int(height * resize_factor)
    display_width = int(width * resize_factor)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (display_width, display_height))

    print(f"Creating layer video with {num_layers} slices...")
    for z in range(num_layers):
        img_slice = volume[:, :, z]
        img_u8 = normalize_to_uint8(img_slice, vmin=ri_base, vmax=ri_base + ri_delta)
        img_resized = cv2.resize(img_u8, (display_width, display_height), interpolation=cv2.INTER_NEAREST)
        img_color = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

        img_min = float(np.min(img_slice))
        img_max = float(np.max(img_slice))
        img_mean = float(np.mean(img_slice))
        img_nonzero = int(np.count_nonzero(img_slice > ri_base))

        cv2.putText(
            img_color,
            f"Z-Layer: {z:02d}/{num_layers - 1:02d}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img_color,
            f"Min/Max: {img_min:.3f}/{img_max:.3f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img_color,
            f"Mean: {img_mean:.4f}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img_color,
            f"Foreground: {img_nonzero}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img_color,
            "UCDAVIS",
            (10, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        writer.write(img_color)

    writer.release()
    print(f"Layer video saved to: {video_path}")


def save_ucdavis_dataset(
    save_root="dataset",
    shape_x=512,
    shape_y=512,
    layer=14,
    dx=0.33,
    dy=0.33,
    dz=1.5,
    letters="UCDavis",
    num_light_positions=1500,
    light_grid_size=40,
):
    volume, saved_letter_slices = generate_ucdavis_volume(
        shape_x=shape_x,
        shape_y=shape_y,
        layer=layer,
        letters=letters,
    )

    name = f"ucdavis_dx{dx}"
    save_dir = os.path.join(save_root, name)
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "RI_gt.npy"), volume)
    np.save(os.path.join(save_dir, "ucdavis.npy"), volume)
    tiff.imwrite(
        os.path.join(save_dir, f"{name}.tif"),
        volume.transpose(2, 0, 1).astype(np.float32),
    )

    light_loc = generate_light_positions(
        num_positions=num_light_positions,
        grid_size=light_grid_size,
    )
    np.save(os.path.join(save_dir, "new_location1024org.npy"), light_loc)

    layer_video_path = os.path.join(save_dir, f"{name}_layers.mp4")
    create_ucdavis_video(volume, layer_video_path, fps=3, resize_factor=0.5)

    params_info = {
        "shape_xyz": [shape_x, shape_y, layer],
        "dx_um": dx,
        "dy_um": dy,
        "dz_um": dz,
        "letters": list(letters),
        "letter_slices": saved_letter_slices,
        "ri_base": 1.33,
        "ri_delta": 0.03,
        "num_light_positions": int(light_loc.shape[0]),
        "light_grid_size": int(light_grid_size),
    }
    with open(os.path.join(save_dir, "parameters.json"), "w", encoding="utf-8") as f:
        json.dump(params_info, f, indent=2)

    print(f"volume shape: {volume.shape}")
    print(f"volume dtype: {volume.dtype}")
    print(f"volume min/max: {float(np.min(volume)):.6f}/{float(np.max(volume)):.6f}")
    print(f"letter slices: {saved_letter_slices}")
    print(f"light positions shape: {light_loc.shape}")
    print(f"layer video: {layer_video_path}")
    print(f"saved to: {save_dir}")
    return volume, save_dir


def main():
    parser = argparse.ArgumentParser(description="Generate the UCDavis RI dataset.")
    parser.add_argument("--shape-x", type=int, default=512)
    parser.add_argument("--shape-y", type=int, default=512)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--dx", type=float, default=0.33)
    parser.add_argument("--dy", type=float, default=0.33)
    parser.add_argument("--dz", type=float, default=1.5)
    parser.add_argument("--letters", default="UCDavis")
    parser.add_argument("--save-root", default="dataset")
    parser.add_argument("--num-light-positions", type=int, default=1500)
    parser.add_argument("--light-grid-size", type=int, default=40)
    args = parser.parse_args()

    save_ucdavis_dataset(
        save_root=args.save_root,
        shape_x=args.shape_x,
        shape_y=args.shape_y,
        layer=args.layer,
        dx=args.dx,
        dy=args.dy,
        dz=args.dz,
        letters=args.letters,
        num_light_positions=args.num_light_positions,
        light_grid_size=args.light_grid_size,
    )


if __name__ == "__main__":
    main()
