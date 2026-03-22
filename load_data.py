import os

import cv2
import numpy as np


def load_phase_data(data_path, calib=0):
    del calib
    image_path = os.path.join(data_path, "new_img1024org.npy")
    light_path = os.path.join(data_path, "new_location1024org.npy")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Rendered image stack not found: {image_path}")
    if not os.path.exists(light_path):
        raise FileNotFoundError(f"Light position file not found: {light_path}")

    images = np.load(image_path).astype(np.float32)
    light_loc = np.load(light_path).astype(np.float32)
    return images, light_loc


def process_traning_data_simu(images, light_loc_gt, shuffle_idx, i_batch, batch):
    start = i_batch * batch
    end = min(start + batch, len(shuffle_idx))
    light_loc_ids = shuffle_idx[start:end]
    light_loc_training = light_loc_gt[light_loc_ids]
    intensity = images[light_loc_ids]
    return light_loc_training, intensity, light_loc_ids


def video_generate(volume, folder_path, data_type="img", fps=8):
    os.makedirs(folder_path, exist_ok=True)
    output_path = os.path.join(folder_path, f"{data_type}.mp4")

    if volume.ndim != 3:
        return output_path

    height, width = volume.shape[1], volume.shape[2]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for frame in volume:
        frame = frame.astype(np.float32)
        frame_min = float(np.min(frame))
        frame_max = float(np.max(frame))
        if frame_max > frame_min:
            frame = (frame - frame_min) / (frame_max - frame_min)
        else:
            frame = np.zeros_like(frame)
        frame_u8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR))

    writer.release()
    return output_path
