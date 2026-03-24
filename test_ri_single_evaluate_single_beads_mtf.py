import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
try:
    import tifffile
except ImportError:
    tifffile = None

matplotlib.use("Agg")
from matplotlib import pyplot as plt


MTF_LOG_FLOOR = 1e-4


def orient_volume_to_yxz(volume: np.ndarray) -> tuple[np.ndarray, str]:
    """Return a volume ordered as (y, x, z)."""
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape={volume.shape}")

    s0, s1, s2 = volume.shape
    if s0 <= s1 and s0 <= s2:
        return volume.transpose(1, 2, 0), "zyx_to_yxz"
    if s2 <= s0 and s2 <= s1:
        return volume, "yxz"
    return volume, "assume_yxz"


def build_psf_like_volume(volume: np.ndarray) -> tuple[np.ndarray, float, str]:
    """
    Convert raw RI-like volume to positive PSF-like response.
    Background is removed with median; polarity is auto-detected.
    """
    background = float(np.median(volume))
    signal = volume - background

    if abs(float(np.min(signal))) > abs(float(np.max(signal))):
        signal = -signal
        polarity = "inverted_negative_peak_to_positive"
    else:
        polarity = "positive_peak"

    psf_like = np.clip(signal, 0.0, None)
    return psf_like, background, polarity


def crop_around_center(
    volume: np.ndarray,
    center_yxz: tuple[int, int, int],
    half_xy: int,
    half_z: int,
) -> tuple[np.ndarray, dict]:
    cy, cx, cz = center_yxz
    ny, nx, nz = volume.shape

    y0 = max(cy - half_xy, 0)
    y1 = min(cy + half_xy + 1, ny)
    x0 = max(cx - half_xy, 0)
    x1 = min(cx + half_xy + 1, nx)
    z0 = max(cz - half_z, 0)
    z1 = min(cz + half_z + 1, nz)

    crop = volume[y0:y1, x0:x1, z0:z1]
    bounds = {"y0": int(y0), "y1": int(y1), "x0": int(x0), "x1": int(x1), "z0": int(z0), "z1": int(z1)}
    return crop, bounds


def hann_window_3d(shape: tuple[int, int, int]) -> np.ndarray:
    wy = np.hanning(shape[0]) if shape[0] > 1 else np.ones(1, dtype=np.float32)
    wx = np.hanning(shape[1]) if shape[1] > 1 else np.ones(1, dtype=np.float32)
    wz = np.hanning(shape[2]) if shape[2] > 1 else np.ones(1, dtype=np.float32)
    return wy[:, None, None] * wx[None, :, None] * wz[None, None, :]


def radial_average_xy(
    mtf_xy: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    kx, ky = np.meshgrid(fx, fy, indexing="xy")
    kr = np.sqrt(kx**2 + ky**2)
    kr_flat = kr.ravel()
    val_flat = mtf_xy.ravel()

    r_max = float(min(np.max(np.abs(fx)), np.max(np.abs(fy))))
    edges = np.linspace(0.0, r_max, int(bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    idx = np.digitize(kr_flat, edges) - 1
    valid = (idx >= 0) & (idx < bins)
    idx_valid = idx[valid]
    val_valid = val_flat[valid]

    sums = np.bincount(idx_valid, weights=val_valid, minlength=bins)
    counts = np.bincount(idx_valid, minlength=bins)
    radial = sums / np.maximum(counts, 1)

    keep = counts > 0
    return centers[keep], radial[keep]


def crop_mtf_section_edges(
    data: np.ndarray,
    axis0: np.ndarray,
    axis1: np.ndarray,
    margin_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Crop the outer margin from a 2D MTF section and its matching axes."""
    if not 0.0 <= margin_fraction < 0.5:
        raise ValueError(f"margin_fraction must be in [0, 0.5), got {margin_fraction}")

    n0, n1 = data.shape
    trim0 = int(np.floor(n0 * margin_fraction))
    trim1 = int(np.floor(n1 * margin_fraction))

    if trim0 * 2 >= n0 or trim1 * 2 >= n1:
        raise ValueError(
            f"MTF crop is too aggressive for shape={data.shape}, margin_fraction={margin_fraction}"
        )

    cropped = data[trim0 : n0 - trim0, trim1 : n1 - trim1]
    axis0_cropped = axis0[trim0 : n0 - trim0]
    axis1_cropped = axis1[trim1 : n1 - trim1]
    crop_info = {
        "margin_fraction_per_side": float(margin_fraction),
        "total_fraction_removed": float(2.0 * margin_fraction),
        "trim_axis0_pixels_per_side": int(trim0),
        "trim_axis1_pixels_per_side": int(trim1),
        "cropped_shape": [int(v) for v in cropped.shape],
    }
    return cropped, axis0_cropped, axis1_cropped, crop_info


def cutoff_frequency(freq: np.ndarray, mtf: np.ndarray, target: float) -> float | None:
    if len(freq) == 0:
        return None
    if mtf[0] < target:
        return None

    for i in range(1, len(freq)):
        if mtf[i] <= target:
            x0, x1 = freq[i - 1], freq[i]
            y0, y1 = mtf[i - 1], mtf[i]
            if y1 == y0:
                return float(x1)
            return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))
    return None


def interpolate_crossing(x0: float, y0: float, x1: float, y1: float, target: float) -> float:
    if y1 == y0:
        return 0.5 * (x0 + x1)
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


def compute_fwhm_from_profile(profile: np.ndarray, spacing_um: float) -> dict | None:
    profile = np.asarray(profile, dtype=np.float64)
    if profile.ndim != 1 or profile.size == 0:
        return None

    peak_idx = int(np.argmax(profile))
    peak_value = float(profile[peak_idx])
    if peak_value <= 0:
        return None

    half_max = 0.5 * peak_value

    left = peak_idx
    while left > 0 and profile[left] >= half_max:
        left -= 1
    left_cross = (
        float(peak_idx)
        if left == peak_idx
        else interpolate_crossing(left, profile[left], left + 1, profile[left + 1], half_max)
    )

    right = peak_idx
    while right < profile.size - 1 and profile[right] >= half_max:
        right += 1
    right_cross = (
        float(peak_idx)
        if right == peak_idx
        else interpolate_crossing(right - 1, profile[right - 1], right, profile[right], half_max)
    )

    return {
        "peak_index": peak_idx,
        "peak_value": peak_value,
        "half_max_value": float(half_max),
        "left_cross_pixel": float(left_cross),
        "right_cross_pixel": float(right_cross),
        "fwhm_pixel": float(right_cross - left_cross),
        "fwhm_um": float((right_cross - left_cross) * spacing_um),
    }


def save_curve_csv(path: Path, x: np.ndarray, y: np.ndarray, x_name: str = "freq_cyc_per_um") -> None:
    arr = np.column_stack([x, y])
    header = f"{x_name},mtf"
    np.savetxt(path, arr, delimiter=",", header=header, comments="")


def save_line_plot(
    path: Path,
    freq: np.ndarray,
    mtf_line: np.ndarray,
    xlabel: str,
    color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(freq, np.maximum(mtf_line, MTF_LOG_FLOOR), linewidth=3, color=color)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yscale("log")
    ax.set_ylim(MTF_LOG_FLOOR, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def load_colormap0627_bgr() -> np.ndarray:
    palette = np.load("colormap0627.npy")
    if palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError(f"Unexpected colormap0627.npy shape: {palette.shape}")
    if float(np.max(palette)) <= 1.0:
        palette = palette * 255.0
    palette = np.clip(palette, 0, 255).astype(np.uint8)
    return palette[:, [2, 1, 0]]


def apply_palette_to_gray(gray_u8: np.ndarray, palette_bgr: np.ndarray) -> np.ndarray:
    palette_bgr = np.asarray(palette_bgr, dtype=np.uint8)
    if palette_bgr.ndim != 2 or palette_bgr.shape[1] != 3:
        raise ValueError(f"Unexpected palette shape: {palette_bgr.shape}")
    indices = np.round(gray_u8.astype(np.float32) * (len(palette_bgr) - 1) / 255.0).astype(np.int32)
    return palette_bgr[indices]


def save_heatmap_opencv(
    path: Path,
    data: np.ndarray,
    cv2_colormap: int = cv2.COLORMAP_TURBO,
    palette_bgr: np.ndarray | None = None,
) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    data = np.clip(data, 0.0, 1.0)

    scale = max(4, min(12, int(420 / max(data.shape))))
    heat_u8 = np.round(np.flipud(data) * 255.0).astype(np.uint8)
    if palette_bgr is None:
        heat_color = cv2.applyColorMap(heat_u8, cv2_colormap)
    else:
        heat_color = apply_palette_to_gray(heat_u8, palette_bgr)
    heat_color = cv2.resize(
        heat_color,
        (data.shape[1] * scale, data.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imwrite(str(path), heat_color)
    return heat_color


def save_section_opencv(path: Path, data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    data_min = float(np.min(data))
    data_max = float(np.max(data))
    if data_max > data_min:
        data = (data - data_min) / (data_max - data_min)
    else:
        data = np.zeros_like(data, dtype=np.float32)

    return save_heatmap_opencv(path, data, palette_bgr=load_colormap0627_bgr())


def save_colorbar_opencv(
    path: Path,
    height: int,
    width: int = 48,
    cv2_colormap: int = cv2.COLORMAP_TURBO,
    palette_bgr: np.ndarray | None = None,
) -> np.ndarray:
    colorbar_u8 = np.linspace(255, 0, height, dtype=np.uint8)[:, None]
    colorbar_u8 = np.repeat(colorbar_u8, width, axis=1)
    if palette_bgr is None:
        colorbar = cv2.applyColorMap(colorbar_u8, cv2_colormap)
    else:
        colorbar = apply_palette_to_gray(colorbar_u8, palette_bgr)
    cv2.imwrite(str(path), colorbar)
    return colorbar


def save_heatmap_plot(
    path: Path,
    data: np.ndarray,
    extent: list[float],
    cmap: str = "turbo",
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.imshow(
        data,
        cmap=cmap,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def log1p_unit_scale(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    data = np.clip(data, 0.0, None)
    return np.log1p(data) / np.log(2.0)


def evaluate_single_bead_mtf(args: argparse.Namespace) -> dict:
    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.suffix.lower() == ".npy":
        raw = np.load(in_path)
    else:
        if tifffile is None:
            raise RuntimeError("tifffile is required for TIFF input, but it is not installed.")
        raw = tifffile.imread(in_path)
    raw = np.asarray(raw, dtype=np.float32)
    volume, orientation = orient_volume_to_yxz(raw)

    psf_full, background, polarity = build_psf_like_volume(volume)

    if args.center is not None:
        center = (int(args.center[0]), int(args.center[1]), int(args.center[2]))
    else:
        center = tuple(int(v) for v in np.unravel_index(np.argmax(psf_full), psf_full.shape))

    psf_crop, bounds = crop_around_center(psf_full, center, args.half_xy, args.half_z)
    raw_crop, _ = crop_around_center(volume, center, args.half_xy, args.half_z)

    if np.max(psf_crop) <= 0:
        raise RuntimeError("PSF crop has no positive signal after preprocessing. Check center/polarity.")

    if args.no_window:
        psf_for_fft = psf_crop.copy()
        window_used = False
    else:
        psf_for_fft = psf_crop * hann_window_3d(psf_crop.shape)
        window_used = True

    total = float(np.sum(psf_for_fft))
    if total <= 0:
        raise RuntimeError("PSF crop sum is zero after windowing.")
    psf_for_fft /= total

    otf = np.fft.fftn(psf_for_fft)
    otf = np.fft.fftshift(otf)
    mtf = np.abs(otf).astype(np.float64)

    ny, nx, nz = mtf.shape
    cy, cx, cz = ny // 2, nx // 2, nz // 2
    dc = float(mtf[cy, cx, cz])
    if dc <= 0:
        raise RuntimeError("DC component is non-positive, cannot normalize MTF.")
    mtf /= dc

    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=args.dx))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=args.dy))
    fz = np.fft.fftshift(np.fft.fftfreq(nz, d=args.dz))

    mtf_xy_full = mtf[:, :, cz]
    mtf_xz_full = mtf[cy, :, :]
    mtf_xy, fy_cropped, fx_cropped, mtf_xy_crop_info = crop_mtf_section_edges(
        mtf_xy_full,
        fy,
        fx,
        args.mtf_crop_fraction,
    )
    mtf_xz, fx_xz_cropped, fz_cropped, mtf_xz_crop_info = crop_mtf_section_edges(
        mtf_xz_full,
        fx,
        fz,
        args.mtf_crop_fraction,
    )
    freq_radial, mtf_radial = radial_average_xy(mtf_xy, fx_cropped, fy_cropped, bins=args.bins)

    cy_xy, cx_xy = mtf_xy.shape[0] // 2, mtf_xy.shape[1] // 2
    cx_xz, cz_xz = mtf_xz.shape[0] // 2, mtf_xz.shape[1] // 2
    fx_pos = fx_cropped[cx_xy:]
    fz_pos = fz_cropped[cz_xz:]
    mtf_x_full = mtf_xy[cy_xy, :]
    mtf_z_full = mtf_xz[cx_xz, :]
    mtf_x = mtf_xy[cy_xy, cx_xy:]
    mtf_z = mtf_xz[cx_xz, cz_xz:]

    mtf50 = cutoff_frequency(freq_radial, mtf_radial, 0.5)
    mtf10 = cutoff_frequency(freq_radial, mtf_radial, 0.1)

    mtf_xy_log = log1p_unit_scale(mtf_xy)
    mtf_xz_log = log1p_unit_scale(mtf_xz)

    save_curve_csv(out_dir / "mtf_xy_radial.csv", freq_radial, mtf_radial)
    save_curve_csv(out_dir / "mtf_x_line.csv", fx_pos, mtf_x)
    save_curve_csv(out_dir / "mtf_z_line.csv", fz_pos, mtf_z)
    save_curve_csv(out_dir / "mtf_x_line_full.csv", fx_cropped, mtf_x_full)
    save_curve_csv(out_dir / "mtf_z_line_full.csv", fz_cropped, mtf_z_full)

    np.save(out_dir / "single_bead_psf_crop.npy", psf_crop.astype(np.float32))
    np.save(out_dir / "mtf_xy_section_linear.npy", mtf_xy.astype(np.float32))
    np.save(out_dir / "mtf_xz_section_linear.npy", mtf_xz.astype(np.float32))
    np.save(out_dir / "mtf_xy_section.npy", mtf_xy_log.astype(np.float32))
    np.save(out_dir / "mtf_xz_section.npy", mtf_xz_log.astype(np.float32))

    if tifffile is not None:
        tifffile.imwrite(out_dir / "single_bead_psf_crop.tif", psf_crop.astype(np.float32))
        tifffile.imwrite(out_dir / "mtf_xy_section_linear.tif", mtf_xy.astype(np.float32))
        tifffile.imwrite(out_dir / "mtf_xz_section_linear.tif", mtf_xz.astype(np.float32))
        tifffile.imwrite(out_dir / "mtf_xy_section.tif", mtf_xy_log.astype(np.float32))
        tifffile.imwrite(out_dir / "mtf_xz_section.tif", mtf_xz_log.astype(np.float32))

    psf_xy = psf_crop[:, :, psf_crop.shape[2] // 2]
    psf_xz = psf_crop[psf_crop.shape[0] // 2, :, :]
    raw_xy = raw_crop[:, :, raw_crop.shape[2] // 2]
    fft_xy_log = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(psf_xy))))

    np.save(out_dir / "psf_xy_section.npy", psf_xy.astype(np.float32))
    np.save(out_dir / "psf_xz_section.npy", psf_xz.astype(np.float32))

    if tifffile is not None:
        tifffile.imwrite(out_dir / "psf_xy_section.tif", psf_xy.astype(np.float32))
        tifffile.imwrite(out_dir / "psf_xz_section.tif", psf_xz.astype(np.float32))

    psf_peak_yxz = tuple(int(v) for v in np.unravel_index(np.argmax(psf_crop), psf_crop.shape))
    py, px, pz = psf_peak_yxz
    psf_x_profile = psf_crop[py, :, pz]
    psf_y_profile = psf_crop[:, px, pz]
    psf_z_profile = psf_crop[py, px, :]
    fwhm_x = compute_fwhm_from_profile(psf_x_profile, args.dx)
    fwhm_y = compute_fwhm_from_profile(psf_y_profile, args.dy)
    fwhm_z = compute_fwhm_from_profile(psf_z_profile, args.dz)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes.ravel()

    ax[0].imshow(raw_xy, cmap="gray", origin="lower")
    ax[0].set_title("Raw ROI (XY)")
    ax[0].set_axis_off()

    ax[1].imshow(psf_xy, cmap="gray", origin="lower")
    ax[1].set_title("PSF-like ROI (XY)")
    ax[1].set_axis_off()

    ax[2].imshow(fft_xy_log, cmap="magma", origin="lower")
    ax[2].set_title("log(1+|FFT2|) of PSF XY")
    ax[2].set_axis_off()

    ax[3].plot(freq_radial, np.maximum(mtf_radial, MTF_LOG_FLOOR), linewidth=2, label="Radial XY MTF")
    if mtf50 is not None:
        ax[3].axvline(mtf50, color="tab:green", linestyle="--", linewidth=1, label=f"MTF50={mtf50:.3f}")
    if mtf10 is not None:
        ax[3].axvline(mtf10, color="tab:red", linestyle="--", linewidth=1, label=f"MTF10={mtf10:.3f}")
    ax[3].set_xlabel("Spatial Frequency (cycles/um)")
    ax[3].set_ylabel("MTF")
    ax[3].set_yscale("log")
    ax[3].set_ylim(MTF_LOG_FLOOR, 1.05)
    ax[3].grid(alpha=0.3)
    ax[3].legend(loc="upper right", fontsize=8)

    ax[4].plot(fx_pos, np.maximum(mtf_x, MTF_LOG_FLOOR), linewidth=2)
    ax[4].set_xlabel("fx (cycles/um)")
    ax[4].set_ylabel("MTF")
    ax[4].set_yscale("log")
    ax[4].set_ylim(MTF_LOG_FLOOR, 1.05)
    ax[4].grid(alpha=0.3)
    ax[4].set_title("MTF Along X Axis")

    ax[5].plot(fz_pos, np.maximum(mtf_z, MTF_LOG_FLOOR), linewidth=2, color="tab:orange")
    ax[5].set_xlabel("fz (cycles/um)")
    ax[5].set_ylabel("MTF")
    ax[5].set_yscale("log")
    ax[5].set_ylim(MTF_LOG_FLOOR, 1.05)
    ax[5].grid(alpha=0.3)
    ax[5].set_title("MTF Along Z Axis")

    fig.suptitle("Single Small Bead FFT/MTF Evaluation", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "single_bead_mtf_overview.png", dpi=220)
    plt.close(fig)

    xy_canvas = save_heatmap_opencv(
        out_dir / "single_bead_mtf_xy_section.png",
        mtf_xy_log,
        cv2_colormap=cv2.COLORMAP_TURBO,
    )
    xz_canvas = save_heatmap_opencv(
        out_dir / "single_bead_mtf_xz_section.png",
        mtf_xz_log.T,
        cv2_colormap=cv2.COLORMAP_TURBO,
    )
    colorbar_canvas = save_colorbar_opencv(
        out_dir / "single_bead_mtf_colorbar.png",
        height=max(int(xy_canvas.shape[0]), int(xz_canvas.shape[0])),
        cv2_colormap=cv2.COLORMAP_TURBO,
    )
    save_section_opencv(
        out_dir / "single_bead_psf_xy_section.png",
        psf_xy,
    )
    save_section_opencv(
        out_dir / "single_bead_psf_xz_section.png",
        psf_xz.T,
    )
    save_heatmap_plot(
        out_dir / "single_bead_mtf_xy_section_plot.png",
        mtf_xy_log,
        [float(fx_cropped[0]), float(fx_cropped[-1]), float(fy_cropped[0]), float(fy_cropped[-1])],
        cmap="turbo",
    )
    save_heatmap_plot(
        out_dir / "single_bead_mtf_xz_section_plot.png",
        mtf_xz_log.T,
        [float(fx_xz_cropped[0]), float(fx_xz_cropped[-1]), float(fz_cropped[0]), float(fz_cropped[-1])],
        cmap="turbo",
    )
    save_line_plot(
        out_dir / "single_bead_mtf_x_line_full.png",
        fx_cropped,
        mtf_x_full,
        xlabel="fx (cycles/um)",
        color="tab:blue",
    )
    save_line_plot(
        out_dir / "single_bead_mtf_z_line_full.png",
        fz_cropped,
        mtf_z_full,
        xlabel="fz (cycles/um)",
        color="tab:orange",
    )

    if args.show:
        try:
            cv2.namedWindow("MTF XY Section", cv2.WINDOW_NORMAL)
            cv2.namedWindow("MTF XZ Section", cv2.WINDOW_NORMAL)
            cv2.namedWindow("MTF Colorbar", cv2.WINDOW_NORMAL)
            cv2.imshow("MTF XY Section", xy_canvas)
            cv2.imshow("MTF XZ Section", xz_canvas)
            cv2.imshow("MTF Colorbar", colorbar_canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as exc:
            print(f"OpenCV imshow failed: {exc}")

    summary = {
        "input_path": str(in_path),
        "output_dir": str(out_dir),
        "raw_shape": [int(v) for v in raw.shape],
        "volume_shape_yxz": [int(v) for v in volume.shape],
        "orientation_action": orientation,
        "background_median": background,
        "polarity": polarity,
        "center_yxz": [int(v) for v in center],
        "crop_bounds_yxz": bounds,
        "crop_shape_yxz": [int(v) for v in psf_crop.shape],
        "dx_um": float(args.dx),
        "dy_um": float(args.dy),
        "dz_um": float(args.dz),
        "window_used": bool(window_used),
        "mtf_crop_fraction_per_side": float(args.mtf_crop_fraction),
        "mtf_crop_total_fraction": float(2.0 * args.mtf_crop_fraction),
        "mtf_xy_crop_info": mtf_xy_crop_info,
        "mtf_xz_crop_info": mtf_xz_crop_info,
        "mtf_section_saved_as": "log1p_unit_scale",
        "mtf_section_linear_backup_saved": True,
        "mtf50_xy_radial_cyc_per_um": None if mtf50 is None else float(mtf50),
        "mtf10_xy_radial_cyc_per_um": None if mtf10 is None else float(mtf10),
        "psf_peak_yxz_in_crop": [int(v) for v in psf_peak_yxz],
        "fwhm_x": fwhm_x,
        "fwhm_y": fwhm_y,
        "fwhm_z": fwhm_z,
        "fwhm_xy_mean_um": (
            None
            if fwhm_x is None or fwhm_y is None
            else float(0.5 * (fwhm_x["fwhm_um"] + fwhm_y["fwhm_um"]))
        ),
        "raw_min": float(np.min(volume)),
        "raw_max": float(np.max(volume)),
        "raw_mean": float(np.mean(volume)),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MTF from a single small bead via FFT.")
    parser.add_argument(
        "--input",
        type=str,
        default="./fdt_revise_data/beads07x01_2/ri.tif",
        help="Input TIFF volume path.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./fdt_revise_data/beads07x01_2/evaluation",
        help="Output directory for plots/curves/summary.",
    )
    parser.add_argument("--dx", type=float, default=0.2, help="Voxel size in x (um).")
    parser.add_argument("--dy", type=float, default=0.2, help="Voxel size in y (um).")
    parser.add_argument("--dz", type=float, default=0.2, help="Voxel size in z (um).")
    parser.add_argument("--half-xy", type=int, default=24, help="Half crop size in x/y around bead center.")
    parser.add_argument("--half-z", type=int, default=12, help="Half crop size in z around bead center.")
    parser.add_argument(
        "--center",
        nargs=3,
        type=int,
        default=None,
        metavar=("Y", "X", "Z"),
        help="Optional manual bead center in y x z index.",
    )
    parser.add_argument("--bins", type=int, default=180, help="Number of bins for radial MTF.")
    parser.add_argument(
        "--mtf-crop-fraction",
        type=float,
        default=1.0 / 3.0,
        help="Crop this fraction from each side of the MTF XY/XZ sections.",
    )
    parser.add_argument("--no-window", action="store_true", help="Disable 3D Hann window before FFT.")
    parser.add_argument("--show", action="store_true", help="Display XY/XZ MTF sections via OpenCV imshow.")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    summary = evaluate_single_bead_mtf(args)
    print("Single bead MTF evaluation done.")
    print(f"Input: {summary['input_path']}")
    print(f"Output dir: {summary['output_dir']}")
    print(f"Detected center (y,x,z): {tuple(summary['center_yxz'])}")
    print(f"Crop size (y,x,z): {tuple(summary['crop_shape_yxz'])}")
    print(
        "MTF section crop: "
        f"{summary['mtf_crop_fraction_per_side']:.4f} per side, "
        f"{summary['mtf_crop_total_fraction']:.4f} total"
    )
    print(
        "Crop section sizes: "
        f"PSF XY={tuple(summary['crop_shape_yxz'][:2])}, "
        f"PSF XZ={tuple((summary['crop_shape_yxz'][1], summary['crop_shape_yxz'][2]))}"
    )
    print(f"MTF50 (radial XY): {summary['mtf50_xy_radial_cyc_per_um']} cycles/um")
    print(f"MTF10 (radial XY): {summary['mtf10_xy_radial_cyc_per_um']} cycles/um")
    if summary["fwhm_x"] is not None:
        print(f"FWHM x: {summary['fwhm_x']['fwhm_um']:.4f} um")
    if summary["fwhm_y"] is not None:
        print(f"FWHM y: {summary['fwhm_y']['fwhm_um']:.4f} um")
    if summary["fwhm_z"] is not None:
        print(f"FWHM z: {summary['fwhm_z']['fwhm_um']:.4f} um")
    if summary["fwhm_xy_mean_um"] is not None:
        print(f"Mean lateral FWHM: {summary['fwhm_xy_mean_um']:.4f} um")


if __name__ == "__main__":
    main()
