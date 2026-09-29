"""How many cosine coefficients does a dance need?

Represents each joint trajectory as a DCT-II series and measures reconstruction
error against the number of retained coefficients K. The DCT (even extension)
rather than the DFT (periodic extension) because a dance does not return to its
starting pose: forcing periodicity introduces a seam discontinuity whose
truncation produces Gibbs ringing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.fft import dct, idct

MIN_VISIBILITY = 0.5


def load_tracks(path: Path) -> tuple[np.ndarray, float]:
    """Return (T, J, 2) array of normalised joint coords, and fps.

    Frames where a joint is occluded are filled by linear interpolation along
    time, so the transform never sees a spurious jump to zero.
    """
    meta = json.loads(path.read_text())
    frames = meta["frames"]
    n_joints = len(meta["landmark_names"])
    xy = np.full((len(frames), n_joints, 2), np.nan, dtype=np.float64)
    for t, frame in enumerate(frames):
        if not frame["landmarks"]:
            continue
        for j, lm in enumerate(frame["landmarks"]):
            if lm["visibility"] >= MIN_VISIBILITY:
                xy[t, j] = (lm["x"], lm["y"])

    times = np.arange(len(frames), dtype=np.float64)
    for j in range(n_joints):
        for c in range(2):
            col = xy[:, j, c]
            good = ~np.isnan(col)
            if good.sum() < 2:
                col[:] = 0.0 if not good.any() else col[good][0]
            else:
                col[:] = np.interp(times, times[good], col[good])
    return xy, float(meta["fps"])


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "out_paquita/motion.json")
    xy, fps = load_tracks(path)
    n_frames, n_joints, _ = xy.shape
    signal = xy.reshape(n_frames, -1)            # (T, 66) independent channels
    coeffs = dct(signal, axis=0, norm="ortho")   # (T, 66)

    total_energy = float((coeffs**2).sum())
    print(f"{path}: {n_frames} frames x {n_joints} joints @ {fps:g}fps")
    print(f"raw numbers: {signal.size:,}\n")

    # Reconstruction error as a function of retained coefficients.
    print(f"{'K':>6}{'RMSE (px@1080)':>16}{'max err px':>12}{'energy %':>10}"
          f"{'numbers':>10}{'compression':>13}")
    for k in (2, 4, 8, 16, 32, 64, 128, 256, 512):
        if k > n_frames:
            break
        truncated = np.zeros_like(coeffs)
        truncated[:k] = coeffs[:k]
        approx = idct(truncated, axis=0, norm="ortho")
        err = np.abs(approx - signal)
        rmse_px = float(np.sqrt((err**2).mean())) * 1080
        max_px = float(err.max()) * 1080
        energy = 100.0 * float((coeffs[:k]**2).sum()) / total_energy
        kept = k * signal.shape[1]
        print(f"{k:>6}{rmse_px:>16.2f}{max_px:>12.1f}{energy:>10.4f}"
              f"{kept:>10,}{signal.size/kept:>12.1f}x")

    # Smallest K meeting perceptual thresholds (1080-wide frame).
    print()
    for target_px in (5.0, 2.0, 1.0):
        for k in range(1, n_frames + 1):
            truncated = np.zeros_like(coeffs)
            truncated[:k] = coeffs[:k]
            rmse = float(np.sqrt(((idct(truncated, axis=0, norm="ortho")
                                   - signal) ** 2).mean())) * 1080
            if rmse <= target_px:
                print(f"RMSE <= {target_px:>4.1f}px needs K = {k:>4}  "
                      f"({k * signal.shape[1]:,} numbers, "
                      f"{signal.size / (k * signal.shape[1]):.0f}x compression)")
                break


if __name__ == "__main__":
    main()
