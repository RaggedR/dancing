"""Differentiate the fitted dance and find its structure.

Two questions, both answerable only because the motion is now a closed-form
function rather than a table of samples:

1. WHERE ARE THE HELD POSITIONS?  A balance is d/dt(pose) ~ 0 sustained over an
   interval. Differentiating the cosine series term by term gives exact joint
   velocities; finite differences on raw pose data would amplify tracker noise
   while shrinking the very signal being measured.

2. HOW MANY DIMENSIONS DOES THE DANCE OCCUPY?  PCA over translation- and
   scale-normalised poses gives the eigen-poses the choreography actually moves
   through, and the intrinsic dimension of that motion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from interpolate_motion import CosineSeries, fill_gaps, load, reject_outliers

MIN_BALANCE_S = 0.30      # shorter than this is a passing slow moment, not a hold
BALANCE_SPEED_FRAC = 0.5  # of median speed


def normalise_poses(xy: np.ndarray, names: list[str]) -> np.ndarray:
    """Remove translation and scale so PCA sees shape, not where she stood.

    Centre on the hip midpoint, scale by torso length. Without this the leading
    component would merely encode her position on the stage.
    """
    hip = (xy[:, names.index("left_hip")] + xy[:, names.index("right_hip")]) / 2
    sho = (xy[:, names.index("left_shoulder")] + xy[:, names.index("right_shoulder")]) / 2
    torso = np.linalg.norm(sho - hip, axis=1, keepdims=True)
    torso = np.where(torso < 1e-6, np.nan, torso)
    return (xy - hip[:, None, :]) / torso[:, None]


def find_balances(times_s: np.ndarray, speed: np.ndarray, fps: float):
    """Contiguous intervals where speed stays below a fraction of the median."""
    thr = BALANCE_SPEED_FRAC * float(np.median(speed))
    quiet = speed < thr
    runs, start = [], None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) / fps >= MIN_BALANCE_S:
                runs.append((times_s[start], times_s[i - 1], speed[start:i].mean()))
            start = None
    if start is not None and (len(quiet) - start) / fps >= MIN_BALANCE_S:
        runs.append((times_s[start], times_s[-1], speed[start:].mean()))
    return runs, thr


def draw_pose(ax, pose: np.ndarray, connections, colour: str, lw=1.4, alpha=1.0):
    for a, b in connections:
        ax.plot([pose[a, 0], pose[b, 0]], [pose[a, 1], pose[b, 1]],
                color=colour, lw=lw, alpha=alpha, solid_capstyle="round")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cutoff-hz", type=float, default=6.0)
    args = ap.parse_args()

    xy, fps, connections, meta = load(args.motion)
    names = meta["landmark_names"]
    width = meta["width"]
    xy, n_bad = reject_outliers(xy)
    xy = fill_gaps(xy)
    n_frames = xy.shape[0]

    signal = xy.reshape(n_frames, -1)
    n_keep = max(2, int(round(2 * n_frames * args.cutoff_hz / fps)))
    series = CosineSeries(signal, n_keep)

    # --- 1. analytic velocity ------------------------------------------------
    t = np.arange(n_frames, dtype=float)
    vel = series.derivative(t).reshape(n_frames, -1, 2)      # units/frame
    speed_px_s = np.linalg.norm(vel, axis=2).mean(axis=1) * fps * width
    times_s = t / fps

    # Absolute speed conflates three things: the body changing shape, the dancer
    # travelling across the stage, and the camera moving. A balance held while
    # travelling reads as fast. Refit in translation/scale-normalised pose space
    # to isolate SHAPE change -- the quantity a held position actually makes small.
    norm_xy = fill_gaps(normalise_poses(xy, names))
    shape_series = CosineSeries(norm_xy.reshape(n_frames, -1), n_keep)
    shape_vel = shape_series.derivative(t).reshape(n_frames, -1, 2)
    shape_speed = np.linalg.norm(shape_vel, axis=2).mean(axis=1) * fps  # torso/s

    balances, thr = find_balances(times_s, shape_speed, fps)

    print(f"frames {n_frames} @ {fps:g}fps, {n_bad} outliers repaired, K={n_keep}")
    print(f"mean joint speed: {speed_px_s.mean():7.1f} px/s   "
          f"median {np.median(speed_px_s):.1f}   peak {speed_px_s.max():.1f}")
    print(f"balance threshold: {thr:.3f} torso/s, min duration {MIN_BALANCE_S}s\n")
    print(f"shape speed: median {np.median(shape_speed):.3f} torso/s")
    print(f"{len(balances)} held positions detected:")
    for a, b, s in balances:
        print(f"   {a:6.2f}s -> {b:6.2f}s   ({b - a:4.2f}s)   mean shape speed {s:6.3f} torso/s")

    # --- 2. PCA over normalised poses ---------------------------------------
    norm = normalise_poses(xy, names)
    flat = norm.reshape(n_frames, -1)
    flat = flat[~np.isnan(flat).any(axis=1)]
    mean_pose = flat.mean(axis=0)
    centred = flat - mean_pose
    _, sv, vt = np.linalg.svd(centred, full_matrices=False)
    var = sv**2
    cum = np.cumsum(var) / var.sum()

    print(f"\nPCA over {flat.shape[0]} normalised poses in {flat.shape[1]}-dim space:")
    for target in (0.80, 0.90, 0.95, 0.99):
        k = int(np.searchsorted(cum, target) + 1)
        print(f"  {target:.0%} of pose variance: {k:>3} components")
    print("  leading components: " +
          ", ".join(f"PC{i+1} {100*var[i]/var.sum():.1f}%" for i in range(6)))

    # --- 3. figure -----------------------------------------------------------
    fig = plt.figure(figsize=(15, 9), facecolor="white")
    gs = fig.add_gridspec(3, 6, height_ratios=[1.1, 1.0, 1.3], hspace=0.45, wspace=0.3)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(times_s, speed_px_s / speed_px_s.max(), lw=0.7, color="#bbb",
            label="absolute speed (conflates travel + camera)")
    ax.plot(times_s, shape_speed / shape_speed.max(), lw=0.9, color="#1f4e79",
            label="shape speed (translation/scale removed)")
    ax.axhline(thr / shape_speed.max(), color="#c00", ls="--", lw=0.8,
               label=f"threshold {thr:.2f} torso/s")
    for a, b, _ in balances:
        ax.axvspan(a, b, color="#ffd24d", alpha=0.55, lw=0)
    ax.set_title("Analytic velocity from the differentiated series "
                 f"({len(balances)} held positions shaded)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("speed (normalised)")
    ax.set_xlim(times_s[0], times_s[-1]); ax.legend(loc="upper right", fontsize=8)

    ax = fig.add_subplot(gs[1, :3])
    ax.plot(np.arange(1, len(cum) + 1), 100 * cum, marker="o", ms=3, color="#1f4e79")
    for target, c in ((0.90, "#c00"), (0.95, "#e8a"), (0.99, "#888")):
        ax.axhline(100 * target, color=c, ls="--", lw=0.8)
    ax.set_xlim(0, 30); ax.set_ylim(0, 101)
    ax.set_title("Intrinsic dimension: cumulative pose variance")
    ax.set_xlabel("principal components"); ax.set_ylabel("% variance")

    ax = fig.add_subplot(gs[1, 3:])
    ax.bar(np.arange(1, 13), 100 * var[:12] / var.sum(), color="#1f4e79")
    ax.set_title("Variance per component"); ax.set_xlabel("PC"); ax.set_ylabel("%")

    mp = mean_pose.reshape(-1, 2)
    for i in range(6):
        ax = fig.add_subplot(gs[2, i])
        comp = vt[i].reshape(-1, 2) * np.sqrt(var[i] / len(flat))
        draw_pose(ax, mp - 2 * comp, connections, "#c0392b", alpha=0.85)
        draw_pose(ax, mp + 2 * comp, connections, "#1f4e79", alpha=0.85)
        draw_pose(ax, mp, connections, "#999", lw=0.8, alpha=0.6)
        ax.set_title(f"PC{i+1}  {100*var[i]/var.sum():.1f}%", fontsize=9)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")

    fig.suptitle("Structure of the variation: velocity, held positions, eigen-poses",
                 fontsize=13)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
