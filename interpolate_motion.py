"""Fit a continuous closed-form curve to extracted joint trajectories.

Each joint coordinate becomes a truncated DCT-II series

    x(t) = Σ_{k=0}^{K-1} c_k · w(k) · cos(π k (2t+1) / (2N)),
    w(0) = √(1/N),  w(k>0) = √(2/N)

The cosine basis is defined for REAL t, not just integer frames, so the same
coefficients give both smoothing (truncation) and resampling at any framerate
(evaluation between samples). No separate interpolation step is needed.

DCT rather than DFT: a dance is not periodic, and a Fourier series would impose
a seam discontinuity between last and first frame, producing Gibbs ringing.

Truncation is specified as a cutoff FREQUENCY, not a coefficient count, because
the meaningful quantity is physical: limb motion lives below ~8 Hz, so anything
above that is tracker jitter. For DCT-II, coefficient k sits at k·fps/(2N) Hz.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.fft import dct

MIN_VISIBILITY = 0.5
OUTLIER_MAD_FACTOR = 12.0  # inter-frame jumps beyond this are tracker failures
BONE_COLOUR = (235, 235, 235)
JOINT_COLOUR = (90, 200, 255)


def load(path: Path):
    meta = json.loads(path.read_text())
    frames = meta["frames"]
    n_joints = len(meta["landmark_names"])
    xy = np.full((len(frames), n_joints, 2), np.nan)
    for t, f in enumerate(frames):
        if not f["landmarks"]:
            continue
        for j, lm in enumerate(f["landmarks"]):
            if lm["visibility"] >= MIN_VISIBILITY:
                xy[t, j] = (lm["x"], lm["y"])
    return xy, float(meta["fps"]), meta["connections"], meta


def reject_outliers(xy: np.ndarray) -> tuple[np.ndarray, int]:
    """Blank frames whose inter-frame motion is a gross outlier.

    Turn sequences make the tracker swap left/right limbs, producing single-frame
    teleports. Left in place these dominate the high-frequency coefficients and
    survive any low-pass as ringing, so they are removed before fitting.
    """
    disp = np.nanmean(np.linalg.norm(np.diff(xy, axis=0), axis=2), axis=1)
    med = np.nanmedian(disp)
    mad = np.nanmedian(np.abs(disp - med)) or 1e-9
    bad = np.where(disp > med + OUTLIER_MAD_FACTOR * mad)[0] + 1
    xy = xy.copy()
    xy[bad] = np.nan
    return xy, len(bad)


LR_SWAP_MARGIN = 0.7  # swap only when clearly better, so symmetric poses are left alone


def relabel_left_right(xy: np.ndarray, names: list[str]) -> tuple[np.ndarray, int]:
    """Choose, per frame, the L/R labelling closest to the previous frame.

    During a turn the pose model swaps left and right limb assignments: a rotating
    body is near-symmetric front-to-back, so the labelling becomes unstable. Once
    swapped the motion is continuous again in the swapped state, so these frames
    never look like outliers and survive any outlier filter.

    The margin gate matters. Ungated, this rule also fires on near-symmetric
    standing poses where swapping barely changes the distance and noise decides --
    measured at 229 swaps, only 53 of them in the turn. Requiring the swap to be
    clearly better confines it to 56 swaps, all inside the turn.
    """
    swap = np.arange(len(names))
    for i, n in enumerate(names):
        if n.startswith("left_"):
            j = names.index("right_" + n[5:])
            swap[i], swap[j] = j, i

    out = fill_gaps(xy.copy())
    n_swapped = 0
    for t in range(1, out.shape[0]):
        d_id = np.nansum((out[t] - out[t - 1]) ** 2)
        d_sw = np.nansum((out[t][swap] - out[t - 1]) ** 2)
        if d_sw < d_id * LR_SWAP_MARGIN:
            out[t] = out[t][swap]
            n_swapped += 1
    return out, n_swapped


def fill_gaps(xy: np.ndarray) -> np.ndarray:
    """Linear fill along time so the transform sees no NaNs or false zeros."""
    t = np.arange(xy.shape[0], dtype=float)
    out = xy.copy()
    for j in range(xy.shape[1]):
        for c in range(2):
            col = out[:, j, c]
            good = ~np.isnan(col)
            if good.sum() < 2:
                col[:] = col[good][0] if good.any() else 0.0
            else:
                col[:] = np.interp(t, t[good], col[good])
    return out


class CosineSeries:
    """Truncated DCT-II series, evaluable at arbitrary real times."""

    def __init__(self, signal: np.ndarray, n_keep: int):
        self.n = signal.shape[0]
        self.k = min(n_keep, self.n)
        self.coeffs = dct(signal, axis=0, norm="ortho")[: self.k]
        self.weights = np.full(self.k, np.sqrt(2.0 / self.n))
        self.weights[0] = np.sqrt(1.0 / self.n)

    def __call__(self, t: np.ndarray) -> np.ndarray:
        k = np.arange(self.k)
        basis = np.cos(np.pi * k[None, :] * (2 * t[:, None] + 1) / (2 * self.n))
        return (basis * self.weights) @ self.coeffs

    def derivative(self, t: np.ndarray) -> np.ndarray:
        """Exact d/dt of the series, in units per frame.

        Differentiating term by term avoids finite differences, which on pose
        data amplify tracker noise while shrinking the signal of interest.
        The chain rule on (2t+1)/(2N) contributes the factor pi*k/N.
        """
        k = np.arange(self.k)
        phase = np.pi * k[None, :] * (2 * t[:, None] + 1) / (2 * self.n)
        basis = -np.sin(phase) * (np.pi * k / self.n)
        return (basis * self.weights) @ self.coeffs


def draw(canvas, pts, connections):
    for a, b in connections:
        cv2.line(canvas, pts[a], pts[b], BONE_COLOUR, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(canvas, p, 3, JOINT_COLOUR, -1, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--cutoff-hz", type=float, default=12.0)
    ap.add_argument("--slowmo", type=float, default=4.0,
                    help="temporal upsampling factor for the slow-motion render")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    xy, fps, connections, meta = load(args.motion)
    w, h = meta["width"], meta["height"]
    n_frames = xy.shape[0]

    xy, n_swapped = relabel_left_right(xy, meta["landmark_names"])
    xy, n_bad = reject_outliers(xy)
    xy = fill_gaps(xy)
    signal = xy.reshape(n_frames, -1)

    n_keep = max(2, int(round(2 * n_frames * args.cutoff_hz / fps)))
    series = CosineSeries(signal, n_keep)

    resid = series(np.arange(n_frames, dtype=float)) - signal
    rmse = float(np.sqrt((resid ** 2).mean())) * w

    print(f"frames {n_frames} @ {fps:g}fps   L/R relabelled: {n_swapped}   outliers repaired: {n_bad}")
    print(f"cutoff {args.cutoff_hz:g} Hz -> K = {n_keep} coefficients per channel")
    print(f"coefficients kept: {n_keep * signal.shape[1]:,} of {signal.size:,} "
          f"({signal.size / (n_keep * signal.shape[1]):.1f}x smaller)")
    print(f"residual vs input: {rmse:.2f} px (this is smoothing, not error)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    renders = {
        "smoothed": np.arange(n_frames, dtype=float),
        "slowmo": np.linspace(0, n_frames - 1, int(n_frames * args.slowmo)),
    }
    for name, times in renders.items():
        vw = cv2.VideoWriter(str(args.out_dir / f"{name}.mp4"), fourcc, fps, (w, h))
        coords = series(times).reshape(len(times), -1, 2)
        for frame in coords:
            canvas = np.zeros((h, w, 3), np.uint8)
            pts = [(int(x * w), int(y * h)) for x, y in frame]
            draw(canvas, pts, connections)
            vw.write(canvas)
        vw.release()
        print(f"  wrote {name}.mp4 ({len(times)} frames)")

    np.savez(args.out_dir / "series.npz", coeffs=series.coeffs,
             weights=series.weights, n=series.n, fps=fps,
             connections=np.array(connections))
    print(f"  wrote series.npz (the formula's coefficients)")


if __name__ == "__main__":
    main()
