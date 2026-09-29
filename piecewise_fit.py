"""Fit the motion in segments, each with its own cutoff, blended at the seams.

A single global cutoff cannot serve a whole variation: sustained phrases want
heavy smoothing (6 Hz) while a pirouette carries real limb motion far above that
and turns to mush. This fits each region separately and crossfades between them.

Segments overlap, and each contributes a raised-cosine weight across its overlap
so the handover is spread over ~0.2s. Without that, two independent fits disagree
at the seam -- each chose its coefficients knowing only its own samples -- and the
splice itself becomes a visible jump.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from interpolate_motion import (CosineSeries, fill_gaps, load, reject_outliers,
                                relabel_left_right)

BONE = (235, 235, 235)
JOINT = (90, 200, 255)


def raised_cosine(n: int) -> np.ndarray:
    """Smooth 0->1 ramp; its complement sums to exactly 1 with it."""
    return 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / max(n - 1, 1))


def piecewise(signal: np.ndarray, fps: float, spans, overlap: int):
    """spans: list of (start_frame, end_frame, cutoff_hz). Returns blended fit."""
    n_frames = signal.shape[0]
    acc = np.zeros_like(signal)
    wsum = np.zeros(n_frames)
    report = []

    for (a, b, hz) in spans:
        lo, hi = max(0, a - overlap), min(n_frames, b + overlap)
        seg = signal[lo:hi]
        n_seg = seg.shape[0]
        k = max(2, min(n_seg, int(round(2 * n_seg * hz / fps))))
        series = CosineSeries(seg, k)
        fitted = series(np.arange(n_seg, dtype=float))

        w = np.ones(n_seg)
        if lo > 0:                       # ramp in
            m = min(overlap, n_seg // 2)
            w[:m] = raised_cosine(m)
        if hi < n_frames:                # ramp out
            m = min(overlap, n_seg // 2)
            w[-m:] = raised_cosine(m)[::-1]

        acc[lo:hi] += fitted * w[:, None]
        wsum[lo:hi] += w
        report.append((a / fps, b / fps, hz, k, n_seg))

    # any frame not covered by a span keeps its input
    bare = wsum < 1e-9
    wsum[bare] = 1.0
    acc[bare] = signal[bare]
    return acc / wsum[:, None], report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--turn", nargs=2, type=float, default=[18.80, 20.60],
                    metavar=("START_S", "END_S"))
    ap.add_argument("--cutoff-calm", type=float, default=6.0)
    ap.add_argument("--cutoff-turn", type=float, default=20.0)
    ap.add_argument("--overlap", type=float, default=0.24, help="crossfade seconds")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    xy, fps, connections, meta = load(args.motion)
    names = meta["landmark_names"]
    w_px, h_px = meta["width"], meta["height"]
    xy, n_sw = relabel_left_right(xy, names)
    xy, n_bad = reject_outliers(xy)
    xy = fill_gaps(xy)
    n_frames = xy.shape[0]
    signal = xy.reshape(n_frames, -1)

    t0, t1 = int(args.turn[0] * fps), int(args.turn[1] * fps)
    overlap = int(args.overlap * fps)
    spans = [(0, t0, args.cutoff_calm),
             (t0, t1, args.cutoff_turn),
             (t1, n_frames, args.cutoff_calm)]

    fitted, report = piecewise(signal, fps, spans, overlap)

    print(f"{n_frames} frames @ {fps:g}fps   relabelled {n_sw}   outliers {n_bad}")
    print(f"crossfade {args.overlap:g}s ({overlap} frames)\n")
    print(f"{'span':>18}{'cutoff':>9}{'K':>6}{'frames':>8}")
    for a, b, hz, k, n_seg in report:
        print(f"{a:7.2f}s -{b:7.2f}s{hz:8.0f}Hz{k:6}{n_seg:8}")

    turn = slice(t0, t1)
    resid = np.abs(fitted - signal) * w_px
    print(f"\nresidual: turn {resid[turn].mean():.2f}px   "
          f"elsewhere {np.delete(resid, np.r_[turn], axis=0).mean():.2f}px")

    # seam continuity: inter-frame jump at the splice vs typical
    jump = np.linalg.norm(np.diff(fitted, axis=0), axis=1) * w_px
    typical = float(np.median(jump))
    for name, f in (("splice in", t0), ("splice out", t1)):
        local = jump[max(0, f - 2):f + 2].max()
        print(f"  {name:11} max jump {local:6.2f}px   (median across clip "
              f"{typical:.2f}px, ratio {local/typical:.2f}x)")

    vw = cv2.VideoWriter(str(args.out_dir / "spliced.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_px, h_px))
    coords = fitted.reshape(n_frames, -1, 2)
    for f in coords:
        canvas = np.zeros((h_px, w_px, 3), np.uint8)
        pts = [(int(x * w_px), int(y * h_px)) for x, y in f]
        for a_, b_ in connections:
            cv2.line(canvas, pts[a_], pts[b_], BONE, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(canvas, p, 3, JOINT, -1, cv2.LINE_AA)
        vw.write(canvas)
    vw.release()
    print(f"\nwrote {args.out_dir / 'spliced.mp4'}")


if __name__ == "__main__":
    main()
