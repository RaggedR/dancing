"""Fit the 3D world landmarks to a cosine series and render an orbiting camera.

Same representation as the 2D pipeline, now over 99 channels (33 joints x 3 axes).
The orbit is the point: if the depth axis were fabricated, rotating the camera
would reveal a flat sheet. A body that stays solid from every angle is the proof
that the third dimension carries real information.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from interpolate_motion import CosineSeries, fill_gaps

BONE = (235, 235, 235)
JOINT = (90, 200, 255)


def load3d(path: Path):
    d = json.loads(path.read_text())
    n = d["frame_count"]
    names = d["landmark_names"]
    xyz = np.full((n, len(names), 3), np.nan)
    for t, f in enumerate(d["frames"]):
        if f["world"]:
            for j, lm in enumerate(f["world"]):
                xyz[t, j] = (lm["x"], lm["y"], lm["z"])
    return xyz, float(d["fps"]), [tuple(c) for c in d["connections"]], names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion3d", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cutoff-hz", type=float, default=12.0)
    ap.add_argument("--size", type=int, default=720)
    ap.add_argument("--orbit-deg", type=float, default=360.0,
                    help="total camera rotation across the clip")
    args = ap.parse_args()

    xyz, fps, conn, names = load3d(args.motion3d)
    n, j, _ = xyz.shape
    flat = fill_gaps(xyz.reshape(n, j, 3).reshape(n, -1).reshape(n, j * 3, 1)
                     ).reshape(n, -1) if False else None

    # fill per-channel along time
    sig = xyz.reshape(n, -1).copy()
    t_idx = np.arange(n, dtype=float)
    for c in range(sig.shape[1]):
        col = sig[:, c]
        good = ~np.isnan(col)
        col[:] = np.interp(t_idx, t_idx[good], col[good]) if good.sum() > 1 else 0.0

    K = max(2, int(round(2 * n * args.cutoff_hz / fps)))
    series = CosineSeries(sig, K)
    smooth = series(t_idx).reshape(n, j, 3)
    resid = np.abs(smooth - xyz.reshape(n, j, 3))
    print(f"{n} frames, {j} joints, K={K} ({args.cutoff_hz:g} Hz)")
    print(f"residual {np.nanmean(resid)*1000:.1f} mm mean")

    S = args.size
    vw = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (S, S))
    scale = S * 0.34
    for i in range(n):
        p = smooth[i].copy()
        p -= p.mean(axis=0)                      # centre the body
        a = np.radians(args.orbit_deg * i / n)   # orbit angle
        ca, sa = np.cos(a), np.sin(a)
        x = p[:, 0] * ca + p[:, 2] * sa          # rotate about vertical axis
        y = p[:, 1]
        img = np.zeros((S, S, 3), np.uint8)
        pts = [(int(S / 2 + x[k] * scale), int(S / 2 + y[k] * scale)) for k in range(j)]
        for a_, b_ in conn:
            cv2.line(img, pts[a_], pts[b_], BONE, 2, cv2.LINE_AA)
        for q in pts:
            cv2.circle(img, q, 3, JOINT, -1, cv2.LINE_AA)
        cv2.putText(img, f"{i/fps:5.2f}s   camera {np.degrees(a):5.0f}deg",
                    (14, S - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
        vw.write(img)
    vw.release()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
