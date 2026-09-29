"""Extract 2D body motion and an isolated-body matte from a dance video.

A single inference pass per frame yields both the 33-point BlazePose skeleton and
a person segmentation mask: MediaPipe already decides person-vs-background while
locating the body, so `enable_segmentation` is nearly free. A separate matting
model would cost a second pass and could disagree with the pose about the body.

Pinned to mediapipe 0.10.x deliberately -- the 1.0 release crashes on macOS ARM
inside DrishtiMetalHelper at graph construction, even with the CPU delegate.

Outputs (into --out-dir):
    motion.json      per-frame landmarks, normalised + pixel coords
    body.mp4         source pixels, background removed
    silhouette.mp4   white body matte on black
    wireframe.mp4    skeleton lines on black
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

LANDMARK_NAMES = tuple(lm.name.lower() for lm in mp_pose.PoseLandmark)

MASK_THRESHOLD = 0.5
MASK_FEATHER_PX = 5  # odd; softens the matte edge so limbs do not alias
MIN_VISIBILITY = 0.5  # below this a joint is unreliable and is not drawn

BONE_COLOUR = (235, 235, 235)
JOINT_COLOUR = (90, 200, 255)


def matte_from_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Turn a confidence mask into a feathered 0..1 alpha map at frame size."""
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    alpha = (mask > MASK_THRESHOLD).astype(np.float32)
    if MASK_FEATHER_PX > 1:
        alpha = cv2.GaussianBlur(alpha, (MASK_FEATHER_PX, MASK_FEATHER_PX), 0)
    return alpha


def draw_wireframe(canvas: np.ndarray, points: list[tuple[int, int, float]]) -> None:
    """Draw bones then joints, skipping landmarks the model is unsure about."""
    for a, b in mp_pose.POSE_CONNECTIONS:
        xa, ya, va = points[a]
        xb, yb, vb = points[b]
        if va < MIN_VISIBILITY or vb < MIN_VISIBILITY:
            continue
        cv2.line(canvas, (xa, ya), (xb, yb), BONE_COLOUR, 2, cv2.LINE_AA)
    for x, y, v in points:
        if v < MIN_VISIBILITY:
            continue
        cv2.circle(canvas, (x, y), 3, JOINT_COLOUR, -1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("out"), type=Path)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all")
    parser.add_argument("--complexity", type=int, default=2, choices=(0, 1, 2),
                        help="2 = heavy/most accurate")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total = min(total, args.max_frames)
    print(f"{width}x{height} @ {fps:g}fps, {total} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = {
        name: cv2.VideoWriter(str(args.out_dir / f"{name}.mp4"), fourcc, fps,
                              (width, height))
        for name in ("body", "silhouette", "wireframe")
    }

    frames_out: list[dict] = []
    detected = 0
    index = 0

    # static_image_mode=False keeps temporal state: the model tracks rather than
    # re-detecting each frame, which is faster and markedly less jittery.
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=args.complexity,
        enable_segmentation=True,
        smooth_landmarks=True,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame_bgr = capture.read()
            if not ok or (args.max_frames and index >= args.max_frames):
                break

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = pose.process(rgb)

            body = np.zeros_like(frame_bgr)
            silhouette = np.zeros_like(frame_bgr)
            wireframe = np.zeros_like(frame_bgr)

            if result.segmentation_mask is not None:
                alpha = matte_from_mask(result.segmentation_mask, width, height)
                a3 = alpha[:, :, None]
                body = (frame_bgr.astype(np.float32) * a3).astype(np.uint8)
                silhouette = (np.float32(255.0) * a3).astype(np.uint8).repeat(3, axis=2)

            record: dict = {"frame": index, "t": round(index / fps, 4),
                            "landmarks": None}
            if result.pose_landmarks:
                lms = result.pose_landmarks.landmark
                points = [
                    (int(lm.x * width), int(lm.y * height), float(lm.visibility))
                    for lm in lms
                ]
                draw_wireframe(wireframe, points)
                record["landmarks"] = [
                    {
                        "name": LANDMARK_NAMES[i],
                        "x": round(float(lm.x), 5),
                        "y": round(float(lm.y), 5),
                        "z": round(float(lm.z), 5),
                        "visibility": round(float(lm.visibility), 4),
                        "px": points[i][0],
                        "py": points[i][1],
                    }
                    for i, lm in enumerate(lms)
                ]
                detected += 1
            frames_out.append(record)

            writers["body"].write(body)
            writers["silhouette"].write(silhouette)
            writers["wireframe"].write(wireframe)

            index += 1
            if index % 50 == 0:
                print(f"  {index}/{total} frames, {detected} with a pose",
                      flush=True)

    capture.release()
    for writer in writers.values():
        writer.release()

    payload = {
        "source": str(args.video),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": index,
        "frames_with_pose": detected,
        "landmark_names": list(LANDMARK_NAMES),
        "connections": sorted(tuple(c) for c in mp_pose.POSE_CONNECTIONS),
        "note": "x,y normalised to [0,1]; z is weak monocular depth, hip-relative",
        "frames": frames_out,
    }
    (args.out_dir / "motion.json").write_text(json.dumps(payload))
    print(f"done: {index} frames, pose found in {detected} "
          f"({100.0 * detected / max(index, 1):.1f}%)")


if __name__ == "__main__":
    main()
