"""Extract metric 3D body landmarks (pose_world_landmarks) from a video.

pose_landmarks gives image coordinates whose z is a weak hip-relative guess.
pose_world_landmarks gives a metric estimate in metres with the origin at the hip
midpoint -- a genuine 3D body pose, though with no global trajectory: it says how
the body is configured, not where it stands.

This matters for rotation. A pirouette is rotation about the vertical axis, which
2D projection discards entirely; world landmarks estimate the depth axis directly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
NAMES = tuple(lm.name.lower() for lm in mp_pose.PoseLandmark)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--complexity", type=int, default=2, choices=(0, 1, 2))
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, detected, index = [], 0, 0

    with mp_pose.Pose(static_image_mode=False, model_complexity=args.complexity,
                      enable_segmentation=False, smooth_landmarks=True,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = pose.process(rgb)

            rec = {"frame": index, "t": round(index / fps, 4), "world": None}
            if res.pose_world_landmarks:
                rec["world"] = [
                    {"name": NAMES[i], "x": round(float(lm.x), 5),
                     "y": round(float(lm.y), 5), "z": round(float(lm.z), 5),
                     "visibility": round(float(lm.visibility), 4)}
                    for i, lm in enumerate(res.pose_world_landmarks.landmark)
                ]
                detected += 1
            frames.append(rec)
            index += 1
            if index % 200 == 0:
                print(f"  {index}/{total}", flush=True)

    cap.release()
    args.out.write_text(json.dumps({
        "source": str(args.video), "fps": fps, "frame_count": index,
        "frames_with_pose": detected, "landmark_names": list(NAMES),
        "connections": [list(c) for c in sorted(mp_pose.POSE_CONNECTIONS)],
        "units": "metres, origin at hip midpoint; no global translation",
        "frames": frames,
    }))
    print(f"done: {index} frames, 3D pose in {detected} "
          f"({100.0*detected/max(index,1):.1f}%)")


if __name__ == "__main__":
    main()
