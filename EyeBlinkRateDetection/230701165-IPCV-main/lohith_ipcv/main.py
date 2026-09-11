"""
Eye Blink Rate Detection for fatigue monitoring.
Uses dlib HOG face detector + 68-landmark shape predictor and classical EAR (Eye Aspect Ratio).
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import dlib

from ear_utils import combined_ear, landmarks_to_eyes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EAR-based blink rate / fatigue hint (webcam or video).")
    p.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0).",
    )
    p.add_argument(
        "--video",
        type=str,
        default="",
        help="Path to a video file instead of webcam.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="",
        help="Path to shape_predictor_68_face_landmarks.dat (default: models/... in repo).",
    )
    p.add_argument(
        "--ear-thresh",
        type=float,
        default=0.25,
        help="EAR below this is treated as eye closed (tune per camera/distance).",
    )
    p.add_argument(
        "--ear-consec",
        type=int,
        default=2,
        help="Consecutive frames below threshold required to register a closed phase.",
    )
    p.add_argument(
        "--blink-window-sec",
        type=float,
        default=60.0,
        help="Rolling window (seconds) for blinks-per-minute estimate.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    model_path = Path(args.model) if args.model else repo_root / "models" / "shape_predictor_68_face_landmarks.dat"
    if not model_path.is_file():
        print(f"Missing model: {model_path}")
        print("Run: python download_model.py")
        raise SystemExit(1)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(model_path))

    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video: {args.video}")
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise SystemExit(f"Could not open camera index {args.camera}")

    ear_thresh = args.ear_thresh
    consec = max(1, args.ear_consec)
    window_sec = max(5.0, args.blink_window_sec)

    blink_timestamps: deque[float] = deque()
    closed_counter = 0
    total_blinks = 0
    fps_smooth = 0.0
    prev_t = time.perf_counter()

    win = "Eye Blink Rate (EAR + dlib 68) — q to quit"

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        now = time.time()
        while blink_timestamps and (now - blink_timestamps[0]) > window_sec:
            blink_timestamps.popleft()
        bpm = (len(blink_timestamps) / window_sec) * 60.0 if window_sec > 0 else 0.0

        ear_text = "--"
        for rect in rects:
            shape = predictor(gray, rect)
            left, right = landmarks_to_eyes(shape)
            ear = combined_ear(left, right)
            ear_text = f"{ear:.3f}"

            if ear < ear_thresh:
                closed_counter += 1
            else:
                if closed_counter >= consec:
                    total_blinks += 1
                    blink_timestamps.append(time.time())
                closed_counter = 0

            # draw landmarks for eyes only
            for (x, y) in left:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)
            for (x, y) in right:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)
            cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 1)

            cv2.putText(
                frame,
                f"EAR: {ear_text}  thresh: {ear_thresh}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Blinks (session): {total_blinks}",
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Blink rate (~{int(window_sec)}s window): {bpm:.1f} / min",
                (10, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "Info only — not medical diagnosis; tune --ear-thresh for your setup",
                (10, 118),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 255),
                1,
            )
            break  # first face only

        if not rects:
            closed_counter = 0
            cv2.putText(
                frame,
                "No face — center in frame",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Blinks (session): {total_blinks}",
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 180, 180),
                2,
            )
            cv2.putText(
                frame,
                f"Blink rate (~{int(window_sec)}s window): {bpm:.1f} / min",
                (10, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 180, 180),
                2,
            )

        t = time.perf_counter()
        dt = t - prev_t
        prev_t = t
        if dt > 0:
            inst_fps = 1.0 / dt
            fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps if fps_smooth > 0 else inst_fps
        cv2.putText(
            frame,
            f"FPS: {fps_smooth:.1f}",
            (10, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            1,
        )

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
