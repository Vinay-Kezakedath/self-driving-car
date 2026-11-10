"""Command-line entry point.

Examples
--------
    python main.py                                   # sample video, show window
    python main.py --no-display -o output/demo.mp4   # head-less, write a video
    python main.py --source my_dashcam.mp4 --detector yolo --focal 1100
    python main.py --source 0                        # webcam
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2

from src.config import AppConfig, CameraConfig
from src.object_detection import make_detector
from src.pipeline import Pipeline

SAMPLE = "data/sample_road.mp4"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OpenCV + YOLO self-driving-car perception demo")
    ap.add_argument("--source", default=SAMPLE, help="video file or webcam index (default: bundled sample)")
    ap.add_argument("-o", "--output", default=None, help="write annotated video here (mp4)")
    ap.add_argument("--no-display", action="store_true", help="do not open a window (head-less / servers)")
    ap.add_argument("--detector", choices=["auto", "yolo", "classical", "none"], default="auto",
                    help="auto = classical for the synthetic sample, YOLO for everything else")
    ap.add_argument("--weights", default="models/yolov8n.pt", help="YOLO weights file (downloaded if missing)")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    ap.add_argument("--focal", type=float, default=None, help="camera focal length in pixels (for distance estimates)")
    ap.add_argument("--detect-every", type=int, default=1, help="run object detection every N frames (speed-up)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = whole video)")
    ap.add_argument("--src-roi", type=float, nargs=8, metavar=("X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4"),
                    help="perspective trapezoid TL TR BR BL in pixels of the source frame (real videos)")
    ap.add_argument("--roi-length-m", type=float, default=None,
                    help="real-world length (m) covered by the trapezoid; sets the forward metres-per-pixel (real videos)")
    return ap.parse_args()


def main() -> int:
    a = parse_args()
    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"error: cannot open {a.source!r}. Run `python generate_sample_video.py` first?", file=sys.stderr)
        return 1
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # ---- per-video calibration presets (data/real/presets.json) --------------
    # If the source is a file listed there, use its stored calibration unless the
    # user passed the corresponding flag explicitly.
    preset_file = Path("data/real/presets.json")
    if preset_file.exists() and not a.source.isdigit():
        preset = json.loads(preset_file.read_text()).get(Path(a.source).name)
        if preset:
            a.src_roi = a.src_roi or preset.get("src_roi")
            a.roi_length_m = a.roi_length_m or preset.get("roi_length_m")
            a.focal = a.focal or preset.get("focal")
            print(f"using calibration preset for {Path(a.source).name}")

    # ---- configuration ----------------------------------------------------
    cfg = AppConfig()
    is_sample = str(a.source) == SAMPLE
    focal = a.focal or CameraConfig().focal_px
    if a.src_roi:                                      # user-calibrated trapezoid
        from dataclasses import replace
        pts = tuple((a.src_roi[i], a.src_roi[i + 1]) for i in range(0, 8, 2))
        cfg = replace(cfg, lane=replace(cfg.lane, src=pts, ref_size=(W, H)))
    if a.roi_length_m:                                 # forward scale of the bird's-eye view
        from dataclasses import replace
        cfg = replace(cfg, lane=replace(cfg.lane, ym_per_pix=a.roi_length_m / cfg.lane.warp_size[1]))
    kind = a.detector if a.detector != "auto" else ("classical" if is_sample else "yolo")
    detector = make_detector(kind, a.weights, a.conf, focal, CameraConfig().horizon_y * H // 720)
    print(f"source={a.source} {W}x{H}@{fps:.0f}fps  detector={kind}")

    pipe = Pipeline(cfg, (W, H), detector, fps, a.detect_every)
    writer = None
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(a.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    show = not a.no_display and bool(os.environ.get("DISPLAY") or sys.platform == "darwin" or os.name == "nt")
    if not a.no_display and not show:
        print("(no display found - running head-less; use -o to save a video)")

    # ---- main loop ---------------------------------------------------------
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or (a.max_frames and n >= a.max_frames):
            break
        r = pipe.step(frame)
        if writer:
            writer.write(r.annotated)
        if show:
            cv2.imshow("self-driving-car  (q / ESC to quit, space to pause)", r.annotated)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            if k == ord(" "):
                cv2.waitKey(0)
        n += 1
        if n % 60 == 0:
            print(f"frame {n:5d}  {r.command.state.value:5s} {r.command.target_speed_kmh:4.0f} km/h  steer {r.command.steering_deg:+5.1f}  {r.fps:4.1f} fps")
    cap.release()
    if writer:
        writer.release()
        print("saved", a.output)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
