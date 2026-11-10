"""Calibration helper: check a perspective trapezoid (``--src-roi``) on a frame.

    python calibrate.py --source data/real/project_video.mp4 --frame 40 \
        --src-roi 585 460 695 460 1127 720 203 720 -o output/calib.png

Writes an image with two panels:
  left  : the frame with the trapezoid drawn on the road
  right : the bird's-eye view. GOOD calibration  =  on a straight road both lane
          lines are vertical, parallel and sit ON the two green guide lines
          (the guides mark where the lane is expected: x = 520 and 760).
If the lines tilt outwards (/ \\) the trapezoid top is too wide; if they tilt
inwards (\\ /) it is too narrow. Adjust the 8 numbers and run again.
Corner order: top-left, top-right, bottom-right, bottom-left (pixels).
"""
import argparse

import cv2
import numpy as np

from src.config import LaneConfig
from src.lane_detection import LaneDetector

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--source", required=True)
ap.add_argument("--frame", type=int, default=40, help="frame index to inspect (pick a straight road)")
ap.add_argument("--src-roi", type=float, nargs=8, required=True)
ap.add_argument("-o", "--output", default="output/calib.png")
a = ap.parse_args()

cap = cv2.VideoCapture(a.source)
cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
ok, frame = cap.read()
assert ok, f"cannot read frame {a.frame} of {a.source}"
H, W = frame.shape[:2]

pts = tuple((a.src_roi[i], a.src_roi[i + 1]) for i in range(0, 8, 2))
from dataclasses import replace
det = LaneDetector(replace(LaneConfig(), src=pts, ref_size=(W, H)), (W, H))

left = frame.copy()
cv2.polylines(left, [np.int32(pts)], True, (0, 255, 255), 2, cv2.LINE_AA)
right = det.warp(frame)
for x in (det.cfg.dst[0][0], det.cfg.dst[1][0]):                      # expected lane-line positions
    cv2.line(right, (int(x), 0), (int(x), right.shape[0]), (0, 255, 0), 1)
right = cv2.resize(right[:, 200:1080], None, fx=H / right.shape[0], fy=H / right.shape[0])
left_w = int(left.shape[1] * 1.0)
cv2.imwrite(a.output, np.hstack([left, right]))
print("wrote", a.output)
