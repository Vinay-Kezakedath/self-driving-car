"""Drawing helpers: lane overlay, bounding boxes, HUD, steering wheel, mini-map.

Everything here only *draws*; no decisions are made. Keeping drawing separate
from logic means the pipeline can run head-less (no GUI) with the same code.
"""
from __future__ import annotations

import cv2
import numpy as np

from .decision import Command, State
from .lane_detection import LaneDetector, LaneResult
from .object_detection import Detection

# BGR colours
GREEN, RED, AMBER, WHITE, BLACK, CYAN = (80, 220, 80), (60, 60, 240), (0, 190, 255), (245, 245, 245), (20, 20, 20), (230, 220, 60)
STATE_COLOR = {State.DRIVE: GREEN, State.SLOW: AMBER, State.STOP: RED}
LABEL_COLOR = {"car": CYAN, "truck": CYAN, "bus": CYAN, "person": AMBER, "traffic light": (200, 120, 255), "stop sign": RED}


def _text(img, s, org, scale=0.6, color=WHITE, thick=1):
    """Text with a dark outline so it is readable on any background."""
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_lane(img: np.ndarray, det: LaneDetector, res: LaneResult, state: State) -> None:
    """Translucent lane polygon + the two fitted lane lines, projected back
    from the bird's-eye view to the camera image with the inverse homography."""
    poly = det.lane_polygon_image(res)
    if poly is None:
        return
    layer = img.copy()
    cv2.fillPoly(layer, [poly.astype(np.int32)], STATE_COLOR[state])
    cv2.addWeighted(layer, 0.30, img, 0.70, 0, img)                     # alpha blend
    n = len(poly) // 2
    cv2.polylines(img, [poly[:n].astype(np.int32)], False, (0, 0, 255), 4, cv2.LINE_AA)       # left line
    cv2.polylines(img, [poly[n:].astype(np.int32)], False, (255, 120, 0), 4, cv2.LINE_AA)     # right line


def draw_detections(img: np.ndarray, dets: list[Detection], cmd: Command) -> None:
    """Boxes with label + range. In-path objects get a thick red outline."""
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = d.box
        col = LABEL_COLOR.get(d.label, WHITE)
        thick = 3 if i in cmd.in_path else 1
        cv2.rectangle(img, (x1, y1), (x2, y2), RED if i in cmd.in_path else col, thick)
        tag = d.label + (f":{d.state}" if d.state else "")
        if np.isfinite(d.distance_m):
            tag += f" {d.distance_m:.0f}m"
        _text(img, tag, (x1, max(14, y1 - 6)), 0.5, col)


def draw_steering_wheel(img: np.ndarray, center: tuple[int, int], radius: int, angle_deg: float) -> None:
    """A little steering wheel rotated by ``angle_deg`` (x ~12: gives a visible lock)."""
    a = np.radians(angle_deg * 12)                  # exaggerate for visibility
    cv2.circle(img, center, radius, (40, 40, 40), -1, cv2.LINE_AA)
    cv2.circle(img, center, radius, WHITE, 3, cv2.LINE_AA)
    for k in (0, 2.094, 4.189):                      # three spokes 120 deg apart
        p = (int(center[0] + radius * 0.95 * np.sin(a + k + np.pi)), int(center[1] - radius * 0.95 * np.cos(a + k + np.pi)))
        cv2.line(img, center, p, WHITE, 3, cv2.LINE_AA)
    top = (int(center[0] + radius * np.sin(a)), int(center[1] - radius * np.cos(a)))
    cv2.circle(img, top, 5, RED, -1, cv2.LINE_AA)    # red marker = "12 o'clock"


def draw_minimap(img: np.ndarray, det: LaneDetector, res: LaneResult, origin: tuple[int, int], size=(240, 270)) -> None:
    """Bird's-eye inset: thresholded mask + fitted lane lines + car marker."""
    w, h = size
    if res.warped_binary is None:
        return
    wb = cv2.cvtColor(res.warped_binary, cv2.COLOR_GRAY2BGR)
    if res.detected:
        ys = np.arange(0, wb.shape[0], 8)
        for fit, col in ((res.left_fit, (0, 0, 255)), (res.right_fit, (255, 120, 0)), ((res.left_fit + res.right_fit) / 2, GREEN)):
            pts = np.column_stack([np.polyval(fit, ys), ys]).astype(np.int32)
            cv2.polylines(wb, [pts], False, col, 5, cv2.LINE_AA)
    # crop the interesting central band of the warped image and shrink it
    ww = wb.shape[1]
    crop = wb[:, ww // 2 - 260: ww // 2 + 260]
    cv2.rectangle(crop, (crop.shape[1] // 2 - 12, crop.shape[0] - 40), (crop.shape[1] // 2 + 12, crop.shape[0] - 2), WHITE, -1)  # ego car
    mini = cv2.resize(crop, (w, h))
    x0, y0 = origin
    img[y0:y0 + h, x0:x0 + w] = mini
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), WHITE, 2)
    _text(img, "bird's-eye", (x0 + 6, y0 + 18), 0.5)


def draw_hud(img: np.ndarray, det: LaneDetector, res: LaneResult, cmd: Command, fps: float, n_dets: int) -> None:
    """Top banner (state + reason), telemetry panel and steering wheel."""
    H, W = img.shape[:2]
    col = STATE_COLOR[cmd.state]
    # --- top banner ---------------------------------------------------------
    cv2.rectangle(img, (0, 0), (W, 46), (25, 25, 25), -1)
    cv2.rectangle(img, (0, 0), (14, 46), col, -1)
    _text(img, f"{cmd.state.value}", (26, 32), 0.95, col, 2)
    _text(img, cmd.reason, (170, 31), 0.7)
    _text(img, f"{fps:4.1f} FPS | {n_dets} obj", (W - 210, 31), 0.55)
    # --- telemetry panel ----------------------------------------------------
    px, py = 14, 62
    panel = img.copy()
    cv2.rectangle(panel, (px - 6, py - 4), (px + 300, py + 150), (20, 20, 20), -1)
    cv2.addWeighted(panel, 0.55, img, 0.45, 0, img)
    if res.detected:
        direction = "straight" if res.radius_m > 2000 else ("right" if res.curvature_m > 0 else "left")
        rad = "inf" if res.radius_m > 2000 else f"{res.radius_m:.0f} m"
        side = "centred" if abs(res.offset_m) < 0.05 else (f"{abs(res.offset_m):.2f} m {'right' if res.offset_m > 0 else 'left'}")
        lines = [f"Lane      : {'coasting' if res.coasting else 'locked'}", f"Curve     : {direction}  R={rad}",
                 f"Position  : {side}", f"Lane width: {res.lane_width_m:.2f} m"]
    else:
        lines = ["Lane      : LOST", "Curve     : -", "Position  : -", "Lane width: -"]
    lines += [f"Speed     : {cmd.target_speed_kmh:4.0f} km/h", f"Steering  : {cmd.steering_deg:+5.1f} deg"]
    for i, s in enumerate(lines):
        _text(img, s, (px, py + 18 + i * 22), 0.55)
    # --- steering wheel + mini-map ------------------------------------------
    draw_steering_wheel(img, (W - 90, H - 90), 56, cmd.steering_deg)
    draw_minimap(img, det, res, (W - 250, 60))


def render(frame: np.ndarray, det: LaneDetector, res: LaneResult, dets: list[Detection],
           cmd: Command, fps: float) -> np.ndarray:
    """Compose the final annotated frame (returns a new image)."""
    out = frame.copy()
    draw_lane(out, det, res, cmd.state)
    draw_detections(out, dets, cmd)
    draw_hud(out, det, res, cmd, fps, len(dets))
    return out
