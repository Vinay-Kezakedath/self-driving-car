"""Classical computer-vision lane detection (OpenCV + NumPy only).

Pipeline, one call to :meth:`LaneDetector.process` per frame::

    BGR frame
      │ 1. threshold()      colour (white / yellow) + Sobel-x edges  -> binary mask
      │ 2. warp             perspective transform -> bird's-eye view
      │ 3. find_lane_pixels histogram + sliding windows (or fast "search around
      │                     previous fit") -> pixels belonging to left / right line
      │ 4. fit              2nd-order polynomial x = A*y^2 + B*y + C per line,
      │                     sanity-checked and temporally smoothed
      │ 5. measure          curvature radius (m) + vehicle offset from lane centre (m)
      ▼ 6. LaneResult       (polygon for drawing, numbers for the planner)

Coordinate conventions (important for the maths!):
    * "warped" image: x to the right, y downwards, the car is at the BOTTOM
      centre (x = warp_w/2, y = warp_h-1). Moving forward = decreasing y.
    * Signed curvature: positive = road bends to the RIGHT, negative = LEFT.
    * Vehicle offset: positive = car is to the RIGHT of the lane centre.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import LaneConfig


@dataclass
class LaneResult:
    """Everything the rest of the system needs to know about the lane."""

    detected: bool                       # True if a usable lane model exists
    left_fit: np.ndarray | None = None   # polynomial coeffs (A, B, C) in warped px
    right_fit: np.ndarray | None = None
    curvature_m: float = float("inf")    # signed 1/R: >0 right bend. inf radius = straight
    radius_m: float = float("inf")       # |R| in metres
    offset_m: float = 0.0                # + = car right of centre
    heading_rad: float = 0.0             # lane direction relative to the car (+ = lane goes right)
    lane_width_m: float = 0.0
    binary: np.ndarray | None = None     # thresholded mask (original view) - for debugging
    warped_binary: np.ndarray | None = None  # bird's-eye mask - for debugging
    coasting: bool = False               # True when re-using an old fit (lane lost)
    left_pts: np.ndarray | None = field(default=None, repr=False)   # pixels used for the fit
    right_pts: np.ndarray | None = field(default=None, repr=False)


class LaneDetector:
    """Stateful detector (state = previous fit, used for tracking + smoothing)."""

    def __init__(self, cfg: LaneConfig | None = None, frame_size: tuple[int, int] = (1280, 720)):
        self.cfg = cfg or LaneConfig()
        self.frame_w, self.frame_h = frame_size
        self._build_transform()
        self._left: np.ndarray | None = None     # smoothed left polynomial
        self._right: np.ndarray | None = None
        self._lost = 0                           # consecutive frames without a good fit

    # ------------------------------------------------------------ transform --
    def _build_transform(self) -> None:
        """Compute the homography M (image -> bird's-eye) and its inverse.

        The trapezoid corners in ``cfg.src`` are defined for ``cfg.ref_size``;
        we rescale them if the video has a different resolution.
        """
        rw, rh = self.cfg.ref_size
        sx, sy = self.frame_w / rw, self.frame_h / rh
        src = np.float32([(x * sx, y * sy) for x, y in self.cfg.src])
        dst = np.float32(self.cfg.dst)
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        self.src_poly = src          # kept for drawing the calibration ROI

    # ------------------------------------------------------------ 1. threshold --
    @staticmethod
    def threshold(frame: np.ndarray) -> np.ndarray:
        """Return a uint8 mask (0/255) of pixels that look like lane paint.

        Three cues are OR-ed together so that one failing (e.g. white paint in
        shadow) is compensated by another:
          * white paint   : high lightness in HLS space,
          * yellow paint  : hue window in HSV space,
          * vertical edges: Sobel-x of the lightness channel (paint edges are
                            vertical in the road image), only where bright enough.
        """
        blur = cv2.GaussianBlur(frame, (5, 5), 0)          # kill asphalt grain
        hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # white: any hue, light, low-ish saturation; threshold adapts to exposure
        l = hls[:, :, 1]
        white_thr = max(150, int(np.percentile(l[l.shape[0] // 2:], 99.0) * 0.82))
        white = cv2.inRange(hls, (0, white_thr, 0), (255, 255, 110))
        # yellow: OpenCV hue is 0..179 (yellow ~ 20-35)
        yellow = cv2.inRange(hsv, (15, 90, 100), (38, 255, 255))
        # gradient: horizontal derivative of lightness, normalised to 0..255
        sobel = np.abs(cv2.Sobel(l, cv2.CV_32F, 1, 0, ksize=3))
        sobel = np.uint8(255 * sobel / max(sobel.max(), 1e-6))
        edges = cv2.inRange(sobel, 60, 255) & cv2.inRange(l, 110, 255)

        mask = white | yellow | edges
        # crop away everything above the horizon region: lane lines never live there
        mask[: int(mask.shape[0] * 0.55)] = 0
        return mask

    def warp(self, img: np.ndarray, flags: int = cv2.INTER_LINEAR) -> np.ndarray:
        """Bird's-eye view of ``img``."""
        return cv2.warpPerspective(img, self.M, self.cfg.warp_size, flags=flags)

    # ----------------------------------------------------- 3. find lane pixels --
    def _histogram_bases(self, wb: np.ndarray) -> tuple[int | None, int | None]:
        """Find the starting x of the left and right lane line.

        We sum the bottom half of the bird's-eye mask column-wise: lane lines
        show up as peaks. Instead of "the biggest peak on each half" (which
        would grab e.g. a solid yellow line two lanes away) we choose, on each
        side of the car, the peak CLOSEST to the car's centre = the ego lane.
        """
        h, w = wb.shape
        hist = (wb[h // 2:] > 0).sum(axis=0).astype(np.float32)
        hist = np.convolve(hist, np.ones(15, np.float32) / 15, mode="same")  # smooth
        thr = max(8.0, 0.18 * hist.max())
        # local maxima above threshold (a peak = max within +-25 px)
        peaks = [x for x in range(25, w - 25)
                 if hist[x] >= thr and hist[x] == hist[x - 25: x + 26].max()]
        centre = w // 2
        left = [p for p in peaks if p < centre]
        right = [p for p in peaks if p >= centre]
        lb = max(left) if left else None          # nearest to the centre
        rb = min(right) if right else None
        return lb, rb

    def _sliding_window(self, wb: np.ndarray, x_base: int):
        """Walk up the image in ``n_windows`` slices, re-centring a window on
        the mean x of the lit pixels it contains. Returns pixel coords (ys, xs)."""
        c = self.cfg
        h = wb.shape[0]
        nz_y, nz_x = wb.nonzero()
        win_h = h // c.n_windows
        x_cur = x_base
        keep = []
        for i in range(c.n_windows):
            y_hi, y_lo = h - i * win_h, h - (i + 1) * win_h
            sel = ((nz_y >= y_lo) & (nz_y < y_hi) &
                   (nz_x >= x_cur - c.margin) & (nz_x < x_cur + c.margin)).nonzero()[0]
            keep.append(sel)
            if len(sel) > c.minpix:                 # enough evidence -> follow the line
                x_cur = int(nz_x[sel].mean())
        keep = np.concatenate(keep)
        return nz_y[keep], nz_x[keep]

    def _search_around_fit(self, wb: np.ndarray, fit: np.ndarray):
        """Fast path: keep lit pixels within +-margin of last frame's polynomial."""
        nz_y, nz_x = wb.nonzero()
        x_pred = np.polyval(fit, nz_y)
        sel = np.abs(nz_x - x_pred) < self.cfg.margin
        return nz_y[sel], nz_x[sel]

    # ------------------------------------------------------------ 4. fitting --
    @staticmethod
    def _fit(ys: np.ndarray, xs: np.ndarray) -> np.ndarray | None:
        """Least-squares x = A*y^2 + B*y + C; None if too few / too narrow data."""
        if len(ys) < 150 or np.ptp(ys) < 120:
            return None
        return np.polyfit(ys, xs, 2)

    def _plausible(self, left: np.ndarray, right: np.ndarray) -> bool:
        """Reject fits that violate basic road geometry:
        lane width must be roughly constant and close to the expected one."""
        h = self.cfg.warp_size[1]
        ys = np.array([h - 1, h * 0.5, h * 0.1])
        widths = np.polyval(right, ys) - np.polyval(left, ys)
        exp = self.cfg.lane_width_px
        return bool(np.all(widths > 0.6 * exp) and np.all(widths < 1.5 * exp))

    def _smooth(self, old: np.ndarray | None, new: np.ndarray) -> np.ndarray:
        """Exponential moving average of polynomial coefficients (jitter filter)."""
        if old is None:
            return new
        a = self.cfg.smooth_alpha
        return a * new + (1 - a) * old

    # ---------------------------------------------------------- 5. measuring --
    def _measure(self, left: np.ndarray, right: np.ndarray) -> dict:
        """Convert the pixel polynomials to real-world numbers."""
        c = self.cfg
        h = c.warp_size[1]
        xm, ym = c.xm_per_pix, c.ym_per_pix
        centre_fit = (left + right) / 2.0          # the lane centre-line polynomial
        y_car = h - 1

        # pixel -> metre: x_m = xm*x, y_m = ym*y  =>  A_m = A*xm/ym^2, B_m = B*xm/ym
        A, B = centre_fit[0] * xm / ym ** 2, centre_fit[1] * xm / ym
        y_m = y_car * ym
        slope = 2 * A * y_m + B                    # dx/dy in metres at the car
        # Standard curvature formula  k = x'' / (1 + x'^2)^(3/2)
        kappa = 2 * A / (1 + slope ** 2) ** 1.5    # + = bends right (see module doc)

        lane_centre_px = (np.polyval(left, y_car) + np.polyval(right, y_car)) / 2.0
        car_px = c.warp_size[0] / 2.0              # camera is at the image centre
        offset = (car_px - lane_centre_px) * xm    # + = car right of lane centre
        width = (np.polyval(right, y_car) - np.polyval(left, y_car)) * xm
        # y grows downward, "forward" is -y, so lane heading = -dx/dy
        heading = float(np.arctan(-slope))
        return dict(curvature=float(kappa), offset=float(offset), heading=heading, width=float(width))

    # --------------------------------------------------------------- driver --
    def process(self, frame: np.ndarray) -> LaneResult:
        """Run the whole pipeline on one BGR frame."""
        binary = self.threshold(frame)
        wb = self.warp(binary, cv2.INTER_NEAREST)

        left_pts = right_pts = None
        # --- fast tracking mode: reuse last fit when we have a recent one ----
        if self._left is not None and self._right is not None and self._lost == 0:
            ly, lx = self._search_around_fit(wb, self._left)
            ry, rx = self._search_around_fit(wb, self._right)
            lf, rf = self._fit(ly, lx), self._fit(ry, rx)
            if lf is not None and rf is not None:
                left_pts, right_pts = (ly, lx), (ry, rx)
        # --- otherwise (re)acquire with histogram + sliding windows ----------
        if left_pts is None:
            lb, rb = self._histogram_bases(wb)
            w = self.cfg.lane_width_px
            # If only one line is visible, hallucinate the other one lane away.
            if lb is None and rb is not None:
                lb = int(rb - w)
            if rb is None and lb is not None:
                rb = int(lb + w)
            if lb is not None and rb is not None:
                ly, lx = self._sliding_window(wb, lb)
                ry, rx = self._sliding_window(wb, rb)
                left_pts, right_pts = (ly, lx), (ry, rx)

        new_l = new_r = None
        if left_pts is not None:
            new_l, new_r = self._fit(*left_pts), self._fit(*right_pts)
            # one-sided: if just one polynomial succeeded, mirror it by a lane width
            if new_l is None and new_r is not None:
                new_l = new_r.copy(); new_l[2] -= self.cfg.lane_width_px
            elif new_r is None and new_l is not None:
                new_r = new_l.copy(); new_r[2] += self.cfg.lane_width_px

        ok = new_l is not None and new_r is not None and self._plausible(new_l, new_r)
        if ok:
            self._left = self._smooth(self._left, new_l)
            self._right = self._smooth(self._right, new_r)
            self._lost = 0
        else:
            self._lost += 1
            if self._lost > self.cfg.max_lost_frames:   # give up: forget the model
                self._left = self._right = None

        if self._left is None or self._right is None:
            return LaneResult(detected=False, binary=binary, warped_binary=wb)

        m = self._measure(self._left, self._right)
        return LaneResult(
            detected=True, left_fit=self._left, right_fit=self._right,
            curvature_m=m["curvature"], radius_m=(1 / abs(m["curvature"]) if abs(m["curvature"]) > 1e-6 else float("inf")),
            offset_m=m["offset"], heading_rad=m["heading"], lane_width_m=m["width"],
            binary=binary, warped_binary=wb, coasting=not ok,
            left_pts=None if left_pts is None else np.column_stack(left_pts[::-1]),
            right_pts=None if right_pts is None else np.column_stack(right_pts[::-1]))

    # ----------------------------------------------------- geometry helpers --
    def lane_polygon_image(self, res: LaneResult, n: int = 40) -> np.ndarray | None:
        """Lane area as an (N,2) polygon in ORIGINAL image coordinates."""
        if not res.detected:
            return None
        ys = np.linspace(0, self.cfg.warp_size[1] - 1, n)
        lx, rx = np.polyval(res.left_fit, ys), np.polyval(res.right_fit, ys)
        pts = np.vstack([np.column_stack([lx, ys]), np.column_stack([rx, ys])[::-1]]).astype(np.float32)
        return cv2.perspectiveTransform(pts[None], self.Minv)[0]

    def point_in_lane(self, res: LaneResult, x: float, y: float, margin_m: float = 0.0) -> bool | None:
        """Is image point (x, y) inside the ego lane? None if undecidable
        (no lane model, or the point lies outside the warped region)."""
        if not res.detected:
            return None
        wx, wy = cv2.perspectiveTransform(np.float32([[[x, y]]]), self.M)[0, 0]
        if not (0 <= wy < self.cfg.warp_size[1]):
            return None
        m = margin_m / self.cfg.xm_per_pix
        return bool(np.polyval(res.left_fit, wy) - m <= wx <= np.polyval(res.right_fit, wy) + m)
