"""Shared configuration: camera model, lane geometry and driving parameters."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CameraConfig:
    """Pin-hole camera model (also used by the synthetic road simulator)."""

    width: int = 1280
    height: int = 720
    focal_px: float = 600.0      # focal length in pixels
    height_m: float = 1.3        # camera height above the road
    horizon_y: int = 380         # image row of the horizon


@dataclass(frozen=True)
class LaneConfig:
    """Perspective transform + lane-fitting parameters.

    ``src`` is a trapezoid (as fractions of the image) lying on the road plane
    in front of the car. It is mapped to a rectangle (bird's-eye view).
    The defaults match the synthetic sample video; for a real dash-cam
    video, check/tune them with ``calibrate.py`` (see README).
    """

    # trapezoid corners as (x, y) in *pixels* for a 1280x720 frame:
    # top-left, top-right, bottom-right, bottom-left
    src: tuple = ((586, 420), (694, 420), (1077, 700), (203, 700))
    # destination rectangle (in the warped image): lane spans x 520..760
    dst: tuple = ((520, 0), (760, 0), (760, 720), (520, 720))
    lane_width_m: float = 3.5
    ref_size: tuple = (1280, 720)      # frame size the src/dst were defined for
    warp_size: tuple = (1280, 720)
    ym_per_pix: float = 17.0 / 720     # metres per warped pixel (forward)
    n_windows: int = 12
    margin: int = 70
    minpix: int = 40
    smooth_alpha: float = 0.35         # EMA factor for polynomial smoothing
    max_lost_frames: int = 15          # frames to coast on the last good fit

    @property
    def xm_per_pix(self) -> float:
        return self.lane_width_m / (self.dst[1][0] - self.dst[0][0])

    @property
    def lane_width_px(self) -> float:
        return float(self.dst[1][0] - self.dst[0][0])


@dataclass(frozen=True)
class DriveConfig:
    """Decision / control parameters."""

    cruise_speed_kmh: float = 50.0
    slow_speed_kmh: float = 20.0
    stop_distance_m: float = 9.0       # in-path obstacle closer than this -> STOP
    slow_distance_m: float = 28.0      # in-path obstacle closer than this -> SLOW
    pedestrian_stop_m: float = 12.0
    traffic_light_stop_m: float = 35.0
    curve_radius_slow_m: float = 150.0  # tighter than this -> slow for the curve
    wheelbase_m: float = 2.7
    max_steer_deg: float = 35.0
    stanley_gain: float = 1.2
    max_accel_kmh_s: float = 10.0
    max_decel_kmh_s: float = 30.0
    max_steer_rate_deg_s: float = 60.0


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    lane: LaneConfig = field(default_factory=LaneConfig)
    drive: DriveConfig = field(default_factory=DriveConfig)
