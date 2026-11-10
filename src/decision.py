"""Behaviour planning + steering control.

Input : lane geometry (``LaneResult``) and the list of ``Detection`` objects.
Output: a :class:`Command` = (state, target speed, steering angle, reason).

The planner is a small, readable *rule-based* state machine (not learned),
which makes every decision explainable on the HUD:

    priority   rule
    --------   ---------------------------------------------------------------
    1 (STOP)   red/amber traffic light closer than ``traffic_light_stop_m``
    2 (STOP)   obstacle in our lane closer than ``stop_distance_m``
    3 (STOP)   pedestrian in/next to our lane closer than ``pedestrian_stop_m``
    4 (SLOW)   obstacle in our lane closer than ``slow_distance_m``
    5 (SLOW)   tight curve (radius < ``curve_radius_slow_m``)
    6 (SLOW)   lane lost (no model) -> creep
    7 (DRIVE)  otherwise cruise

Steering uses the *Stanley controller* idea (used by Stanford's DARPA winner)::

    delta = heading_error + atan( k * cross_track_error / (v + eps) ) + feedforward

with feedforward = atan(wheelbase * curvature) so the car anticipates bends.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from .config import DriveConfig
from .lane_detection import LaneDetector, LaneResult
from .object_detection import VEHICLES, Detection


class State(str, Enum):
    DRIVE = "DRIVE"
    SLOW = "SLOW"
    STOP = "STOP"


@dataclass
class Command:
    """Output of the planner for the current frame."""

    state: State
    target_speed_kmh: float
    steering_deg: float                 # + = steer right, - = left
    reason: str
    lead: Detection | None = None       # closest in-path obstacle (for the HUD)
    in_path: list = field(default_factory=list)   # ids (indices) of in-path detections


class DrivingPlanner:
    """Stateful: keeps the last steering value to rate-limit it, and the last
    commanded speed to apply acceleration limits (comfort + realism)."""

    def __init__(self, cfg: DriveConfig | None = None, lane: LaneDetector | None = None, fps: float = 30.0):
        self.cfg = cfg or DriveConfig()
        self.lane = lane
        self.dt = 1.0 / fps
        self.speed_kmh = 0.0          # "virtual" ego speed after accel limits
        self.steer_deg = 0.0

    # ------------------------------------------------------------- perception --
    def _in_path(self, res: LaneResult, d: Detection, frame_w: int) -> bool:
        """Is this object in (or about to enter) our lane?

        Primary test: project the object's ground-contact point into the
        bird's-eye view and compare with the lane polynomials. For objects
        beyond the warped region (too far) fall back to a central corridor
        in the image (middle 30 % of the width).
        """
        x, y = d.bottom_center
        if self.lane is not None:
            inside = self.lane.point_in_lane(res, x, y, margin_m=0.3 if d.label == "person" else 0.0)
            if inside is not None:
                return inside
        return abs(x - frame_w / 2) < 0.15 * frame_w

    # ---------------------------------------------------------------- steering --
    def _steering(self, res: LaneResult) -> float:
        """Stanley-style steering angle in degrees (+ right)."""
        if not res.detected:
            return 0.0                                     # no lane: go straight
        c = self.cfg
        v = max(self.speed_kmh / 3.6, 2.0)                 # m/s, floor avoids /0 at standstill
        cte = -res.offset_m                                # + when lane centre is to our right
        correction = math.atan2(c.stanley_gain * cte, v)   # pulls us back to the centre
        feedforward = math.atan(c.wheelbase_m * res.curvature_m)   # anticipate the bend
        delta = math.degrees(res.heading_rad + correction + feedforward)
        return max(-c.max_steer_deg, min(c.max_steer_deg, delta))

    # ---------------------------------------------------------------- planning --
    def plan(self, res: LaneResult, dets: list[Detection], frame_w: int) -> Command:
        c = self.cfg
        state, target, reason = State.DRIVE, c.cruise_speed_kmh, "lane clear"
        lead: Detection | None = None
        in_path_ids: list[int] = []

        # --- which detections matter? (ordered nearest first) ----------------
        order = sorted(range(len(dets)), key=lambda i: dets[i].distance_m)
        for i in order:
            d = dets[i]
            if d.label in VEHICLES or d.label == "person":
                if self._in_path(res, d, frame_w):
                    in_path_ids.append(i)
                    if lead is None:
                        lead = d

        def escalate(new_state: State, speed: float, why: str) -> None:
            """Keep the MOST restrictive state (STOP > SLOW > DRIVE)."""
            nonlocal state, target, reason
            rank = {State.DRIVE: 0, State.SLOW: 1, State.STOP: 2}
            if rank[new_state] > rank[state] or (new_state == state and speed < target):
                state, target, reason = new_state, speed, why

        # --- rule: traffic lights ------------------------------------------
        for d in dets:
            if d.label == "traffic light" and d.state in ("red", "amber") and d.distance_m < c.traffic_light_stop_m:
                escalate(State.STOP, 0.0, f"{d.state} light {d.distance_m:.0f} m")
            if d.label == "stop sign" and d.distance_m < c.traffic_light_stop_m:
                escalate(State.STOP, 0.0, f"stop sign {d.distance_m:.0f} m")

        # --- rule: obstacle in path ---------------------------------------
        if lead is not None:
            dist = lead.distance_m
            stop_at = c.pedestrian_stop_m if lead.label == "person" else c.stop_distance_m
            if dist < stop_at:
                escalate(State.STOP, 0.0, f"{lead.label} in path {dist:.0f} m")
            elif dist < c.slow_distance_m:
                # scale the speed with the gap: closer -> slower (linear ramp)
                frac = (dist - stop_at) / max(c.slow_distance_m - stop_at, 1e-6)
                escalate(State.SLOW, max(5.0, c.slow_speed_kmh * frac + 5.0), f"{lead.label} ahead {dist:.0f} m")

        # --- rule: curve / lane state -------------------------------------
        if not res.detected:
            escalate(State.SLOW, c.slow_speed_kmh, "lane lost")
        elif res.radius_m < c.curve_radius_slow_m:
            # v = sqrt(a_lat * R) with a comfortable lateral accel of 2.5 m/s^2
            v_curve = math.sqrt(2.5 * res.radius_m) * 3.6
            escalate(State.SLOW, max(15.0, min(c.cruise_speed_kmh, v_curve)), f"curve R={res.radius_m:.0f} m")

        # --- apply comfort limits to speed and steering ---------------------
        dv = target - self.speed_kmh
        self.speed_kmh += max(-c.max_decel_kmh_s * self.dt, min(c.max_accel_kmh_s * self.dt, dv))
        steer = self._steering(res)
        ds = steer - self.steer_deg
        lim = c.max_steer_rate_deg_s * self.dt
        self.steer_deg += max(-lim, min(lim, ds))

        return Command(state, self.speed_kmh, self.steer_deg, reason, lead, in_path_ids)
