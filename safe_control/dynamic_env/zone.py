#!/usr/bin/env python3
"""
finish_zone_sweep.py
====================
Compact lawnmower sweep placed ENTIRELY IN FRONT of the robot at the
moment it reaches the goal.  The sweep rectangle is oriented along the
robot's heading so it never goes behind the arrival point.

  Robot arrives here (R), heading right (yaw=0):

        R ─────────────┐
        │  row 1  ───► │
        │  ◄───  row 2 │
        │  row 3  ───► │
        └──────────────┘
            2m forward
"""

import math
from typing import List, Tuple, Optional

import numpy as np
import rospy
from geometry_msgs.msg import Twist


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class FinishZoneSweep:
    """
    Tiny lawnmower sweep placed in front of the robot.

    The rectangle extends from the robot's current position forward
    by ``2 * forward_dist`` along its heading, and ``half_h`` to each
    side perpendicular to the heading.

    Parameters
    ----------
    robot_pos : (x, y)
        Robot position when the sweep begins.
    robot_yaw : float
        Robot heading in radians when the sweep begins.
    forward_dist : float
        How far forward the rectangle extends (metres).
    half_w : float
        Not used directly — kept for param compatibility.  Forward
        extent is controlled by ``forward_dist``.
    half_h : float
        Half-width perpendicular to heading (metres).  Default 1.0 ->
        2m side-to-side coverage.
    num_rows : int
        Number of passes across the rectangle.
    speed : float
        Forward speed during the sweep (m/s).
    wp_tol : float
        Distance to consider a waypoint reached (m).
    k_omega : float
        Proportional gain for heading correction.
    w_max : float
        Maximum angular velocity (rad/s).
    """

    def __init__(
        self,
        robot_pos: Tuple[float, float],
        robot_yaw: float,
        forward_dist: float = 1.0,
        half_w: float = 1.0,
        half_h: float = 1.0,
        num_rows: int = 3,
        speed: float = 0.5,
        wp_tol: float = 0.3,
        k_omega: float = 2.5,
        w_max: float = 1.5,
    ):
        self.speed = speed
        self.wp_tol = wp_tol
        self.k_omega = k_omega
        self.w_max = w_max

        rx, ry = robot_pos

        # Unit vectors: forward along heading, lateral perpendicular
        fwd_x = math.cos(robot_yaw)
        fwd_y = math.sin(robot_yaw)
        lat_x = -math.sin(robot_yaw)   # 90 deg left of heading
        lat_y =  math.cos(robot_yaw)

        # The rectangle goes from the robot position forward by 2*forward_dist,
        # and half_h to each side (perpendicular to heading).
        # "along" ranges from 0 (at robot) to 2*forward_dist (far end)
        # "across" ranges from -half_h to +half_h

        along_vals  = np.linspace(0.2, 2.0 * forward_dist, num_rows)
        across_left  = -half_h
        across_right =  half_h

        self.waypoints: List[Tuple[float, float]] = []

        for i, along in enumerate(along_vals):
            if i % 2 == 0:
                # left to right
                p1 = (rx + fwd_x * along + lat_x * across_left,
                      ry + fwd_y * along + lat_y * across_left)
                p2 = (rx + fwd_x * along + lat_x * across_right,
                      ry + fwd_y * along + lat_y * across_right)
            else:
                # right to left
                p1 = (rx + fwd_x * along + lat_x * across_right,
                      ry + fwd_y * along + lat_y * across_right)
                p2 = (rx + fwd_x * along + lat_x * across_left,
                      ry + fwd_y * along + lat_y * across_left)

            self.waypoints.append(p1)
            self.waypoints.append(p2)

        # End at the center of the far edge (straight ahead)
        end_x = rx + fwd_x * 2.0 * forward_dist
        end_y = ry + fwd_y * 2.0 * forward_dist
        self.waypoints.append((end_x, end_y))

        self.current_wp_idx = 0
        self.is_complete = False

        rospy.loginfo(
            f"[FinishSweep] {len(self.waypoints)} waypoints, "
            f"heading={math.degrees(robot_yaw):.1f}deg, "
            f"forward={2*forward_dist:.1f}m, lateral=+/-{half_h:.1f}m")
        for idx, (wx, wy) in enumerate(self.waypoints):
            rospy.loginfo(f"  wp[{idx}] = ({wx:.2f}, {wy:.2f})")

    def compute_cmd(self, robot_x: float, robot_y: float,
                    robot_yaw: float) -> Optional[Twist]:
        """Return Twist to drive toward current waypoint, or None if done."""
        if self.is_complete or self.current_wp_idx >= len(self.waypoints):
            self.is_complete = True
            return None

        wx, wy = self.waypoints[self.current_wp_idx]
        dx = wx - robot_x
        dy = wy - robot_y
        dist = math.hypot(dx, dy)

        # Waypoint reached - advance
        if dist < self.wp_tol:
            rospy.loginfo(
                f"[FinishSweep] wp {self.current_wp_idx}/"
                f"{len(self.waypoints)-1} reached")
            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.waypoints):
                self.is_complete = True
                rospy.loginfo("[FinishSweep] Sweep complete!")
                return None
            wx, wy = self.waypoints[self.current_wp_idx]
            dx = wx - robot_x
            dy = wy - robot_y
            dist = math.hypot(dx, dy)

        # Proportional steering toward waypoint
        desired_yaw = math.atan2(dy, dx)
        heading_err = normalize_angle(desired_yaw - robot_yaw)

        omega = self.k_omega * heading_err
        omega = max(-self.w_max, min(self.w_max, omega))

        # Slow down on sharp turns
        speed = self.speed * max(0.2, math.cos(heading_err))

        cmd = Twist()
        cmd.linear.x = max(0.05, speed)
        cmd.angular.z = omega

        rospy.loginfo_throttle(1.0,
            f"[FinishSweep] wp={self.current_wp_idx}/{len(self.waypoints)-1} "
            f"dist={dist:.2f} herr={math.degrees(heading_err):.0f}deg")
        return cmd