#!/usr/bin/env python3
"""
occupancy_memory_map.py
=======================
Lightweight 2D occupancy grid that accumulates LIDAR wall detections over
time to build a persistent spatial memory of the enclosure.

Purpose
-------
When the robot lacks an IMU, odometry drift during aggressive turns inside
the enclosure corrupts the heading estimate.  By accumulating wall hits into
an occupancy grid, the robot can:

  1.  Detect the enclosure walls (top, bottom, left) and the open side.
  2.  Estimate the vertical (Y) extent of the opening on the right side.
  3.  Compute the centre of the opening as the true exit waypoint.
  4.  Track which edge the robot is exiting from, so the final goal can
      be adjusted even if odometry-based heading is wrong.

Integration
-----------
Drop-in for ``cbf_qp_ros_node.py``.  The node creates an OccupancyMemoryMap
instance, feeds every LIDAR scan into ``update()``, and periodically calls
``get_exit_goal()`` to obtain a corrected goal that replaces the static
``self.goal``.

Coordinate convention
---------------------
Everything is in the **world / odom frame** (same frame as robot.X).
The grid covers a configurable bounding box around the expected workspace.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import rospy


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WallEstimate:
    """Axis-aligned wall segment detected from the occupancy grid."""
    axis: str           # 'horizontal' or 'vertical'
    position: float     # coordinate along the perpendicular axis (e.g. y for horizontal)
    start: float        # start along the parallel axis
    end: float          # end along the parallel axis
    hit_count: int = 0  # number of cells that contributed


@dataclass
class OpeningEstimate:
    """Gap detected along the exit side of the enclosure."""
    y_min: float        # lower edge of the opening (world Y)
    y_max: float        # upper edge of the opening (world Y)
    x_position: float   # X coordinate of the exit wall
    center: Tuple[float, float] = field(init=False)

    def __post_init__(self):
        self.center = (self.x_position, (self.y_min + self.y_max) / 2.0)


# ---------------------------------------------------------------------------
# Occupancy grid
# ---------------------------------------------------------------------------

class OccupancyMemoryMap:
    """
    2D log-odds occupancy grid with wall / opening extraction.

    Parameters
    ----------
    origin : tuple (x, y)
        Bottom-left corner of the grid in world coordinates.
    size : tuple (width, height)
        Extent of the grid in metres.
    resolution : float
        Cell size in metres (default 0.1 m → 10 cm).
    log_odds_hit : float
        Log-odds increment when a cell is observed occupied.
    log_odds_miss : float
        Log-odds decrement for a free-space ray-cast cell (not used yet,
        but reserved for future ray tracing).
    log_odds_max / log_odds_min : float
        Clamping bounds for the log-odds values.
    wall_threshold : float
        Minimum occupancy probability to consider a cell as "wall".
    min_wall_length_cells : int
        Minimum contiguous run of occupied cells to qualify as a wall.
    exit_side : str
        Which side of the enclosure has the opening: 'right', 'left',
        'top', or 'bottom'.  Default 'right' (positive X direction).
    exit_goal_offset : float
        How far outside the opening (in metres) to place the exit goal.
    """

    def __init__(
        self,
        origin: Tuple[float, float] = (-5.0, -2.0),
        size: Tuple[float, float] = (20.0, 12.0),
        resolution: float = 0.1,
        log_odds_hit: float = 0.7,
        log_odds_miss: float = -0.3,
        log_odds_max: float = 5.0,
        log_odds_min: float = -2.0,
        wall_threshold: float = 0.6,
        min_wall_length_cells: int = 8,
        exit_side: str = 'right',
        exit_goal_offset: float = 1.5,
    ):
        self.origin = np.array(origin, dtype=float)
        self.size = np.array(size, dtype=float)
        self.resolution = resolution

        self.cols = int(math.ceil(size[0] / resolution))
        self.rows = int(math.ceil(size[1] / resolution))

        # Log-odds grid (initialised to 0 → P=0.5 = unknown)
        self._grid = np.zeros((self.rows, self.cols), dtype=np.float32)
        self._lock = threading.Lock()

        # Log-odds parameters
        self._l_hit = log_odds_hit
        self._l_miss = log_odds_miss
        self._l_max = log_odds_max
        self._l_min = log_odds_min

        # Wall extraction parameters
        self._wall_thresh = wall_threshold
        self._min_wall_len = min_wall_length_cells
        self._exit_side = exit_side
        self._exit_goal_offset = exit_goal_offset

        # Cached results (updated periodically, not every tick)
        self._walls: list[WallEstimate] = []
        self._opening: Optional[OpeningEstimate] = None
        self._exit_goal: Optional[np.ndarray] = None

        # Track robot trajectory for exit-direction estimation
        self._trajectory: list[Tuple[float, float]] = []
        self._max_traj_len = 2000  # keep last N poses

        # Update counter — only re-extract walls every K scans
        self._scan_count = 0
        self._extract_every = 20

        rospy.loginfo(
            f"[OccupancyMap] Grid {self.cols}x{self.rows} cells, "
            f"resolution={resolution}m, origin={origin}, size={size}, "
            f"exit_side={exit_side}"
        )

    # ------------------------------------------------------------------
    # World ↔ grid coordinate transforms
    # ------------------------------------------------------------------

    def _world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert world (x, y) to grid (col, row)."""
        col = int((wx - self.origin[0]) / self.resolution)
        row = int((wy - self.origin[1]) / self.resolution)
        return col, row

    def _grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """Convert grid (col, row) to world (x, y) — cell centre."""
        wx = self.origin[0] + (col + 0.5) * self.resolution
        wy = self.origin[1] + (row + 0.5) * self.resolution
        return wx, wy

    def _in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.cols and 0 <= row < self.rows

    # ------------------------------------------------------------------
    # Grid update
    # ------------------------------------------------------------------

    def update(self, obstacle_points_world: np.ndarray,
               robot_x: float, robot_y: float):
        """
        Feed a batch of obstacle points (Nx2 or Nx5, world frame) into the
        grid.  Also records the robot position for trajectory tracking.

        Parameters
        ----------
        obstacle_points_world : ndarray, shape (N, 2+)
            Each row is at least [x, y, ...].  Only the first two columns
            are used.
        robot_x, robot_y : float
            Current robot position in the world frame.
        """
        # Record trajectory
        self._trajectory.append((robot_x, robot_y))
        if len(self._trajectory) > self._max_traj_len:
            self._trajectory.pop(0)

        if obstacle_points_world.shape[0] == 0:
            return

        pts = obstacle_points_world[:, :2]

        with self._lock:
            for px, py in pts:
                c, r = self._world_to_grid(px, py)
                if self._in_bounds(c, r):
                    self._grid[r, c] = min(
                        self._grid[r, c] + self._l_hit, self._l_max)

        self._scan_count += 1
        if self._scan_count % self._extract_every == 0:
            self._extract_walls_and_opening()

    # ------------------------------------------------------------------
    # Probability query
    # ------------------------------------------------------------------

    def _prob(self, log_odds: float) -> float:
        """Convert log-odds to probability."""
        return 1.0 - 1.0 / (1.0 + math.exp(log_odds))

    def get_occupancy_prob(self, wx: float, wy: float) -> float:
        """Return occupancy probability at a world point."""
        c, r = self._world_to_grid(wx, wy)
        if not self._in_bounds(c, r):
            return 0.0
        return self._prob(self._grid[r, c])

    # ------------------------------------------------------------------
    # Wall and opening extraction
    # ------------------------------------------------------------------

    def _extract_walls_and_opening(self):
        """
        Analyse the occupancy grid to find walls and the exit opening.

        Strategy:
          1. Create a binary "wall" image from the occupancy grid.
          2. Project onto the X-axis (column sums) and Y-axis (row sums)
             to find dominant horizontal and vertical lines.
          3. For the exit side, scan along the exit wall to find the gap.
        """
        with self._lock:
            grid_copy = self._grid.copy()

        # Binary wall mask
        prob_grid = 1.0 - 1.0 / (1.0 + np.exp(grid_copy))
        wall_mask = (prob_grid > self._wall_thresh).astype(np.uint8)

        walls = []

        # ----- Detect horizontal walls (constant Y) -----
        # Sum each row along columns → high sum = horizontal wall
        row_sums = wall_mask.sum(axis=1)
        row_threshold = self._min_wall_len

        in_wall = False
        wall_start_row = 0
        for r in range(self.rows):
            if row_sums[r] >= row_threshold:
                if not in_wall:
                    in_wall = True
                    wall_start_row = r
            else:
                if in_wall:
                    in_wall = False
                    # Merge the band into one wall at the mean row
                    mid_row = (wall_start_row + r - 1) // 2
                    # Find the extent along X
                    band = wall_mask[wall_start_row:r, :]
                    col_hits = band.any(axis=0)
                    cols_with_hits = np.where(col_hits)[0]
                    if len(cols_with_hits) > 0:
                        _, wy = self._grid_to_world(0, mid_row)
                        x_start, _ = self._grid_to_world(cols_with_hits[0], 0)
                        x_end, _   = self._grid_to_world(cols_with_hits[-1], 0)
                        walls.append(WallEstimate(
                            axis='horizontal', position=wy,
                            start=x_start, end=x_end,
                            hit_count=int(col_hits.sum()),
                        ))
        if in_wall:
            mid_row = (wall_start_row + self.rows - 1) // 2
            band = wall_mask[wall_start_row:, :]
            col_hits = band.any(axis=0)
            cols_with_hits = np.where(col_hits)[0]
            if len(cols_with_hits) > 0:
                _, wy = self._grid_to_world(0, mid_row)
                x_start, _ = self._grid_to_world(cols_with_hits[0], 0)
                x_end, _   = self._grid_to_world(cols_with_hits[-1], 0)
                walls.append(WallEstimate(
                    axis='horizontal', position=wy,
                    start=x_start, end=x_end,
                    hit_count=int(col_hits.sum()),
                ))

        # ----- Detect vertical walls (constant X) -----
        col_sums = wall_mask.sum(axis=0)
        col_threshold = self._min_wall_len

        in_wall = False
        wall_start_col = 0
        for c in range(self.cols):
            if col_sums[c] >= col_threshold:
                if not in_wall:
                    in_wall = True
                    wall_start_col = c
            else:
                if in_wall:
                    in_wall = False
                    mid_col = (wall_start_col + c - 1) // 2
                    band = wall_mask[:, wall_start_col:c]
                    row_hits = band.any(axis=1)
                    rows_with_hits = np.where(row_hits)[0]
                    if len(rows_with_hits) > 0:
                        wx, _ = self._grid_to_world(mid_col, 0)
                        _, y_start = self._grid_to_world(0, rows_with_hits[0])
                        _, y_end   = self._grid_to_world(0, rows_with_hits[-1])
                        walls.append(WallEstimate(
                            axis='vertical', position=wx,
                            start=y_start, end=y_end,
                            hit_count=int(row_hits.sum()),
                        ))
        if in_wall:
            mid_col = (wall_start_col + self.cols - 1) // 2
            band = wall_mask[:, wall_start_col:]
            row_hits = band.any(axis=1)
            rows_with_hits = np.where(row_hits)[0]
            if len(rows_with_hits) > 0:
                wx, _ = self._grid_to_world(mid_col, 0)
                _, y_start = self._grid_to_world(0, rows_with_hits[0])
                _, y_end   = self._grid_to_world(0, rows_with_hits[-1])
                walls.append(WallEstimate(
                    axis='vertical', position=wx,
                    start=y_start, end=y_end,
                    hit_count=int(row_hits.sum()),
                ))

        self._walls = walls

        # ----- Find the opening on the exit side -----
        self._opening = self._find_opening(wall_mask, walls)
        if self._opening is not None:
            cx, cy = self._opening.center
            # Place the goal slightly outside the opening
            if self._exit_side == 'right':
                goal_x = cx + self._exit_goal_offset
                goal_y = cy
            elif self._exit_side == 'left':
                goal_x = cx - self._exit_goal_offset
                goal_y = cy
            elif self._exit_side == 'top':
                goal_x = cx
                goal_y = cy + self._exit_goal_offset
            else:  # bottom
                goal_x = cx
                goal_y = cy - self._exit_goal_offset

            self._exit_goal = np.array([goal_x, goal_y], dtype=float)

            rospy.loginfo_throttle(5.0,
                f"[OccupancyMap] Opening detected: "
                f"y=[{self._opening.y_min:.2f}, {self._opening.y_max:.2f}] "
                f"at x={self._opening.x_position:.2f}  "
                f"center=({cx:.2f}, {cy:.2f})  "
                f"exit_goal=({goal_x:.2f}, {goal_y:.2f})")
        else:
            rospy.loginfo_throttle(5.0,
                "[OccupancyMap] Opening not yet detected — need more scans.")

        if walls:
            rospy.loginfo_throttle(10.0,
                f"[OccupancyMap] Detected {len(walls)} wall segments")

    def _find_opening(self, wall_mask: np.ndarray,
                      walls: list[WallEstimate]) -> Optional[OpeningEstimate]:
        """
        Find the gap in the exit-side wall.

        For exit_side='right':
          - Identify the two longest horizontal walls (top and bottom of
            the enclosure).
          - The opening is where the right side has no vertical wall
            connecting them.
          - The Y-extent of the opening is determined by scanning the
            right-most column region for gaps.
        """
        # Find the two dominant horizontal walls (top and bottom boundaries)
        h_walls = [w for w in walls if w.axis == 'horizontal']
        if len(h_walls) < 2:
            return None

        # Sort by Y position
        h_walls.sort(key=lambda w: w.position)
        bottom_wall = h_walls[0]
        top_wall = h_walls[-1]

        if self._exit_side == 'right':
            # The exit X is roughly the maximum X extent of the horizontal walls
            # But the opening is where the wall *stops*.
            # Look at the right ends of top and bottom walls.
            top_right_x = top_wall.end
            bot_right_x = bottom_wall.end

            # The exit X position is the further-right of the two wall ends
            exit_x = max(top_right_x, bot_right_x)

            # Now scan the column at exit_x to find the gap.
            # The gap is the Y range between the two wall ends where there
            # is no occupied cell.
            exit_col, _ = self._world_to_grid(exit_x, 0)

            # Scan a band of columns around exit_x for robustness
            col_band = 5  # cells
            c_lo = max(0, exit_col - col_band)
            c_hi = min(self.cols, exit_col + col_band + 1)

            # For each row, check if ANY cell in the band is occupied
            band = wall_mask[:, c_lo:c_hi]
            row_occupied = band.any(axis=1)

            # Find gaps (contiguous runs of unoccupied rows)
            # within the Y range of the enclosure
            _, bot_row = self._world_to_grid(0, bottom_wall.position)
            _, top_row = self._world_to_grid(0, top_wall.position)

            # Ensure ordering (row index increases with Y)
            r_lo = min(bot_row, top_row)
            r_hi = max(bot_row, top_row)

            # Find the longest gap in the exit column band
            best_gap_start = None
            best_gap_len = 0
            gap_start = None

            for r in range(r_lo, r_hi + 1):
                if not row_occupied[r]:
                    if gap_start is None:
                        gap_start = r
                else:
                    if gap_start is not None:
                        gap_len = r - gap_start
                        if gap_len > best_gap_len:
                            best_gap_len = gap_len
                            best_gap_start = gap_start
                        gap_start = None

            # Handle gap that runs to the end
            if gap_start is not None:
                gap_len = (r_hi + 1) - gap_start
                if gap_len > best_gap_len:
                    best_gap_len = gap_len
                    best_gap_start = gap_start

            if best_gap_start is None or best_gap_len < 3:
                # Try alternative: the gap might be the area beyond where
                # one wall ends.  E.g. top wall extends to x=8 but bottom
                # wall extends to x=10 — the gap is between y_top_wall
                # and y_top at x > 8.
                #
                # Use the wall endpoints directly.
                # The opening spans from where the shorter wall ends to
                # where the longer wall is.

                # If top wall is shorter on the right, the gap is at the top
                if top_right_x < bot_right_x:
                    # Gap from top_wall.position upward... but actually
                    # the gap is to the RIGHT of top_right_x.
                    # The opening Y range is [bottom_wall.position, top_wall.position]
                    # at x = exit_x where top wall has ended
                    return OpeningEstimate(
                        y_min=bottom_wall.position,
                        y_max=top_wall.position,
                        x_position=exit_x,
                    )
                elif bot_right_x < top_right_x:
                    return OpeningEstimate(
                        y_min=bottom_wall.position,
                        y_max=top_wall.position,
                        x_position=exit_x,
                    )
                else:
                    return None

            # Convert gap rows back to world Y
            _, y_min = self._grid_to_world(0, best_gap_start)
            _, y_max = self._grid_to_world(0, best_gap_start + best_gap_len)

            return OpeningEstimate(
                y_min=y_min,
                y_max=y_max,
                x_position=exit_x,
            )

        # For other exit sides, mirror the logic (left/top/bottom)
        # For now, only 'right' is fully implemented.
        return None

    # ------------------------------------------------------------------
    # Exit direction estimation from trajectory
    # ------------------------------------------------------------------

    def estimate_exit_direction(self) -> Optional[float]:
        """
        Estimate the direction the robot is heading as it exits, based on
        the last segment of the recorded trajectory.

        Returns the angle in radians (world frame), or None if not enough
        trajectory data.
        """
        if len(self._trajectory) < 10:
            return None

        # Use the last 20 poses to compute a direction vector
        n = min(20, len(self._trajectory))
        recent = self._trajectory[-n:]
        p0 = np.array(recent[0])
        p1 = np.array(recent[-1])
        delta = p1 - p0
        if np.linalg.norm(delta) < 0.05:
            return None
        return float(math.atan2(delta[1], delta[0]))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_exit_goal(self) -> Optional[np.ndarray]:
        """
        Return the corrected goal position (centre of the detected opening,
        offset outward), or None if the opening hasn't been detected yet.
        """
        return self._exit_goal

    def get_opening(self) -> Optional[OpeningEstimate]:
        """Return the detected opening, if any."""
        return self._opening

    def get_walls(self) -> list[WallEstimate]:
        """Return the list of detected wall segments."""
        return list(self._walls)

    def get_robot_trajectory(self) -> list[Tuple[float, float]]:
        """Return the recorded robot trajectory."""
        return list(self._trajectory)

    def has_robot_exited(self, robot_x: float, robot_y: float) -> bool:
        """
        Check if the robot has passed through the opening to the outside.
        Uses the detected opening position.
        """
        if self._opening is None:
            return False

        if self._exit_side == 'right':
            return robot_x > self._opening.x_position + 0.3
        elif self._exit_side == 'left':
            return robot_x < self._opening.x_position - 0.3
        elif self._exit_side == 'top':
            return robot_y > self._opening.y_position + 0.3
        else:
            return robot_y < self._opening.y_position - 0.3

    def get_enclosure_bounds(self) -> Optional[dict]:
        """
        Return a dict with the enclosure boundary estimates:
        {
          'x_min': ..., 'x_max': ...,
          'y_min': ..., 'y_max': ...,
          'exit_x': ...,
          'opening_y_min': ..., 'opening_y_max': ...,
          'opening_center': (x, y),
        }
        Returns None if not enough walls detected.
        """
        h_walls = [w for w in self._walls if w.axis == 'horizontal']
        v_walls = [w for w in self._walls if w.axis == 'vertical']

        if len(h_walls) < 2:
            return None

        h_walls.sort(key=lambda w: w.position)
        bounds = {
            'y_min': h_walls[0].position,
            'y_max': h_walls[-1].position,
        }

        if v_walls:
            v_walls.sort(key=lambda w: w.position)
            bounds['x_min'] = v_walls[0].position
            if len(v_walls) > 1:
                bounds['x_max'] = v_walls[-1].position

        if self._opening:
            bounds['exit_x'] = self._opening.x_position
            bounds['opening_y_min'] = self._opening.y_min
            bounds['opening_y_max'] = self._opening.y_max
            bounds['opening_center'] = self._opening.center

        return bounds

    def debug_grid_snapshot(self) -> np.ndarray:
        """Return a copy of the probability grid for debugging / visualization."""
        with self._lock:
            return 1.0 - 1.0 / (1.0 + np.exp(self._grid.copy()))