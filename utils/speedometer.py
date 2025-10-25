""" 
Author : Md Hasib Al Muzdadid Haque Himel

Copyright (c) 2025, Hasib Al Muzdadid
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""


import numpy as np
import math
from collections import defaultdict, deque


class LocalScaleSpeedometer:
# Computes speed using local pixel -> meter scale computed from homography mapper
# Uses update_with_centroid(frame_idx, track_id, (cx, cy)) every frame per tracked object
# Uses get_speed(track_id) to read smoothed speed (kph int)
    def __init__(self, mapper, fps, unit=3.6, window=5, max_kph=300):
        self.mapper = mapper
        self.fps = float(fps)
        self.unit = float(unit)         # m/s -> km/h multiplier (3.6)
        self.window = int(window)       # how many centroids to keep for smoothing (deque)
        self.max_kph = float(max_kph)
        self.pos_hist = defaultdict(lambda: deque(maxlen=self.window))   # stores image centroids (cx,cy)
        self.speed_hist = defaultdict(lambda: deque(maxlen=8))           # recent speed estimates for smoothing

    def _local_mpp(self, point):
    # Estimates meters per pixel at image point (cx, cy) by mapping (cx,cy), (cx+1,cy), (cx,cy+1); returns (mpp_x, mpp_y) in meters/pixel
        cx, cy = int(round(point[0])), int(round(point[1]))
        img_pts = np.array([[cx, cy], [cx + 1, cy], [cx, cy + 1]], dtype=np.float32)
        try:
            world_pts = self.mapper.map(img_pts)  # shape (3,2)
        except Exception:
            # fallback small epsilon to avoid division by zero
            return (1e-6, 1e-6)

        w00 = world_pts[0]
        wx = world_pts[1]
        wy = world_pts[2]
        mpp_x = float(np.linalg.norm(wx - w00))
        mpp_y = float(np.linalg.norm(wy - w00))
        if mpp_x == 0:
            mpp_x = 1e-6
        if mpp_y == 0:
            mpp_y = 1e-6
        return (mpp_x, mpp_y)

    def update_with_centroid(self, frame_idx:int, track_id:int, centroid:tuple):
        tid = int(track_id)
        cx, cy = int(round(centroid[0])), int(round(centroid[1]))
        self.pos_hist[tid].append((cx, cy))

        # need at least two points to compute motion
        if len(self.pos_hist[tid]) < 2:
            return

        (x_prev, y_prev), (x_curr, y_curr) = self.pos_hist[tid][-2], self.pos_hist[tid][-1]
        dx_px = float(x_curr - x_prev)
        dy_px = float(y_curr - y_prev)

        # compute local meters-per-pixel at the midpoint
        mid = ((x_prev + x_curr) / 2.0, (y_prev + y_curr) / 2.0)
        mpp_x, mpp_y = self._local_mpp(mid)

        dx_m = dx_px * mpp_x
        dy_m = dy_px * mpp_y
        ds_m = math.hypot(dx_m, dy_m)

        # meters per second (distance per frame * fps)
        m_s = ds_m * self.fps
        kph = m_s * self.unit

        # clip extreme spikes and fallback to previous if too large
        if kph < 0:
            kph = 0.0
        if kph > self.max_kph:
            if self.speed_hist[tid]:
                kph = float(self.speed_hist[tid][-1])
            else:
                kph = float(min(kph, self.max_kph))

        # smoothing: push and keep history; median used on get_speed
        self.speed_hist[tid].append(kph)

    def get_speed(self, track_id:int):
        tid = int(track_id)
        if not self.speed_hist[tid]:
            return 0
        arr = np.array(self.speed_hist[tid], dtype=float)
        return int(round(float(np.median(arr))))

    def reset(self, track_id:int):
        tid = int(track_id)
        if tid in self.pos_hist:
            self.pos_hist[tid].clear()
        if tid in self.speed_hist:
            self.speed_hist[tid].clear()