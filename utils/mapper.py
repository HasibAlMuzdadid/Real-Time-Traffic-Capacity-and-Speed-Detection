""" 
Author : Md Hasib Al Muzdadid Haque Himel

Copyright (c) 2025, Hasib Al Muzdadid
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""


import numpy as np
import cv2 as cv


class Cam2WorldMapper:
# maps points from image to world coordinates using perspective transformation
    def __init__(self):
        self.M = None       # Transformation matrix

    def __call__(self, image_pts):
        return self.map(image_pts)

    def find_perspective_transform(self, image_pts, world_pts):
        image_pts = np.float32(image_pts).reshape(-1, 1, 2)
        world_pts = np.float32(world_pts).reshape(-1, 1, 2)
        self.M = cv.getPerspectiveTransform(image_pts, world_pts)
        return self.M

    def map(self, image_pts):
        if self.M is None:
            raise ValueError("Perspective transformation has not been estimated!")
        image_pts = np.float32(image_pts).reshape(-1, 1, 2)
        return cv.perspectiveTransform(image_pts, self.M).reshape(-1, 2)