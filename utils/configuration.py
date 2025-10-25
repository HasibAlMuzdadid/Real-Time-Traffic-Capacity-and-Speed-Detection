""" 
Author : Md Hasib Al Muzdadid Haque Himel

Copyright (c) 2025, Hasib Al Muzdadid
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""


from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = "yolo11m.pt"
    video_path: str = "./input/video/input_video.mp4"
    output_video_path: str = "./output/video/output_annotated.mp4"
    compressed_video_path: str = "./output/video/output_annotated_compressed.mp4"
    output_insights_path: str = "./output/Insights"
    csv_path: str = "./output/vehicle_speeds.csv"
    
    # Detection parameters
    conf_threshold: float = 0.4
    classes: tuple = (2, 5, 7)  # Car, Bus, Truck

    # Counting line positions
    in_line_y: int = 700       # entry line (upward movement)
    out_line_y: int = 500      # exit line (downward movement)

    # Blink effect
    blink_duration: int = 5   # frames
    line_thickness: int = 3

    # Speed tracking
    max_reuse_gap: int = 30
    min_track_frames_for_speed: int = 3

    # Visualization
    font_scale: float = 0.9
    font_thickness: int = 2

    # Four image points
    imgpointAx: int = 830
    imgpointAy: int = 410
    imgpointBx: int = 1090
    imgpointBy: int = 410
    imgpointCx: int = 1920
    imgpointCy: int = 850
    imgpointDx: int = 0
    imgpointDy: int = 850

    imgpointA: tuple = (imgpointAx, imgpointAy)
    imgpointB: tuple = (imgpointBx, imgpointBy)
    imgpointC: tuple = (imgpointCx, imgpointCy)
    imgpointD: tuple = (imgpointDx, imgpointDy)

    # Real world distance
    road_width: int = 32
    road_distance: int = 50

    # Custom color palette for object tracking
    colors: tuple = ("#007fff", "#0072e6", "#0066cc", "#0059b3", "#004c99", "#004080", "#003366", "#00264d")
    