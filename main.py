""" 
Author : Md Hasib Al Muzdadid Haque Himel

Copyright (c) 2025, Hasib Al Muzdadid
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""


import argparse
from utils.configuration import Config
from utils.detector import VehicleCounter
from utils.insight import InsightGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle Speed and Count Detection")
    parser.add_argument("--model_name", type=str, default=None, help="YOLO model name")
    parser.add_argument("--video_path", type=str, default=None, help="Path to input video")
    parser.add_argument("--output_video_path", type=str, default=None, help="Path to save annotated video")
    parser.add_argument("--compressed_video_path", type=str, default=None, help="Path to save compressed annotated video")
    parser.add_argument("--output_insights_path", type=str, default=None, help="Path to save insight image")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to save csv file")
    parser.add_argument("--conf_threshold", type=float, default=0.4, help="Confidence threshold")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config()

    if args.model_name: cfg.model_name = args.model_name
    if args.video_path: cfg.video_path = args.video_path
    if args.output_video_path: cfg.output_video_path = args.output_video_path
    if args.compressed_video_path: cfg.compressed_video_path = args.compressed_video_path
    if args.output_insights_path: cfg.output_insights_path = args.output_insights_path
    if args.csv_path: cfg.csv_path = args.csv_path
    if args.conf_threshold: cfg.conf_threshold = args.conf_threshold

    vehicle_counter = VehicleCounter(cfg)
    vehicle_counter.run()

    generate_insight = InsightGenerator(cfg)
    generate_insight.plot_speed_distribution(save_figure=True)
    generate_insight.plot_vehicle_stats(save_figure=True)


if __name__ == "__main__":
    main()