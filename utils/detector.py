""" 
Author : Md Hasib Al Muzdadid Haque Himel

Copyright (c) 2025, Hasib Al Muzdadid
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""


import numpy as np
import csv
import subprocess
import cv2 as cv
from collections import defaultdict
import supervision as sv
from ultralytics import YOLO
from utils.speedometer import LocalScaleSpeedometer
from utils.configuration import Config
from utils.mapper import Cam2WorldMapper


class VehicleCounter:
    def __init__(self, cfg=Config):
        self.cfg = cfg
        self.yolo = None
        self.speedometer = None        
        self.last_seen_frame = dict()                # {byte_track_id: last_frame_index_seen}
        self.first_seen_frame = dict()               # {byte_track_id: first_frame_index_when_current_instance_started}
        self.track_history = defaultdict(list)       # centroids for optional counting
        self.seen_unique_labels = set()              # set of unique labels assigned (for summary)
        self.unique_id_counter = defaultdict(int)    
        self.tracker_to_unique_label = dict()        # maps current ByteTrack id -> "Car#5"
        self.count_in = 0
        self.count_out = 0
        self.counted_in_ids = set()
        self.counted_out_ids = set()
        self.in_blink_frames = 0
        self.out_blink_frames = 0
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.color_in_bgr = sv.Color.from_hex("#004080").as_bgr()
        self.color_out_bgr = sv.Color.from_hex("#f78923").as_bgr()
        self.color_blink_bgr = sv.Color.from_hex("#ffffff").as_bgr()
        self.font_color_bgr = sv.Color.from_hex("#004c99").as_bgr()
        self.color_palette = sv.ColorPalette(list(map(sv.Color.from_hex, self.cfg.colors)))   

    def run(self):
        self.yolo = YOLO(self.cfg.model_name, task="detect")

        video_info = sv.VideoInfo.from_video_path(self.cfg.video_path)
        fps = video_info.fps
        width, height = video_info.resolution_wh
        width, height = round(width / 32) * 32, round(height / 32) * 32     # YOLO expects the image size to be a multiple of 32
        mid_x = width // 2

        image_pts = [self.cfg.imgpointA, self.cfg.imgpointB, self.cfg.imgpointC, self.cfg.imgpointD]
        world_pts = [(0, 0), (self.cfg.road_width, 0), (self.cfg.road_width, self.cfg.road_distance), (0, self.cfg.road_distance)]
        mapper = Cam2WorldMapper()
        mapper.find_perspective_transform(image_pts, world_pts)

        self.speedometer = LocalScaleSpeedometer(mapper=mapper, fps=int(fps))

        # Polygonal zone that masks out detected objects that are outside it
        poly = np.array([(self.cfg.imgpointDx, self.cfg.imgpointAy), (self.cfg.imgpointCx, self.cfg.imgpointBy), (self.cfg.imgpointCx, self.cfg.imgpointCy), (self.cfg.imgpointDx, self.cfg.imgpointDy)])
        zone = sv.PolygonZone(poly, (sv.Position.TOP_CENTER, sv.Position.BOTTOM_CENTER))

        csvfile = open(self.cfg.csv_path, "w", newline="")
        writer = csv.DictWriter(csvfile, fieldnames=["frame", "tracker_id", "class", "speed_kph", "cx", "cy"])
        writer.writeheader()

        bbox_annotator = sv.BoxAnnotator(color=self.color_palette, thickness=2, color_lookup=sv.ColorLookup.TRACK)
        trace_annotator = sv.TraceAnnotator(color=self.color_palette, position=sv.Position.CENTER, thickness=2, trace_length=int(fps), color_lookup=sv.ColorLookup.TRACK)
        label_annotator = sv.RichLabelAnnotator(color=self.color_palette, border_radius=2, font_size=16, color_lookup=sv.ColorLookup.TRACK, text_padding=6)

        frame_idx = 0
        with sv.VideoSink(self.cfg.output_video_path, video_info) as sink:
            for frame in sv.get_video_frames_generator(self.cfg.video_path):
                frame_idx += 1

                result = self.yolo.track(
                    frame,
                    conf=self.cfg.conf_threshold,
                    classes=self.cfg.classes,
                    imgsz=(height, width),
                    persist=True,
                    verbose=False,
                    tracker="bytetrack.yaml",
                )

                det = sv.Detections.from_ultralytics(result[0])
                det = det[zone.trigger(detections=det)]         # filter by polygon zone

                labels = []
                trace_ids = det.tracker_id if len(det) > 0 else []

                self._update_speeds(trace_ids=trace_ids, det=det, frame_idx=frame_idx, writer=writer, labels=labels)
                self._update_counts(trace_ids=trace_ids, mid_x=mid_x)

                # ---------- Annotations ----------
                frame_rgb = cv.cvtColor(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2RGB)

                if len(det) > 0:
                    frame_rgb = bbox_annotator.annotate(frame_rgb, det)
                    frame_rgb = trace_annotator.annotate(frame_rgb, det)
                    # label_annotator expects labels ordered as detections list
                    if labels:
                        frame_rgb = label_annotator.annotate(frame_rgb, det, labels=labels)
                    else:
                        frame_rgb = label_annotator.annotate(frame_rgb, det)

                annotated_frame = self._draw_annotations(frame=frame_rgb, width=width, mid_x=mid_x)
                sink.write_frame(annotated_frame)

        csvfile.close()

        compressed = self.cfg.compressed_video_path
        subprocess.run(
            [
                "ffmpeg",
                "-i", self.cfg.output_video_path,
                "-crf", "18",
                "-preset", "veryfast",
                "-vcodec", "libx264",
                compressed,
                "-loglevel", "quiet",
                "-y",
            ]
        )
        print(f"Completed — unique vehicles assigned: {len(self.seen_unique_labels)}. CSV saved to: {self.cfg.csv_path}")

    def _update_speeds(self, trace_ids, det, frame_idx, writer, labels):
        # safety: convert xyxy and class arrays to numpy lists/arrays
        xyxy = np.array(det.xyxy) if det.xyxy is not None else np.zeros((0,4))
        class_ids = np.array(det.class_id) if det.class_id is not None else np.zeros((len(trace_ids),), dtype=int)

        for i, byte_tid in enumerate(list(trace_ids)):
            byte_tid = int(byte_tid)

            # bounding box -> centroid
            if i >= len(xyxy):
                continue
            x1, y1, x2, y2 = map(int, xyxy[i])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # ----------------- Unique label assignment -----------------
            class_id = int(class_ids[i]) if i < len(class_ids) else None
            class_name = {2: "Car", 5: "Bus", 7: "Truck"}.get(class_id, "Vehicle")

            # If this ByteTrack id seen before and short gap -> reuse current mapping
            if byte_tid in self.tracker_to_unique_label:
                gap = frame_idx - self.last_seen_frame.get(byte_tid, frame_idx)
                if gap > self.cfg.max_reuse_gap:
                    # ByteTrack id reappeared after a long gap -> assign a new global unique label
                    self.unique_id_counter[class_name] += 1
                    new_label = f"{class_name}#{self.unique_id_counter[class_name]}"
                    self.tracker_to_unique_label[byte_tid] = new_label
            # else: keep existing mapping (it's still the same instance)
            else:
                # first time seeing this ByteTrack id -> assign a new global unique label
                self.unique_id_counter[class_name] += 1
                new_label = f"{class_name}#{self.unique_id_counter[class_name]}"
                self.tracker_to_unique_label[byte_tid] = new_label

            unique_label = self.tracker_to_unique_label[byte_tid]
            self.seen_unique_labels.add(unique_label)

            # ------------- ID reuse protection for speed history -------------
            if byte_tid not in self.last_seen_frame:
                # new first appearance
                self.first_seen_frame[byte_tid] = frame_idx
                self.speedometer.reset(byte_tid)
                self.track_history[byte_tid].clear()
            else:
                gap = frame_idx - self.last_seen_frame[byte_tid]
                if gap > self.cfg.max_reuse_gap:
                    # treat as new instance of same ByteTrack id (already assigned new unique_label above)
                    self.first_seen_frame[byte_tid] = frame_idx
                    self.speedometer.reset(byte_tid)
                    self.track_history[byte_tid].clear()

            self.last_seen_frame[byte_tid] = frame_idx

            # --------- update centroid history used for counting/robustness ----------
            self.track_history[byte_tid].append((cx, cy))
            if len(self.track_history[byte_tid]) > 30:
                self.track_history[byte_tid] = self.track_history[byte_tid][-30:]

            # --------- compute speed after stable tracking ----------
            seen_duration = frame_idx - self.first_seen_frame.get(byte_tid, frame_idx)
            if seen_duration >= self.cfg.min_track_frames_for_speed:
                # Use centroid-based local-scale speed estimator (LocalScaleSpeedometer)
                self.speedometer.update_with_centroid(frame_idx, byte_tid, (cx, cy))
                current_speed = self.speedometer.get_speed(byte_tid)
            else:
                current_speed = 0

            # label to display on video
            labels.append(f"{unique_label} {current_speed} km/h")

            # write CSV row (unique_label used instead of ByteTrack id)
            writer.writerow({
                "frame": frame_idx,
                "tracker_id": unique_label,
                "class": class_name,
                "speed_kph": current_speed,
                "cx": cx,
                "cy": cy
            })

    def _update_counts(self, trace_ids, mid_x):
        for byte_tid in trace_ids:
            if byte_tid not in self.track_history:
                continue
        
            curr_cx, curr_cy = self.track_history[byte_tid][-1]
            prev_cx, prev_cy = self.track_history[byte_tid][-2] if len(self.track_history[byte_tid]) >= 2 else (None, None)

            # ---------- IN count (left half, moving UP or first frame) ----------
            if byte_tid not in self.counted_in_ids and curr_cx <= mid_x:
                if prev_cy is None:
                    # first detection frame, vehicle already above IN line
                    if curr_cy <= self.cfg.in_line_y:
                        self.count_in += 1
                        self.counted_in_ids.add(byte_tid)
                else:
                    # line crossed upwardy:
                    if prev_cy > self.cfg.in_line_y >= curr_cy:       
                        self.count_in += 1
                        self.counted_in_ids.add(byte_tid)
                        self.in_blink_frames = self.cfg.blink_duration        # trigger blink

            # ---------- OUT count (right half, moving DOWN or already below OUT line at first detection) ----------
            if byte_tid not in self.counted_out_ids and curr_cx > mid_x:
                if prev_cy is None:
                    # first detection and already below or on OUT line
                    if curr_cy >= self.cfg.out_line_y:
                        self.count_out += 1
                        self.counted_out_ids.add(byte_tid)
                else:
                    # normal downward crossing
                    if prev_cy <= self.cfg.out_line_y <= curr_cy:
                        self.count_out += 1
                        self.counted_out_ids.add(byte_tid)
                        self.out_blink_frames = self.cfg.blink_duration        # trigger blink

    def _draw_annotations(self, frame, width, mid_x):
        # IN line blink
        overlay = frame.copy()
        if self.in_blink_frames > 0:
            cv.line(overlay, (0, self.cfg.in_line_y), (mid_x-50, self.cfg.in_line_y), self.color_blink_bgr, 6)
            frame = cv.addWeighted(overlay, 0.7, frame, 0.3, 0)
            self.in_blink_frames -= 1
        else:
            cv.line(frame, (0, self.cfg.in_line_y), (mid_x-50, self.cfg.in_line_y), self.color_in_bgr, self.cfg.line_thickness)

        # OUT line blink
        overlay = frame.copy()
        if self.out_blink_frames > 0:
            cv.line(overlay, (mid_x+20, self.cfg.out_line_y), (width, self.cfg.out_line_y), self.color_blink_bgr, 6)
            frame = cv.addWeighted(overlay, 0.7, frame, 0.3, 0)
            self.out_blink_frames -= 1
        else:
            cv.line(frame, (mid_x+20, self.cfg.out_line_y), (width, self.cfg.out_line_y), self.color_out_bgr, self.cfg.line_thickness)

        # overlay text
        cv.putText(frame, f"Vehicles Entered: {self.count_in}", (40, 60),
                   self.font, self.cfg.font_scale, self.font_color_bgr, self.cfg.font_thickness)
        cv.putText(frame, f"Vehicles Left: {self.count_out}", (40, 100),
                   self.font, self.cfg.font_scale, self.font_color_bgr, self.cfg.font_thickness)
        cv.putText(frame, f"Total Vehicles: {len(self.seen_unique_labels)}", (40, 140),
                   self.font, self.cfg.font_scale, self.font_color_bgr, self.cfg.font_thickness)
        return frame