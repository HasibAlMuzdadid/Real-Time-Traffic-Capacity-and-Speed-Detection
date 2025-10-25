""" 
Author : Md Hasib Al Muzdadid Haque Himel

Copyright (c) 2025, Hasib Al Muzdadid
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from utils.configuration import Config
import supervision as sv
import warnings
import os

warnings.simplefilter("ignore")


class InsightGenerator:
    def __init__(self, cfg=Config):
        self.cfg = cfg
        self.fps = None
        self.csv_path = self.cfg.csv_path
        self.output_insights_path = self.cfg.output_insights_path
        os.makedirs(self.output_insights_path, exist_ok=True)

    def plot_speed_distribution(self, save_figure=False):
    # Plots smoothed vehicle speed traces and speed distribution
        video_info = sv.VideoInfo.from_video_path(self.cfg.video_path)
        self.fps = video_info.fps

        df = self._load_and_clean()
        df = self._smooth_speeds(df=df)
        wide_df = self._pivot_and_clip(df=df)

        fig, axes = plt.subplots(2, 1, figsize=(20, 10), tight_layout=True)

        sns.lineplot(data=wide_df, palette="viridis", linewidth=1.25, ax=axes[0])
        axes[0].set_xlabel("Frame", color="#000000")
        axes[0].set_ylabel("Speed (km/h)", color="#000000")
        axes[0].set_ylim(10, 140)
        axes[0].get_legend().set_visible(False)
        axes[0].set_title("Vehicle Speed Traces (Smoothed)", color="#000000", loc="center", pad=40)
        axes[0].tick_params(colors="#000000")

        sns.kdeplot(wide_df.to_numpy().ravel(), fill=True, color="#004080", linewidth=1, ax=axes[1])
        axes[1].set_xlabel("Speed (km/h)", color="#000000")
        axes[1].set_ylabel("Density", color="#000000")
        axes[1].set_title("Speed Distribution Across All Vehicles", color="#000000", loc="center", pad=20)
        axes[1].tick_params(colors="#000000")

        if save_figure:
            figure_path = os.path.join(self.output_insights_path, "speed_distribution.png")
            plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_vehicle_stats(self, save_figure=False):
    # Plots average, and max speed per vehicle in subplots
        video_info = sv.VideoInfo.from_video_path(self.cfg.video_path)
        self.fps = video_info.fps

        df = self._load_and_clean()
        df = self._smooth_speeds(df=df)    
        
        vehicle_stats = df.groupby("tracker_id")["speed_smooth"].agg(
            avg_speed="mean",
            max_speed="max"
        ).reset_index()

        fig, axes = plt.subplots(2, 1, figsize=(20, 10), tight_layout=True)
        palette = sns.color_palette("viridis", n_colors=len(vehicle_stats))

        # Average Speed
        sns.barplot(data=vehicle_stats, x="tracker_id", y="avg_speed", palette=palette, ax=axes[0])
        axes[0].set_title("Average Speed per Vehicle", color="#000000", fontsize=14)
        axes[0].set_xlabel("Vehicle ID", color="#000000")
        axes[0].set_ylabel("Speed (km/h)", color="#000000")
        axes[0].tick_params(colors="#000000", rotation=90)

        # Max Speed
        sns.barplot(data=vehicle_stats, x="tracker_id", y="max_speed", palette=palette, ax=axes[1])
        axes[1].set_title("Max Speed per Vehicle", color="#000000", fontsize=14)
        axes[1].set_xlabel("Vehicle ID", color="#000000")
        axes[1].set_ylabel("Speed (km/h)", color="#000000")
        axes[1].tick_params(colors="#000000", rotation=90)

        if save_figure:
            figure_path = os.path.join(self.output_insights_path, "vehicle_speed_stats.png")
            plt.savefig(figure_path, dpi=300, bbox_inches="tight")        
        plt.show()

    def _load_and_clean(self):
    # Loads and cleans dataframe    
        df = pd.read_csv(self.csv_path)
        df = df.dropna(subset=["speed_kph"])
        df = df[df["speed_kph"] > 0]
        df = df.sort_values(["tracker_id", "frame"]).reset_index(drop=True)
        return df

    def _safe_savgol(self, x):
    # Smooths a pandas Series using Savitzky-Golay filter safely
        n = len(x)
        if n < 5:
            return pd.Series(x, index=x.index)
        window = min(self.fps, n if n % 2 == 1 else n - 1)
        if window < 3:
            window = 3
        smoothed = savgol_filter(x, window_length=window, polyorder=2)
        return pd.Series(smoothed, index=x.index)

    def _smooth_speeds(self, df):
    # Applies safe smoothing per vehicle and adds 'speed_smooth' column
        smooth_series = df.groupby("tracker_id")["speed_kph"].apply(lambda x: self._safe_savgol(x))
        smooth_series = smooth_series.reset_index(level=0, drop=True)
        df["speed_smooth"] = smooth_series
        return df

    def _pivot_and_clip(self, df):
    # Pivot to wide layout (frames x vehicles) and clip extreme speeds
        wide_df = df.pivot(index="frame", columns="tracker_id", values="speed_smooth")
        wide_df = wide_df.clip(
            lower=float(np.nanpercentile(wide_df, 1)),
            upper=float(np.nanpercentile(wide_df, 99))
        )
        return wide_df
