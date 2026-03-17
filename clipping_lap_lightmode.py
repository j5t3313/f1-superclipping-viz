import os

import fastf1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from scipy.signal import savgol_filter


CACHE_DIR = "cache"
YEAR = 2026
GP = "China"
SESSION_TYPE = "Q"

THROTTLE_THRESHOLD = 98
DECEL_THRESHOLD = -0.05
FRAME_STEP = 3
FPS = 10
DPI = 150
OUTPUT = "clipping_lap.gif"
SAVGOL_WINDOW = 13
SAVGOL_POLY = 2
MIN_CLIP_RUN = 5

BG = "#ffffff"
TRACK_GHOST = "#e0e0e0"
ACCEL = "#00a878"
CLIP = "#d6002a"
NEUTRAL = "#b0b0b0"
THROTTLE_FILL = "#000000"
TEXT = "#1a1a1a"
GRID = "#e0e0e0"
SUBTEXT = "#4a4a4a"


def filter_short_runs(mask, min_length):
    filtered = mask.copy()
    i = 0
    while i < len(filtered):
        if filtered[i]:
            j = i
            while j < len(filtered) and filtered[j]:
                j += 1
            if (j - i) < min_length:
                filtered[i:j] = False
            i = j
        else:
            i += 1
    return filtered


def load_lap():
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)
    session = fastf1.get_session(YEAR, GP, SESSION_TYPE)
    session.load(telemetry=True, laps=True, weather=False)

    lap = session.laps.pick_fastest()
    car = lap.get_car_data().add_distance()
    pos = lap.get_pos_data()

    merged = pd.merge_asof(
        car.sort_values("Time"),
        pos[["Time", "X", "Y"]].sort_values("Time"),
        on="Time",
        direction="nearest",
    )

    for col in ["Speed", "Throttle", "X", "Y"]:
        merged[col] = merged[col].interpolate(method="linear")
    merged[["nGear", "Brake"]] = merged[["nGear", "Brake"]].ffill()
    merged = merged.dropna(subset=["Speed", "Throttle", "X", "Y", "Distance"]).reset_index(drop=True)

    merged["Speed"] = savgol_filter(merged["Speed"].values, SAVGOL_WINDOW, SAVGOL_POLY)

    merged["dSpeed"] = np.gradient(merged["Speed"].values, merged["Distance"].values)
    raw_clip = (
        (merged["Throttle"] >= THROTTLE_THRESHOLD) & (merged["dSpeed"] < DECEL_THRESHOLD)
    ).values
    merged["is_clipping"] = filter_short_runs(raw_clip, MIN_CLIP_RUN)

    meta = {
        "driver": lap["Driver"],
        "team": lap["Team"],
        "lap_time": str(lap["LapTime"]).split(" ")[-1][:10],
    }

    return merged, meta


def segment_color(row):
    if row["is_clipping"]:
        return CLIP
    if row["Throttle"] >= THROTTLE_THRESHOLD and row["dSpeed"] >= 0:
        return ACCEL
    return NEUTRAL


def build_figure(df, meta):
    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.8, 1], hspace=0.22,
        left=0.06, right=0.94, top=0.93, bottom=0.06,
    )
    ax_map = fig.add_subplot(gs[0])
    ax_spd = fig.add_subplot(gs[1])

    for ax in [ax_map, ax_spd]:
        ax.set_facecolor(BG)
        ax.tick_params(colors=SUBTEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)

    ax_map.plot(df["X"], df["Y"], color=TRACK_GHOST, linewidth=7, solid_capstyle="round", zorder=1)
    ax_map.set_aspect("equal")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(False)

    fig.suptitle(
        f"{meta['driver']}  |  {meta['team']}  |  {GP} {YEAR} Qualifying  |  {meta['lap_time']}",
        color=TEXT, fontsize=13, fontweight="bold", y=0.97,
    )

    ax_spd.set_xlim(df["Distance"].min(), df["Distance"].max())
    ax_spd.set_ylim(0, df["Speed"].max() * 1.1)
    ax_spd.set_xlabel("Distance (m)", color=SUBTEXT, fontsize=9)
    ax_spd.set_ylabel("Speed (km/h)", color=TEXT, fontsize=10)
    ax_spd.grid(True, color=GRID, alpha=0.4, linewidth=0.5)

    ax_thr = ax_spd.twinx()
    ax_thr.set_ylim(0, 110)
    ax_thr.set_ylabel("Throttle %", color=SUBTEXT, fontsize=9)
    ax_thr.tick_params(colors=SUBTEXT, labelsize=8)
    for spine in ax_thr.spines.values():
        spine.set_color(GRID)

    return fig, ax_map, ax_spd, ax_thr


def run(df, meta):
    seg_colors = [segment_color(row) for _, row in df.iterrows()]

    fig, ax_map, ax_spd, ax_thr = build_figure(df, meta)

    trail_collection = None
    car_dot, = ax_map.plot([], [], "o", markersize=7, color=ACCEL, zorder=10)

    speed_line, = ax_spd.plot([], [], color=ACCEL, linewidth=1.4, zorder=5)
    throttle_artist = None
    clip_artist = None

    spd_text = ax_spd.text(
        0.015, 0.93, "", transform=ax_spd.transAxes,
        color=TEXT, fontsize=12, fontweight="bold", va="top",
    )
    clip_text = ax_spd.text(
        0.985, 0.93, "", transform=ax_spd.transAxes,
        color=CLIP, fontsize=12, fontweight="bold", va="top", ha="right",
    )

    legend_elements = [
        plt.Line2D([0], [0], color=ACCEL, linewidth=3, label="Deploying"),
        plt.Line2D([0], [0], color=CLIP, linewidth=3, label="Super-clipping"),
        plt.Line2D([0], [0], color=NEUTRAL, linewidth=3, label="Partial / coast"),
    ]
    ax_map.legend(
        handles=legend_elements, loc="upper right",
        fontsize=11, facecolor=BG, edgecolor=GRID,
        labelcolor=TEXT, framealpha=0.9,
    )

    indices = list(range(0, len(df), FRAME_STEP))
    if indices[-1] != len(df) - 1:
        indices.append(len(df) - 1)

    def update(frame_num):
        nonlocal trail_collection, throttle_artist, clip_artist

        idx = indices[frame_num]
        sub = df.iloc[: idx + 1]

        if trail_collection is not None:
            trail_collection.remove()

        pts = np.column_stack([sub["X"].values, sub["Y"].values]).reshape(-1, 1, 2)
        if len(pts) > 1:
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            trail_collection = LineCollection(
                segs, colors=seg_colors[:idx], linewidths=2.8, zorder=2
            )
            ax_map.add_collection(trail_collection)

        car_dot.set_data([df["X"].iloc[idx]], [df["Y"].iloc[idx]])
        car_dot.set_color(CLIP if df["is_clipping"].iloc[idx] else ACCEL)

        speed_line.set_data(sub["Distance"], sub["Speed"])

        if throttle_artist is not None:
            throttle_artist.remove()
        throttle_artist = ax_thr.fill_between(
            sub["Distance"], 0, sub["Throttle"], alpha=0.12, color=THROTTLE_FILL,
            edgecolor=THROTTLE_FILL, linewidth=0.8,
        )

        if clip_artist is not None:
            clip_artist.remove()
        clip_mask = sub["is_clipping"].values
        if np.any(clip_mask):
            clip_artist = ax_spd.fill_between(
                sub["Distance"],
                0,
                df["Speed"].max() * 1.1,
                where=clip_mask,
                alpha=0.1,
                color=CLIP,
                zorder=1,
            )
        else:
            clip_artist = None

        spd_text.set_text(f"{sub['Speed'].iloc[-1]:.0f} km/h")
        clip_text.set_text("SUPER-CLIPPING" if df["is_clipping"].iloc[idx] else "")

        return ()

    anim = animation.FuncAnimation(
        fig, update, frames=len(indices), interval=1000 / FPS, blit=False
    )
    anim.save(OUTPUT, writer="pillow", fps=FPS, dpi=DPI)
    plt.close(fig)


def main():
    df, meta = load_lap()
    run(df, meta)


if __name__ == "__main__":
    main()