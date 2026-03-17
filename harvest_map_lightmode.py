import os

import fastf1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from scipy.signal import savgol_filter


CACHE_DIR = "cache"
YEAR = 2026
GP = "China"
SESSION_TYPE = "Q"

THROTTLE_FULL = 95
THROTTLE_COAST = 5
DECEL_THRESHOLD = -0.05
FRAME_STEP = 3
FPS = 10
DPI = 150
OUTPUT = "harvest_map.gif"
SAVGOL_WINDOW = 13
SAVGOL_POLY = 2
MIN_STATE_RUN = 5

BG = "#ffffff"
TRACK_GHOST = "#e0e0e0"

STATE_DEPLOY = "DEPLOY"
STATE_SUPERCLIP = "SUPERCLIP"
STATE_HARVEST = "HARVEST"
STATE_BRAKE = "BRAKE"

PALETTE = {
    STATE_DEPLOY: "#1565c0",
    STATE_SUPERCLIP: "#d6002a",
    STATE_HARVEST: "#e68a00",
    STATE_BRAKE: "#7a7a7a",
}

LABELS = {
    STATE_DEPLOY: "Deploying",
    STATE_SUPERCLIP: "Super-clipping",
    STATE_HARVEST: "Harvesting (partial throttle)",
    STATE_BRAKE: "Braking regen",
}

TEXT = "#1a1a1a"
SUBTEXT = "#4a4a4a"
GRID = "#e0e0e0"


def filter_short_runs(states, min_length):
    filtered = list(states)
    i = 0
    while i < len(filtered):
        j = i
        while j < len(filtered) and filtered[j] == filtered[i]:
            j += 1
        if (j - i) < min_length and i > 0:
            filtered[i:j] = [filtered[i - 1]] * (j - i)
        i = j
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

    raw_states = classify_states(merged)
    merged["state"] = filter_short_runs(raw_states, MIN_STATE_RUN)

    meta = {
        "driver": lap["Driver"],
        "team": lap["Team"],
        "lap_time": str(lap["LapTime"]).split(" ")[-1][:10],
    }

    return merged, meta


def classify_states(df):
    states = []
    for _, row in df.iterrows():
        if row["Brake"] > 0:
            states.append(STATE_BRAKE)
        elif row["Throttle"] >= THROTTLE_FULL and row["dSpeed"] < DECEL_THRESHOLD:
            states.append(STATE_SUPERCLIP)
        elif row["Throttle"] >= THROTTLE_FULL and row["dSpeed"] >= 0:
            states.append(STATE_DEPLOY)
        else:
            states.append(STATE_HARVEST)
    return states


def build_figure(df, meta):
    fig, ax = plt.subplots(figsize=(12, 10), facecolor=BG)
    ax.set_facecolor(BG)

    ax.plot(df["X"], df["Y"], color=TRACK_GHOST, linewidth=8, solid_capstyle="round", zorder=1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.suptitle(
        f"Energy State Map  |  {meta['driver']}  |  {meta['team']}  |  {GP} {YEAR} Qualifying",
        color=TEXT, fontsize=13, fontweight="bold", y=0.96,
    )

    fig.text(
        0.5, 0.925, meta["lap_time"],
        ha="center", color=SUBTEXT, fontsize=10,
    )

    legend_elements = [
        Patch(facecolor=PALETTE[s], edgecolor="none", label=LABELS[s])
        for s in [STATE_DEPLOY, STATE_SUPERCLIP, STATE_HARVEST, STATE_BRAKE]
    ]
    ax.legend(
        handles=legend_elements, loc="upper right",
        fontsize=11, facecolor=BG, edgecolor=GRID,
        labelcolor=TEXT, framealpha=0.9,
    )

    return fig, ax


def run(df, meta):
    seg_colors = [PALETTE[s] for s in df["state"]]
    fig, ax = build_figure(df, meta)

    trail_collection = None
    car_dot, = ax.plot([], [], "o", markersize=8, color="#1a1a1a", zorder=10)

    spd_text = ax.text(
        0.015, 0.05, "", transform=ax.transAxes,
        color=TEXT, fontsize=14, fontweight="bold", va="bottom", family="monospace",
    )
    state_text = ax.text(
        0.015, 0.10, "", transform=ax.transAxes,
        color=TEXT, fontsize=11, fontweight="bold", va="bottom", ha="left",
    )

    pct_texts = {}
    y_start = 0.82
    for i, s in enumerate([STATE_DEPLOY, STATE_SUPERCLIP, STATE_HARVEST, STATE_BRAKE]):
        pct_texts[s] = ax.text(
            0.985, y_start - i * 0.045, "", transform=ax.transAxes,
            color=PALETTE[s], fontsize=10, va="bottom", ha="right", family="monospace",
        )

    indices = list(range(0, len(df), FRAME_STEP))
    if indices[-1] != len(df) - 1:
        indices.append(len(df) - 1)

    def update(frame_num):
        nonlocal trail_collection

        idx = indices[frame_num]
        sub = df.iloc[: idx + 1]

        if trail_collection is not None:
            trail_collection.remove()

        pts = np.column_stack([sub["X"].values, sub["Y"].values]).reshape(-1, 1, 2)
        if len(pts) > 1:
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            trail_collection = LineCollection(
                segs, colors=seg_colors[:idx], linewidths=3.5, zorder=2
            )
            ax.add_collection(trail_collection)

        current_state = df["state"].iloc[idx]
        car_dot.set_data([df["X"].iloc[idx]], [df["Y"].iloc[idx]])
        car_dot.set_color(PALETTE[current_state])

        spd_text.set_text(f"{sub['Speed'].iloc[-1]:.0f} km/h")
        state_text.set_text(LABELS[current_state])
        state_text.set_color(PALETTE[current_state])

        total = len(sub)
        state_counts = pd.Series(sub["state"].values).value_counts()
        for s, txt in pct_texts.items():
            pct = state_counts.get(s, 0) / total * 100
            txt.set_text(f"{LABELS[s]}: {pct:.1f}%")

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