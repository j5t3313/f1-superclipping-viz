# F1 2026 Super-Clipping Visualizations

Animated visualizations of super-clipping and energy state behavior in the 2026 Formula 1 season, built with FastF1 telemetry data and Matplotlib.

## Scripts

### clipping_lap.py

Dual-panel animated GIF. The upper panel renders the circuit from car position data with a progressive color-coded trail. The lower panel draws a synchronized speed trace against distance with a secondary throttle axis. Regions where the car is decelerating on full throttle (the super-clipping signature) are highlighted on both panels simultaneously.

Trail color mapping:
- Teal: deploying (full throttle, accelerating)
- Red: super-clipping (full throttle, decelerating)
- Gray: partial throttle or coasting

### harvest_map.py

Single-panel animated GIF. The circuit trail is progressively color-coded by a four-state energy classification, with running percentage breakdowns and a live speed readout.

State classification:
- Blue: deploying (full throttle, accelerating)
- Red: super-clipping (full throttle, decelerating)
- Yellow: harvesting on partial throttle
- Gray: braking regen

## Detection Logic

Super-clipping is identified where two conditions are simultaneously true:

1. Throttle is at or above the configured threshold (98% in `clipping_lap.py`, 95% in `harvest_map.py`)
2. The first derivative of speed with respect to distance is below the configured deceleration threshold (-0.05 km/h per meter)

The speed signal is smoothed with a Savitzky-Golay filter (window 13, polynomial order 2) before the gradient is computed. A run-length filter removes detections shorter than 5 consecutive samples to suppress noise.

## Data Pipeline

Both scripts share the same data loading approach:

1. Load the fastest qualifying lap via `session.laps.pick_fastest()`
2. Merge car telemetry (`get_car_data().add_distance()`) with position data (`get_pos_data()`) using `pd.merge_asof` on the Time column
3. Apply linear interpolation to Speed, Throttle, X, Y
4. Apply forward fill to nGear, Brake
5. Drop any remaining null rows
6. Apply Savitzky-Golay smoothing to Speed
7. Compute `dSpeed` via `np.gradient(Speed, Distance)`
8. Classify and filter states

## Configuration

All tunable parameters are defined as constants at the top of each script.

| Parameter | clipping_lap.py | harvest_map.py | Description |
|---|---|---|---|
| `YEAR` | 2026 | 2026 | Season year |
| `GP` | China | China | Grand prix name (must match FastF1 event naming) |
| `SESSION_TYPE` | Q | Q | Session type (Q, R, FP1, FP2, FP3, S, SQ) |
| `THROTTLE_THRESHOLD` / `THROTTLE_FULL` | 98 | 95 | Minimum throttle % to qualify as full throttle |
| `DECEL_THRESHOLD` | -0.05 | -0.05 | dSpeed/dDistance cutoff for deceleration detection |
| `SAVGOL_WINDOW` | 13 | 13 | Savitzky-Golay filter window length (must be odd) |
| `SAVGOL_POLY` | 2 | 2 | Savitzky-Golay polynomial order |
| `MIN_CLIP_RUN` / `MIN_STATE_RUN` | 5 | 5 | Minimum consecutive samples for a detection to survive filtering |
| `FRAME_STEP` | 3 | 3 | Sample every Nth telemetry point for animation frames |
| `FPS` | 30 | 30 | GIF playback frame rate |
| `DPI` | 150 | 150 | Output resolution |

## Setup

```
pip install -r requirements.txt
```

## Usage

```
python clipping_lap.py
python harvest_map.py
```

Each script creates a `cache/` directory on first run for FastF1 data caching. Output GIFs are written to the working directory.

## Adapting to Other Circuits

Change the `GP` constant to any valid FastF1 event name. Circuits with long straights and limited heavy braking zones (Melbourne, Monza, Spa) will show more pronounced super-clipping. You may need to adjust `DECEL_THRESHOLD` per circuit depending on telemetry sample density.

## Dependencies

- Python 3.9+
- FastF1 3.8.1+ (required for 2026 season telemetry parsing)
- Matplotlib
- NumPy
- pandas
- SciPy
