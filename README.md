# SkyCore W1

A from-scratch, open-source **small global weather forecasting model** — a tiny GraphCast-style network. W1 is the shared base of the SkyCore "We Models" project: both the code and the weights are open, and it is trained **only on openly licensed (public-domain / CC0) data**.

> This repository is the community base. The rule for the "We" line: the base is **W-series only** (never S/D), and training data must be **openly licensed (CC0) only**.

---

## What it is

- **Global gridded forecasting** (a small GraphCast-style model). The Earth is split into a 2.5° grid (73 latitudes × 144 longitudes); the model predicts the atmosphere 6 hours ahead, then rolls that forward autoregressively out to 10 days.
- **17 predicted variables**: at three upper levels (850 / 500 / 250 hPa) — geopotential height, temperature, eastward wind, northward wind — plus, at the surface — sea-level pressure, temperature, wind, and precipitable water.
- **~0.7M (690k) parameters** (width 96, depth 8). Trains from scratch in a few hours on a single ordinary GPU.

W1 aims to be **light and easy to fork** — a starting point for your own experiments. The accuracy-focused larger models are a separate line (S1 / D1).

---

## Measured skill

RMSE (lower is better) on the test period **2025–2026**, 130 forecasts. For reference we also show *persistence* ("tomorrow = today") and *climatology* ("the seasonal normal").

| Variable | 1 day | 3 days | 5 days | persistence (1d) | climatology |
|---|---|---|---|---|---|
| z500 geopotential height (m) | 11.7 | 35.1 | 59.1 | 59.9 | 84.2 |
| t850 lower-air temperature (°C) | 1.0 | 1.9 | 2.8 | 3.0 | 3.4 |
| slp sea-level pressure (Pa) | 144 | 360 | 573 | 582 | 724 |
| pwat precipitable water (mm) | 2.0 | 4.1 | 5.6 | 5.6 | 6.3 |

- **Beats both persistence and climatology at every lead and variable** — it genuinely forecasts the weather.
- Anomaly correlation (ACC ≥ 0.6, the usual "useful forecast" threshold) holds to about **6 days** for z500 and about **5.5 days** for t850 and slp.

> **Note on comparability.** These numbers are measured on W1's own test data on a coarse 2.5° grid. They are **not directly comparable** to GraphCast or operational high-resolution models, which use finer grids and different ground truth. Read them as confirmation that W1 is a working weather model, not as a leaderboard ranking.

---

## How it works

- **Input**: the last two states (now and 6 hours ago) + time cues (season, local solar time, latitude/longitude) + terrain and land/sea mask.
- **Output**: the 6-hour *tendency*, added to the current state to get the next one (learning the change is easier than learning the field).
- **Backbone**: a ConvNeXt-style convolutional net (7×7 depthwise conv → GroupNorm → 1×1 expand ×4 → GELU → project back, repeated 8 times). Because the Earth is a sphere, longitude is **circular-padded** and the poles are **replicate-padded**.
- **Loss**: latitude-area-weighted mean squared error, normalized by the spread of the 6-hour change.
- **Two training phases**: first one step ahead, then a **4-step rollout** at 1/10 the learning rate (so errors that grow when the model is fed its own output are penalized).
- **Memory tricks**: mixed precision (AMP), gradient checkpointing, an automatic pre-flight that measures the largest batch that fits on the GPU, and mid-run recovery that halves the batch on an out-of-memory error.

---

## Training data

- Openly licensed (public-domain / CC0) analysis fields, fetched on a 1° grid and regridded to 2.5° with light smoothing. 6-hourly, from 2021 to the present.
- Train: 2021–2024 / Test: 2025–2026.

Keep data in W1 **openly licensed (CC0) only** — that is the rule of the W line.

---

## Usage

Python plus (optionally) a GPU for training. Everything — data preparation, training, scoring, forecasting — is in the single file `skycore.py`.

```bash
# prepare -> train -> score, all at once (data is placed under --data)
python skycore.py all --model w1 --data ./data --out ./out

# or step by step
python skycore.py prepare  --model w1 --data ./data
python skycore.py train    --model w1 --data ./data --out ./out
python skycore.py evaluate --model w1 --data ./data --out ./out

# forecast with the trained model
python skycore.py forecast --model w1 --data ./data --out ./out
```

Missing libraries (numpy / torch / eccodes, …) are installed on first run. The first data download takes a while; later runs reuse it.

### Using the released weights

`w1/model.pt` is the trained checkpoint behind the scores above. Load it and run the forecast command with `--out` pointing at the folder that holds it.

---

## Files

| File | Role |
|---|---|
| `skycore.py` | Everything: data prep, model, training, scoring, forecasting |
| `skycore_jax.py` | JAX training backend (for TPU; verified to compute the same thing as the PyTorch version — weights are interchangeable) |
| `test_backends.py` | Test that the PyTorch and JAX backends agree |
| `w1/model.pt` | Trained W1 weights |
| `w1/scores.json` | Full RMSE / ACC tables behind this card |

---

## License

- Code and weights are both **Apache-2.0**. Use, modify, and redistribute freely.
- Trained only on openly licensed (public-domain / CC0) data.
