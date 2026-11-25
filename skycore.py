# SkyCore: a small global weather forecasting model, written from scratch.
#
# One file on purpose: the same file runs on a laptop (smoke test) and as a
# Kaggle script kernel (full training on a free GPU).
#
#   python skycore.py prepare  --years 2019 2020 --channels surface
#   python skycore.py train    --hours 0.05
#   python skycore.py evaluate
#   python skycore.py forecast --init 2020-12-01T00
#
# Models:  s1 = standard, d1 = twice as deep and tuned on longer rollouts,
#          w1 = small base model for the community, trained on GFS only.
# Data:    r1  = NCEP/NCAR Reanalysis 1 (NOAA PSL), 2.5 degree, 6-hourly
#          gfs = NOAA GFS analyses from the NOAA Open Data Dissemination
#                bucket on AWS (CC0), 1 degree, regridded to 2.5 degree
# No pretrained weights and no third-party model code are used.

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np

PSL = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis"
EPOCH = datetime(1900, 1, 1, tzinfo=timezone.utc)
NLAT, NLON = 73, 144  # 90N..90S, 0E..357.5E

# name: (subdir, file stem, netCDF variable, level in hPa or None, rough mean, rough std)
# The rough mean/std are only used to store values compactly as float16.
# The statistics used for training are measured from the training years.
CHANNELS = {
    "z850": ("pressure", "hgt", "hgt", 850, 1450.0, 120.0),
    "z500": ("pressure", "hgt", "hgt", 500, 5550.0, 250.0),
    "z250": ("pressure", "hgt", "hgt", 250, 10400.0, 450.0),
    "t850": ("pressure", "air", "air", 850, 275.0, 15.0),
    "t500": ("pressure", "air", "air", 500, 253.0, 13.0),
    "t250": ("pressure", "air", "air", 250, 222.0, 8.0),
    "u850": ("pressure", "uwnd", "uwnd", 850, 1.5, 8.0),
    "u500": ("pressure", "uwnd", "uwnd", 500, 6.5, 12.0),
    "u250": ("pressure", "uwnd", "uwnd", 250, 13.0, 18.0),
    "v850": ("pressure", "vwnd", "vwnd", 850, 0.0, 6.0),
    "v500": ("pressure", "vwnd", "vwnd", 500, 0.0, 9.0),
    "v250": ("pressure", "vwnd", "vwnd", 250, 0.0, 13.0),
    "slp": ("surface", "slp", "slp", None, 101100.0, 1300.0),
    "tsfc": ("surface", "air.sig995", "air", None, 278.0, 21.0),
    "usfc": ("surface", "uwnd.sig995", "uwnd", None, 0.0, 6.0),
    "vsfc": ("surface", "vwnd.sig995", "vwnd", None, 0.0, 5.0),
    "pwat": ("surface", "pr_wtr.eatm", "pr_wtr", None, 18.0, 16.0),
}
SURFACE = ["slp", "tsfc", "usfc", "vsfc", "pwat"]
PRESETS = {"all": list(CHANNELS), "surface": SURFACE}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def fetch(url, dest, tries=40):  # patient: the server can be away for many minutes
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    tmp = dest + ".part"
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "skycore/0.1"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
                want = r.headers.get("Content-Length")
            if want and os.path.getsize(tmp) != int(want):
                raise IOError("download cut short")
            os.replace(tmp, dest)
            return dest
        except Exception as e:  # network errors are expected on long runs
            log(f"  retry {k + 1}/{tries} {url}: {e}")
            time.sleep(min(60, 5 * (k + 1)))
    raise RuntimeError(f"download failed: {url}")


def read_nc(path, var, levels=None):
    try:
        import netCDF4
    except ImportError:  # not preinstalled on every hosted notebook image
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "netCDF4"])
        import netCDF4

    with netCDF4.Dataset(path) as ds:
        v = ds.variables[var]
        t = np.asarray(ds.variables["time"][:], dtype=np.float64)  # hours since 1800-01-01
        lat = np.asarray(ds.variables["lat"][:])
        if levels is None:
            a = v[:]
        else:
            lev = list(np.asarray(ds.variables["level"][:]).astype(int))
            a = v[:, [lev.index(x) for x in levels]]
    a = np.ma.filled(a.astype(np.float32), np.nan)
    assert lat[0] > lat[-1], "expected north-to-south latitude"
    hours = t - (EPOCH - datetime(1800, 1, 1, tzinfo=timezone.utc)).total_seconds() / 3600.0
    return a, np.round(hours).astype(np.int64)


def prepare_year(year, names, data_dir, raw_dir, keep_raw=False):
    out = os.path.join(data_dir, f"r1_{year}.npy")
    if os.path.exists(out):
        return out
    files = {}
    for n in names:
        sub, stem, var, lev, _, _ = CHANNELS[n]
        files.setdefault((sub, stem, var), []).append((n, lev))
    paths = {}
    with ThreadPoolExecutor(9) as ex:
        futs = {
            k: ex.submit(fetch, f"{PSL}/{k[0]}/{k[1]}.{year}.nc", os.path.join(raw_dir, f"{k[1]}.{year}.nc"))
            for k in files
        }
        for k, f in futs.items():
            paths[k] = f.result()
    fields, hours = {}, None
    for k, items in files.items():
        levels = [lev for _, lev in items] if items[0][1] is not None else None
        a, h = read_nc(paths[k], k[2], levels)
        if hours is None or len(h) < len(hours):
            hours = h
        for i, (n, _) in enumerate(items):
            fields[n] = a[:, i] if levels is not None else a
    T = len(hours)
    arr = np.empty((T, len(names), NLAT, NLON), dtype=np.float16)
    for c, n in enumerate(names):
        f = fields[n][:T]
        if not np.isfinite(f).all():
            raise ValueError(f"{n} {year}: non-finite values")
        arr[:, c] = (f - CHANNELS[n][4]) / CHANNELS[n][5]
    if not (np.diff(hours) == 6).all():
        raise ValueError(f"{year}: time axis is not 6-hourly")
    np.save(out, arr)
    np.save(os.path.join(data_dir, f"hours_{year}.npy"), hours)
    if not keep_raw:
        for p in paths.values():
            os.remove(p)
    return out


def prepare(years, names, data_dir, keep_raw=False):
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    for stem, var in (("land", "land"), ("hgt.sfc", "hgt")):
        p = fetch(f"{PSL}/surface/{stem}.nc", os.path.join(raw_dir, f"{stem}.nc"))
        a, _ = read_nc(p, var)
        np.save(os.path.join(data_dir, f"static_{var}.npy"), a[0].astype(np.float32))
    t0 = time.time()
    for y in years:
        prepare_year(y, names, data_dir, raw_dir, keep_raw)
        log(f"prepared {y}  ({time.time() - t0:.0f}s elapsed)")
    with open(os.path.join(data_dir, "meta.json"), "w") as f:
        json.dump({"channels": names, "years": list(years), "source": "r1"}, f)


# --------------------------------------------------------------------------
# data: GFS analyses (NOAA Open Data Dissemination, CC0)
# --------------------------------------------------------------------------

GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
GFS_FIRST = datetime(2021, 1, 1, tzinfo=timezone.utc)
_SIG, _COL = "0.995 sigma level", "entire atmosphere (considered as a single layer)"
GFS_FIELDS = {
    **{f"{k}{p}": (v, f"{p} mb") for k, v in (("z", "HGT"), ("t", "TMP"), ("u", "UGRD"), ("v", "VGRD"))
       for p in (850, 500, 250)},
    "slp": ("PRMSL", "mean sea level"), "tsfc": ("TMP", _SIG), "usfc": ("UGRD", _SIG),
    "vsfc": ("VGRD", _SIG), "pwat": ("PWAT", _COL),
    "land": ("LAND", "surface"), "orog": ("HGT", "surface"),
}


def http_get(url, byte_range=None, tries=4):
    for k in range(tries):
        try:
            h = {"User-Agent": "skycore/0.1"}
            if byte_range:
                h["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            time.sleep(2 * (k + 1))
        except Exception:
            time.sleep(2 * (k + 1))
    return None


def decode_grib(msg):
    try:
        import eccodes
    except ImportError:
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "eccodes"])
        import eccodes
    h = eccodes.codes_new_from_message(msg)
    try:
        ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
        return eccodes.codes_get_values(h).reshape(nj, ni).astype(np.float32)
    finally:
        eccodes.codes_release(h)


def to_grid(a):
    """1 degree (181x360, north to south) -> 2.5 degree (73x144).

    A light 1-2-1 smoothing removes scales the coarse grid cannot hold,
    then values are read off by linear interpolation.
    """
    a = 0.25 * np.roll(a, 1, 1) + 0.5 * a + 0.25 * np.roll(a, -1, 1)
    p = np.concatenate([a[:1], a, a[-1:]], 0)
    a = 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:]
    y = np.arange(NLAT) * 2.5
    x = np.arange(NLON) * 2.5
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    y1, x1 = np.minimum(y0 + 1, 180), (x0 + 1) % 360
    fy, fx = (y - y0)[:, None], (x - x0)[None, :]
    top = a[y0][:, x0] * (1 - fx) + a[y0][:, x1] * fx
    bot = a[y1][:, x0] * (1 - fx) + a[y1][:, x1] * fx
    return (top * (1 - fy) + bot * fy).astype(np.float32)


def gfs_fields(t, names):
    """Analysis fields at time t, regridded. None if the cycle is not in the bucket."""
    d, hh = t.strftime("%Y%m%d"), t.strftime("%H")
    for sub in ("atmos/", ""):
        url = f"{GFS}/gfs.{d}/{hh}/{sub}gfs.t{hh}z.pgrb2.1p00.f000"
        idx = http_get(url + ".idx")
        if idx:
            break
    else:
        return None
    rows = [r.split(":") for r in idx.decode().strip().splitlines()]
    offs = [int(r[1]) for r in rows]
    out = {}
    for n in names:
        var, lev = GFS_FIELDS[n]
        hit = [i for i, r in enumerate(rows) if r[3] == var and r[4] == lev and r[5] == "anl"]
        if not hit:
            return None
        i = hit[0]
        end = offs[i + 1] - 1 if i + 1 < len(offs) else ""
        want = offs[i + 1] - offs[i] if i + 1 < len(offs) else None
        field = None
        for _ in range(3):  # a cut-off or damaged message is fetched again, then the cycle is skipped
            msg = http_get(url, (offs[i], end))
            if not msg or msg[-4:] != b"7777" or (want and len(msg) != want):
                continue
            try:
                field = decode_grib(msg)
            except Exception:
                continue
            if field.shape == (181, 360):
                break
            field = None
        if field is None:
            return None
        out[n] = to_grid(field)
    return out


def prepare_gfs_year(year, names, data_dir, limit=None):
    out = os.path.join(data_dir, f"gfs_{year}.npy")
    if os.path.exists(out):
        return
    t = max(datetime(year, 1, 1, tzinfo=timezone.utc), GFS_FIRST)
    stop = min(datetime(year + 1, 1, 1, tzinfo=timezone.utc), datetime.now(timezone.utc) - timedelta(days=2))
    times = []
    while t < stop and (limit is None or len(times) < limit):
        times.append(t)
        t += timedelta(hours=6)

    def one(t):
        f = gfs_fields(t, names)
        if f is None or not all(np.isfinite(f[n]).all() for n in names):
            return None
        return np.stack([(f[n] - CHANNELS[n][4]) / CHANNELS[n][5] for n in names]).astype(np.float16)

    with ThreadPoolExecutor(16) as ex:
        frames = list(ex.map(one, times))
    keep = [i for i, f in enumerate(frames) if f is not None]
    log(f"gfs {year}: {len(keep)}/{len(times)} analyses")
    np.save(out, np.stack([frames[i] for i in keep]))
    np.save(os.path.join(data_dir, f"hours_{year}.npy"),
            np.array([int((times[i] - EPOCH).total_seconds() // 3600) for i in keep], dtype=np.int64))


def prepare_gfs(years, names, data_dir, limit=None):
    os.makedirs(data_dir, exist_ok=True)
    f = gfs_fields(datetime(2024, 1, 1, tzinfo=timezone.utc), ["land", "orog"])
    np.save(os.path.join(data_dir, "static_land.npy"), f["land"])
    np.save(os.path.join(data_dir, "static_hgt.npy"), f["orog"])
    t0 = time.time()
    for y in years:
        prepare_gfs_year(y, names, data_dir, limit)
        log(f"prepared {y}  ({time.time() - t0:.0f}s elapsed)")
    with open(os.path.join(data_dir, "meta.json"), "w") as fh:
        json.dump({"channels": names, "years": list(years), "source": "gfs"}, fh)


class Archive:
    """All prepared years, addressed by one global time index."""

    def __init__(self, data_dir, years=None):
        meta = json.load(open(os.path.join(data_dir, "meta.json")))
        self.names = meta["channels"]
        self.years = [y for y in meta["years"] if years is None or y in years]
        src = meta.get("source", "r1")
        self.parts = [np.load(os.path.join(data_dir, f"{src}_{y}.npy"), mmap_mode="r") for y in self.years]
        self.hours = np.concatenate([np.load(os.path.join(data_dir, f"hours_{y}.npy")) for y in self.years])
        self.starts = np.cumsum([0] + [len(p) for p in self.parts])
        self.rough_mean = np.array([CHANNELS[n][4] for n in self.names], dtype=np.float32)
        self.rough_std = np.array([CHANNELS[n][5] for n in self.names], dtype=np.float32)
        self.land = np.load(os.path.join(data_dir, "static_land.npy"))
        self.orog = np.load(os.path.join(data_dir, "static_hgt.npy"))

    def windows(self, a, b, n):
        """Start indices in [a, b) of runs of n frames with no missing time."""
        i = np.arange(a, b - n + 1)
        return i[self.hours[i + n - 1] - self.hours[i] == 6 * (n - 1)]

    def __len__(self):
        return int(self.starts[-1])

    def frame(self, i):
        k = int(np.searchsorted(self.starts, i, side="right") - 1)
        return self.parts[k][i - self.starts[k]]

    def frames(self, i, n):
        return np.stack([self.frame(j) for j in range(i, i + n)])

    def year_range(self, y0, y1):
        """Global index range [a, b) covering years y0..y1 inclusive."""
        a = self.starts[self.years.index(y0)]
        b = self.starts[self.years.index(y1) + 1]
        return int(a), int(b)


def when(hours):
    return EPOCH + timedelta(hours=int(hours))


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def build_model(n_state, n_forcing, width, depth):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    def sphere_pad(x, p):
        # longitude wraps around; latitude is clamped at the poles
        x = F.pad(x, (p, p, 0, 0), mode="circular")
        return F.pad(x, (0, 0, p, p), mode="replicate")

    class Block(nn.Module):
        def __init__(self, w):
            super().__init__()
            self.dw = nn.Conv2d(w, w, 7, groups=w)
            self.norm = nn.GroupNorm(1, w)
            self.up = nn.Conv2d(w, 4 * w, 1)
            self.down = nn.Conv2d(4 * w, w, 1)
            self.scale = nn.Parameter(torch.full((1, w, 1, 1), 1e-2))

        def forward(self, x):
            h = self.dw(sphere_pad(x, 3))
            h = self.down(F.gelu(self.up(self.norm(h))))
            return x + self.scale * h

    class Net(nn.Module):
        """Predicts the 6-hour change of the state from the last two states."""

        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(2 * n_state + n_forcing, width, 3)
            self.blocks = nn.Sequential(*[Block(width) for _ in range(depth)])
            self.norm = nn.GroupNorm(1, width)
            self.head = nn.Conv2d(width, n_state, 3)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            self.save_memory = False  # training: recompute block activations in the backward pass

        def forward(self, prev, cur, forcing):
            x = torch.cat([cur, cur - prev, forcing], 1)
            x = self.stem(sphere_pad(x, 1))
            for blk in self.blocks:
                if self.save_memory and x.requires_grad:
                    x = checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x)
            return self.head(sphere_pad(self.norm(x), 1))

    return Net()


class Physics:
    """Normalisation, forcings and the time-stepping rule, on one device."""

    N_FORCING = 10

    def __init__(self, stats, land, orog, device):
        import torch

        self.torch = torch
        self.dev = device
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)
        self.mean = t(stats["mean"]).view(1, -1, 1, 1)
        self.std = t(stats["std"]).view(1, -1, 1, 1)
        self.dstd = t(stats["dstd"]).view(1, -1, 1, 1)  # std of 6h change, in normalised units
        lat = torch.linspace(90, -90, NLAT, device=device) * math.pi / 180
        lon = torch.arange(NLON, device=device) * (2 * math.pi / NLON)
        self.lon = lon.view(1, 1, 1, NLON)
        w = torch.cos(lat).clamp_min(math.cos(math.radians(89.375)) / 2)
        self.area = (w / w.mean()).view(1, 1, NLAT, 1)
        ones = torch.ones(1, 1, NLAT, NLON, device=device)
        self.static = torch.cat(
            [
                torch.sin(lat).view(1, 1, NLAT, 1) * ones,
                torch.cos(lat).view(1, 1, NLAT, 1) * ones,
                torch.sin(self.lon) * ones,
                torch.cos(self.lon) * ones,
                t(land).view(1, 1, NLAT, NLON),
                t(orog).view(1, 1, NLAT, NLON) / 1000.0,
            ],
            1,
        )

    def normalise(self, rough):
        """float16 archive values -> normalised state."""
        return (rough.to(self.dev).float() - self.mean) / self.std

    def forcing(self, hours):
        torch = self.torch
        h = torch.as_tensor(hours, device=self.dev, dtype=torch.float32).view(-1, 1, 1, 1)
        season = 2 * math.pi * (h / 24.0 % 365.2425) / 365.2425
        local = 2 * math.pi * (h % 24.0) / 24.0 + self.lon
        B = h.shape[0]
        ones = torch.ones(B, 1, NLAT, NLON, device=self.dev)
        return torch.cat(
            [
                torch.sin(season) * ones,
                torch.cos(season) * ones,
                torch.sin(local) * torch.ones(B, 1, NLAT, 1, device=self.dev),
                torch.cos(local) * torch.ones(B, 1, NLAT, 1, device=self.dev),
                self.static.expand(B, -1, -1, -1),
            ],
            1,
        )

    def step(self, model, prev, cur, hours):
        return cur + model(prev, cur, self.forcing(hours)) * self.dstd

    def loss(self, pred, target):
        return (((pred - target) / self.dstd) ** 2 * self.area).mean()


def measure_stats(arc, a, b, n=600, seed=0):
    """Mean/std per channel and std of the 6-hour change, from training years."""
    rng = np.random.default_rng(seed)
    ok = arc.windows(a, b, 2)
    idx = np.sort(rng.choice(ok, size=min(n, len(ok)), replace=False))
    x = np.stack([arc.frame(i) for i in idx]).astype(np.float32)
    y = np.stack([arc.frame(i + 1) for i in idx]).astype(np.float32)
    mean = x.mean((0, 2, 3))
    std = x.std((0, 2, 3))
    dstd = (y - x).std((0, 2, 3)) / std
    return {"mean": mean.tolist(), "std": std.tolist(), "dstd": dstd.tolist()}


def make_loader(arc, a, b, steps, batch, workers, shuffle=True):
    import torch

    starts = arc.windows(a, b, steps + 2)

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(starts)

        def __getitem__(self, i):
            j = int(starts[i])
            return torch.from_numpy(arc.frames(j, steps + 2)), int(arc.hours[j + 1])

    return torch.utils.data.DataLoader(
        DS(), batch_size=batch, shuffle=shuffle, num_workers=workers, drop_last=True,
        persistent_workers=workers > 0, pin_memory=True,
    )


def rollout_loss(model, phys, frames, hrs, steps, amp):
    import torch

    x = phys.normalise(frames)
    prev, cur = x[:, 0], x[:, 1]
    loss = 0.0
    with torch.autocast("cuda", enabled=amp):
        for s in range(steps):
            nxt = phys.step(model, prev, cur, hrs + 6 * s)
            loss = loss + phys.loss(nxt.float(), x[:, 2 + s]) / steps
            prev, cur = cur, nxt
    return loss


def is_oom(e):
    return "out of memory" in str(e).lower()


def fit_batch(model, phys, steps, batch, n_state):
    """Largest batch (halving from the requested one) whose training step fits on the GPU."""
    import torch

    if phys.dev != "cuda":
        return batch
    total = torch.cuda.get_device_properties(0).total_memory
    while True:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            frames = torch.zeros(batch, steps + 2, n_state, NLAT, NLON, dtype=torch.float16)
            hrs = torch.full((batch,), 1_000_000, dtype=torch.long)
            rollout_loss(model, phys, frames, hrs, steps, True).backward()
            peak = torch.cuda.max_memory_allocated()
            model.zero_grad(set_to_none=True)
            if peak < 0.8 * total or batch == 1:
                log(f"rollout={steps} batch={batch} fits: peak {peak / 2 ** 30:.1f} of {total / 2 ** 30:.1f} GB")
                return batch
        except RuntimeError as e:
            if not is_oom(e) or batch == 1:
                raise
            model.zero_grad(set_to_none=True)
        batch = max(1, batch // 2)


def phase_plan(args):
    return [(1, args.batch, args.lr, args.hours * (1 - args.rollout_share)),
            (args.rollout, max(2, args.batch // args.rollout), args.lr / 10, args.hours * args.rollout_share)]


def preflight_memory(args):
    """Before any data is fetched: build the model and check that its training steps fit."""
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    C = len(PRESETS[args.channels])
    zero = np.zeros((NLAT, NLON), dtype=np.float32)
    phys = Physics({"mean": [0.0] * C, "std": [1.0] * C, "dstd": [1.0] * C}, zero, zero, dev)
    model = build_model(C, Physics.N_FORCING, args.width, args.depth).to(dev).to(memory_format=torch.channels_last)
    model.save_memory = True
    args.fitted = {steps: fit_batch(model, phys, steps, batch, C) for steps, batch, _, hours in phase_plan(args)
                   if hours > 0}
    log(f"preflight: device={dev} batch sizes {args.fitted}")
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()


def train(args):
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    arc = Archive(args.data)
    ty0, ty1 = args.train_years
    a, b = arc.year_range(ty0, ty1)
    os.makedirs(args.out, exist_ok=True)
    stats = measure_stats(arc, a, b)
    phys = Physics(stats, arc.land, arc.orog, dev)
    model = build_model(len(arc.names), Physics.N_FORCING, args.width, args.depth).to(dev)
    model = model.to(memory_format=torch.channels_last)
    model.save_memory = True
    log(f"device={dev} channels={len(arc.names)} train={ty0}-{ty1} ({b - a} steps) "
        f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    amp = dev == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)
    cfg = {"model": args.model, "channels": arc.names, "width": args.width, "depth": args.depth, "stats": stats,
           "train_years": [ty0, ty1]}

    def save(tag):
        torch.save({"model": model.state_dict(), "cfg": cfg}, os.path.join(args.out, tag))

    # phase 1: one step ahead.  phase 2: several steps unrolled, so errors that
    # grow when the model is fed its own output are penalised.
    fitted = getattr(args, "fitted", {})
    fake_oom = os.environ.get("SKYCORE_TEST_OOM")  # test hook: pretend the GPU filled up once
    history = []
    for steps, batch, lr, hours in phase_plan(args):
        if hours <= 0:
            continue
        batch = fitted.get(steps) or fit_batch(model, phys, steps, batch, len(arc.names))
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        budget, t0, it, run, seen = hours * 3600, time.time(), 0, None, 0
        done = False
        while not done:
            loader = make_loader(arc, a, b, steps, batch, args.workers)
            for frames, hrs in loader:
                frac = (time.time() - t0) / budget
                if frac >= 1:
                    done = True
                    break
                warm = min(1.0, (it + 1) / 500)
                for g in opt.param_groups:
                    g["lr"] = lr * warm * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * frac)))
                try:
                    if fake_oom and it == 3:
                        fake_oom = None
                        raise RuntimeError("CUDA out of memory (test)")
                    loss = rollout_loss(model, phys, frames, hrs, steps, amp)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                except RuntimeError as e:  # out of memory mid-run: carry on with half the batch
                    if not is_oom(e) or batch == 1:
                        raise
                    loss = None
                    opt.zero_grad(set_to_none=True)
                    if dev == "cuda":
                        torch.cuda.empty_cache()
                    batch = max(1, batch // 2)
                    log(f"out of memory, continuing with batch={batch}")
                    break
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
                it += 1
                seen += batch
                if it % args.log_every == 0:
                    log(f"rollout={steps} it={it} loss={run:.4f} {100 * frac:.0f}% "
                        f"{seen / (time.time() - t0):.0f} samples/s")
                    history.append({"rollout": steps, "it": it, "loss": run})
                if it % 2000 == 0:
                    save("model.pt")
        save("model.pt")
        log(f"phase rollout={steps} finished after {it} iterations, loss={run}")
    json.dump(history, open(os.path.join(args.out, "train_log.json"), "w"))


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def climatology(arc, a, b):
    """Mean state per (day of year, hour), smoothed over +-10 days. Rough units."""
    C = len(arc.names)
    s = np.zeros((366, 4, C, NLAT, NLON), dtype=np.float32)
    n = np.zeros((366, 4), dtype=np.float32)
    for i in range(a, b):
        d = when(arc.hours[i])
        k = d.timetuple().tm_yday - 1
        s[k, d.hour // 6] += arc.frame(i)
        n[k, d.hour // 6] += 1
    out = np.empty_like(s)
    for k in range(366):
        idx = [(k + j) % 366 for j in range(-10, 11)]
        out[k] = s[idx].sum(0) / np.maximum(n[idx].sum(0), 1)[:, None, None, None]
    return out


def load_model(out_dir, dev):
    """Loads model.pt, or model.npz + model.json as written by the JAX backend."""
    import torch

    pt = os.path.join(out_dir, "model.pt")
    if os.path.exists(pt):
        ck = torch.load(pt, map_location=dev)
        cfg, state = ck["cfg"], ck["model"]
    else:
        cfg = json.load(open(os.path.join(out_dir, "model.json")))
        state = {k: torch.from_numpy(v) for k, v in np.load(os.path.join(out_dir, "model.npz")).items()}
    model = build_model(len(cfg["channels"]), Physics.N_FORCING, cfg["width"], cfg["depth"]).to(dev)
    model.load_state_dict(state)
    return model.eval(), cfg


def area_weights():
    w = np.maximum(np.cos(np.radians(np.linspace(90, -90, NLAT))), math.cos(math.radians(89.375)) / 2)
    return (w / w.mean()).astype(np.float32)


def score(arc, cfg, args, advance):
    """Roll forecasts out over the test years and compare with the archive.

    advance(prev, cur, hours) -> next state; normalised float32 arrays (B, C, lat, lon).
    """
    a, b = arc.year_range(*cfg["train_years"])
    log("building climatology from training years")
    clim = climatology(arc, a, b)
    ta, tb = arc.year_range(*args.test_years)
    L = args.lead_days * 4
    inits = [int(i) + 1 for i in arc.windows(ta, tb, L + 2)[::args.init_every]]
    log(f"test={args.test_years} forecasts={len(inits)} lead={args.lead_days} days")
    C = len(arc.names)
    area = area_weights()
    mean = np.asarray(cfg["stats"]["mean"], dtype=np.float32).reshape(1, C, 1, 1)
    std = np.asarray(cfg["stats"]["std"], dtype=np.float32).reshape(1, C, 1, 1)
    se = {k: np.zeros((L, C)) for k in ("model", "persistence", "climatology")}
    acc = np.zeros((L, C))

    def wmean(f):  # area-weighted mean over the globe, per forecast and channel
        return (f * area[None, None, :, None]).mean((2, 3))

    def clim_at(i):
        d = when(arc.hours[i])
        return clim[d.timetuple().tm_yday - 1, d.hour // 6]

    for s in range(0, len(inits), args.batch):
        idx = inits[s:s + args.batch]
        x = (np.stack([arc.frames(i - 1, 2) for i in idx]).astype(np.float32) - mean[:, None]) / std[:, None]
        prev, cur = x[:, 0], x[:, 1]
        start = np.stack([arc.frame(i) for i in idx]).astype(np.float32)
        hrs = np.array([int(arc.hours[i]) for i in idx], dtype=np.int64)
        for l in range(L):
            cur, prev = np.asarray(advance(prev, cur, hrs + 6 * l), dtype=np.float32), cur
            truth = np.stack([arc.frame(i + l + 1) for i in idx]).astype(np.float32)
            cl = np.stack([clim_at(i + l + 1) for i in idx])
            fc = cur * std + mean  # archive units
            se["model"][l] += wmean((fc - truth) ** 2).sum(0)
            se["persistence"][l] += wmean((start - truth) ** 2).sum(0)
            se["climatology"][l] += wmean((cl - truth) ** 2).sum(0)
            fa, ta_ = fc - cl, truth - cl
            acc[l] += (wmean(fa * ta_) / np.sqrt(wmean(fa ** 2) * wmean(ta_ ** 2))).sum(0)
    n = len(inits)
    res = {"model": cfg.get("model"), "channels": arc.names, "n_forecasts": n,
           "test_years": list(args.test_years), "lead_hours": [6 * (l + 1) for l in range(L)],
           "rmse": {k: (np.sqrt(v / n) * arc.rough_std).tolist() for k, v in se.items()},
           "acc": (acc / n).tolist()}
    json.dump(res, open(os.path.join(args.out, "scores.json"), "w"))
    print_scores(res)
    return res


def evaluate(args):
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    arc = Archive(args.data)
    model, cfg = load_model(args.out, dev)
    assert cfg["channels"] == arc.names
    phys = Physics(cfg["stats"], arc.land, arc.orog, dev)
    t = lambda x: torch.as_tensor(x, device=dev)

    def advance(prev, cur, hours):
        with torch.no_grad():
            return phys.step(model, t(prev), t(cur), t(hours)).cpu().numpy()

    return score(arc, cfg, args, advance)


def print_scores(res):
    show = [c for c in ("z500", "t850", "tsfc", "slp") if c in res["channels"]]
    for c in show:
        k = res["channels"].index(c)
        print(f"\n{c}: RMSE by lead time (test years {res['test_years']}, {res['n_forecasts']} forecasts)")
        print(f"{'lead':>6} {'model':>10} {'persist':>10} {'climate':>10} {'ACC':>6}")
        for l, h in enumerate(res["lead_hours"]):
            if h in (6, 24, 72, 120, 168, 240):
                r = res["rmse"]
                print(f"{h:>5}h {r['model'][l][k]:>10.2f} {r['persistence'][l][k]:>10.2f} "
                      f"{r['climatology'][l][k]:>10.2f} {res['acc'][l][k]:>6.3f}")
    sys.stdout.flush()


def forecast(args):
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    arc = Archive(args.data)
    model, cfg = load_model(args.out, dev)
    phys = Physics(cfg["stats"], arc.land, arc.orog, dev)
    t = datetime.strptime(args.init, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
    h = int((t - EPOCH).total_seconds() // 3600)
    i = int(np.searchsorted(arc.hours, h))
    if i < 1 or i >= len(arc) or arc.hours[i] != h or arc.hours[i - 1] != h - 6:
        raise SystemExit(f"{args.init} is not in the prepared archive")
    x = phys.normalise(torch.from_numpy(arc.frames(i - 1, 2)[None]))
    prev, cur = x[:, 0], x[:, 1]
    out = []
    with torch.no_grad():
        for l in range(args.lead_days * 4):
            cur, prev = phys.step(model, prev, cur, torch.tensor([h + 6 * l])), cur
            out.append((cur * phys.std + phys.mean).cpu().numpy()[0])
    out = np.stack(out) * arc.rough_std[None, :, None, None] + arc.rough_mean[None, :, None, None]
    path = os.path.join(args.out, f"forecast_{args.init.replace(':', '')}.npz")
    np.savez_compressed(path, fields=out.astype(np.float32), channels=np.array(arc.names),
                        init=args.init, lead_hours=np.arange(1, len(out) + 1) * 6,
                        lat=np.linspace(90, -90, NLAT), lon=np.arange(NLON) * 2.5)
    log(f"wrote {path}  shape={out.shape}")


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------

MODELS = {
    "s1": dict(source="r1", width=192, depth=12, rollout=4, rollout_share=0.25,
               years=[1979, 2025], train_years=[1979, 2022], test_years=[2024, 2025]),
    "d1": dict(source="r1", width=192, depth=24, rollout=8, rollout_share=0.35,
               years=[1979, 2025], train_years=[1979, 2022], test_years=[2024, 2025]),
    "w1": dict(source="gfs", width=96, depth=8, rollout=4, rollout_share=0.25,
               years=[2021, 2026], train_years=[2021, 2024], test_years=[2025, 2026]),
}
KAGGLE_MODEL = "s1"
KAGGLE_HOURS = "5.0"
KAGGLE_BACKEND = "torch"
KAGGLE_PREFLIGHT_ONLY = False


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="s1", choices=list(MODELS), help="sets the defaults of the options below")
    p.add_argument("--source", default=None, choices=["r1", "gfs"])
    p.add_argument("--backend", default="torch", choices=["torch", "jax"])
    p.add_argument("--limit", type=int, default=None, help="gfs only: analyses per year (for quick tests)")
    p.add_argument("command", choices=["prepare", "train", "evaluate", "forecast", "all"])
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="out")
    p.add_argument("--years", type=int, nargs=2, default=None, help="first and last year to prepare")
    p.add_argument("--channels", default="all", choices=list(PRESETS))
    p.add_argument("--keep-raw", action="store_true")
    p.add_argument("--train-years", type=int, nargs=2, default=None)
    p.add_argument("--test-years", type=int, nargs=2, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hours", type=float, default=6.0, help="training time budget")
    p.add_argument("--rollout", type=int, default=None)
    p.add_argument("--rollout-share", type=float, default=None)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--lead-days", type=int, default=10)
    p.add_argument("--init-every", type=int, default=21, help="spacing of test forecasts in 6h steps")
    p.add_argument("--init", default=None, help="forecast start, e.g. 2025-01-15T00")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    for k, v in MODELS[args.model].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.command == "all" and args.backend == "torch":
        preflight_memory(args)
    if args.command in ("prepare", "all"):
        years = range(args.years[0], args.years[1] + 1)
        if args.source == "gfs":
            prepare_gfs(years, PRESETS[args.channels], args.data, args.limit)
        else:
            prepare(years, PRESETS[args.channels], args.data, args.keep_raw)
    jx = None
    if args.backend == "jax":
        jx = sys.modules[__name__] if "train_jax" in globals() else __import__("skycore_jax")
    if args.command in ("train", "all"):
        jx.train_jax(args) if jx else train(args)
    if args.command in ("evaluate", "all"):
        jx.evaluate_jax(args) if jx else evaluate(args)
    if args.command == "forecast":
        forecast(args)


if __name__ == "__main__":
    if len(sys.argv) == 1 and os.path.isdir("/kaggle/working"):
        # Hosted script kernel: full run. Prepared data goes to scratch space,
        # only the model and the scores are kept as kernel output.
        if KAGGLE_BACKEND == "jax":
            n_dev, kind = describe_devices()
            if kind == "cpu":
                raise SystemExit("JAX sees no accelerator here; not starting a long run on CPU")
            preflight(KAGGLE_MODEL)
            if KAGGLE_PREFLIGHT_ONLY:
                sys.exit(0)
        main(["all", "--model", KAGGLE_MODEL, "--backend", KAGGLE_BACKEND, "--data", "/kaggle/tmp/skycore-data",
              "--out", "/kaggle/working", "--hours", KAGGLE_HOURS, "--workers", "3"])
    else:
        main()
