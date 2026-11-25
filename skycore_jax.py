# SkyCore, JAX backend: the same model and training rule as skycore.py,
# written for accelerators that JAX drives well (TPU). Weights use the same
# names and shapes as the PyTorch model, so either backend can load them.

import json
import math
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np

try:
    import skycore as sc
except ImportError:  # single-file build: the core lives in this very module
    sc = sys.modules[__name__]

NLAT, NLON, N_FORCING = 73, 144, 10


def init_params(seed, n_state, width, depth):
    """Same initialisation as torch.nn.Conv2d / GroupNorm defaults."""
    rng = np.random.default_rng(seed)

    def conv(name, out_c, in_c, k, groups=1):
        fan_in = in_c // groups * k * k
        lim = 1.0 / math.sqrt(fan_in)
        p[name + ".weight"] = rng.uniform(-lim, lim, (out_c, in_c // groups, k, k)).astype(np.float32)
        p[name + ".bias"] = rng.uniform(-lim, lim, (out_c,)).astype(np.float32)

    def norm(name):
        p[name + ".weight"] = np.ones(width, np.float32)
        p[name + ".bias"] = np.zeros(width, np.float32)

    p = {}
    conv("stem", width, 2 * n_state + N_FORCING, 3)
    for i in range(depth):
        b = f"blocks.{i}"
        conv(b + ".dw", width, width, 7, groups=width)
        norm(b + ".norm")
        conv(b + ".up", 4 * width, width, 1)
        conv(b + ".down", width, 4 * width, 1)
        p[b + ".scale"] = np.full((1, width, 1, 1), 1e-2, np.float32)
    norm("norm")
    p["head.weight"] = np.zeros((n_state, width, 3, 3), np.float32)
    p["head.bias"] = np.zeros((n_state,), np.float32)
    return p


def make_net(width, depth):
    import jax
    import jax.numpy as jnp
    from jax import lax

    def conv(p, name, x, groups=1):
        y = lax.conv_general_dilated(x, p[name + ".weight"], (1, 1), "VALID",
                                     dimension_numbers=("NCHW", "OIHW", "NCHW"), feature_group_count=groups)
        return y + p[name + ".bias"][None, :, None, None]

    def sphere_pad(x, k):
        # longitude wraps around; latitude is clamped at the poles
        x = jnp.concatenate([x[..., -k:], x, x[..., :k]], -1)
        return jnp.pad(x, ((0, 0), (0, 0), (k, k), (0, 0)), mode="edge")

    def norm(p, name, x):
        m = x.mean((1, 2, 3), keepdims=True)
        v = ((x - m) ** 2).mean((1, 2, 3), keepdims=True)
        return (x - m) * lax.rsqrt(v + 1e-5) * p[name + ".weight"][None, :, None, None] \
            + p[name + ".bias"][None, :, None, None]

    def net(p, prev, cur, forcing):
        x = jnp.concatenate([cur, cur - prev, forcing], 1)
        x = conv(p, "stem", sphere_pad(x, 1))
        for i in range(depth):
            b = f"blocks.{i}"
            h = conv(p, b + ".dw", sphere_pad(x, 3), groups=width)
            h = conv(p, b + ".up", norm(p, b + ".norm", h))
            h = conv(p, b + ".down", jax.nn.gelu(h, approximate=False))
            x = x + p[b + ".scale"] * h
        return conv(p, "head", sphere_pad(norm(p, "norm", x), 1))

    return net


def make_physics(stats, land, orog):
    """Returns (normalise, forcing, dstd, area) as JAX functions and constants."""
    import jax.numpy as jnp

    f32 = lambda a: jnp.asarray(np.asarray(a, dtype=np.float32))
    mean = f32(stats["mean"]).reshape(1, -1, 1, 1)
    std = f32(stats["std"]).reshape(1, -1, 1, 1)
    dstd = f32(stats["dstd"]).reshape(1, -1, 1, 1)
    lat = np.linspace(90, -90, NLAT) * math.pi / 180
    lon = np.arange(NLON) * (2 * math.pi / NLON)
    area = f32(sc.area_weights()).reshape(1, 1, NLAT, 1)
    ones = np.ones((1, 1, NLAT, NLON), np.float32)
    static = f32(np.concatenate([
        np.sin(lat).reshape(1, 1, NLAT, 1) * ones, np.cos(lat).reshape(1, 1, NLAT, 1) * ones,
        np.sin(lon).reshape(1, 1, 1, NLON) * ones, np.cos(lon).reshape(1, 1, 1, NLON) * ones,
        np.asarray(land, np.float32).reshape(1, 1, NLAT, NLON),
        np.asarray(orog, np.float32).reshape(1, 1, NLAT, NLON) / 1000.0], 1))
    lon_j = f32(lon).reshape(1, 1, 1, NLON)

    def normalise(rough):
        return (rough.astype(jnp.float32) - mean) / std

    def forcing(hours):
        h = hours.astype(jnp.float32).reshape(-1, 1, 1, 1)
        season = 2 * math.pi * (h / 24.0 % 365.2425) / 365.2425
        local = 2 * math.pi * (h % 24.0) / 24.0 + lon_j
        B = h.shape[0]
        full = jnp.ones((B, 1, NLAT, NLON), jnp.float32)
        return jnp.concatenate([
            jnp.sin(season) * full, jnp.cos(season) * full,
            jnp.sin(local) * full, jnp.cos(local) * full,
            jnp.broadcast_to(static, (B,) + static.shape[1:])], 1)

    return normalise, forcing, dstd, area


def make_loss(net, phys, steps):
    normalise, forcing, dstd, area = phys

    def loss_fn(p, frames, hours):
        # frames: (B, steps + 2, C, lat, lon) in archive units; hours: time of frames[:, 1]
        B, T = frames.shape[:2]
        x = normalise(frames.reshape((B * T,) + frames.shape[2:])).reshape(frames.shape)
        prev, cur = x[:, 0], x[:, 1]
        loss = 0.0
        for s in range(steps):
            nxt = cur + net(p, prev, cur, forcing(hours + 6 * s)) * dstd
            loss = loss + (((nxt - x[:, 2 + s]) / dstd) ** 2 * area).mean() / steps
            prev, cur = cur, nxt
        return loss

    return loss_fn


def make_train_step(loss_fn, weight_decay=0.05, b1=0.9, b2=0.999, eps=1e-8, clip=1.0):
    """One AdamW step with global-norm gradient clipping, on every device at once."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    tmap = jax.tree_util.tree_map

    @partial(jax.pmap, axis_name="dev")
    def step(p, opt, frames, hours, lr):
        loss, g = jax.value_and_grad(loss_fn)(p, frames, hours)
        g = lax.pmean(g, "dev")
        loss = lax.pmean(loss, "dev")
        gnorm = jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g)))
        g = tmap(lambda x: x * jnp.minimum(1.0, clip / (gnorm + 1e-6)), g)
        t = opt["t"] + 1
        m = tmap(lambda a, x: b1 * a + (1 - b1) * x, opt["m"], g)
        v = tmap(lambda a, x: b2 * a + (1 - b2) * x * x, opt["v"], g)
        c1, c2 = 1 - b1 ** t, 1 - b2 ** t
        p = tmap(lambda w, a, b: w * (1 - lr * weight_decay) - lr * (a / c1) / (jnp.sqrt(b / c2) + eps), p, m, v)
        return p, {"m": m, "v": v, "t": t}, loss

    return step


def replicate(tree, n):
    import jax
    import jax.numpy as jnp

    return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(jnp.asarray(x), (n,) + np.shape(x)), tree)


def first_replica(tree):
    import jax

    return {k: np.asarray(v[0]) for k, v in tree.items()} if isinstance(tree, dict) else jax.tree_util.tree_map(
        lambda x: np.asarray(x[0]), tree)


def batches(arc, starts, n_frames, batch, seed, readers=32, ahead=4):
    """Endless shuffled batches, read on background threads."""
    q = queue.Queue(maxsize=ahead)

    def produce():
        rng = np.random.default_rng(seed)
        with ThreadPoolExecutor(readers) as ex:
            while True:
                order = rng.permutation(starts)
                for k in range(0, len(order) - batch + 1, batch):
                    ix = order[k:k + batch]
                    q.put((np.stack(list(ex.map(lambda j: arc.frames(int(j), n_frames), ix))),
                           arc.hours[ix + 1].astype(np.float32)))

    threading.Thread(target=produce, daemon=True).start()
    while True:
        yield q.get()


def describe_devices():
    import jax

    devs = jax.devices()
    kind = devs[0].platform
    sc.log(f"jax {jax.__version__}: {len(devs)} x {kind} ({devs[0]})")
    return len(devs), kind


def half_ok():
    """Whether float16 arrays can be handed to the device (halves the transfer)."""
    import jax
    import jax.numpy as jnp

    try:
        jax.jit(lambda a: a.astype(jnp.float32).sum())(np.ones((2, 2), np.float16)).block_until_ready()
        return True
    except Exception as e:
        sc.log(f"float16 input not supported here, using float32: {type(e).__name__}")
        return False


def train_jax(args, arc=None, budget_scale=1.0):
    import jax

    n_dev, kind = describe_devices()
    arc = arc or sc.Archive(args.data)
    ty0, ty1 = args.train_years
    a, b = arc.year_range(ty0, ty1)
    os.makedirs(args.out, exist_ok=True)
    stats = sc.measure_stats(arc, a, b)
    phys = make_physics(stats, arc.land, arc.orog)
    net = make_net(args.width, args.depth)
    params = init_params(0, len(arc.names), args.width, args.depth)
    n_par = sum(v.size for v in params.values())
    sc.log(f"channels={len(arc.names)} train={ty0}-{ty1} ({b - a} steps) params={n_par / 1e6:.2f}M devices={n_dev}")
    cfg = {"model": args.model, "channels": arc.names, "width": args.width, "depth": args.depth,
           "stats": stats, "train_years": [ty0, ty1], "backend": "jax"}
    json.dump(cfg, open(os.path.join(args.out, "model.json"), "w"))
    dtype = np.float16 if half_ok() else np.float32

    def save(p):
        tmp = os.path.join(args.out, "model.tmp.npz")
        np.savez(tmp, **first_replica(p))
        os.replace(tmp, os.path.join(args.out, "model.npz"))

    p = replicate(params, n_dev)
    history = []
    phases = [(1, args.batch, args.lr, args.hours * (1 - args.rollout_share)),
              (args.rollout, max(1, args.batch // args.rollout), args.lr / 10, args.hours * args.rollout_share)]
    for steps, per_dev, lr, hours in phases:
        if hours <= 0:
            continue
        step = make_train_step(make_loss(net, phys, steps))
        zeros = jax.tree_util.tree_map(lambda x: np.zeros_like(x), params)
        opt = replicate({"m": zeros, "v": zeros, "t": np.zeros((), np.float32)}, n_dev)
        data = batches(arc, arc.windows(a, b, steps + 2), steps + 2, per_dev * n_dev, seed=steps)
        budget, it, run, t0 = hours * 3600 * budget_scale, 0, None, None
        while True:
            frames, hrs = next(data)
            frames = frames.astype(dtype, copy=False).reshape((n_dev, per_dev) + frames.shape[1:])
            hrs = hrs.reshape(n_dev, per_dev)
            if t0 is None:
                frac = 0.0
            else:
                frac = (time.time() - t0) / budget
                if frac >= 1:
                    break
            warm = min(1.0, (it + 1) / 500)
            cur_lr = lr * warm * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * frac)))
            p, opt, loss = step(p, opt, frames, hrs, np.full((n_dev,), cur_lr, np.float32))
            if t0 is None:  # the first call compiles; do not count it against the budget
                loss = float(loss[0])
                t0 = time.time()
                sc.log(f"rollout={steps}: compiled, first loss={loss:.4f}")
            it += 1
            if it % args.log_every == 0:
                loss = float(loss[0])
                run = loss if run is None else 0.9 * run + 0.1 * loss
                sc.log(f"rollout={steps} it={it} loss={run:.4f} {100 * frac:.0f}% "
                       f"{(it - 1) * per_dev * n_dev / (time.time() - t0):.0f} samples/s")
                history.append({"rollout": steps, "it": it, "loss": run})
            if it % 2000 == 0:
                save(p)
        save(p)
        sc.log(f"phase rollout={steps} finished after {it} iterations, loss={run}")
    json.dump(history, open(os.path.join(args.out, "train_log.json"), "w"))


def evaluate_jax(args):
    import jax

    arc = sc.Archive(args.data)
    cfg = json.load(open(os.path.join(args.out, "model.json")))
    assert cfg["channels"] == arc.names
    params = {k: v for k, v in np.load(os.path.join(args.out, "model.npz")).items()}
    _, forcing, dstd, _ = make_physics(cfg["stats"], arc.land, arc.orog)
    net = make_net(cfg["width"], cfg["depth"])
    step = jax.jit(lambda p, prev, cur, hours: cur + net(p, prev, cur, forcing(hours)) * dstd)
    params = jax.device_put(params)
    return sc.score(arc, cfg, args, lambda prev, cur, hours: step(params, prev, cur, hours.astype(np.float32)))


def preflight(model="d1", per_dev=32, iters=6):
    """Compile and time the real training step on random data, before any download."""
    import jax

    n_dev, kind = describe_devices()
    m = sc.MODELS[model]
    C = len(sc.CHANNELS)
    stats = {"mean": [0.0] * C, "std": [1.0] * C, "dstd": [0.3] * C}
    phys = make_physics(stats, np.zeros((NLAT, NLON)), np.zeros((NLAT, NLON)))
    net = make_net(m["width"], m["depth"])
    params = init_params(0, C, m["width"], m["depth"])
    dtype = np.float16 if half_ok() else np.float32
    rng = np.random.default_rng(0)
    for steps, pd in ((1, per_dev), (m["rollout"], max(1, per_dev // m["rollout"]))):
        step = make_train_step(make_loss(net, phys, steps))
        zeros = jax.tree_util.tree_map(lambda x: np.zeros_like(x), params)
        p = replicate(params, n_dev)
        opt = replicate({"m": zeros, "v": zeros, "t": np.zeros((), np.float32)}, n_dev)
        frames = rng.standard_normal((n_dev, pd, steps + 2, C, NLAT, NLON)).astype(dtype)
        hrs = np.full((n_dev, pd), 1.0e6, np.float32)
        lr = np.full((n_dev,), 1e-4, np.float32)
        t0 = time.time()
        p, opt, loss = step(p, opt, frames, hrs, lr)
        first = float(loss[0])
        t1 = time.time()
        for _ in range(iters):
            p, opt, loss = step(p, opt, frames, hrs, lr)
        last = float(loss[0])
        dt = (time.time() - t1) / iters
        sc.log(f"preflight rollout={steps}: compile {t1 - t0:.0f}s, {dt * 1000:.0f} ms/step, "
               f"{pd * n_dev / dt:.0f} samples/s, loss {first:.3f} -> {last:.3f}")
    return n_dev, kind
