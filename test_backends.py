# Checks that the JAX backend computes the same thing as the PyTorch model:
# forward pass, rollout loss, gradients and one optimiser step.
#   python test_backends.py

import numpy as np
import torch

import skycore as sc
import skycore_jax as sj

W, D, C, B, STEPS = 24, 3, len(sc.CHANNELS), 2, 3
rng = np.random.default_rng(1)
stats = {"mean": rng.normal(0, 0.2, C).tolist(), "std": rng.uniform(0.8, 1.2, C).tolist(),
         "dstd": rng.uniform(0.2, 0.5, C).tolist()}
land = (rng.random((sc.NLAT, sc.NLON)) > 0.7).astype(np.float32)
orog = rng.uniform(0, 3000, (sc.NLAT, sc.NLON)).astype(np.float32)
frames = rng.standard_normal((B, STEPS + 2, C, sc.NLAT, sc.NLON)).astype(np.float16)
hours = np.array([1_100_000, 1_100_006 + 24 * 200], dtype=np.int64)

# random weights everywhere (the default zero head would hide differences)
params = sj.init_params(0, C, W, D)
params["head.weight"] = rng.normal(0, 0.05, params["head.weight"].shape).astype(np.float32)
params["head.bias"] = rng.normal(0, 0.05, params["head.bias"].shape).astype(np.float32)

# ---- PyTorch side
model = sc.build_model(C, sc.Physics.N_FORCING, W, D)
assert set(model.state_dict()) == set(params), set(model.state_dict()) ^ set(params)
model.load_state_dict({k: torch.from_numpy(v) for k, v in params.items()})
phys = sc.Physics(stats, land, orog, "cpu")


def torch_loss():
    x = phys.normalise(torch.from_numpy(frames))
    prev, cur, loss = x[:, 0], x[:, 1], 0.0
    for s in range(STEPS):
        nxt = phys.step(model, prev, cur, torch.from_numpy(hours) + 6 * s)
        loss = loss + phys.loss(nxt, x[:, 2 + s]) / STEPS
        prev, cur = cur, nxt
    return loss


x = phys.normalise(torch.from_numpy(frames))
with torch.no_grad():
    t_fwd = model(x[:, 0], x[:, 1], phys.forcing(torch.from_numpy(hours))).numpy()
    t_forc = phys.forcing(torch.from_numpy(hours)).numpy()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
loss = torch_loss()
opt.zero_grad()
loss.backward()
t_grad = {k: v.grad.numpy().copy() for k, v in model.named_parameters()}
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
t_after = {k: v.detach().numpy() for k, v in model.named_parameters()}
t_loss2 = float(torch_loss())

# ---- JAX side
import jax  # noqa: E402

jphys = sj.make_physics(stats, land, orog)
net = sj.make_net(W, D)
normalise, forcing, dstd, area = jphys
jx = normalise(frames.reshape((-1,) + frames.shape[2:])).reshape(frames.shape)
j_forc = np.asarray(forcing(hours.astype(np.float32)))
j_fwd = np.asarray(net(params, jx[:, 0], jx[:, 1], forcing(hours.astype(np.float32))))
loss_fn = sj.make_loss(net, jphys, STEPS)
j_loss, j_grad = jax.value_and_grad(loss_fn)(params, frames, hours.astype(np.float32))
step = sj.make_train_step(loss_fn)
zeros = {k: np.zeros_like(v) for k, v in params.items()}
p1, _, _ = step(sj.replicate(params, 1), sj.replicate({"m": zeros, "v": zeros, "t": np.zeros((), np.float32)}, 1),
                frames[None], hours.astype(np.float32)[None], np.full((1,), 1e-3, np.float32))
j_after = sj.first_replica(p1)
j_loss2 = float(loss_fn(j_after, frames, hours.astype(np.float32)))


def rel(a, b):
    return float(np.abs(a - b).max() / (np.abs(a).max() + 1e-12))


checks = {
    "forcing": rel(t_forc, j_forc),
    "forward": rel(t_fwd, j_fwd),
    "loss": abs(float(loss) - float(j_loss)) / float(loss),
    "gradient (worst tensor)": max(rel(t_grad[k], np.asarray(j_grad[k])) for k in t_grad),
    "weights after one step": max(rel(t_after[k], j_after[k]) for k in t_after),
    "loss after one step": abs(t_loss2 - j_loss2) / t_loss2,
}
print(f"loss torch={float(loss):.6f} jax={float(j_loss):.6f}; after one step torch={t_loss2:.6f} jax={j_loss2:.6f}")
bad = 0
for k, v in checks.items():
    ok = v < 2e-3
    bad += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {k}: relative difference {v:.2e}")
raise SystemExit(1 if bad else 0)
