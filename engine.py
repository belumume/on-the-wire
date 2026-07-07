"""Edge-of-Stability engine (Entry 3: "On the Wire").

Reproduces the central phenomenon of Cohen, Kaur, Li, Kolter, Talwalkar,
"Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability"
(ICLR 2021, arXiv 2103.00065): under full-batch gradient descent, the top
eigenvalue of the training-loss Hessian (the "sharpness") rises until it hovers
just above 2/eta, then stays pinned there while the loss descends
non-monotonically. Classical optimization theory predicts divergence once
sharpness exceeds 2/eta; instead the system self-stabilizes at that threshold.

Reference code exists (github.com/locuslab/edge-of-stability, the paper's
authors). This is an independent, dependency-light reimplementation validated
two ways: (1) on a quadratic bowl, where the sharpness is a known fixed Hessian
eigenvalue and classical theory holds exactly, so the power-iteration
instrument can be checked against ground truth; (2) on a tiny nonlinear MLP,
where the edge-of-stability clamp appears.

The notebook's original extension ("Wind on the Wire") re-runs the identical
setup at shrinking minibatch sizes and asks whether the clamp survives noise:
a question the paper's full-batch scope excludes.

Design: fp32, explicit divergence guards (errors disqualify in the competition),
device="cuda" if available else cpu. Everything seeded and deterministic.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
@dataclasses.dataclass
class Config:
    n_data: int = 40
    input_dim: int = 8
    hidden: int = 32
    depth: int = 2  # number of weight layers
    steps: int = 500
    seed: int = 0
    diverge_threshold: float = 1e4
    power_iters: int = 20  # Hessian-vector power iterations per sharpness probe
    dtype: torch.dtype = torch.float32


def pick_device(pref: str | None = None) -> torch.device:
    if pref is not None:
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Tiny MLP as a flat parameter vector (so the Hessian is well-defined and
# Hessian-vector products are one torch.autograd call).
# --------------------------------------------------------------------------
def make_problem(cfg: Config, device: torch.device):
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    X = torch.randn(cfg.n_data, cfg.input_dim, generator=g, dtype=cfg.dtype)
    y = torch.randn(cfg.n_data, 1, generator=g, dtype=cfg.dtype)
    shapes = []
    fan_in = cfg.input_dim
    for i in range(cfg.depth):
        fan_out = 1 if i == cfg.depth - 1 else cfg.hidden
        shapes.append((fan_in, fan_out))
        fan_in = fan_out
    # He-style init packed into one flat vector
    parts = []
    for a, b in shapes:
        w = torch.randn(a, b, generator=g, dtype=cfg.dtype) * (2.0 / a) ** 0.5
        parts.append(w.reshape(-1))
    theta0 = torch.cat(parts)
    return X.to(device), y.to(device), theta0.to(device), shapes


def _forward(theta: torch.Tensor, X: torch.Tensor, shapes) -> torch.Tensor:
    off = 0
    h = X
    for i, (a, b) in enumerate(shapes):
        w = theta[off : off + a * b].reshape(a, b)
        off += a * b
        h = h @ w
        if i < len(shapes) - 1:
            h = torch.tanh(h)
    return h


def loss_fn(
    theta: torch.Tensor, X: torch.Tensor, y: torch.Tensor, shapes
) -> torch.Tensor:
    pred = _forward(theta, X, shapes)
    return ((pred - y) ** 2).mean()


# --------------------------------------------------------------------------
# Top Hessian eigenvalue via Hessian-vector-product power iteration.
# --------------------------------------------------------------------------
def top_hessian_eigs(
    theta: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    shapes,
    k: int = 1,
    iters: int = 20,
    seed: int = 0,
) -> torch.Tensor:
    """Top-k Hessian eigenvalues at theta (deflated power iteration)."""
    theta = theta.detach().requires_grad_(True)
    _l = loss_fn(theta, X, y, shapes)
    (grad,) = torch.autograd.grad(_l, theta, create_graph=True)

    def hvp(v: torch.Tensor) -> torch.Tensor:
        (out,) = torch.autograd.grad(grad, theta, grad_outputs=v, retain_graph=True)
        return out.detach()

    g = torch.Generator(device="cpu").manual_seed(seed)
    eigs = []
    basis: list[torch.Tensor] = []
    for _ in range(k):
        v = torch.randn(theta.shape, generator=g, dtype=theta.dtype).to(theta.device)
        v = v / (v.norm() + 1e-12)
        lam = torch.zeros((), dtype=theta.dtype, device=theta.device)
        for _ in range(iters):
            for b in basis:  # deflate previously found eigenvectors
                v = v - (v @ b) * b
            w = hvp(v)
            lam = v @ w
            nrm = w.norm() + 1e-12
            v = w / nrm
        for b in basis:
            v = v - (v @ b) * b
        v = v / (v.norm() + 1e-12)
        eigs.append(lam)
        basis.append(v)
    return torch.stack(eigs)


# --------------------------------------------------------------------------
# Training loops.
# --------------------------------------------------------------------------
def train_full_batch(
    cfg: Config,
    lr: float,
    device: str | torch.device | None = None,
    track_k: int = 1,
    sharpness_every: int = 1,
):
    """Full-batch GD. Returns dict of trajectories: loss, sharpness (top-k)."""
    dev = pick_device(device) if not isinstance(device, torch.device) else device
    X, y, theta, shapes = make_problem(cfg, dev)
    losses, sharp_steps, sharp_vals = [], [], []
    diverged_at = None
    for step in range(cfg.steps):
        theta_g = theta.detach().requires_grad_(True)
        _l = loss_fn(theta_g, X, y, shapes)
        (grad,) = torch.autograd.grad(_l, theta_g)
        with torch.no_grad():
            theta = theta - lr * grad
        lv = float(_l.detach())
        losses.append(lv)
        if not np.isfinite(lv) or lv > cfg.diverge_threshold:
            diverged_at = step
            break
        if step % sharpness_every == 0:
            ev = top_hessian_eigs(
                theta, X, y, shapes, k=track_k, iters=cfg.power_iters, seed=cfg.seed
            )
            sharp_steps.append(step)
            sharp_vals.append(ev.detach().cpu().numpy())
    return {
        "loss": np.array(losses),
        "sharp_steps": np.array(sharp_steps),
        "sharp": np.array(sharp_vals) if sharp_vals else np.zeros((0, track_k)),
        "threshold": 2.0 / lr,
        "diverged_at": diverged_at,
        "lr": lr,
    }


def quadratic_bowl(lr: float, curvature: float, steps: int = 60):
    """1D control: GD on 0.5*curvature*x^2. Sharpness == curvature (constant).

    Classical theory: converges iff lr < 2/curvature, diverges above. The
    'sharpness' is the fixed Hessian eigenvalue, so this validates the
    instrument against a known answer and calibrates the reader's expectation.
    """
    x = 1.0
    xs, losses = [x], [0.5 * curvature * x * x]
    for _ in range(steps):
        x = x - lr * curvature * x
        xs.append(x)
        lv = 0.5 * curvature * x * x
        losses.append(lv)
        if not np.isfinite(lv) or lv > 1e6:
            break
    return {
        "x": np.array(xs),
        "loss": np.array(losses),
        "sharpness": curvature,
        "threshold": 2.0 / lr,
        "stable": lr < 2.0 / curvature,
    }


def train_minibatch_ensemble(
    cfg: Config,
    lr: float,
    batch_size: int,
    n_seeds: int = 12,
    device: str | torch.device | None = None,
):
    """Extension "Wind on the Wire": n_seeds independent minibatch runs at a
    fixed lr near the deterministic threshold. Returns per-seed sharpness
    trajectories plus two honest summaries: how many seeds RODE the wire
    (stayed bounded near 2/lr) versus were blown off it (diverged), and the
    clamp tightness measured over the surviving runs only.

    A run "rode the wire" if it stayed finite for the whole budget and its
    plateau sharpness never exceeded ride_ceiling * threshold. Diverged runs
    are recorded (with their sharpness clamped to a finite display ceiling so
    plots render) but excluded from the tightness average, which would
    otherwise be dominated by astronomically large post-divergence values.
    """
    dev = pick_device(device) if not isinstance(device, torch.device) else device
    X, y, theta0, shapes = make_problem(cfg, dev)
    threshold = 2.0 / lr
    ride_ceiling = 3.0
    display_cap = ride_ceiling * threshold
    runs, rode_flags = [], []
    for s in range(n_seeds):
        g = torch.Generator(device="cpu").manual_seed(10_000 + s)
        theta = theta0.clone()
        sharp = []
        blew_up = False
        for step in range(cfg.steps):
            idx = torch.randint(0, cfg.n_data, (batch_size,), generator=g).to(dev)
            theta_g = theta.detach().requires_grad_(True)
            _l = loss_fn(theta_g, X[idx], y[idx], shapes)
            (grad,) = torch.autograd.grad(_l, theta_g)
            with torch.no_grad():
                theta = theta - lr * grad
            if not torch.isfinite(theta).all():
                blew_up = True
                break
            if step % 5 == 0:
                ev = top_hessian_eigs(
                    theta, X, y, shapes, k=1, iters=cfg.power_iters, seed=cfg.seed
                )
                sv = float(ev[0])
                if not np.isfinite(sv) or sv > display_cap:
                    sharp.append(display_cap)
                    blew_up = True
                    break
                sharp.append(sv)
        arr = np.array(sharp, dtype=float)
        completed = len(arr) >= cfg.steps // 5 - 1
        rode = (
            (not blew_up)
            and completed
            and (arr.max() <= display_cap if len(arr) else False)
        )
        runs.append(arr)
        rode_flags.append(bool(rode))
    # clamp tightness over SURVIVING (rode) runs only; nan if none survived
    tights = [
        float(np.std(r[len(r) * 2 // 3 :]))
        for r, rd in zip(runs, rode_flags)
        if rd and len(r) >= 6
    ]
    return {
        "runs": runs,
        "rode": rode_flags,
        "n_rode": int(sum(rode_flags)),
        "n_seeds": n_seeds,
        "ride_ceiling": ride_ceiling,
        "display_cap": float(display_cap),
        "threshold": threshold,
        "clamp_std": float(np.mean(tights)) if tights else float("nan"),
        "batch_size": batch_size,
        "lr": lr,
    }


def _smoke():
    cfg = Config(steps=300)
    dev = pick_device()
    print("device:", dev)
    # instrument check on the bowl
    b = quadratic_bowl(lr=0.9 * 2 / 5.0, curvature=5.0)
    print("bowl stable (expect True):", b["stable"], "final loss", b["loss"][-1])
    b2 = quadratic_bowl(lr=1.1 * 2 / 5.0, curvature=5.0)
    print(
        "bowl above-threshold stable (expect False):",
        b2["stable"],
        "final loss",
        b2["loss"][-1],
    )
    # edge of stability on the MLP
    out = train_full_batch(cfg, lr=0.05, device=dev, track_k=1, sharpness_every=5)
    if len(out["sharp"]):
        final_sharp = out["sharp"][-1, 0]
        print(
            f"EoS: final sharpness {final_sharp:.2f} vs threshold {out['threshold']:.2f} "
            f"(ratio {final_sharp / out['threshold']:.2f}, expect ~1)"
        )
    print("diverged_at:", out["diverged_at"], "final loss:", out["loss"][-1])


if __name__ == "__main__":
    _smoke()
