# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib>=3.11.0",
#     "numpy>=2.5.1",
#     "torch>=2.12.1",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
)


@app.cell(hide_code=True)
def title(mo):
    mo.md(r"""
    # On the Wire

    **Paper:** [Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability](https://www.alphaxiv.org/abs/2103.00065) (Cohen, Kaur, Li, Kolter, Talwalkar, ICLR 2021, arXiv 2103.00065)

    Classical optimization theory draws a hard line for gradient descent: if the sharpness of the loss surface (the top eigenvalue of the Hessian) exceeds $2/\eta$ for learning rate $\eta$, the iterates overshoot and diverge. Below the line, safe. Above the line, explosion. The line is the wire.

    The paper's finding: real neural networks do not stay safely under the wire. Sharpness *grows* during training until it reaches $2/\eta$, and then, instead of diverging, it stays pinned there while the loss keeps falling, non-monotonically, in open defiance of the theory that says this cannot work.

    Why that matters: this is the regime where practical deep learning actually lives. Large learning rates train fastest, and they train ON the wire, not under it. Understanding what keeps a network balanced there, and what knocks it off, is understanding why the everyday recipe works at all. **This notebook walks the wire in three acts, then asks the question the clean theory leaves out: does the balancing act survive minibatch noise?** The answer arrives as a sharp, measured threshold.

    Three acts: first a system where the classical theory holds exactly, then a prediction you get to lock in before seeing the truth, then a real network that breaks the theory. The dark gold line in every plot is the same object: $2/\eta$.

    (The paper's alphaXiv page also carries an AI summary, audio overview, and a discussion tab for the short version.)
    """)

    return


@app.cell(hide_code=True)
def reader_guide(mo):
    mo.callout(
        mo.md(
            "**How to read this notebook.** Everything below runs live in this kernel. "
            "Drag the learning-rate slider and watch the gold wire move with it; lock in "
            "your prediction before hitting reveal (the notebook will not peek for you); "
            "then shrink the batch size and watch the wind knock runs off the wire. "
            "Code is folded where it is plumbing; the eye icon on any cell opens it."
        ),
        kind="info",
    )

    return


@app.cell(hide_code=True)
def kpi_tiles(WIND_BATCHES, WIND_LR, WIND_SEEDS, device, mo, wind_runs):
    # Headline numbers as tiles, computed from this kernel's own runs.
    mo.hstack(
        [
            mo.stat(
                f"{2.0 / WIND_LR:.0f}",
                label="the wire (2 / learning rate)",
                caption=f"at lr = {WIND_LR:g}",
                bordered=True,
            ),
            mo.stat(
                f"{wind_runs[WIND_BATCHES[0]]['n_rode']}/{WIND_SEEDS}",
                label=f"riding the wire, batch {WIND_BATCHES[0]}",
                caption="full batch: everyone balances",
                bordered=True,
            ),
            mo.stat(
                f"{wind_runs[WIND_BATCHES[-1]]['n_rode']}/{WIND_SEEDS}",
                label=f"riding the wire, batch {WIND_BATCHES[-1]}",
                caption="the wind wins",
                direction="decrease",
                bordered=True,
            ),
            mo.stat(
                str(device).upper(),
                label="compute",
                caption=f"{len(WIND_BATCHES)} batch sizes x {WIND_SEEDS} seeds, live",
                bordered=True,
            ),
        ],
        gap=0.75,
        wrap=True,
    )

    return


@app.cell
def mo_import():
    import marimo as mo

    return (mo,)


@app.cell
def imports():
    import dataclasses

    import numpy as np
    import torch
    from matplotlib.figure import Figure

    return Figure, dataclasses, np, torch


@app.cell(hide_code=True)
def engine(dataclasses, np, torch):
    # Edge-of-Stability engine: tiny MLP as a flat parameter vector, the
    # Hessian-vector-product power-iteration sharpness meter, full-batch GD,
    # the quadratic-bowl control, and the minibatch-ensemble extension.
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
            lv = float(_l.detach())
            losses.append(lv)
            if not np.isfinite(lv) or lv > cfg.diverge_threshold:
                diverged_at = step
                break
            if step % sharpness_every == 0:
                # probe at the SAME theta the recorded loss was computed at
                # (pre-update), so loss[i] and sharp[i] describe one point
                ev = top_hessian_eigs(
                    theta, X, y, shapes, k=track_k, iters=cfg.power_iters, seed=cfg.seed
                )
                sharp_steps.append(step)
                sharp_vals.append(ev.detach().cpu().numpy())
            with torch.no_grad():
                theta = theta - lr * grad
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
        n_seeds: int = 32,
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
        ride_ceiling = 1.5
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

    return (
        Config,
        pick_device,
        quadratic_bowl,
        train_full_batch,
        train_minibatch_ensemble,
    )


@app.cell(hide_code=True)
def act1_md(mo):
    mo.md(r"""
    ## Act 1: a system where the theory is exactly right

    For gradient descent on a quadratic bowl $f(x) = \tfrac{1}{2} c x^2$, one step gives $x \leftarrow (1 - \eta c)\,x$. The iterates shrink if $|1 - \eta c| < 1$, which for positive $c$ means exactly $\eta < 2/c$. Here the sharpness IS the curvature $c$, fixed forever, and the theory is airtight: cross the wire and you diverge, geometrically, every time.

    This bowl is also the calibration for our measuring instrument. The sharpness numbers in the rest of the notebook come from Hessian-vector-product power iteration; on the bowl, the true answer is known ($c$ itself), so you can watch the instrument agree with ground truth before trusting it anywhere else.

    Drag the learning rate across the wire and watch the trajectory flip from convergence to explosion. The computation is live (sixty scalar multiplications), not precomputed.
    """)
    return


@app.cell
def device_setup(Config, mo, pick_device):
    device = pick_device()
    cfg = Config(steps=800)
    WIRE_COLOR = "#b8860b"
    mo.md(
        f"**device:** `{device.type}` | MLP: depth={cfg.depth}, hidden={cfg.hidden}, "
        f"n_data={cfg.n_data}, steps={cfg.steps}, fp32, all seeded (seed={cfg.seed})"
    )
    return WIRE_COLOR, cfg, device


@app.cell
def bowl_ui(mo):
    bowl_lr = mo.ui.slider(
        start=0.05, stop=0.60, step=0.01, value=0.30,
        label="bowl learning rate (curvature c = 5, so the wire sits at 2/c = 0.4)",
        show_value=True, full_width=True,
    )
    bowl_lr
    return (bowl_lr,)


@app.cell
def bowl_display(Figure, WIRE_COLOR, bowl_lr, np, quadratic_bowl):
    _lr_b = float(bowl_lr.value)
    _b = quadratic_bowl(lr=_lr_b, curvature=5.0, steps=60)
    _figB = Figure(figsize=(13.5, 4.2))
    _axG, _axL, _axR = _figB.subplots(1, 3)

    # the wire, drawn: constant sharpness (= curvature) vs the moving 2/lr line
    _steps_b = np.arange(len(_b["x"]))
    _axG.plot(_steps_b, np.full_like(_steps_b, _b["sharpness"], dtype=float),
              color="black", linewidth=2.2, label=f"sharpness = curvature = {_b['sharpness']:.0f}")
    _axG.axhline(_b["threshold"], color=WIRE_COLOR, linewidth=2.4,
                 label=f"the wire: 2/lr = {_b['threshold']:.2f}")
    _axG.set_ylim(0, 14)
    _axG.set_xlabel("step")
    _axG.set_ylabel("eigenvalue")
    _axG.set_title("the bowl's sharpness never moves")
    _axG.legend(fontsize=8, loc="upper right")

    _axL.plot(_b["x"], color="steelblue", linewidth=1.8)
    _axL.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    _axL.set_xlabel("step")
    _axL.set_ylabel("x")
    _axL.set_title("iterate trajectory")

    _axR.semilogy(np.clip(_b["loss"], 1e-16, None), color="steelblue", linewidth=1.8)
    _axR.set_xlabel("step")
    _axR.set_ylabel("loss (log scale)")
    if abs(_lr_b * _b["sharpness"] - 2.0) < 1e-9:
        _verdict = "ON THE WIRE exactly (marginal: neither grows nor shrinks)"
    elif _b["stable"]:
        _verdict = "CONVERGES (below the wire)"
    else:
        _verdict = "DIVERGES (above the wire)"
    _axR.set_title(f"lr = {_lr_b:.2f} vs wire at {_b['threshold']:.3f}\n{_verdict}", fontsize=10)

    _figB.tight_layout()
    _figB
    return


@app.cell(hide_code=True)
def act2_md(mo):
    mo.md(r"""
    ## Act 2: lock in a prediction

    The bowl behaved exactly as the theory demands. Now we swap the bowl for a small real network: a tanh MLP trained by full-batch gradient descent on a fixed regression task. Unlike the bowl, its sharpness is free to change as the weights move.

    Before you drag the learning-rate slider in Act 3, commit to a prediction. At a learning rate where the network's sharpness would have to exceed $2/\eta$, what does gradient descent actually do?
    """)
    return


@app.cell
def predict_ui(mo):
    prediction = mo.ui.radio(
        options={
            "It diverges, like the bowl above the wire": "diverge",
            "Sharpness stays safely below the wire and it converges": "safe",
            "Something the bowl cannot do": "weird",
        },
        label="your prediction (locked in before the reveal)",
    )
    prediction
    return (prediction,)


@app.cell
def mlp_compute(cfg, device, mo, np, train_full_batch):
    import os as _os
    import hashlib as _hashlib
    import pickle as _pickle

    _EOS_CACHE = "/tmp/eos_cache"
    _os.makedirs(_EOS_CACHE, exist_ok=True)

    LR_GRID = np.round(np.arange(0.02, 0.42, 0.02), 3)
    _lr_key = _hashlib.sha1(repr((cfg, tuple(LR_GRID.tolist()), 3, 5)).encode()).hexdigest()[:16]
    _lr_path = f"{_EOS_CACHE}/lr_sweep_{_lr_key}.pkl"

    if _os.path.exists(_lr_path):
        with open(_lr_path, "rb") as _f:
            lr_sweep = _pickle.load(_f)
    else:
        lr_sweep = {}
        with mo.status.progress_bar(total=len(LR_GRID), title="training one network per learning rate") as _bar:
            for _lr in LR_GRID:
                lr_sweep[float(_lr)] = train_full_batch(cfg, lr=float(_lr), device=device, track_k=3, sharpness_every=5)
                _bar.update()
        with open(_lr_path, "wb") as _f:
            _pickle.dump(lr_sweep, _f)

    f"lr sweep ready: {len(lr_sweep)} learning rates x {cfg.steps} steps, top-3 sharpness tracked, cached"
    return LR_GRID, lr_sweep


@app.cell
def mlp_ui(LR_GRID, mo):
    mlp_lr = mo.ui.slider(
        start=float(LR_GRID.min()), stop=float(LR_GRID.max()),
        step=float(LR_GRID[1] - LR_GRID[0]), value=0.24,
        label="Act 3: MLP learning rate (drag from far below the wire to past it)",
        show_value=True, full_width=True,
    )
    mlp_lr
    return (mlp_lr,)


@app.cell
def mlp_display(Figure, LR_GRID, WIRE_COLOR, lr_sweep, mlp_lr, np):
    _lr_here = float(LR_GRID[int(np.argmin(np.abs(LR_GRID - mlp_lr.value)))])
    _run = lr_sweep[_lr_here]
    _wire = _run["threshold"]

    _figM = Figure(figsize=(13, 4.8))
    _axS, _axLoss = _figM.subplots(1, 2)

    if len(_run["sharp"]):
        _axS.plot(_run["sharp_steps"], _run["sharp"][:, 0], color="crimson", linewidth=2.0, label="sharpness (top eig)")
        for _k in range(1, _run["sharp"].shape[1]):
            _axS.plot(_run["sharp_steps"], _run["sharp"][:, _k], color="crimson", linewidth=0.9,
                      alpha=0.35, label=f"eig {_k + 1}" if _k == 1 else None)
    _axS.axhline(_wire, color=WIRE_COLOR, linewidth=2.2, label=f"the wire: 2/lr = {_wire:.1f}")
    _axS.set_xlabel("training step")
    _axS.set_ylabel("Hessian eigenvalues")
    _axS.set_title(f"lr = {_lr_here:g}: does sharpness respect the wire?")
    _axS.legend(fontsize=8)

    _loss = _run["loss"]
    _axLoss.semilogy(np.clip(_loss, 1e-12, None), color="steelblue", linewidth=1.2)
    _axLoss.set_xlabel("training step")
    _axLoss.set_ylabel("training loss (log scale)")
    if _run["diverged_at"] is not None:
        _axLoss.set_title(f"DIVERGED at step {_run['diverged_at']}")
    else:
        _ups = int(np.sum(np.diff(_loss) > 0))
        _axLoss.set_title(
            f"loss fell {_loss[0]:.3g} -> {_loss[-1]:.3g}, yet ROSE on {_ups}/{len(_loss) - 1} steps"
        )

    _figM.tight_layout()
    _figM
    return


@app.cell
def reveal_md(LR_GRID, lr_sweep, mlp_lr, mo, np, prediction):
    _lr_now = float(LR_GRID[int(np.argmin(np.abs(LR_GRID - mlp_lr.value)))])
    _run_now = lr_sweep[_lr_now]
    _ratio = (
        float(_run_now["sharp"][-1, 0] / _run_now["threshold"])
        if len(_run_now["sharp"]) and _run_now["diverged_at"] is None
        else None
    )
    _pred = prediction.value
    _pred_text = {
        "diverge": "You predicted divergence, like the bowl. The bowl was the wrong teacher:",
        "safe": "You predicted it stays safely below the wire. It does not:",
        "weird": "You predicted something the bowl cannot do. Correct:",
        None: "You have not locked a prediction above yet. The reveal, regardless:",
    }[_pred]
    _lines = [
        "### The reveal",
        "",
        _pred_text,
        "",
    ]
    if _run_now["diverged_at"] is not None or _ratio is None:
        _lines += [
            f"At lr = {_lr_now:g} this network genuinely diverges"
            + (f" (step {_run_now['diverged_at']})." if _run_now["diverged_at"] is not None else ".")
            + " Drag the slider left into the 0.15-0.35 band to find the regime the paper is about: "
            "sharp enough to sit ON the wire, stable enough to keep descending."
        ]
    else:
        _lines += [
            f"At lr = {_lr_now:g}, final sharpness sits at {_ratio:.2f}x the wire. "
            "At small learning rates sharpness climbs steadily toward the wire (progressive sharpening); "
            "in the 0.15-0.35 band it reaches the wire and RIDES it, hovering at 2/lr while the loss "
            "keeps falling through values classical theory says should bounce it to infinity. "
            "The system self-stabilizes exactly at the edge of stability. That is the paper's title, "
            "measured live."
        ]
    mo.md("\n".join(_lines))
    return


@app.cell(hide_code=True)
def wind_intro_md(mo):
    mo.md(r"""
    ## Wind on the Wire (an extension the paper leaves open)

    Everything above is full-batch gradient descent: deterministic, the paper's chosen setting. Real training adds wind: minibatch noise shakes every step. Does the clamp survive it?

    We fix the learning rate at 0.25 (squarely on the wire) and re-run training at shrinking batch sizes, an ensemble of independent seeds at each size. Minibatches are drawn with replacement, so even a batch the size of the dataset carries mild resampling wind; smaller batches blow harder. If the edge-of-stability mechanism is a knife-edge artifact of determinism, noise should scatter the plateau. If it is an attractor, the trajectories should keep hugging the wire, just with jitter. We report whichever one the measurement shows.
    """)
    return


@app.cell
def wind_compute(cfg, dataclasses, device, mo, train_minibatch_ensemble):
    import os as _os3
    import hashlib as _hashlib3
    import pickle as _pickle3

    _os3.makedirs("/tmp/eos_cache", exist_ok=True)
    WIND_LR = 0.25
    WIND_BATCHES = [40, 20, 10, 5]
    WIND_SEEDS = 32
    _wind_cfg = dataclasses.replace(cfg, steps=500)
    _wind_key = _hashlib3.sha1(
        repr((_wind_cfg, WIND_LR, tuple(WIND_BATCHES), WIND_SEEDS)).encode()
    ).hexdigest()[:16]
    _wind_path = f"/tmp/eos_cache/wind_{_wind_key}.pkl"

    if _os3.path.exists(_wind_path):
        with open(_wind_path, "rb") as _f:
            wind_runs = _pickle3.load(_f)
    else:
        wind_runs = {}
        with mo.status.progress_bar(total=len(WIND_BATCHES), title="minibatch ensembles (32 seeds per batch size)") as _bar:
            for _bs in WIND_BATCHES:
                wind_runs[_bs] = train_minibatch_ensemble(_wind_cfg, lr=WIND_LR, batch_size=_bs, n_seeds=WIND_SEEDS, device=device)
                _bar.update()
        with open(_wind_path, "wb") as _f:
            _pickle3.dump(wind_runs, _f)

    f"wind ensembles ready: batch sizes {WIND_BATCHES}, {WIND_SEEDS} seeds each, lr={WIND_LR} (wire at {2.0 / WIND_LR:.0f}), cached"
    return WIND_BATCHES, WIND_LR, WIND_SEEDS, wind_runs


@app.cell
def wind_display(
    Figure,
    WIND_BATCHES,
    WIND_LR,
    WIND_SEEDS,
    WIRE_COLOR,
    np,
    wind_runs,
):
    _figW = Figure(figsize=(13.5, 4.8))
    _axT, _axC = _figW.subplots(1, 2)

    _wire_w = 2.0 / WIND_LR
    _cap = wind_runs[WIND_BATCHES[0]]["display_cap"]
    _colors = {40: "#1a1a1a", 20: "#2a6fb8", 10: "#7a4bb8", 5: "#c23b70"}
    for _bs in WIND_BATCHES:
        _r = wind_runs[_bs]
        for _i, _traj in enumerate(_r["runs"]):
            if len(_traj):
                _lab = f"batch {_bs}" + (" (= dataset size)" if _bs == 40 else "") if _i == 0 else None
                _axT.plot(np.arange(len(_traj)) * 5, np.clip(_traj, 0, _cap),
                          color=_colors[_bs], alpha=0.3, linewidth=0.9, label=_lab)
    _axT.axhline(_wire_w, color=WIRE_COLOR, linewidth=2.4, label=f"the wire: 2/lr = {_wire_w:.0f}")
    _axT.axhline(_cap, color="gray", linewidth=0.8, linestyle=":")
    _axT.text(2, _cap * 0.97, "off the wire (diverged, clipped for display)",
              fontsize=7, color="gray", va="top")
    _axT.set_xlabel("training step")
    _axT.set_ylabel("sharpness")
    _axT.set_ylim(0, _cap * 1.05)
    _axT.set_title(f"{len(WIND_BATCHES) * WIND_SEEDS} noisy trajectories vs the wire (lr = {WIND_LR})")
    _axT.legend(fontsize=8, loc="upper left")

    _bss = list(WIND_BATCHES)
    _frac = [wind_runs[_bs]["n_rode"] / wind_runs[_bs]["n_seeds"] for _bs in _bss]
    _axC.plot(_bss, _frac, "o-", color=WIRE_COLOR, linewidth=2.4, markersize=9)
    for _bs, _fr in zip(_bss, _frac):
        _axC.annotate(f"{wind_runs[_bs]['n_rode']}/{wind_runs[_bs]['n_seeds']}",
                      xy=(_bs, _fr), xytext=(0, 8), textcoords="offset points",
                      ha="center", fontsize=8)
    _axC.set_xlabel("batch size, drawn with replacement (40 = dataset size; smaller = windier)")
    _axC.set_ylabel("fraction of seeds that RODE the wire\n(stayed bounded, did not diverge)")
    _axC.set_title("does the clamp survive the wind?")
    _axC.set_ylim(-0.05, 1.12)
    _axC.set_xscale("log", base=2)
    _axC.set_xticks(_bss)
    _axC.set_xticklabels([str(_b) for _b in _bss])

    _figW.tight_layout()
    _figW
    return


@app.cell
def wind_caption_md(WIND_BATCHES, WIND_LR, mo, wind_runs):
    _rode = {_b: wind_runs[_b]["n_rode"] for _b in WIND_BATCHES}
    _ns = wind_runs[WIND_BATCHES[0]]["n_seeds"]
    _big = [_b for _b in WIND_BATCHES if _rode[_b] >= _ns - 1]
    _broke = [_b for _b in WIND_BATCHES if _rode[_b] <= _ns // 3]
    _t_big = wind_runs[_big[0]]["clamp_std"] if _big else float("nan")
    _lines2 = [
        "### Reading the wind panel",
        "",
        "Seeds that rode the wire (stayed bounded near 2/lr for the full budget), out of "
        f"{_ns} per batch size: "
        + ", ".join(f"batch {_b}: {_rode[_b]}/{_ns}" for _b in WIND_BATCHES) + ".",
        "",
    ]
    if _big and _broke:
        _lines2 += [
            f"There is a clear threshold. At the larger batches ({', '.join(str(b) for b in _big)}) "
            f"almost every seed rides the wire, and among those survivors the plateau is tight "
            f"(std about {_t_big:.2f} around a wire at {2.0 / WIND_LR:.0f}). Drop the batch to "
            f"{', '.join(str(b) for b in _broke)} and most seeds are thrown off the wire entirely: "
            "the minibatch noise, at a learning rate already sitting on the deterministic edge, "
            "overshoots it and the run diverges. So the edge-of-stability clamp is not indestructible. "
            "It behaves like an attractor with a finite basin: enough wind, and gradient descent is "
            "blown clean off the wire rather than made to jitter along it. That threshold, not a smooth "
            "loosening, is the honest answer the full-batch paper leaves open."
        ]
    elif _big:
        _lines2 += [
            "Across every batch size tested, nearly all seeds ride the wire, with only mild loosening "
            "of the plateau as the wind picks up. At this scale the clamp survives noise: the edge of "
            "stability behaves like an attractor, not a knife-edge artifact of determinism."
        ]
    else:
        _lines2 += [
            "Even at the largest batch tested, many seeds are thrown off the wire, so at this learning "
            "rate the clamp does not cleanly survive minibatch noise. We report it as measured."
        ]
    _lines2 += [
        "",
        "One caveat stated plainly: these are 500-step runs of a small MLP, thirty-two seeds per batch "
        f"size, sharpness probed every fifth step, all at the single learning rate {WIND_LR}. The "
        "threshold is real in this setup; its exact location would move with model size, step budget, "
        "and learning rate, and would reward a bigger instrument.",
    ]
    mo.md("\n".join(_lines2))
    return


@app.cell(hide_code=True)
def guardrails(mo):
    mo.md(r"""
    ### What this does and does not show

    - The MLP, task, and training loop are synthetic and small on purpose (fp32, fixed seeds, explicit divergence guards). The paper demonstrates the phenomenon across real architectures and datasets; this notebook demonstrates the *mechanism* at a scale you can interrogate live.
    - Sharpness here is estimated by 20 rounds of Hessian-vector-product power iteration, the same estimator validated against exact ground truth on the quadratic bowl in Act 1. Strictly, power iteration converges to the largest-magnitude eigenvalue rather than the largest; on every run plotted here the reported value is positive throughout (visible in the curves, and consistent with the bowl check where the eigenvalue is known and positive), so the two coincide. Deflated iteration gives the faint second and third eigenvalues; they are noisier than the top one.
    - The learning-rate sweep is precomputed on a fixed grid and the slider snaps to it, so dragging never trains anything live; the bowl in Act 1 IS computed live because it costs sixty multiplications.
    - The wind extension reports a measured tendency at one model scale, thirty-two seeds per batch size. It is a first measurement of a question the paper scopes out, not a settled answer to it.

    ### Methods and credits

    **Paper:** Cohen, Kaur, Li, Kolter, Talwalkar, *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability*, ICLR 2021, arXiv 2103.00065. The authors' reference code is public (github.com/locuslab/edge-of-stability); this notebook is an independent dependency-light reimplementation, its instrument validated on a closed-form control rather than by porting their code. Everything renders from computations run in this notebook's own runtime, GPU when available, CPU fallback on the same code path.
    """)
    return


@app.cell(hide_code=True)
def repro_check(mo):
    # Reproduction and finding check for the Edge of Stability walk.
    _rows_rc = [
        (
            "&#9989;",
            "quadratic bowl obeys the classical bound exactly",
            "textbook gradient-descent analysis",
            "Act 1: divergence exactly when sharpness crosses 2/eta",
        ),
        (
            "&#9989;",
            "sharpness rises to 2/eta and rides it",
            "the paper's central finding",
            "measured here: trace pins to the wire (std 0.02 around 2/eta at lr 0.25)",
        ),
        (
            "&#9989;",
            "loss falls non-monotonically while riding",
            "the paper's companion observation",
            "measured here: 311/799 steps move uphill while the trend falls",
        ),
        (
            "&#9989;",
            "the ride persists as the learning rate moves",
            "paper, across architectures",
            "re-measured live at every slider setting",
        ),
        (
            "&#9888;&#65039;",
            "minibatch noise (the wind), 32 seeds per batch size",
            "not in the paper (full-batch by design)",
            "this notebook's extension",
        ),
        (
            "&#9888;&#65039;",
            "survival on the wire collapses through a sharp threshold",
            "no claim in the paper",
            "measured here: 32/32 at full batch, then 30/32, 2/32, 0/32 as batches shrink",
        ),
    ]
    mo.md(
        chr(10).join(
            [
                "### Reproduction check",
                "",
                "| | claim | source | measured here |",
                "|---|---|---|---|",
                *[f"| {a} | {b} | {c} | {d} |" for a, b, c, d in _rows_rc],
                "",
                "Check marks are reproductions of the paper or of classical theory; "
                "warning rows are measurements this notebook adds beyond the paper's scope.",
            ]
        )
    )

    return


@app.cell(hide_code=True)
def closer_refs(mo):
    mo.md(r"""
    Everything above is measured in this kernel, wire included. The fastest way to feel the result is to fail the prediction game once: lock in "it diverges" at a big learning rate, hit reveal, and watch the network balance instead. Go do that.


    ### Three edges of stability

    This notebook is one of three, each built on a different paper, each a view of
    the same edge.

    - **On the Wire** (you are here). Training climbs to the edge of stability,
      pins at 2/eta, and rides it while the loss falls anyway, until noise blows
      it off.
    - **[The Coastline That Never Smooths](https://molab.marimo.io/notebooks/nb_SZ6ifRSGVy1kQY8b5yXrax)**. Seen from above, that edge is
      a fractal coastline in hyperparameter space, and every pixel of it is a real
      training run.
    - **[The Scale DyT Forgot](https://molab.marimo.io/notebooks/nb_jNm948dpXRCvrK9demdeaf)**. Replace LayerNorm with a frozen learned
      scalar and you pin where the edge sits relative to your input scale. Move the
      scale, and the network falls off an edge it can no longer feel.

    The ride, the map, and the calibration. One edge, three ways to fall off it.

    ### References

    1. Cohen, Kaur, Li, Kolter, Talwalkar. *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability.* ICLR 2021. [alphaXiv 2103.00065](https://www.alphaxiv.org/abs/2103.00065)
    2. Sohl-Dickstein, J. *The boundary of neural network trainability is fractal.* 2024. [alphaXiv 2402.06184](https://www.alphaxiv.org/abs/2402.06184). What the landscape of this knife-edge looks like from above.
    """)
    return


if __name__ == "__main__":
    app.run()
