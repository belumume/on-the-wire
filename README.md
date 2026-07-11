# On the Wire

A marimo notebook built on "Gradient Descent on Neural Networks Typically
Occurs at the Edge of Stability" (Cohen, Kaur, Li, Kolter, Talwalkar, ICLR
2021, arXiv 2103.00065), for the alphaXiv x marimo molab Notebook Competition #2.

Three acts: a quadratic bowl where classical 2/eta theory holds exactly (and
calibrates the sharpness meter against a known answer), a prediction you lock
in, then a real network that defies it, riding the 2/eta wire while its loss
falls non-monotonically. The extension asks whether the clamp survives
minibatch noise; the measured answer is a threshold: below a critical batch
size, the wind blows training off the wire entirely.

Live notebook: https://molab.marimo.io/notebooks/nb_QtxPsWxaZSpnRU2J8qWZWm

Run locally:

    uvx marimo edit --sandbox notebook.py

A GPU speeds up the precomputed learning-rate sweep; CPU runs the same code
path. `engine.py` is the standalone engine. `__marimo__/` carries rendered
outputs.

## What this adds to the paper

The reproduction is the ground floor: the MLP rides 2/eta with std 0.02 while the
loss falls non-monotonically (311 of 799 steps move uphill while the trend falls,
counted live), and a quadratic bowl where classical theory holds exactly calibrates
the sharpness estimator against a known answer before it touches the network.

What the deterministic analysis leaves open is minibatch noise: does the pinning
survive it? Measured here with 32 independent runs per batch size, where survival
means sharpness is still pinned at 2/eta at the end rather than the run diverging.
Full batch: all 32 hold. Then 30/32. Then 2/32. Then none. Almost the whole collapse
lands in one step. The threshold is sharp, and the notebook shows where it sits.

## The trilogy

Three notebooks, one per paper, three views of the same edge:

- **On the Wire** (this one): training rides the stability boundary rather than avoiding it.
- **The Coastline That Never Smooths**: that boundary is fractal. https://molab.marimo.io/notebooks/nb_SZ6ifRSGVy1kQY8b5yXrax
- **The Scale DyT Forgot**: a frozen scalar pins where it sits relative to your input scale. https://molab.marimo.io/notebooks/nb_jNm948dpXRCvrK9demdeaf
