# On the Wire

A marimo notebook built on "Gradient Descent on Neural Networks Typically
Occurs at the Edge of Stability" (Cohen, Kaur, Li, Kolter, Talwalkar, ICLR
2021, arXiv 2103.00065), for the marimo x alphaXiv Notebook Competition #2.

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
