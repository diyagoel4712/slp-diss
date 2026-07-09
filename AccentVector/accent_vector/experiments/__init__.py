"""Experiment harness for the Accent Vector dissertation plan.

Each module maps onto a research question / experiment code in EXPERIMENTS.md:

    grid.py             build the synthesis grid (accent x alpha)      [A1]
    rq1_reproduction.py alpha-monotonicity + identity retention        [E1.1-1.2]
    rq2_geometry.py     weight-space map + output-space RSA/Mantel      [E2.1-2.3]
    rq3_decomposition.py segmental (PPG-KL) vs suprasegmental (F0)      [E3.1-3.3]
    rq3_layers.py       layer localisation of the accent vector        [E3.4]
    rq5_bias.py         relative WER, gender split, metric agreement    [E5.1-5.2]
    common.py           shared math (cosine, Mantel, nPVI, gap-closure)

Every analysis reads the ONE synthesis grid produced by grid.py; only grid.py
(and the fine-tunes upstream of it) need a GPU.
"""
