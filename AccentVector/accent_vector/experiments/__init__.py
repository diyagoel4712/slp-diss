"""Experiment harness for the Accent Vector dissertation plan.

Each module maps onto a research question / experiment code in EXPERIMENTS.md:

    grid.py             build the synthesis grid (accent x speaker x alpha)  [A1]
    aggregate.py        pool per-speaker metric CSVs across speakers    [A1]
    rq1_reproduction.py alpha-monotonicity + identity retention        [E1.1-1.2]
    rq2_geometry.py     weight-space map + output-space RSA/Mantel      [E2.1-2.3]
    rq3_decomposition.py segmental (PPG-KL) vs suprasegmental (F0)      [E3.1-3.3]
    rq3_layers.py       layer localisation of the accent vector        [E3.4]
    rq6_temporal.py      accent-vector trajectory over fine-tuning       [E6.1]
    viz_temporal.py     dashboard video of that trajectory              [E6.1]
    shared.py           helpers reused by 2+ RQ modules (eval bridge, grid IO,
                        cosine/MDS geometry, threshold-onset)

Every analysis reads the ONE synthesis grid produced by grid.py; only grid.py
(and the fine-tunes upstream of it) need a GPU.
"""
