"""Accent Vector on F5-TTS.

A port of the Accent Vector method (Lertpetchpun et al., 2026) to the F5-TTS
backbone, adapted from Expressive-Vectors. The task-vector formulation is
backbone-agnostic:

    tau_accent = theta_ft - theta_pre          # extract  (paper Eq. 1-3)
    theta      = theta_pre + alpha * tau        # scale    (paper Eq. 4)
    theta      = theta_pre + sum_i a_i * tau_i  # compose  (paper Eq. 5-6)

See ../ADAPTATION_PLAN.md for the design rationale and the deviations from the
paper's XTTS-v2 recipe (notably: F5-TTS has no language-ID token).
"""
