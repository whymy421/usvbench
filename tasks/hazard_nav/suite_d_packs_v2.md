# Suite D: thrust-imbalance protocol v2 — extended-dose pack

Frozen 2026-08-03 BEFORE any v2 evaluation ran, per the same rule as v1:
pack boundaries may not move after seeing results.

## Why v2 exists

The v1 packs (train {0.0, 0.02, 0.04}, interp {0.01, 0.03, 0.05}, extrap
{0.08, 0.12, 0.16}) produced a flat table: the pack-trained champion holds
95-100% SR everywhere, including at 4x the training maximum, with path
creeping only 25.6 -> 28.7 m. Per-episode randomization regularized the
policy far past the frozen extrapolation band, so v1's extrap pack does not
discriminate. That result stands and is reported as-is under v1.

## v2 extended-dose pack (owner-approved 2026-08-03)

- Extended extrapolation: `{0.30, 0.50}`
  (7.5x and 12.5x the training maximum; at 0.50 the starboard motor delivers
  3x the port thrust, a yaw moment of ~16.9 N m under a full-surge command
  against a yaw budget of ~12 N m -- straight-line travel now demands more
  than half the total yaw authority.)
- Champions: the SAME v1-trained checkpoints (s42 agent_13824, s43 agent_5632).
  No retraining; v2 is a pure evaluation extension.
- Procedure: scripts/eval_imbalance.py, 128 episodes x eval seeds {42, 123}
  per value per champion, level 0, report SR / collision rate / median path.
- Falsifiable expectation (pre-registered): if v2 is also flat (>90% at 0.50),
  the imbalance axis is declared non-discriminating for this carrier task at
  any physically sensible dose, and Suite D's discriminating axis must come
  from motor time-constant or asymmetric drag instead.
