# Suite D: thrust-imbalance protocol

Suite D uses the forced-crossing carrier task and changes only the fixed
left/right thrust-gain mismatch. The frozen packs are:

- Train: `{0.0, 0.02, 0.04}`
- Interpolation: `{0.01, 0.03, 0.05}`
- Extrapolation: `{0.08, 0.12, 0.16}`

Train on `Isaac-USV-HazardCrossImb-Direct-v1`. It samples the train pack
uniformly per environment at every episode reset.

Evaluate the selected checkpoint through `scripts/eval_imbalance.py`, which
disables the choice tuple and applies one scalar imbalance at a time. For every
value in every pack, run 128 episodes at each of evaluation seeds 42 and 123.
Report success rate (`SR`), collision episodes/rate, and path-length statistics
separately for every value and seed; do not pool values before reporting them.

The train, interpolation, and extrapolation pack boundaries were frozen on
2026-08-03, before any Suite D run. They may not be moved, relabeled, or have
values added or removed after seeing results.
