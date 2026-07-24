# HazardNav v11 showcase

The packaged v11 checkpoint, three rollout videos, independent evaluation, and
the comparison against `jinshi-brady/benchmark-v2` are available here:

- [Showcase and reproduction commands](artifacts/hazard_nav_v11_showcase/README.md)
- [Detailed comparison with the student branch](artifacts/hazard_nav_v11_showcase/COMPARISON_WITH_JINSHI_BRADY.md)
- [Fixed-level evaluation JSON](artifacts/hazard_nav_v11_showcase/eval/v11_best_level1_seed42.json)

The runnable task id is `Isaac-USV-HazardNav-Direct-v3`. The included checkpoint
is `tasks/hazard_nav/checkpoints/hazard_nav_v11_best_s42.pt`.

Important: v11 improves goal entry and obstacle navigation but does not solve
stern-first travel. The measured pre-goal reverse-velocity ratio is 0.8944.
