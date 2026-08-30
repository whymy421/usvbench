# USVBench

A reproducible simulation benchmark for deadline-aware autonomous surface vessel
control, built on NVIDIA Isaac Lab.

This repository is the artefact for the MSc thesis *Sim2Real Benchmarking for
Multimodal Marine Robotics* (Arif Abubhakkar, UCL Department of Mechanical
Engineering, 2026). It contains the environment, the three verification
instruments, the evaluation harness, and the raw data behind every figure and
table in the thesis.

---

## What this is

USVBench defines two tasks for unmanned surface vessels:

- **P1, waypoint patrol.** A catamaran visits a cyclic waypoint sequence. Used
  here as a diagnostic control condition. **No validated score is reported for
  P1** — see the thesis, Section 7.4.
- **P2, deadline-aware multi-casualty rescue.** Four casualties spawn at 80–200 m
  with independent countdown deadlines of 40–90 s. A four-point tour spans about
  500 m and needs roughly 42 s of transit against a shortest deadline of 40 s, so
  the vessel provably cannot reach all of them. Triage is forced, not optional.

The headline result on P2 is a rescue rate of **0.770 ± 0.036** across three
seeds against a threshold of 0.70, exceeding a scripted pure-pursuit controller
with perfect state access by 0.188.

> **Read this before using the number.** The reported figure selects the best
> checkpoint per seed on the same episodes used to report it, which is an
> optimistic estimator. Re-scoring one selected checkpoint over independent
> episodes returned 0.872 against the 0.934 that had selected it, so the
> selection bias is of order 0.06. Thesis Section 9.5 sets this out in full.

## Why it exists

Nine defects were found during development, spanning the physics model, the
evaluation harness and the task specification. None raised an exception,
produced a NaN, or emitted a warning. Two concealed one another and produced a
plausible intermediate result; two more were *exactly correct at zero heading*,
which is where every episode begins and where conventional single-axis testing
operates.

The transferable outcome is procedural rather than numerical: a benchmark
specification is incomplete without a demonstrated attainable lower bound, and
evaluation software warrants the same verification usually reserved for the
policy it scores. `docs/PROGRESS.md` carries the full defect register.

---

## Layout

```
rescue_boat/      The P2 environment, config, gym registration and agent config
                  plus the PowerShell run scripts used for every reported run
tools/            Verification instruments and analysis
  probe_physics.py          Open-loop physics probe. Phase 3 sweeps thrust and
                            yaw through all headings and asserts tilt <= 5 deg
  eval_fixed.py             Corrected evaluator. Refuses to score if the
                            observation normaliser cannot be restored
  diagnose_eval_sweep.py    Reproduces the original evaluation defect
  eval_oracle.py            Scripted pure-pursuit baseline and config sweeps
  eval_robustness.py        Sea-state sweep. Implemented, never run
  analyse_behaviour.py      Path efficiency, revolutions, hull-vs-track angle
  analyse_rescue_timing.py  When rescues and expiries occur within an episode
  train_with_eval.py        Training with the corrected checkpoint sweep
  mkfigs.py, mkfig_sim.py   Regenerate every thesis figure from data/
data/             Raw evaluation output behind every figure and table
figures/          The figures as published, vector PDF
docs/             Defect register, results log, asset provenance
```

## Reproducing the results

Requires Isaac Sim 5.1.0, Isaac Lab 0.54.4, skrl 2.1.0. Every reported run was
produced on a single RTX 4060 Laptop GPU with 8 GB of VRAM, at 64 parallel
environments. Cluster compute is not needed and was never used.

```bash
# 1. Verify the vehicle model before trusting anything downstream
python tools/probe_physics.py --headless
#    Phase 3 must PASS: peak tilt <= 5 deg and depth > -5 m through a full
#    heading sweep. Reference run: 2230 deg of cumulative heading change
#    (6.19 turns), peak tilt 1.77 deg, minimum depth -0.60 m.

# 2. Establish what a non-learning controller achieves
python tools/eval_oracle.py --config_sweep --headless

# 3. Train
./rescue_boat/train_v4_locomotion.ps1

# 4. Score. This will refuse to run if the normaliser cannot be restored.
python tools/eval_fixed.py --checkpoint=<path>.pt --headless

# 5. Regenerate the thesis figures from data/
python tools/mkfigs.py
```

PPO hyperparameters are in `data/ppo_hyperparameters.yaml`. They are the skrl
defaults for continuous control with the network narrowed to 256-128-64 to fit
8 GB. **No hyperparameter search was performed**, so the sensitivity of the
reported results to these values is unknown.

## Data

`data/` holds the CSVs behind each figure, named by what they are rather than by
run timestamp:

| File | Thesis reference |
|---|---|
| `eval_sweeps/original_harness_*.csv` (5 seeds) | Fig. 2, grey series. Every checkpoint scores exactly 0.0000 except at 240k and 540k steps, at identical checkpoints in all five seeds |
| `eval_sweeps/corrected_harness_*.csv` (3 seeds) | Fig. 2, navy series. The same weights, re-scored, flat near 0.10 |
| `eval_sweeps/v3_noshaping_*.csv` (3 seeds) | Table 5, without shaping |
| `eval_sweeps/v4_shaping_*.csv` (3 seeds) | Table 5 and 6, with shaping. The headline result |
| `physics_probe_before_correction.csv` | Fig. 3, before. Peak roll 166 deg, minimum depth -43 m |
| `physics_probe_after_correction.csv` | Fig. 3, after. Peak tilt 1.77 deg, holds -0.60 m |
| `oracle_config_sweep.csv` | Fig. 5. Adopted configuration gives 0.582 |
| `oracle_radius_timer_sweep.csv` | Fig. 5, original configuration gives 0.262 |
| `p1_catamaran_eval.csv` | Section 7.4. Pre-correction harness, not a valid result |

## Known limitations

- Everything here is simulated. No physical trial was run and no sim-to-real
  transfer claim is made.
- Three seeds is a small sample. No significance test is claimed and none would
  be meaningful at n = 3.
- The final policy still completes about 12 hull revolutions per 120 s episode,
  down from 19 without locomotion shaping. That is an improvement, not physical
  vessel behaviour.
- P1 is unscored.
- `eval_robustness.py` is implemented but has never been run.
- Section 8 of the thesis is titled "Validation and verification" but contains
  verification only. Nothing here is validated against Fossen, published RIB
  manoeuvring data, or a second simulator.

## Asset licence

**`rescue_boat.usd` is not included in this repository.** The source CAD was
obtained during the project and its redistribution terms have not been
confirmed. See `docs/ASSET_PROVENANCE.md`. A parametric hull generated from
published dimensions is the better long-term answer for a shared benchmark, and
`tools/make_rescue_boat_usd.py` is the starting point for that.

## Citation

If this is used, please cite the thesis. A DOI will be minted on release.

## Acknowledgements

Supervised by Dr Yao Zhang. Yutong Song identified two of the physics defects,
the heading-invariance error and the double-rotated wrench, both of which are
exactly correct at zero heading and had survived my own testing for that reason.
