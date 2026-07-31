# USVBench — Arif's Task Plan (week-by-week)

> **From**: Yutong (PhD, UCL Mech Eng) · **To**: Arif
> **Updated**: 2026-06-11 · **Duration**: ~13 weeks + 1 buffer
> **Authorship**: complete P0+P1+P2 with merged PRs and 3-seed baselines → **second author on the USVBench paper** (target NeurIPS Datasets & Benchmarks 2026).

---

## The project in one paragraph

We're building **USVBench**, an open RL benchmark for surface vessels with real marine
physics (buoyancy, damping, currents). I've built and validated two calm reference tasks
(ROV + boat). **Your job** is to add new vessel × task combinations so the benchmark
covers a wider range of dynamics. You start by reproducing my two reference tasks to
confirm your pipeline works, then build ~2 new tasks of your own.

**You have a lot of freedom on what those two tasks are.** Below I suggest a fast
**catamaran** and a large **cruise ship** (they nicely extend the range from agile to
heavy), and the week-by-week plan is written around them so you have a concrete path to
follow. But if you have your own idea — a different vessel, a different task (docking,
obstacle avoidance, formation, station-keeping, racing…) — **go with it.** Just send me
a one-paragraph proposal first so we make sure it fits the benchmark and isn't redundant
with what I'm building. The examples are a default, not a requirement.

---

## Vessel × task spectrum (whole benchmark)

| Vessel | Task | Owner | Status |
|--------|------|-------|--------|
| ROV | calm nav | Yutong | ✅ done (Task A, **4.48** tgt/ep) |
| 5 m monohull | calm nav | Yutong | ✅ done (Task B, **3.75** tgt/ep) |

> Baselines are maintained in one place only: [`TASKS.md`](../TASKS.md#current-baselines).
> Both are the deterministic `eval_benchmark.py` score of the checkpoint shipped in this
> repo, on the realistic-dynamics physics merged 2026-07-31.
| **Arif's task 1** | *e.g.* catamaran high-speed patrol | **Arif (P1)** | ⬜ TODO (your call) |
| **Arif's task 2** | *e.g.* ~100 m ship harbor approach | **Arif (P2)** | ⬜ TODO (your call) |

---

## Week-by-week

| Week | Phase | Goal | Deliverable |
|------|-------|------|-------------|
| **1** | P0 | Setup + reproduce **Task A (ROV calm)** | ≥ 3.5 tgt/ep on the deterministic eval |
| **2** | P0 | Reproduce **Task B (boat calm)**, 3 seeds | 3 runs ≥ 3.0 tgt/ep; Friday: "P0 done" |
| **3** | P1 | Get a **catamaran USD**, verify forward-axis with a debug thrust | USD in `assets/`, axis noted |
| **4** | P1 | Fork `boat_calm_nav` → `catamaran_patrol`, make it **float + move** (no RL yet) | stable physics demo |
| **5** | P1 | Single-target navigation learns on the catamaran | wandb run reaching targets |
| **6** | P1 | Switch to **waypoint sequence** (3–5 ordered waypoints) | env + reward implemented |
| **7** | P1 | Reward tuning for high cruise speed between waypoints | speed > 1 m/s, waypoints hit |
| **8** | P1 | **3-seed baselines + STARTER + PR** for catamaran | PR opened (catamaran done) |
| **9** | P2 | Get a **cruise-ship USD**, physics stable at mass ≈ 10 t (no NaN) | float demo, no blow-up |
| **10** | P2 | Fork → `cruise_docking`, ship **moves + turns** under scaled thrust | stable moving demo |
| **11** | P2 | **Harbor-approach** task: straight channel + 50 m harbor circle | env + reward implemented |
| **12** | P2 | Reward tuning + training (longer episodes, 5000 iter) | wandb run entering harbor |
| **13** | P2 | **3-seed baselines + STARTER + PR** for cruise ship | PR opened (cruise done) |
| **14** | Buffer | Polish docs, address PR review comments | both PRs merged |

The P1/P2 weeks above are written for the **example** tasks (catamaran, cruise ship). If
you pick your own tasks, keep the same rhythm — roughly: get the USD + physics stable
(1–2 wks) → task + reward (2–3 wks) → 3-seed baseline + STARTER + PR (1 wk) — and we'll
adjust the schedule together.

**If you fall behind, drop the second task first.** One clean, well-documented task is
worth more than two rushed half-working ones.

---

## P0 — Reproduce my reference tasks (Weeks 1–2)

This proves your Isaac Lab + skrl + wandb pipeline works before you build anything new.

**Week 1 — Task A (ROV), the easy one:**
1. Install Isaac Lab (follow the official docs) and create/activate its conda env.
2. Get repo access (send me your GitHub username) and clone to `~/usvbench`.
3. Copy `tasks/rov_calm_nav/` into `<IsaacLab>/source/isaaclab_tasks/isaaclab_tasks/direct/`.
4. Follow `tasks/rov_calm_nav/STARTER_TASK.md`. Run ~30 min.
5. **Success**: `targets_per_episode` ≥ 3.5 on the deterministic eval. Send me the
   wandb link. (Reference: the shipped `rov_calm_s42.pt` scores **4.48** — see
   [`TASKS.md`](../TASKS.md#current-baselines) for the exact command. Earlier drafts of
   this doc quoted 6.9 from a `rov_calm_current_s42.pt` that was never committed, and
   before that ≥ 20 from the pre-merge physics; both are obsolete.)

**Week 2 — Task B (boat), 3 seeds:**
1. Same install steps for `tasks/boat_calm_nav/`.
2. Read its STARTER carefully — **the reward is different from Task A and the doc explains why**. Use the exact env vars given; do NOT set `FORWARD_TRANSIT=1`.
3. Run seeds 42, 123, 456.
4. **Success**: each run ≥ 3.0 `targets_per_episode`. Send 3 wandb links.

If your numbers are far off, **message me before moving on** — don't power through.

---

## P1 — Your first new task (Weeks 3–8) · *example: catamaran high-speed patrol*

> This is one suggested task. If you have your own idea, swap it in (send me a short
> proposal first) and keep the same rhythm. The example below is fully worked out so you
> have something concrete to start from.

**Example concept**: a 5 m catamaran patrols a sequence of 3–5 waypoints, rewarded for
reaching them quickly. Catamarans glide more (lower damping) and run faster than a monohull.

**Steps**: find a catamaran USD → verify which body axis is "forward" with a debug thrust
→ fork `boat_calm_nav` to `catamaran_patrol` → tune physics (cheat sheet below) → replace
single-target with a waypoint sequence → reward fast transit + waypoint reach → 3 seeds.

**Success criteria**:
- Physics stable (floats, doesn't sink/fly/NaN)
- Public wandb runs + PR with baseline numbers
- Mean speed: **TBD** — see below
- `targets_per_episode` averaged over the 3 seeds: **TBD** — see below

> ⚠️ **The two numeric bars are deliberately not filled in.** The old ones (2.0 tgt/ep,
> mean speed > 1.0 m/s) were calibrated against reference baselines that were roughly 5x
> higher than they are now, before the realistic-dynamics merge. Any number written here
> today would be guesswork, and a fabricated bar is worse than no bar — it gets optimised
> against. They will be set from measurement once the catamaran is ported onto the same
> damping and actuator model as the reference tasks, using the same proportion to the
> reference score that the P0 bars use (~80%). Ping me when the port is done and send the
> deterministic eval number; I will fix the bars then.

---

## P2 — Your second new task (Weeks 9–13) · *example: cruise-ship harbor approach*

> Again just a suggestion — your own idea is welcome. The example pairs well with P1
> because it pushes the heavy/slow end of the dynamics range.

**Example concept**: a ~100 m ship approaches a harbor entrance from open sea; goal is to
enter a 50 m-diameter harbor circle without veering out of the channel.

**Steps**: find a ship USD (harder) → fork your catamaran task → scale physics way up
(mass ~10 t, big displacement, high yaw inertia, ~5000 N thrust) → reward = penalize
lateral deviation from the channel + bonus on entering the circle → longer episodes
(240 s) → train 5000 iter × 3 seeds.

**Success criteria**:
- Physics doesn't blow up (with mass = 10 000 kg, any mistake → NaN)
- Ship enters the harbor circle in ≥ 50% of episodes
- Lateral deviation stays within the channel
- PR merged

---

## Physics cheat sheet

> ⚠️ **Obsolete as of the 2026-07-31 realistic-dynamics merge — do not copy these
> numbers.** Every value below is expressed in the old physics framework: a single
> isotropic `linear_damping` / `angular_damping` pair, PhysX-native damping, and hardcoded
> thrust constants. That framework no longer exists on `main`. The reference tasks now use
> per-DOF linear+quadratic damping with actuator limits in the cfg, and terminal speeds
> that emerge from the force balance rather than from engine caps.
>
> Use `tasks/boat_calm_nav/my_first_task_env_cfg.py` as the template instead: it is a
> surface vessel with the same structure, and its coefficients are Froude-scaled from the
> VRX WAM-V with the derivation written down in the docstring. Scale from there for your
> hull, state the scaling you used in your NOTES.md, and check the terminal surge and yaw
> rates that come out are the ones you intend.
>
> The block below is kept only to show the rough size of each quantity.

Starting points — tune if the vessel sinks, flies, or won't accelerate.

### 5 m catamaran (P1)
```python
mass = 200.0            # kg (dual hulls, heavier than monohull)
rov_volume = 0.4        # m^3 (200/(1000*0.5) for 50% submerged equilibrium)
rov_height = 1.0        # m
linear_damping = 0.5    # PhysX native, low (catamarans glide)
angular_damping = 3.0
thrust_forward = 300.0  # N (more thrust for higher patrol speed)
yaw_torque = 150.0      # Nm
goal_radius = 3.0       # m
max_spawn_distance = 50 # m (patrol covers more area)
min_spawn_distance = 20
```

### ~100 m cruise ship (P2)
```python
mass = 10000.0          # kg (simplified; real ships are far heavier)
rov_volume = 20.0       # m^3 (10000/(1000*0.5) for 50% submerged)
rov_height = 5.0        # m (big draft)
linear_damping = 0.3    # ships coast
angular_damping = 10.0  # but high yaw inertia
thrust_forward = 5000.0 # N (scaled with mass)
yaw_torque = 3000.0     # Nm
goal_radius = 25.0      # m (50 m harbor circle)
max_spawn_distance = 200
min_spawn_distance = 100
episode_length_s = 240.0
```

---

## Important physics note (read before P1)

The current physics is **not marine-grade** — it's a numerical hack so vessels float and
move (mass/volume tuned for force balance, isotropic damping, no metacentric stability or
added mass). **Don't try to fix this** — I'm doing the proper physics refactor in parallel.
For now: just get each vessel to float, move, and learn using the hack.

**Axis trap I hit**: exported hulls often don't match Isaac's axis convention. Our
`boat_calm_nav` has `body-X = stern, body-Y = starboard, body-Z = up` (non-standard).
**Always test forward-axis with a debug thrust before training a new vessel** and note it
in the cfg.

---

## USD sources

| Source | Notes |
|--------|-------|
| GrabCAD (grabcad.com) | Lots of marine CAD, free with registration. Look for STEP. |
| Free3D (free3d.com) | Mixed quality, search "boat"/"ship", .blend/.obj. |
| TurboSquid (free section) | Some free models. |
| 3D Warehouse | SketchUp library; USD conversion is painful. |

Workflow: model → SolidWorks (clean up) → STL → Blender → USD export.

---

## Check-ins & getting help

| When | What |
|------|------|
| Every Friday | one-message status: "did X, blocked on Y, next is Z" |
| End of each phase | 30-min video call to review code + plan |
| Anything urgent | just message me — I reply within 24 h |

**Ask immediately** (don't wait) if: anything NaN-crashes, USD won't load, reward goes
hugely negative and stays, or you don't understand a config field. **Ask within 2 days**
if the physics looks wrong (sinks/flies) or training plateaus.

---

## Authorship

- P0 + **both your tasks** (P1 + P2) with merged PRs + 3-seed baselines → **second author** on the USVBench paper.
- P0 + **one** task → acknowledged in the paper.
- Need to stop early → no problem, just tell me early so I can re-plan.

---

## Reference docs

- `GITHUB_COLLABORATION.md` — git workflow (read before your first commit)
- `tasks/rov_calm_nav/STARTER_TASK.md` — how to run Task A
- `tasks/boat_calm_nav/STARTER_TASK.md` — how to run Task B (and why its reward differs)
- wandb project: **usvbench** — I'll add you.

Questions? Just ask.
