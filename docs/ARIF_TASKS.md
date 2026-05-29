# USVBench — Arif's 3-Month Task Plan

> **From**: Yutong (PhD student, UCL Mech Eng)
> **To**: Arif (project teammate)
> **Date**: 2026-05-29
> **Duration**: 3 months (~13 weeks)
> **Authorship opportunity**: Paper B (USVBench benchmark paper, NeurIPS Datasets & Benchmarks track) — you as second author if you complete the work below.

---

## What this project is

We're building **USVBench** — an open-source RL benchmark for surface vessels. It will be the first benchmark in this domain to cover diverse vessel types and physically-grounded marine dynamics (current published ones, like `leonlime/isaac_underwater`, explicitly admit physics are simplified).

Your role: implement **2 vessel-task combinations** that, together with my existing wave-aware navigation tasks, give USVBench a spectrum from 5m fast boats to 100m large vessels.

---

## Vessel × Task spectrum (whole benchmark)

| Vessel | Task | Who | Status |
|--------|------|-----|--------|
| 5m monohull | calm water nav | You (P0 warmup) | reference baseline exists |
| 5m monohull (boat) | wave-aware nav | Yutong | done (E7) |
| 5m monohull × 3 | multi-USV SAR in waves | Yutong | done (E6) |
| **5m catamaran** | **high-speed patrol** | **You (P1)** | TODO |
| **~100m cruise ship** | **harbor approach** | **You (P2)** | TODO |

This gives the benchmark paper a clean narrative: "covers vessel scale from 5m to 100m and dynamics from agile to heavy".

---

## Your 3 months at a glance

| Phase | Weeks | Task | Deliverable |
|-------|-------|------|------------|
| **P0** | 1-2 | Reproduce `boat_calm` baseline | Wandb run, hit reward(mean) > 0.5 |
| **P1** | 3-8 | Catamaran + high-speed patrol | Working task code + 3-seed baselines |
| **P2** | 9-13 | Cruise ship + harbor approach | Working task code + 3-seed baselines |
| **Polish** | 13-14 | Docs + PR review | Two merged PRs |

If anything slows down, drop P2 first. Don't try to add more tasks.

---

## Phase 0: Reproduce boat_calm baseline (Week 1-2)

**Goal**: get the existing `my_first_task_boat_calm` task running on your machine with results similar to my reference.

**Steps**:
1. Install Isaac Lab on your machine (follow Isaac Lab docs).
2. Get the task code from me: `my_first_task_boat_calm/` folder + `boat_physics.usdc`.
3. Place them in the correct paths:
   - Task: `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/my_first_task_boat_calm/`
   - USD: matching the path in `my_first_task_env_cfg.py` line 19
4. Read `STARTER_TASK.md` in the task folder.
5. Run the training command from STARTER_TASK.md. It takes 30-40 minutes.

**Success criteria**:
- Final wandb `Reward / Instantaneous reward (mean)` > 0.5
- `Episode / Total timesteps (mean)` < 4000 (boat reaches target in most episodes)
- Your wandb link sent to me

If you can't hit these numbers, talk to me before moving on. Don't power through.

---

## Phase 1: Catamaran + high-speed patrol (Week 3-8)

### Concept

A 5m catamaran (two parallel hulls) on patrol — must reach a sequence of waypoints fast.

### Steps

1. **Find/make a catamaran USD** (Week 3, see "USD sources" below)
2. **Fork `my_first_task_boat_calm`** → `my_first_task_catamaran_patrol`
3. **Adjust physics constants** (see cheat sheet below)
4. **Modify the task** — instead of single target, give the boat 3-5 waypoints in sequence
5. **Reward design** — bonus when waypoint reached, encourage high cruise speed between waypoints
6. **Train 3 seeds** (42, 123, 456) × 3000 iter each
7. **Document** (PR with STARTER_TASK.md for this task)

### Success criteria

- Catamaran physics stable (boat floats, doesn't sink or fly)
- Mean speed > 1.0 m/s (catamarans are faster than monohulls)
- All 3 seeds reach at least 2/5 waypoints per episode on average
- Wandb runs are public, PR includes baseline numbers

---

## Phase 2: Cruise ship + harbor approach (Week 9-13)

### Concept

A 100m cruise ship approaches a harbor entrance from the open sea. Goal: get inside a defined "harbor circle" (50m diameter) without veering off the channel.

### Steps

1. **Find/make a cruise ship USD** (much harder — see USD sources)
2. **Fork from your catamaran task** → `my_first_task_cruise_docking`
3. **Adjust physics**: huge mass (~10,000 kg), large displacement (~50 m³), high damping
4. **Adjust thrust**: scale up to ~5,000 N to match the inertia
5. **Reward**: penalize lateral deviation from a straight channel + bonus on entering harbor circle
6. **Episode length** longer (cruise ships are slow): 240 s instead of 120
7. **Train 3 seeds** × 5000 iter (more iter because long-horizon)
8. **Document**

### Success criteria

- Cruise ship physics doesn't blow up (with mass=10000kg, anything wrong = NaN crash)
- Boat reaches harbor circle in at least 50% of episodes
- Lateral deviation stays within channel width
- PR merged

---

## Physics cheat sheet

For each vessel, fill cfg values like this:

### 5m catamaran (your P1)

```python
mass = 200.0        # kg (catamarans are heavier than monohull due to dual hulls)
rov_volume = 0.4    # m^3 (200/(1000*0.5) = 0.4 for 50% submerged equilibrium)
rov_height = 1.0    # m
init_z = 0.0        # at equilibrium
linear_damping = 0.5    # PhysX native, low (catamarans glide more)
angular_damping = 3.0
thrust_forward = 300.0  # N (catamarans need more thrust for higher target speed)
yaw_torque = 150.0      # Nm
goal_radius = 3.0       # m
max_spawn_distance = 50  # m (patrol covers more area)
min_spawn_distance = 20
```

### 100m cruise ship (your P2)

```python
mass = 10000.0      # kg (yes, ten tonnes simplified — real cruise ships are 50k+ but we keep it manageable)
rov_volume = 20.0   # m^3 (10000/(1000*0.5) = 20 for 50% submerged)
rov_height = 5.0    # m (large vessel, big draft)
init_z = 0.0
linear_damping = 0.3    # cruise ships have low damping (they coast)
angular_damping = 10.0  # but high yaw inertia
thrust_forward = 5000.0  # N (scaled with mass)
yaw_torque = 3000.0      # Nm
goal_radius = 25.0  # m (50m harbor circle, radius 25)
max_spawn_distance = 200  # cruise ships travel further
min_spawn_distance = 100
episode_length_s = 240.0  # 4 minutes
```

**Important**: these are starting points. Tune them if the vessel sinks, flies, or doesn't accelerate. Tell me if you're stuck for more than 2 days.

---

## Important physics note (read this before P1)

The current physics is **NOT marine-grade** — it's a numerical hack so vessels can float and move. Specifically:
- Mass and volume are tuned for force balance, not realism
- Damping is isotropic via PhysX native (not the proper anisotropic water damping)
- No metacentric stability (GM restoring), no Froude-Krylov wave force, no added mass

**Don't try to fix this** — I'll do the physics refactor (USVBase v2) in parallel during your P1. When ready, you'll switch P2 to use the new physics.

For now: get the boat to behave (floats + moves + learns) using the hack. Realism comes later.

---

## USD sources

You need 3D models of vessels. Free sources I've used or know about:

| Source | Notes |
|--------|-------|
| **GrabCAD** (grabcad.com) | Engineering CAD library, lots of marine. Free with registration. Look for STEP files. |
| **Free3D** (free3d.com) | Mixed quality. Search "boat" or "ship". Look for .blend or .obj files. |
| **TurboSquid free section** | Some free models. |
| **3D Warehouse** (3dwarehouse.sketchup.com) | SketchUp library. Conversion to USD is painful. |
| **Blender Market** (some free) | Higher quality but mixed licensing. |
| **NASA 3D resources** | Some marine vessels in research collection. |

**Workflow**: model → SolidWorks (clean up) → STL → Blender → USD export

**Trap I hit**: When exporting from SolidWorks, the body axes may not match Isaac's convention. **Test which axis is "forward" with a debug thrust before training**. Our `boat_calm` has `body-X = stern, body-Y = starboard, body-Z = up` (non-standard). Set `forward_vec = [-1, 0, 0]` in env to match.

---

## PR acceptance criteria

For each task PR, must have:

- [ ] Task registered in `__init__.py` with new ID
- [ ] cfg file with physics constants documented (mass, volume, etc commented)
- [ ] env file with reward function commented (each term's purpose)
- [ ] STARTER_TASK.md in task folder
- [ ] At least 3 wandb runs (3 seeds × 3000+ iter) with public links
- [ ] Reproducible: I can clone, run the command, get similar results
- [ ] Physics axis verified — comment in cfg saying which body axis is "forward"

If any of these are missing, I'll request changes before merging.

---

## When to ask for help

**Immediately** (don't wait):
- Anything crashes with NaN
- Boat fails to load (USD import error)
- Reward goes massively negative and stays there
- You don't understand what a config field does

**Within 2 days**:
- Boat physics looks wrong (sinks/flies/oscillates)
- PPO training plateau

**End of week**:
- General status update (1 short message: "did X, blocked on Y, next is Z")

---

## Schedule check-ins

| Frequency | Format |
|-----------|--------|
| Each Friday | 1-message status update (Teams) |
| End of each Phase | Video call (30 min) — review code, plan next |
| Anything urgent | Just message me |

I'll respond within 24 hours unless there's a problem.

---

## Authorship + recognition

If you complete P0 + P1 + P2 with merged PRs and 3-seed baselines, you'll be **second author on Paper B (USVBench)** when we submit to NeurIPS Datasets & Benchmarks 2026.

If you complete only P0 + P1, you'll be acknowledged in the paper (not author).

If you drop out partway, no hard feelings, but please tell me early so I can plan.

---

## Quick reference

- This document: long-term roadmap
- `STARTER_TASK.md` (in task folder): how to run boat_calm
- `BENCHMARK_DESIGN.md`: overall architecture (you don't need to read this for P0/P1)
- `EXPERIMENT_LOG.md`: my full experiment history (for context, not required reading)
- Wandb project: `usv-navigation` — I'll add you as a collaborator
- Repo: TBD (will share when set up)

Questions? Just ask.
