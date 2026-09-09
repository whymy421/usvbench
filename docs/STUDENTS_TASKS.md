# USVBench - Student Task Assignments (Calm-Water Phase)

> **Students**: Shi Jin (Mechanical Engineering, Year 3), Zhang Nianxi
> **Supervisor**: Dr. Zhang
> **Goal**: Extend the USVBench calm-water benchmark with one or two new tasks per student.

---

## Shi Jin's Task List

1. **Reproduce the ROV calm baseline** (Weeks 1-2)
   - Run `Isaac-My-First-Task-Calm-Direct-v1`.
   - Reproduce the **4.48** targets/episode deterministic reference in [`TASKS.md`](../TASKS.md#current-baselines).
   - Confirm the Isaac Lab, skrl, and wandb workflow.

2. **Gazebo sim-to-sim validation** (Weeks 3-4)
   - Export the trained Isaac Sim ROV policy.
   - Reproduce the same task in Gazebo/VRX.
   - Compare Isaac and Gazebo trajectories and performance.
   - Deliver a trajectory comparison and transfer-gap metric.

3. **Benchmark enrichment with a new hull** (Week 5+)
   - Select a new hull such as a catamaran or cruise ship.
   - Validate USD loading and physics stability.
   - Design a new task such as point navigation, route following, or docking.
   - Train and report a 3-seed baseline.
   - Deliver `STARTER_TASK.md` and a merge request.

---

## Zhang Nianxi's Task List

1. **Reproduce the boat calm V26 baseline** (Weeks 1-2)
   - Run `Isaac-USVBench-Boat-Calm-Direct-v1`.
   - Understand why the boat reward uses speed coupling and a reach bonus.
   - Reproduce the **3.75** targets/episode deterministic seed-42 reference in [`TASKS.md`](../TASKS.md#current-baselines).

2. **Complete the boat 3-seed baseline** (Week 3)
   - Run seeds 123 and 456.
   - Compute the 3-seed mean and standard deviation.
   - Update the boat baseline in `README.md` and `TASKS.md`.

3. **Calm-water obstacle avoidance** (Week 4+)
   - Add static obstacles such as rocks or buoys to a calm-water task.
   - Design collision detection and a reward penalty.
   - Train a baseline and validate it over three seeds.
   - Deliver the environment, starter guide, and baseline.

---

## General Requirements

- Each new task should deliver:
  - `tasks/<name>/my_first_task_env.py` and `my_first_task_env_cfg.py`.
  - `tasks/<name>/STARTER_TASK.md` with run steps, success criteria, and environment variables.
  - `tasks/<name>/checkpoints/<name>_s42.pt` as a reference policy.
  - Three wandb runs and the corresponding scores.
- Code quality:
  - Follow the Git workflow in [`docs/GITHUB_COLLABORATION.md`](GITHUB_COLLABORATION.md).
  - Keep experiments single-variable in accordance with the project hard rules.
  - Log runs to `uclusv/usvbench` and check `$env:WANDB_ENTITY`.
- Documentation:
  - Explain why the task uses its reward design.
  - List mass, damping, thrust, and the reasons for their values.
  - Record any shortfall under `Known limitations`.

---

## Progress Tracking

| Student | Task 1 | Task 2 | Task 3 |
|---|---|---|---|
| **Shi Jin** | Reproduce ROV | Gazebo validation | New hull (self-selected) |
| **Zhang Nianxi** | Reproduce boat and add 3-seed results | Obstacle avoidance | - |

Post a one-sentence status update every Friday. Report blockers to Yutong on the same day.
