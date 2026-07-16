# Classical baseline

`classical_baseline.py` is USVBench's zero-learning reference controller. It
uses the same deterministic evaluation protocols as learned policies, but its
control stack is conventional marine line-of-sight guidance plus heading PID:

- bearing error comes from `atan2(cross, dot)`;
- PID maps wrapped heading error to yaw torque;
- thrust is reduced near the goal or hold zone, is zero inside it, and uses a
  small creep command while the vehicle turns.

The controller reads only `obs[0:3]` (`dot`, `cross`, and
normalized distance). Hull-specific forward-axis conventions are already
encoded by those observations, so the same controller is used for both hulls.

## Why every task publishes this number

Every benchmark task should publish the classical baseline beside learned-policy
results. It pins a practical floor/ceiling and exposes tasks whose nominal
learning challenge is already solved by standard feedback control: a gate a PID
can pass is not a learning gate. Use `--protocol throughput` for the fixed-step
targets-per-episode evaluation and `--protocol mission` for completed-episode
SR/SPL evaluation.

```powershell
python scripts/classical_baseline.py --task Isaac-My-First-Task-Calm-Direct-v1 `
  --protocol throughput --num_envs 64 --eval_steps 6000 --headless

python scripts/classical_baseline.py --task Isaac-USV-StationKeep-Direct-v1 `
  --protocol mission --num_envs 64 --episodes 128 --headless
```

Point-navigation distance observations default to a 30 m scale. Station-keeping
defaults to 15 m. Override a nonstandard task with `--dist-scale`.

## Gain tuning

The defaults (`kp=2.0`, `ki=0.0`, `kd=0.5`, slow radius 5 m) were hand-set. Gain
tuning is method-side freedom: classical methods may tune these values just as
learning methods tune their algorithms. Always publish the gains, slow radius,
distance scale, task seed, and tuning procedure with the result. Do not tune on
the reported evaluation episodes.
