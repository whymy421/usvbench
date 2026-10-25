"""Fast GPU build check of the journal envs (G6 diagnostic; writes NO certificate).

For each journal id: build the env with 2 envs through the SAME cfg path as the
evaluators, reset it, step a few control steps with zero actions, close it, and report
PASS or FAIL with the exception text. Exit code 0 only if every id passes.

ONE PYTHON PROCESS PER ID. Isaac Lab's DirectRLEnv creates the process-wide
SimulationContext and refuses a second one while it exists ("Simulation context already
exists", isaaclab/envs/direct_rl_env.py); a failed build can leave that singleton, the USD
stage and Kit extensions half-initialised, and a crash or hang in one id would hide every
later id. A fresh process per id isolates each build and lets a named per-id timeout kill
a hung Kit without losing the other results. The parent process imports no Isaac module.

Cfg path per id (--cfg-path auto, the default) = the evaluator the P5 smoke ran for it:
  hydra  classical_baseline.py (path_following, station_keeping, docking):
         isaaclab_tasks.utils.hydra.hydra_task_config(id, "skrl_cfg_entry_point")
         [registry cfg -> to_dict -> Hydra/OmegaConf -> from_dict], then resolve_protocol
  parse  classical_planner_pid.py / classical_harbour.py / eval_mppi.py / eval_v6_frozen.py
         (every other task): isaaclab_tasks.utils.parse_env_cfg, then resolve_protocol
Both paths: eval_protocol.preflight_protocol (release identity, ambient variables, journal
id selection), register_journal_tasks, resolve_protocol with the P5 cell arguments
(--eval-seed 42 --level 0 --num_envs 2 --episodes 2 [--suite ...] [--set ...]),
gym.make(id, cfg=env_cfg), env.reset(), eval_mppi._detect_mode(env) (the evaluator interface;
a refused family is a FAIL at stage "interface"), the cert_v3.CertificateRun constructor (G6b:
the certificate header checks; FAIL at stage "certificate"; no file written), N x env.step(zeros),
env.close().

Usage (on the GPU host, from the release root, in the Isaac Lab python):
  python -B scripts/gpu_build_check.py --headless --device cuda:0 \
      --release-descriptor <release.json> --d7-pool-dir <pool bundle dir> \
      [--ids <journal id> ...] [--vessel blueboat|wamv|both] [--suite waves=D1 ...] \
      [--steps 5] [--timeout-s 900] [--cfg-path auto|parse|hydra] [--log-dir <dir>]
Without --ids every journal id of the selected vessel(s) is checked (24 for both).
--d7-pool-dir is exported to the workers as USVBENCH_D7_POOL_DIR and, for docking ids, also
passed as --set d7_pool_dir=<dir> (as the P5 queue does). Per-id logs go to --log-dir.
Pure ASCII.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SCRIPTS = Path(__file__).resolve().parent
RESULT_TAG = "G6-BUILD-RESULT "
HYDRA_TASKS = ("path_following", "station_keeping", "docking")
DEFAULT_TIMEOUT_S = 900
DEFAULT_STEPS = 5


# ------------------------------------------------------------------ parent (no Isaac imports)
def journal_ids(vessel="both"):
    """[(task, vessel key, id)] from journal_cfg_v2.JOURNAL_ENV_IDS (no simulator import)."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import eval_protocol
    plans = eval_protocol.journal_plans()
    rows = []
    for (task, key), task_id in plans.JOURNAL_ENV_IDS.items():
        label = "wamv" if key.startswith("wamv") else "blueboat"
        if vessel in ("both", label):
            rows.append((task, key, task_id))
    return rows


def cfg_path_for(task, choice):
    if choice != "auto":
        return choice
    return "hydra" if task in HYDRA_TASKS else "parse"


def worker_command(args, task, task_id, cfg_path):
    cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--task", task_id,
           "--cfg-path", cfg_path, "--steps", str(args.steps), "--num_envs", "2", "--episodes", "2",
           "--level", "0", "--eval-seed", "42", "--vessel", "wamv" if "WAMV" in task_id else "blueboat"]
    if args.headless:
        cmd.append("--headless")
    if args.device:
        cmd += ["--device", args.device]
    if args.release_descriptor:
        cmd += ["--release-descriptor", args.release_descriptor]
    for item in args.extra_sets:
        cmd += ["--set", item]
    if args.d7_pool_dir and task == "docking":
        cmd += ["--set", "d7_pool_dir=" + args.d7_pool_dir]
    if args.suite:
        cmd += ["--suite"] + list(args.suite)
    return cmd


def _kill_tree(proc):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        proc.kill()


def run_one(args, task, task_id, log_dir, command_for=None):
    """Run one worker; returns a result dict (never raises for worker failures)."""
    cfg_path = cfg_path_for(task, args.cfg_path)
    cmd = (command_for or worker_command)(args, task, task_id, cfg_path)
    env = dict(os.environ)
    if args.d7_pool_dir:
        env["USVBENCH_D7_POOL_DIR"] = args.d7_pool_dir
    log = Path(log_dir) / (task_id + ".log")
    start = time.monotonic()
    timed_out = False
    with open(log, "w", encoding="utf-8", errors="replace") as stream:
        stream.write("COMMAND: " + subprocess.list2cmdline(cmd) + "\n\n")
        stream.flush()
        # cwd = the log folder: hydra.main (classical_baseline path) writes its outputs/ there,
        # never into the release tree; imports, assets and pools do not depend on the cwd.
        proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env, cwd=str(log_dir))
        try:
            code = proc.wait(timeout=args.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc)
            code = proc.wait()
    seconds = time.monotonic() - start
    text = log.read_text(encoding="utf-8", errors="replace")
    reported = None
    for line in text.splitlines():
        if line.startswith(RESULT_TAG):
            try:
                reported = json.loads(line[len(RESULT_TAG):])
            except ValueError:
                reported = None
    if timed_out:
        ok, error = False, "TIMEOUT after %d s (--timeout-s, killed)" % args.timeout_s
    elif reported is None:
        ok, error = False, "worker exit %d without a result line (see %s): %s" % (code, log, _last_error(text))
    else:
        ok = bool(reported.get("ok")) and code == 0
        error = reported.get("error") or ("worker exit %d" % code if code != 0 else "")
    return {"task_id": task_id, "task": task, "cfg_path": cfg_path, "ok": ok, "exit_code": code,
            "seconds": round(seconds, 1), "error": error, "stage": (reported or {}).get("stage"),
            "steps": (reported or {}).get("steps"), "mppi_mode": (reported or {}).get("mppi_mode"),
            "log": str(log)}


def _last_error(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if "Error" in line or "Exception" in line or "Traceback" in line:
            return line[:400]
    return lines[-1][:400] if lines else "no output"


def _ascii(text):
    return str(text).encode("ascii", "replace").decode("ascii")


def parent_main(args):
    rows = journal_ids(args.vessel)
    if args.ids:
        known = {task_id: (task, key) for task, key, task_id in rows}
        unknown = [i for i in args.ids if i not in known]
        if unknown:
            raise SystemExit("gpu_build_check: not journal ids of vessel %r: %s" % (args.vessel, ", ".join(unknown)))
        rows = [(known[i][0], known[i][1], i) for i in args.ids]
    log_dir = Path(args.log_dir or (Path.cwd() / ("gpu_build_check_%s" % time.strftime("%Y%m%d_%H%M%S"))))
    log_dir.mkdir(parents=True, exist_ok=True)
    print("gpu_build_check: %d id(s), one process per id, timeout %d s each (--timeout-s), %d zero-action steps, "
          "logs %s" % (len(rows), args.timeout_s, args.steps, log_dir), flush=True)
    results = []
    for task, _key, task_id in rows:
        result = run_one(args, task, task_id, log_dir)
        results.append(result)
        if result["ok"]:
            print("PASS %s [%s] %.1f s mppi_mode=%s" % (task_id, result["cfg_path"], result["seconds"],
                                                       result["mppi_mode"]), flush=True)
        else:
            print("FAIL %s [%s] %.1f s stage=%s: %s" % (task_id, result["cfg_path"], result["seconds"],
                                                        result["stage"], _ascii(result["error"])), flush=True)
    passed = sum(r["ok"] for r in results)
    (log_dir / "summary.json").write_text(json.dumps(results, indent=1) + "\n", encoding="ascii")
    print("SUMMARY: %d/%d PASS, %d FAIL (timeout %d s per id; per-id logs and summary.json in %s)"
          % (passed, len(results), len(results) - passed, args.timeout_s, log_dir), flush=True)
    return 0 if passed == len(results) and results else 1


# ------------------------------------------------------------------ worker (one id, Isaac)
def _report(stage, ok, error="", steps=0, mppi_mode=None):
    print(RESULT_TAG + json.dumps({"stage": stage, "ok": ok, "error": _ascii(error)[:2000], "steps": steps,
                                   "mppi_mode": mppi_mode}), flush=True)


def worker_main(argv):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import evaluator_exit
    from eval_protocol import add_protocol_arguments, preflight_protocol, register_journal_tasks, resolve_protocol
    from isaaclab.app import AppLauncher

    guard = evaluator_exit.Guard()
    state = {"stage": "arguments", "steps": 0}

    def body():
        parser = argparse.ArgumentParser(description="gpu_build_check worker (one journal id)")
        parser.add_argument("--worker", action="store_true")
        parser.add_argument("--task", required=True)
        parser.add_argument("--cfg-path", choices=("parse", "hydra"), required=True)
        parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
        parser.add_argument("--episodes", type=int, default=2)
        parser.add_argument("--level", type=int, default=0)
        parser.add_argument("--eval-seed", type=int, default=42)
        add_protocol_arguments(parser)
        AppLauncher.add_app_launcher_args(parser)
        args, hydra_args = parser.parse_known_args(argv)
        state["stage"] = "preflight"
        preflight_protocol(args, [str(Path(__file__).resolve())] + list(argv), parser)
        if getattr(args, "journal_task_info", None) is None:
            raise ValueError("gpu_build_check checks journal ids only, got %r" % args.task)
        import hydra_outputs
        # the hydra path keeps Hydra's run dir / logs in this worker's log folder (its cwd)
        sys.argv = [sys.argv[0]] + hydra_args + hydra_outputs.run_dir_overrides(
            os.path.join(os.getcwd(), "worker"), hydra_args)
        state["stage"] = "app"
        app = AppLauncher(args).app
        guard.attach_app(app)

        import gymnasium as gym
        import torch
        import isaaclab_tasks  # noqa: F401
        register_journal_tasks()
        task = args.task
        state["stage"] = "cfg"
        if args.cfg_path == "hydra":
            from isaaclab_tasks.utils.hydra import hydra_task_config
            # hydra re-raises the original exception (instead of printing it and calling
            # sys.exit(1)), so the FAIL line carries the exception text.
            os.environ["HYDRA_FULL_ERROR"] = "1"
            holder = {}

            @hydra_task_config(task, "skrl_cfg_entry_point")
            def capture(env_cfg, agent_cfg):
                holder["cfg"] = env_cfg
            capture()
            env_cfg = holder["cfg"]
        else:
            from isaaclab_tasks.utils import parse_env_cfg
            env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=args.num_envs)
        resolve_protocol(env_cfg, args)
        state["stage"] = "env_build"
        env = gym.make(task, cfg=env_cfg, render_mode=None)
        state["stage"] = "reset"
        env.reset()
        base = env.unwrapped
        # The evaluator interface (P5: eval_mppi refused the journal docking env after a clean
        # build): eval_mppi's task-family probe must accept every journal env.
        state["stage"] = "interface"
        import eval_mppi
        state["mppi_mode"] = eval_mppi._detect_mode(base)
        # G6b: the REAL certificate writer's constructor (journal_header path: D16 / D19 / D22 / D23 /
        # D24 preflight checks) on the built env, as every evaluator does; no file is written.
        state["stage"] = "certificate"
        from cert_v3 import CertificateRun
        CertificateRun(base=base, env_cfg=env_cfg, evaluator=str(Path(__file__).resolve()), task=task,
                       controller="gpu_build_check", controller_config={}, checkpoint=None,
                       eval_seed=args.eval_seed, level=args.level, episodes_requested=args.episodes,
                       overrides={}, argv=[str(Path(__file__).resolve())] + list(argv), cli_args=vars(args))
        state["stage"] = "step"
        actions = torch.zeros(tuple(base.action_space.shape), device=base.device)
        for _ in range(int(args.steps)):
            with torch.inference_mode():
                env.step(actions)
            state["steps"] += 1
        state["stage"] = "close"
        env.close()
        state["stage"] = "done"
        _report("done", True, steps=state["steps"], mppi_mode=state.get("mppi_mode"))

    try:
        body()
        code = 0
    except SystemExit as exc:
        code = evaluator_exit.exit_code(exc.code)
        if code:
            _report(state["stage"], False, "SystemExit(%r)" % (exc.code,), state["steps"], state.get("mppi_mode"))
    except BaseException as exc:  # noqa: BLE001 - every failure is a FAIL row
        import traceback
        traceback.print_exc()
        code = 1
        _report(state["stage"], False, "%s: %s" % (type(exc).__name__, exc), state["steps"], state.get("mppi_mode"))
    return guard.finish(code)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ids", nargs="+", default=None, help="journal ids (default: all of --vessel)")
    parser.add_argument("--vessel", choices=("blueboat", "wamv", "both"), default="both")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="zero-action control steps after reset")
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S, help="named per-id timeout")
    parser.add_argument("--cfg-path", choices=("auto", "parse", "hydra"), default="auto")
    parser.add_argument("--suite", nargs="+", default=[], metavar="TOKEN")
    parser.add_argument("--set", dest="extra_sets", action="append", default=[], metavar="FIELD=VALUE")
    parser.add_argument("--d7-pool-dir", default=None)
    parser.add_argument("--release-descriptor", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker" in argv:
        return worker_main(argv)
    args = build_parser().parse_args(argv)
    if args.steps < 1 or args.timeout_s < 1:
        raise SystemExit("gpu_build_check: --steps and --timeout-s must be positive")
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
