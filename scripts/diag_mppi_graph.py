"""G9 GPU equivalence diagnostic: the eager MPPI rollout and its CUDA-graph replay side by side.

One journal id, vessel and num_envs per process (Isaac Lab's DirectRLEnv owns the process-wide
SimulationContext; scripts/gpu_build_check.py runs one process per id for the same reason).
The env and the controller are built EXACTLY as scripts/eval_mppi.py builds them
(eval_protocol.preflight_protocol with the vessel-specific journal id, register_journal_tasks,
parse_env_cfg, resolve_protocol, eval_mppi._detect_mode, eval_mppi.build_mppi_setup); a second
MPPIController with the same parameters and the same dynamics object is wrapped in the
RELEASE's mppi_controller.CudaGraphMPPI (the module must be loaded from this scripts folder).
Every control step reads the bridge once (eval_mppi.rollout_inputs), calls the eager controller
and the graph on the same inputs, compares actions, info tensors, warm starts and noise keys,
and steps the env with the EAGER action; finished envs reset both controllers.

Writes no certificate. The JSON report goes to --out (a new file outside the release root).
Exit codes: 0 every step equal within --tolerance (default 1e-6), keys equal, all finite;
3 a difference above the tolerance, a key mismatch or a non-finite action (report written);
4 the loaded modules differ from the release descriptor (report written); 2 refused arguments
(before the app starts); 1 any other failure, a failed capture included (traceback, no report,
the app is not closed; scripts/evaluator_exit.py G6b rules).

Run on the GPU host from the deployed release (scripts/diag_mppi_graph.ps1 runs a grid with
named timeouts):
  python -B <release>/scripts/diag_mppi_graph.py --task journal:path_hazard --vessel blueboat
      --num_envs 16 --steps 300 --headless --release-descriptor <descriptor> --out <dir>/ph_bb_n16.json
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

SCRIPTS_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import evaluator_exit  # noqa: E402
import eval_mppi  # noqa: E402  (pure Python at import time: cfg_override, evaluator_exit)

DIAG_VERSION = "g9-mppi-graph-diag-v1"
RESULT_TAG = "G9-DIAG-RESULT "
DEFAULT_TOLERANCE = 1.0e-6
EXIT_OK = 0
EXIT_DIFFERENCE = 3
EXIT_PROVENANCE = 4
_GUARD = evaluator_exit.Guard()


class DiagError(ValueError):
    """Refused diagnostic arguments (exit 2 through parser.error)."""


def build_parser():
    """eval_mppi's own parser (same MPPI defaults) plus the diagnostic arguments."""
    parser = eval_mppi._build_parser()
    parser.description = "G9: eager MPPI vs its CUDA-graph replay on the same states (no certificate)."
    parser.set_defaults(episodes=None)
    parser.add_argument("--steps", type=int, default=300, help="control steps compared (default 300)")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help="largest accepted absolute difference (default 1e-6)")
    parser.add_argument("--first-k", type=int, default=5,
                        help="first steps left out of the steady timing medians (step 0 holds the capture)")
    return parser


def validate_diag_args(args, release_root=RELEASE_ROOT):
    """Diagnostic-specific refusals (raise DiagError)."""
    if args.steps < 1:
        raise DiagError("--steps must be >= 1")
    if not (math.isfinite(args.tolerance) and args.tolerance > 0.0):
        raise DiagError("--tolerance must be finite and positive")
    if not 0 <= args.first_k < args.steps:
        raise DiagError("--first-k must lie in [0, --steps)")
    if getattr(args, "mppi_impl", "auto") != "auto":
        raise DiagError("the diagnostic always runs both implementations; do not pass --mppi-impl")
    if not args.out:
        raise DiagError("--out <report.json> is required")
    out = Path(os.path.realpath(args.out))
    root = Path(os.path.realpath(release_root))
    if out == root or root in out.parents:
        raise DiagError("--out must lie outside the release root %s" % root)
    if out.exists() or Path(str(out) + ".partial").exists():
        raise DiagError("--out %s already exists (the diagnostic never overwrites a report)" % out)
    if not out.parent.is_dir():
        raise DiagError("the folder of --out does not exist: %s" % out.parent)
    return out


def tensor_difference(torch, a, b):
    """{max, mean, bitwise} of |a - b| in float64; equal values (inf included) differ by 0."""
    a64, b64 = a.detach().double(), b.detach().double()
    if a64.shape != b64.shape:
        return {"max": math.inf, "mean": math.inf, "bitwise": False}
    same = (a64 == b64) | (torch.isnan(a64) & torch.isnan(b64))
    diff = torch.where(same, torch.zeros_like(a64), (a64 - b64).abs())
    diff = torch.where(torch.isnan(diff), torch.full_like(diff, math.inf), diff)
    count = diff.numel()
    return {"max": float(diff.max()) if count else 0.0, "mean": float(diff.mean()) if count else 0.0,
            "bitwise": bool(torch.equal(a, b) or bool(same.all()))}


def compare_step(torch, action_e, action_g, info_e, info_g, u_e, u_g):
    """One step: action, every info tensor and the warm start, eager vs graph."""
    action = tensor_difference(torch, action_e, action_g)
    info = {}
    for key in sorted(set(info_e) | set(info_g)):
        if key not in info_e or key not in info_g:
            info[key] = {"max": math.inf, "mean": math.inf, "bitwise": False}
        else:
            info[key] = tensor_difference(torch, info_e[key], info_g[key])
    warm = tensor_difference(torch, u_e, u_g)
    finite = bool(torch.isfinite(action_e).all()) and bool(torch.isfinite(action_g).all())
    return {"action_max": action["max"], "action_mean": action["mean"], "action_bitwise": action["bitwise"],
            "info_max": max([v["max"] for v in info.values()] or [0.0]),
            "info": {key: value["max"] for key, value in info.items()},
            "warm_start_max": warm["max"], "actions_finite": finite}


def median(values):
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else 0.5 * (values[mid - 1] + values[mid])


def percentile(values, q):
    """cert_v3.timing_stats rule: linear interpolation between order statistics."""
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    low = int(math.floor(position))
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def side_by_side(env, bridge, mode, eager, graph, steps, torch, sync, clock, read_inputs=None):
    """The measured loop: one bridge read, eager call, graph call (each synchronised and timed),
    env.step with the EAGER action, both controllers reset on the same done mask."""
    read_inputs = read_inputs or eval_mppi.rollout_inputs
    rows = []
    for step in range(steps):
        with torch.inference_mode():
            state, kwargs = read_inputs(bridge, mode)
            expected_keys = eager.noise_keys()
            sync()
            begin = clock()
            action_e, info_e = eager.rollout(state, **kwargs)
            sync()
            middle = clock()
            action_g, info_g = graph.rollout(state, **kwargs)
            sync()
            end = clock()
            row = compare_step(torch, action_e, action_g, info_e, info_g, eager.U, graph.U)
        row.update(step=step, eager_s=middle - begin, graph_s=end - middle,
                   keys_match=(list(graph.last_keys) == list(expected_keys)))
        before = clock()
        result = env.step(action_e)
        sync()
        row["env_s"] = clock() - before
        term, trunc = result[2], result[3]
        done = term | trunc
        done = done.squeeze(-1) if done.dim() > 1 else done
        with torch.inference_mode():
            eager.reset(done)
            graph.reset(done)
        row["resets"] = int(done.sum())
        rows.append(row)
    return rows


def summarize(rows, tolerance, first_k):
    """The report numbers and the verdict (EQUIVALENT only when every check passes)."""
    steady = [row for row in rows if row["step"] >= first_k]
    eager_median = median([row["eager_s"] for row in steady])
    graph_median = median([row["graph_s"] for row in steady])
    summary = {
        "steps": len(rows),
        "tolerance": tolerance,
        "max_abs_action_diff": max([row["action_max"] for row in rows] or [0.0]),
        "mean_abs_action_diff": (sum(row["action_mean"] for row in rows) / len(rows)) if rows else 0.0,
        "steps_over_tolerance": sum(1 for row in rows if not row["action_max"] <= tolerance),
        "bit_identical_steps": sum(1 for row in rows if row["action_bitwise"]),
        "key_mismatch_steps": sum(1 for row in rows if not row["keys_match"]),
        "max_abs_info_diff": max([row["info_max"] for row in rows] or [0.0]),
        "info_steps_over_tolerance": sum(1 for row in rows if not row["info_max"] <= tolerance),
        "max_abs_warm_start_diff": max([row["warm_start_max"] for row in rows] or [0.0]),
        "warm_start_steps_over_tolerance": sum(1 for row in rows if not row["warm_start_max"] <= tolerance),
        "nonfinite_action_steps": sum(1 for row in rows if not row["actions_finite"]),
        "env_resets": sum(row["resets"] for row in rows),
        "first_k": first_k,
        "eager_call_s": {"median": eager_median, "mean": (sum(r["eager_s"] for r in steady) / len(steady))
                         if steady else None, "p95": percentile([r["eager_s"] for r in steady], 0.95)},
        "graph_call_s": {"median": graph_median, "mean": (sum(r["graph_s"] for r in steady) / len(steady))
                         if steady else None, "p95": percentile([r["graph_s"] for r in steady], 0.95)},
        "graph_first_call_s": rows[0]["graph_s"] if rows else None,
        "env_step_s": {"median": median([row["env_s"] for row in steady])},
        "speedup_controller_call": (eager_median / graph_median) if eager_median and graph_median else None,
    }
    failures = [name for name in ("steps_over_tolerance", "key_mismatch_steps", "info_steps_over_tolerance",
                                  "warm_start_steps_over_tolerance", "nonfinite_action_steps") if summary[name]]
    summary["verdict"] = "EQUIVALENT" if not failures else "DIFFERENT"
    summary["failed_checks"] = failures
    return summary


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_report(out, report):
    """JSON to <out>.partial, flushed, re-read, renamed (never left half-written)."""
    out = Path(out)
    partial = Path(str(out) + ".partial")
    with open(partial, "w", encoding="ascii") as stream:
        json.dump(report, stream, indent=1, sort_keys=True, allow_nan=False, default=str)
        stream.flush()
        os.fsync(stream.fileno())
    json.loads(partial.read_text(encoding="ascii"))
    os.replace(partial, out)


def json_safe(value):
    """Replace non-finite floats (allow_nan=False) by strings."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def release_module_check(module, scripts_dir=SCRIPTS_DIR):
    """The graph implementation must be the release's own (loaded from this scripts folder)."""
    loaded = Path(os.path.realpath(module.__file__)).parent
    if loaded != Path(os.path.realpath(scripts_dir)):
        raise RuntimeError("mppi_controller was loaded from %s, not from the release scripts folder %s"
                           % (loaded, scripts_dir))
    for name in ("CudaGraphMPPI", "GraphCaptureError", "MPPI_IMPL_GRAPH"):
        if not hasattr(module, name):
            raise RuntimeError("the release mppi_controller has no %s (not a G9 release)" % name)
    return str(Path(module.__file__).resolve())


def _run_diag():
    from isaaclab.app import AppLauncher
    from eval_protocol import (add_protocol_arguments, preflight_protocol, register_journal_tasks,
                               resolve_protocol)

    original_argv = list(sys.argv)
    parser = build_parser()
    add_protocol_arguments(parser)
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    if args.episodes is None:
        args.episodes = args.num_envs
    preflight_protocol(args, original_argv, parser)
    if getattr(args, "journal_task_info", None) is None:
        parser.error("the G9 diagnostic runs journal ids only (journal:<task> with --vessel, or a journal id)")
    eval_mppi._validate_args(parser, args)
    try:
        out = validate_diag_args(args)
    except DiagError as exc:
        parser.error(str(exc))
    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(args).app
    _GUARD.attach_app(app)

    import torch
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401  (registers the deployed task ids)
    from isaaclab_tasks.utils import parse_env_cfg
    import code_tree
    import mppi_controller as mc

    module_file = release_module_check(mc)
    register_journal_tasks()
    task = args.task
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=args.num_envs)
    applied_overrides = resolve_protocol(env_cfg, args)
    env = gym.make(task, cfg=env_cfg, render_mode=None)
    base = env.unwrapped
    mode = eval_mppi._detect_mode(base)
    setup = eval_mppi.build_mppi_setup(args, env_cfg, base, mode, args.journal_task_info, torch,
                                       mc.MPPIController, mc.MPPIParams, mc.NOMINAL_BLUEBOAT,
                                       mc.NominalBlueBoatDynamics)
    eager = setup.controller
    graph = mc.CudaGraphMPPI(mc.MPPIController(num_envs=base.num_envs, device=base.device, params=setup.params,
                                               dynamics=setup.dynamics),
                             log=lambda message: print(message, flush=True))
    cuda = str(base.device).startswith("cuda")

    def sync():
        if cuda:
            torch.cuda.synchronize(base.device)

    env.reset()
    started = time.time()
    rows = side_by_side(env, setup.bridge, mode, eager, graph, args.steps, torch, sync, time.perf_counter)
    summary = summarize(rows, args.tolerance, args.first_k)
    provenance = {"release_descriptor": args.release_descriptor, "release_id": getattr(args, "release_id", None),
                  "verified": None, "mismatches": []}
    if args.release_descriptor:
        release = code_tree.load_release(args.release_descriptor)
        loaded = code_tree.verify_loaded_modules(release["records"], SCRIPTS_DIR, [RELEASE_ROOT / "tasks"],
                                                 sys.modules)
        provenance.update(verified=bool(loaded.get("verified")), mismatches=list(loaded.get("mismatches", []))[:20],
                          checked=loaded.get("checked"))
    report = {
        "diagnostic": DIAG_VERSION,
        "writes_certificate": False,
        "task_id": task, "journal": args.journal_task_info, "vessel": args.vessel, "mode": mode,
        "num_envs": int(base.num_envs), "device": str(base.device),
        "gpu_name": torch.cuda.get_device_name(base.device) if cuda else None,
        "torch_version": str(torch.__version__),
        "mppi": {**setup.params.as_dict(), "frozen_parameters": setup.frozen_parameters,
                 "model": setup.dynamics.params.get("model"), "vessel_model": getattr(setup.dynamics, "vessel", None)},
        "graph": graph.describe(),
        "implementation_stamp": graph.implementation_stamp(),
        "summary": summary,
        "rows": rows,
        "release_root": str(RELEASE_ROOT),
        "mppi_controller": {"path": module_file, "sha256": file_sha256(module_file)},
        "diagnostic_sha256": file_sha256(__file__),
        "provenance": provenance,
        "reference_environment": getattr(args, "reference_environment", None),
        "seed_registration": getattr(args, "seed_registration", None),
        "applied_overrides": applied_overrides,
        "argv": original_argv,
        "started_unix_s": started, "seconds": time.time() - started,
    }
    write_report(out, json_safe(report))
    line = {"ok": summary["verdict"] == "EQUIVALENT" and provenance["verified"] is not False,
            "verdict": summary["verdict"], "task": task, "vessel": args.vessel, "num_envs": int(base.num_envs),
            "steps": summary["steps"], "max_abs_action_diff": summary["max_abs_action_diff"],
            "mean_abs_action_diff": summary["mean_abs_action_diff"],
            "steps_over_tolerance": summary["steps_over_tolerance"],
            "key_mismatch_steps": summary["key_mismatch_steps"],
            "speedup_controller_call": summary["speedup_controller_call"],
            "captured_graphs": graph.captured_graphs, "recaptures": graph.recaptures,
            "provenance_verified": provenance["verified"], "report": str(out)}
    print(RESULT_TAG + json.dumps(json_safe(line), sort_keys=True), flush=True)
    sys.stdout.flush()
    if summary["verdict"] != "EQUIVALENT":
        raise SystemExit(EXIT_DIFFERENCE)
    if provenance["verified"] is False:
        raise SystemExit(EXIT_PROVENANCE)
    env.close()


if __name__ == "__main__":
    evaluator_exit.run(_run_diag, _GUARD)
