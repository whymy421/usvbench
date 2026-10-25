"""G44 R1c (v2.6.3): best checkpoint by periodic evaluation, beside the final checkpoint.

Every periodic checkpoint of a training run (agent_<t>.pt, 10 % of the budget, listed with its SHA-256 in the run's
training_record.json) is evaluated by the protocol's own evaluator (scripts/eval_v6_frozen.py, calm, 64 x 1; the
launcher runs those evaluations while the training is still running). This tool reads the certificates and keeps
the best checkpoint by evaluation:

  python -B scripts/ppo_checkpoint_eval.py select --run-dir RUN --cert T=CERT.json [--cert T=...] --out-dir NEW_DIR

Rules (ppo_training_fixes.select_best / BEST_RULE): every certificate must be complete and name the SHA-256 of the
run's checkpoint at that timestep; all certificates have the same number of episodes and the same eval seed; best =
highest success count, tie -> the later checkpoint. v2.6.4 (ppo_final_g46.VALIDATION_RULE, owner decisions
2026-10-13 / 10-14): every certificate's header.seed_namespace must be a validation block of the run's purpose
(training record 'purpose'): campaign runs ('training') the selection namespace block selection-02 (seed 11002)
ONLY, and EVERY periodic checkpoint of the record must be evaluated; training-smoke runs selection-02 or the
diagnostic seed 42; a seed of the final_test / pilot / calibration namespaces is always refused (test seeds never
select). best_eval.json also reports the final checkpoint's own evaluation. The record is never changed; NEW_DIR (refused if it exists)
receives best_eval.pt (a byte copy of the chosen checkpoint) and best_eval.json. Which of final / best is the
headline is an owner decision; the evaluation protocol evaluates both. Exit 0 = selected, 2 = refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import ppo_training_fixes as FX  # noqa: E402
import ppo_final_g46 as G46  # noqa: E402

SCHEMA = "usvbench.ppo_best_checkpoint_g44/1"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def select(run_dir, certs, out_dir):
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    if out_dir.exists():
        raise FX.FixError("output directory %s exists (never overwrite)" % out_dir)
    record = json.loads((run_dir / "training_record.json").read_text(encoding="ascii"))
    by_step = {int(c["timestep"]): c for c in record["checkpoints"]}
    purpose = record.get("purpose")
    if purpose not in G46.VALIDATION_BLOCKS:
        raise FX.FixError("no validation rule for the run's purpose %r" % (purpose,))
    if purpose == "training" and sorted(int(s) for s in certs) != sorted(by_step):
        raise FX.FixError("a campaign run selects over EVERY periodic checkpoint: missing %s, extra %s"
                          % (sorted(set(by_step) - set(int(s) for s in certs)),
                             sorted(set(int(s) for s in certs) - set(by_step))))
    evaluations, seeds = [], set()
    for step, cert_path in sorted(certs.items()):
        if step not in by_step:
            raise FX.FixError("timestep %d is not a checkpoint of the run" % step)
        document = json.loads(Path(cert_path).read_text(encoding="utf-8"))
        k, n, sha = FX.certificate_outcome(document)
        if sha != by_step[step]["sha256"].lower():
            raise FX.FixError("certificate %s evaluated %s, the run's checkpoint %d is %s"
                              % (cert_path, sha, step, by_step[step]["sha256"]))
        if sha256(run_dir / by_step[step]["path"]) != by_step[step]["sha256"]:
            raise FX.FixError("checkpoint file %s does not match the record" % by_step[step]["path"])
        header = document.get("header") or {}
        validation_block(header, purpose, cert_path)                       # v2.6.4 validation-seed rule
        seeds.add(json.dumps(header.get("eval_seed", header.get("seed")), sort_keys=True))
        evaluations.append({"timestep": step, "successes": k, "episodes": n, "sha256": sha,
                            "certificate": str(cert_path), "certificate_sha256": sha256(cert_path)})
    if len(seeds) > 1:
        raise FX.FixError("certificates use different eval seeds: %s" % sorted(seeds))
    best = FX.select_best(evaluations)
    final = record.get("final_checkpoint") or {}
    final_eval = [e for e in evaluations if final.get("source_sha256") and e["sha256"] == final["source_sha256"]]
    out_dir.mkdir(parents=True)
    source = run_dir / by_step[best["timestep"]]["path"]
    shutil.copyfile(source, out_dir / "best_eval.pt")
    doc = {"schema": SCHEMA, "rule": FX.BEST_RULE, "run_dir": str(run_dir.resolve()),
           "training_record_sha256": sha256(run_dir / "training_record.json"),
           "evaluations": evaluations, "best": best, "best_copy": "best_eval.pt",
           "best_copy_sha256": sha256(out_dir / "best_eval.pt"),
           "final_checkpoint_sha256": final.get("sha256"),
           "best_is_final": bool(final.get("sha256") and final["sha256"] == best["sha256"]),
           "final_evaluation": final_eval[0] if final_eval else None,
           "purpose": purpose, "validation_rule": G46.VALIDATION_RULE,
           "headline": "owner decision (final or best); both are evaluated"}
    (out_dir / "best_eval.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="ascii")
    return doc


def validation_block(header, purpose, cert_path):
    """v2.6.4: the certificate's seed stamp must be a validation block of the run's purpose (G46.VALIDATION_BLOCKS);
    a final_test / pilot / calibration seed is always refused."""
    stamp = header.get("seed_namespace")
    if not isinstance(stamp, dict):
        raise FX.FixError("certificate %s carries no header.seed_namespace stamp" % cert_path)
    if stamp.get("purpose") in G46.FORBIDDEN_SELECTION_PURPOSES or stamp.get("also_registered_as") in (
            "final_test", "dose_pilots", "crossmachine_calibration"):
        raise FX.FixError("certificate %s was evaluated on a %s seed (%s): test / pilot / calibration seeds never "
                          "select a checkpoint" % (cert_path, stamp.get("purpose"), stamp.get("seed")))
    got = {"namespace": stamp.get("namespace"), "block_id": stamp.get("block_id"), "seed": stamp.get("seed")}
    if got not in [dict(b) for b in G46.VALIDATION_BLOCKS[purpose]]:
        raise FX.FixError("certificate %s seed %s is not a validation block of a %s run (allowed: %s)"
                          % (cert_path, got, purpose, list(G46.VALIDATION_BLOCKS[purpose])))
    return got


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--cert", action="append", default=[], help="TIMESTEP=certificate.json (repeat)")
    s.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)
    try:
        certs = {}
        for item in args.cert:
            step, _, path = item.partition("=")
            certs[int(step)] = path
        doc = select(args.run_dir, certs, args.out_dir)
    except (FX.FixError, ValueError, KeyError, OSError) as exc:
        print("ppo_checkpoint_eval: REFUSED: %s" % exc, file=sys.stderr)
        return 2
    print("ppo_checkpoint_eval: BEST timestep %d (%d/%d), final %s, best_is_final=%s"
          % (doc["best"]["timestep"], doc["best"]["successes"], doc["best"]["episodes"],
             (doc["final_checkpoint_sha256"] or "?")[:12], doc["best_is_final"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
