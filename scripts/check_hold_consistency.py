"""Audit a station-keeping certificate against the invariant that defines it.

The success criterion is not an opinion about the record -- it IS a threshold
on a quantity the environment already tracks:

    tasks/station_keeping/station_keeping_env.py
        self._success.copy_(self._max_hold_steps >= self.required_hold_steps)

So once ``max_hold_s`` is exported alongside ``success``, the certificate
carries its own proof: for every episode,

    success  ==  (max_hold_s >= required_hold_time_s)

must hold exactly. A violation means the two fields were latched at different
moments, or the hold counter was reset before it was read, or the threshold
drifted -- all of which are silent corruptions that no success rate could
reveal.

That makes this the strongest available check on the max_hold_s patch: it does
not compare against a remembered number, it compares the record against its own
definition.

The tolerance is half a control step (1/120 s), because max_hold_s is an
integer step count multiplied by the control period, so the only legitimate
disagreement is float representation right at the boundary: the env latches
``episode_max_hold_s = _max_hold_steps.float() * control_step_s`` in float32,
which puts the 3600-step threshold at 60.0000038 s rather than exactly 60.0.
Half a step clears that by three orders of magnitude and still stays strictly
inside one step, so a record a whole step on the wrong side of the threshold
(3599 steps = 59.9833 s) is reported rather than absorbed.

Usage:
    python scripts/check_hold_consistency.py cert.json [more.json ...]
    python scripts/check_hold_consistency.py cert.json --hold-seconds 60.0

Exit 0 if every certificate is self-consistent, 1 otherwise.
Pure standard library.
"""

import argparse
import glob
import json
import os
import statistics
import sys


DEFAULT_HOLD_S = 60.0
DEFAULT_TOL_S = 1.0 / 120.0


def audit(path, hold_seconds, tol):
    name = os.path.splitext(os.path.basename(path))[0]
    # A glob over a results directory also matches files that are not
    # certificates (gcerts/usv10k_scores.json is a bare list) and a run killed
    # mid-write leaves truncated JSON. Neither is a reason to abandon the rest
    # of the sweep, but neither may pass unnoticed either: they come back as a
    # named status carrying one violation, so the gate fails on them.
    try:
        with open(path, "r", encoding="utf-8") as handle:
            blob = json.load(handle)
    except (OSError, ValueError) as exc:
        return name, "UNREADABLE", 0, 0, [(None, None, None, None, str(exc))]
    raw = blob.get("records") or [] if isinstance(blob, dict) else None
    if not isinstance(raw, list):
        return name, "NOT-A-CERT", 0, 0, [
            (None, None, None, None,
             "top-level JSON is not an object carrying a 'records' list")]

    records = [r for r in raw if isinstance(r, dict)]
    if not raw:
        return name, "EMPTY", 0, 0, []

    have = [r for r in records if r.get("max_hold_s") is not None]
    if not have:
        return name, "NO-FIELD", len(raw), 0, []

    violations = []
    if len(records) != len(raw):
        violations.append((None, None, None, None,
                           f"{len(raw) - len(records)} entr(ies) in 'records' "
                           f"are not objects and were not audited"))
    for r in records:
        held = r.get("max_hold_s")
        if held is None:
            violations.append((r.get("env"), r.get("ep"), None, r.get("success"),
                               "missing max_hold_s"))
            continue
        try:
            held = float(held)
        except (TypeError, ValueError):
            violations.append((r.get("env"), r.get("ep"), None, r.get("success"),
                               f"max_hold_s is not a number: {held!r}"))
            continue
        implied = held >= hold_seconds - tol
        actual = bool(r.get("success"))
        if implied != actual:
            # Right at the boundary a half-step of float noise is legitimate;
            # anything further apart is a real inconsistency.
            if abs(held - hold_seconds) > tol:
                violations.append((r.get("env"), r.get("ep"), held, actual,
                                   f"success={actual} but max_hold_s={held:.4f}"))
    return name, "CHECKED", len(raw), len(have), violations


def describe(path):
    # Same tolerance for junk as audit(): the distribution table is a summary
    # of whatever is readable, and a file it cannot parse has already been
    # named and failed by the gate above.
    try:
        with open(path, "r", encoding="utf-8") as handle:
            blob = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(blob, dict):
        return None
    values = []
    for r in blob.get("records") or []:
        if not isinstance(r, dict) or r.get("max_hold_s") is None:
            continue
        try:
            values.append(float(r["max_hold_s"]))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    return {
        "n": n,
        "min": ordered[0],
        "p25": ordered[n // 4],
        "median": statistics.median(ordered),
        "p75": ordered[(3 * n) // 4],
        "max": ordered[-1],
        "at_zero": sum(1 for v in values if v <= 0.0),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("patterns", nargs="+")
    parser.add_argument("--hold-seconds", type=float, default=DEFAULT_HOLD_S,
                        help="required_hold_time_s of the task (default 60.0)")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL_S,
                        help="boundary tolerance in seconds (default half a "
                             "control step at 60 Hz)")
    args = parser.parse_args(argv)

    paths = []
    for pattern in args.patterns:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        print("no files matched", file=sys.stderr)
        return 2

    # max_hold_s is an integer step count times the control period, so the
    # values are quantised to multiples of one control step (1/60 s at the
    # certified sim.dt=1/120 with decimation=2). The default tolerance is half
    # of that: wide enough for the float32 latch in the env
    # (_max_hold_steps.float() * control_step_s puts the 3600-step boundary at
    # 60.0000038 s), narrow enough that a record one whole step on the wrong
    # side of the threshold, 0.0167 s away, is still caught. A tolerance of a
    # full step or more can no longer tell those two apart.
    if args.tol >= 2.0 * DEFAULT_TOL_S:
        print(f"WARNING: --tol {args.tol:g}s is a full control step or wider, "
              f"so a genuine one-step disagreement can no longer be "
              f"distinguished from float noise at the boundary")
        print()

    bad = 0
    print(f"{'cert':<34}{'status':>10}{'n':>6}{'with field':>12}{'violations':>12}")
    print("-" * 74)
    for path in paths:
        name, status, n, have, violations = audit(path, args.hold_seconds, args.tol)
        if violations or status in ("NO-FIELD", "EMPTY"):
            bad += 1
        print(f"{name:<34}{status:>10}{n:>6}{have:>12}{len(violations):>12}")
        for env, ep, held, success, why in violations[:5]:
            print(f"    env={env} ep={ep}: {why}")

    print()
    print("max_hold_s distribution (seconds; the graded quantity the binary "
          "criterion thresholds at "
          f"{args.hold_seconds:g})")
    print(f"{'cert':<34}{'n':>6}{'min':>8}{'p25':>8}{'median':>8}{'p75':>8}"
          f"{'max':>8}{'at 0':>7}")
    print("-" * 87)
    described = 0
    for path in paths:
        stats = describe(path)
        if not stats:
            continue
        described += 1
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"{name:<34}{stats['n']:>6}{stats['min']:>8.2f}{stats['p25']:>8.2f}"
              f"{stats['median']:>8.2f}{stats['p75']:>8.2f}{stats['max']:>8.2f}"
              f"{stats['at_zero']:>7}")
    if described == 0:
        # An empty table under a header reads like "nothing to report" when it
        # actually means the graded quantity is missing from every file.
        print(f"(none of the {len(paths)} matched file(s) carries max_hold_s)")

    print()
    if bad == 0:
        print("CONSISTENCY GATE=PASS -- every certificate proves its own "
              "success column from max_hold_s")
    else:
        print(f"CONSISTENCY GATE=FAIL -- {bad} certificate(s) inconsistent or "
              f"missing the field")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
