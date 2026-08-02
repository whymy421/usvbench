"""Are two "independent" certification eval seeds actually independent?

Certification reports two eval seeds so that agreement between them means
something. In docking it did not: layouts were drawn from the global torch RNG
(seeded by the trainer, not by cfg), so --eval-seed 42 and --eval-seed 123
produced byte-identical episode streams and the apparent 70.3%/70.3% agreement
was not robustness evidence at all -- it was the same 128 episodes twice.

That defect was invisible in the reported numbers. It is only visible by
comparing the per-episode records, which is what this does. Run it on every
certification pair before quoting the agreement as evidence.

    python check_eval_seed_independence.py cert_x_e42.json cert_x_e123.json
"""

from __future__ import annotations

import argparse
import json
import sys

# Per-episode fields that are a fingerprint of the LAYOUT rather than of the
# policy's luck: two different layout streams cannot agree on these.
FINGERPRINT = ("path_length_m", "min_clearance_m", "tts_s")
TOLERANCE = 1e-6


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first")
    parser.add_argument("second")
    args = parser.parse_args()

    a, b = load(args.first), load(args.second)
    for key in ("task", "level", "checkpoint"):
        if a.get(key) != b.get(key):
            print(f"!! {key} differs: {a.get(key)} vs {b.get(key)}")
            print("   These are not the same experiment; comparison is meaningless.")
            sys.exit(2)
    if a.get("seed") == b.get("seed"):
        print(f"!! both files record eval seed {a.get('seed')} -- not a pair")
        sys.exit(2)

    ra, rb = a["records"], b["records"]
    n = min(len(ra), len(rb))
    print(f"task={a['task']} level={a['level']} seeds={a['seed']} vs {b['seed']}")
    print(f"episodes compared: {n}")

    identical = 0
    for x, y in zip(ra[:n], rb[:n]):
        same = True
        for field in FINGERPRINT:
            u, v = x.get(field), y.get(field)
            if u is None or v is None:
                same = same and (u == v)
            elif abs(float(u) - float(v)) > TOLERANCE:
                same = False
        identical += bool(same)

    sr_a = sum(1 for r in ra[:n] if r["success"]) / max(n, 1)
    sr_b = sum(1 for r in rb[:n] if r["success"]) / max(n, 1)
    print(f"SR: {sr_a:.4f} vs {sr_b:.4f}")
    print(f"episodes with an identical layout fingerprint: {identical}/{n}")
    print()

    if identical == n:
        print("=> FAIL: the two seeds ran THE SAME episodes. Their agreement is")
        print("   arithmetic, not evidence. Layouts are not drawn from a")
        print("   cfg.seed-driven generator -- fix that before certifying.")
        sys.exit(1)
    if identical > 0.05 * n:
        print(f"=> SUSPICIOUS: {identical / n:.1%} of episodes coincide. Expected")
        print("   near zero for genuinely independent layout streams.")
        sys.exit(1)
    print("=> PASS: the streams are independent, so agreement between the two")
    print("   numbers is real robustness evidence.")


if __name__ == "__main__":
    main()
