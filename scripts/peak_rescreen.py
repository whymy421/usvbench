"""Peak-guided dense re-screen: find each run's real champion cheaply.

The coarse every-Nth ladder can step over narrow peaks (observed: a
path_hazard run peaked at step 22400 and was back to zero by 38400, so a
5-point ladder reported 1.6% instead of ~31%). Reading the training-time
success curve costs no GPU, so use it to locate candidate peaks and screen
only the checkpoints near them.

Usage:
    python scripts/peak_rescreen.py --run-dir <dir> --task <gym id> [--top 3]
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import subprocess


def read_tb_scalars(run_dir: str, tag: str) -> list[tuple[int, float]]:
    """Minimal TFEvent reader: (step, value) for one scalar tag, no TB dep."""
    points: list[tuple[int, float]] = []
    for name in sorted(os.listdir(run_dir)):
        if "tfevents" not in name:
            continue
        with open(os.path.join(run_dir, name), "rb") as handle:
            data = handle.read()
        offset = 0
        while offset + 12 <= len(data):
            length = struct.unpack("<Q", data[offset : offset + 8])[0]
            body = data[offset + 12 : offset + 12 + length]
            offset += 12 + length + 4
            if len(body) < length:
                break
            # Scan the serialized Event for "<tag>" followed by a float value.
            index = body.find(tag.encode())
            if index < 0:
                continue
            # step is a varint in field 2; parse leniently by searching the
            # simple_value float that follows the tag string.
            float_index = body.find(b"\x15", index)  # field 3, 32-bit
            if float_index < 0 or float_index + 5 > len(body):
                continue
            value = struct.unpack("<f", body[float_index + 1 : float_index + 5])[0]
            step = 0
            pos = 1
            shift = 0
            while pos < len(body) and pos < 16:
                byte = body[pos]
                if body[pos - 1] == 0x18:  # field 3 varint marker for step
                    step = 0
                    shift = 0
                    cursor = pos
                    while cursor < len(body):
                        chunk = body[cursor]
                        step |= (chunk & 0x7F) << shift
                        cursor += 1
                        if not chunk & 0x80:
                            break
                        shift += 7
                    break
                pos += 1
            points.append((step, value))
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--top", type=int, default=3, help="peak regions to probe")
    parser.add_argument("--width", type=int, default=2, help="checkpoints each side")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--repo", default="/root/autodl-tmp/usvbench")
    parser.add_argument("--python", default="/root/autodl-tmp/envs/isaaclab/bin/python")
    args = parser.parse_args()

    points = read_tb_scalars(args.run_dir, "Info / Episode/success")
    ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    steps = sorted(
        int(re.findall(r"\d+", f)[0])
        for f in os.listdir(ckpt_dir)
        if re.fullmatch(r"agent_\d+\.pt", f)
    )
    if not steps:
        print("no checkpoints")
        return

    if points:
        ranked = sorted(points, key=lambda p: -p[1])[: args.top]
        anchors = [step for step, _ in ranked]
        print(f"training-curve peaks at steps {anchors} "
              f"(values {[round(v, 3) for _, v in ranked]})")
    else:
        anchors = [steps[len(steps) // 3], steps[len(steps) // 2]]
        print(f"no curve found; probing {anchors}")

    wanted: set[int] = set()
    for anchor in anchors:
        nearest = min(range(len(steps)), key=lambda i: abs(steps[i] - anchor))
        low = max(0, nearest - args.width)
        high = min(len(steps), nearest + args.width + 1)
        wanted.update(steps[low:high])

    only = ",".join(str(s) for s in sorted(wanted))
    print(f"dense screen over {len(wanted)} checkpoints: {only}")
    subprocess.run(
        [args.python, f"{args.repo}/scripts/screen_v6_ladder.py",
         "--run-dir", args.run_dir, "--task", args.task, "--only", only,
         "--level", str(args.level), "--eval-seed", "42", "--headless"],
        check=False,
    )


if __name__ == "__main__":
    main()
