"""Reward-design bench: score hand-written scenarios BEFORE training.

Motivation (owner, 2026-07-26): a reward function should be checked the way a
unit test is written -- enumerate the behaviours we want and the behaviours we
fear, compute what the reward actually pays for each, and require that the
behaviour we most want scores highest. Every reward change must re-run this and
show the ranking is still what we intend.

Scenarios are described at the behavioural level (how long, how fast, how close
to obstacles, whether contact occurred) and scored with the SAME formulas the
environment uses, term by term, both undiscounted and discounted.

Run:
    python scripts/reward_scenarios.py            # current hazard-nav reward
    python scripts/reward_scenarios.py --band 1.10
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

# --- environment constants, mirrored from hazard_nav_env_cfg.py --------------
DT = 1.0 / 60.0                # control step (decimation 2 @ 120 Hz sim)
HORIZON_S = 120.0
PROGRESS_SCALE = 20.0          # total positive progress over a full run
PROX_SCALE = 4.1
PROX_FLOOR_M = 0.45            # half beam
SAFE_CLEARANCE_M = 0.90
HALF_BEAM_M = 0.45
CONTACT_ENTRY = 25.0
CONTACT_DWELL = 1.0            # per STEP, not per second
GOAL_ENTRY_BONUS = 50.0
GOAL_TIME_BONUS = 50.0
REVERSE_SCALE = 0.05
SWIFT_SCALE = 0.25
SPEED_SCALE_MPS = 2.0
GAMMA = 0.999
N_RAYS = 36


@dataclass
class Scenario:
    """A behaviour, described the way a human would describe it."""

    name: str
    duration_s: float                  # time until the episode ends for this behaviour
    reached_goal: bool
    progress_fraction: float = 1.0     # fraction of the start-to-goal potential covered
    close_rays: int = 0                # how many of the 36 rays sit at `close_range_m`
    close_range_m: float = 1.35        # their reading, in metres from hull COM
    close_seconds: float = 0.0         # for how long that ray picture holds
    contacts: int = 0                  # number of separate contact entries
    contact_seconds: float = 0.0       # total time in contact
    mean_speed_mps: float = 1.2
    reverse_fraction: float = 0.0      # fraction of time commanding negative surge
    note: str = ""
    terms: dict = field(default_factory=dict)


def prox_cost_per_step(close_rays: int, close_range_m: float, band_m: float) -> float:
    """Per-step proximity contribution to the REWARD (i.e. already negative).

    The env computes a positive `prox_cost` and subtracts it; here we return the
    signed reward contribution directly so scenarios can be summed.
    """
    r = min(max(close_range_m, PROX_FLOOR_M), band_m)
    per_close = math.log(r / band_m)  # negative inside the band
    return PROX_SCALE * DT * (close_rays * per_close) / N_RAYS


def discount_sum(steps: int) -> float:
    """Sum of gamma^t for t in [0, steps)."""
    if steps <= 0:
        return 0.0
    return (1.0 - GAMMA**steps) / (1.0 - GAMMA)


def score(s: Scenario, band_m: float) -> Scenario:
    steps = int(round(s.duration_s / DT))
    close_steps = int(round(s.close_seconds / DT))
    contact_steps = int(round(s.contact_seconds / DT))

    # 1. potential progress -- telescopes, so only the endpoints matter
    progress = PROGRESS_SCALE * s.progress_fraction

    # 2. proximity, paid only while the close-ray picture holds
    prox = prox_cost_per_step(s.close_rays, s.close_range_m, band_m) * close_steps

    # 3. contact ledger
    contact = -(CONTACT_ENTRY * s.contacts) - (CONTACT_DWELL * contact_steps)

    # 4. terminal bonus, only if the goal was reached with a clean prefix
    clean = s.reached_goal and s.contacts == 0
    remaining = max(0.0, 1.0 - s.duration_s / HORIZON_S)
    goal = (GOAL_ENTRY_BONUS + GOAL_TIME_BONUS * remaining) if clean else 0.0

    # 5. behaviour costs
    reverse = -REVERSE_SCALE * DT * steps * s.reverse_fraction  # relu(-surge)^2 ~ 1
    idle = -SWIFT_SCALE * DT * steps * (
        1.0 - min(s.mean_speed_mps / SPEED_SCALE_MPS, 1.0)
    )

    total = progress + prox + contact + goal + reverse + idle

    # Discounted view: progress and costs accrue along the way, the bonus lands
    # at the end. This is what the agent's value function actually optimises.
    disc_all = discount_sum(steps) / max(steps, 1)
    disc_progress = progress * disc_all
    disc_prox = prox * (discount_sum(close_steps) / max(close_steps, 1)) if close_steps else 0.0
    disc_contact = contact * (GAMMA ** (steps // 2))  # contact happens mid-run
    disc_goal = goal * (GAMMA**steps)
    disc_total = disc_progress + disc_prox + disc_contact + disc_goal + (reverse + idle) * disc_all

    s.terms = {
        "progress": progress,
        "proximity": prox,
        "contact": contact,
        "goal_bonus": goal,
        "reverse": reverse,
        "idle": idle,
        "TOTAL": total,
        "DISCOUNTED": disc_total,
    }
    return s


def build_scenarios() -> list[Scenario]:
    """The behaviours we want, and the ones we fear."""
    return [
        # --- what we WANT to be the best ------------------------------------
        Scenario(
            "A 理想:直穿缝心,快,零接触",
            duration_s=20.0, reached_goal=True,
            close_rays=6, close_range_m=1.10, close_seconds=6.0,
            mean_speed_mps=1.6,
            note="穿两个缝,每个约 3 秒,居中所以射线读数 1.10 m",
        ),
        Scenario(
            "B 稳妥:绕开障碍带,慢但干净",
            duration_s=55.0, reached_goal=True,
            close_rays=2, close_range_m=1.30, close_seconds=8.0,
            mean_speed_mps=1.0,
            note="从障碍带外侧绕行",
        ),
        Scenario(
            "C 贴缝:挤最窄的缝,很近但没碰",
            duration_s=26.0, reached_goal=True,
            close_rays=10, close_range_m=0.60, close_seconds=8.0,
            mean_speed_mps=1.3,
            note="四档缝宽,居中时两侧各 0.90 m -> 射线读 0.60 m",
        ),
        # --- what we FEAR ----------------------------------------------------
        Scenario(
            "D 蹭一下抄近路(碰一帧)",
            duration_s=15.0, reached_goal=True,
            close_rays=8, close_range_m=0.70, close_seconds=5.0,
            contacts=1, contact_seconds=DT,
            mean_speed_mps=1.9,
            note="比理想快 5 秒,但碰过 -> 按判据不算成功",
        ),
        Scenario(
            "E 长时间擦墙(接触 1 秒)",
            duration_s=18.0, reached_goal=True,
            close_rays=8, close_range_m=0.60, close_seconds=6.0,
            contacts=1, contact_seconds=1.0,
            mean_speed_mps=1.7,
        ),
        Scenario(
            "F 胆怯:贴场地边缘绕圈,超时",
            duration_s=120.0, reached_goal=False, progress_fraction=0.35,
            close_rays=0, close_seconds=0.0,
            mean_speed_mps=0.9,
            note="从不进入障碍带",
        ),
        Scenario(
            "G 停在缝口不动",
            duration_s=120.0, reached_goal=False, progress_fraction=0.60,
            close_rays=6, close_range_m=1.20, close_seconds=100.0,
            mean_speed_mps=0.05,
            note="到了缝口就不动了",
        ),
        Scenario(
            "H 冲进去撞死(碰撞终止)",
            duration_s=12.0, reached_goal=False, progress_fraction=0.55,
            close_rays=8, close_range_m=0.70, close_seconds=3.0,
            contacts=1, contact_seconds=DT,
            mean_speed_mps=1.9,
            note="题B 第二关观察到的行为",
        ),
        Scenario(
            "I 到圈边不进去(老毛病)",
            duration_s=120.0, reached_goal=False, progress_fraction=0.95,
            close_rays=4, close_range_m=1.25, close_seconds=20.0,
            mean_speed_mps=0.8,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--band", type=float, default=SAFE_CLEARANCE_M + HALF_BEAM_M,
                        help="proximity band cap in metres (default 1.35)")
    args = parser.parse_args()
    band = args.band

    rows = [score(s, band) for s in build_scenarios()]

    print(f"奖励打分台 | 邻近带上限 {band:.2f} m | gamma={GAMMA} | 时限 {HORIZON_S:.0f}s")
    print("=" * 108)
    head = f"{'场景':<28}{'进展':>7}{'邻近':>8}{'接触':>8}{'终点奖':>9}{'倒车':>7}{'迟缓':>8}{'合计':>9}{'折现后':>10}"
    print(head)
    print("-" * 108)
    for s in rows:
        t = s.terms
        print(
            f"{s.name:<28}{t['progress']:>7.1f}{t['proximity']:>8.2f}{t['contact']:>8.1f}"
            f"{t['goal_bonus']:>9.1f}{t['reverse']:>7.2f}{t['idle']:>8.2f}"
            f"{t['TOTAL']:>9.1f}{t['DISCOUNTED']:>10.1f}"
        )

    print()
    print("排名(未折现):")
    for i, s in enumerate(sorted(rows, key=lambda x: -x.terms["TOTAL"]), 1):
        print(f"  {i}. {s.name:<28} {s.terms['TOTAL']:>8.1f}")

    print()
    print("排名(折现后 —— 这才是策略真正优化的):")
    for i, s in enumerate(sorted(rows, key=lambda x: -x.terms["DISCOUNTED"]), 1):
        print(f"  {i}. {s.name:<28} {s.terms['DISCOUNTED']:>8.1f}")

    # --- the checks that must hold --------------------------------------
    by = {s.name[0]: s.terms for s in rows}
    print()
    print("必须成立的判据:")
    checks = [
        ("理想 A 必须排第一(未折现)",
         all(by["A"]["TOTAL"] >= v["TOTAL"] for v in by.values())),
        ("理想 A 必须排第一(折现后)",
         all(by["A"]["DISCOUNTED"] >= v["DISCOUNTED"] for v in by.values())),
        ("蹭一下抄近路 D 必须差于理想 A",
         by["D"]["TOTAL"] < by["A"]["TOTAL"]),
        ("撞死 H 必须差于胆怯超时 F(否则撞船是划算的)",
         by["H"]["TOTAL"] < by["F"]["TOTAL"]),
        ("贴缝 C 必须优于绕行 B(否则奖励在劝退穿缝)",
         by["C"]["TOTAL"] > by["B"]["TOTAL"]),
        ("停着不动 G 必须差于所有到达场景",
         by["G"]["TOTAL"] < min(by[k]["TOTAL"] for k in "ABC")),
    ]
    for label, ok in checks:
        print(f"  [{'通过' if ok else '不通过'}] {label}")


if __name__ == "__main__":
    main()
