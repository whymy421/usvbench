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
    python scripts/reward_scenarios.py --v12      # softened-contact ledger
    python scripts/reward_scenarios.py --all-versions
    python scripts/reward_scenarios.py --band 1.10
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, replace

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
V12_CONTACT_DWELL = 0.1        # per STEP, not per second
GOAL_ENTRY_BONUS = 50.0
GOAL_TIME_BONUS = 50.0
REVERSE_SCALE = 0.05
SWIFT_SCALE = 0.25
SPEED_SCALE_MPS = 2.0
GAMMA = 0.999
N_RAYS = 36

# Owner-certified inputs for the v12 pre-registered anti-brush check. The tax
# is still 2.0/s at a 12 m radius; these episode totals are the measured split.
V12_OPEN_WATER_SCALE = 2.0
V12_OPEN_WATER_RADIUS_M = 12.0
V12_DETOUR_TAX = 25.0
V12_THREAD_TAX = 7.0
CERTIFIED_DETOUR_M = 53.6
CERTIFIED_THREAD_M = 32.9
CERTIFIED_CRUISE_MPS = 1.75


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
    open_water_seconds: float = 0.0    # time satisfying the v9/v12 tax predicate
    mean_speed_mps: float = 1.2
    reverse_fraction: float = 0.0      # fraction of time commanding negative surge
    threads_gap: bool = False      # completed a clean gap passage
    thread_offset_u: float = 0.0   # |offset| / passable half-width
    note: str = ""
    terms: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Ledger:
    """One registered Task-A reward ledger and its config-class overrides."""

    name: str
    description: str
    overrides: dict
    fidelity_note: str = ""


LEDGERS = (
    Ledger("v1/v3", "基线账本", {}),
    Ledger(
        "v4", "策略不变 PBRS（负势函数）", {"pbrs_correct": True},
        "逐步势函数轨迹未被场景记录；数值按本台既有的均匀进展约定投影。",
    ),
    Ledger(
        "v5", "可行性池化；观测改动，账本不变",
        {"obs_feasibility": True, "feasibility_sectors": 9, "observation_space": 15},
        "CPU 奖励台不能测量观测对策略的影响；奖励数值与 v1/v3 完全相同。",
    ),
    Ledger("v6", "半正弦穿缝弧，幅值 5", {"reward_threading_amplitude": 5.0}),
    Ledger(
        "v7", "PBRS + 真终止势函数归零",
        {"pbrs_correct": True, "pbrs_zero_at_terminal": True},
        "逐步势函数轨迹未被场景记录；数值按本台既有的均匀进展约定投影。",
    ),
    Ledger(
        "v8", "非负 PBRS 平移（含超时残余）",
        {"pbrs_correct": True, "pbrs_shift_potential": True},
        "逐步势函数轨迹未被场景记录；按均匀进展投影并保留 -1 重置势；cfg 未启用超时归零。",
    ),
    Ledger(
        "v9", "开放水域税 2.0/秒 @ 12 m",
        {"reward_open_water_scale": 2.0, "open_water_radius_m": 12.0},
        "使用脚本中已认证的穿缝/绕行开放水域暴露总量，不模拟几何谓词。",
    ),
    Ledger(
        "v10", "开放水域税 + 可行性池化；观测改动，账本不变",
        {
            "reward_open_water_scale": 2.0, "open_water_radius_m": 12.0,
            "obs_feasibility": True, "feasibility_sectors": 9,
            "observation_space": 15,
        },
        "池化效果不能由 CPU 奖励台测量；奖励数值与 v9 完全相同。",
    ),
    Ledger(
        "v12", "软账本：接触不停机、终点奖不设清洁门、驻留 0.1/步",
        {
            "reward_open_water_scale": 2.0, "open_water_radius_m": 12.0,
            "contact_terminates": False, "clean_goal_gate": False,
            "reward_contact_dwell_penalty": 0.1,
        },
        "沿用已认证的 v12 暴露与恢复模型；其 B、D、H 是现有 --v12 定义。",
    ),
)


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


def pbrs_progress(s: Scenario, steps: int, mode: str) -> tuple[float, float]:
    """Project the env's stepwise PBRS over this bench's uniform-progress path.

    Scenario stores endpoints, not a distance trace. The existing discounted
    view already treats aggregate progress as uniformly accrued; all-version
    PBRS uses that same explicit convention and reports the limitation.
    """
    previous = -1.0  # hazard_nav_env.py resets this value for every cfg class
    total = 0.0
    discounted = 0.0
    true_terminal = s.reached_goal or s.contacts > 0
    for step in range(steps):
        covered = s.progress_fraction * (step + 1) / max(steps, 1)
        potential = covered if mode == "shifted" else covered - 1.0
        next_potential = potential
        if mode == "negative_terminal" and true_terminal and step == steps - 1:
            next_potential = 0.0
        contribution = PROGRESS_SCALE * (GAMMA * next_potential - previous)
        total += contribution
        discounted += (GAMMA**step) * contribution
        previous = potential
    return total, discounted


def score(s: Scenario, band_m: float, threading: float = 0.0,
          pbrs_timeout_bug: bool = False, v12: bool = False,
          pbrs: str = "raw", open_water_tax: bool = False) -> Scenario:
    steps = int(round(s.duration_s / DT))
    close_steps = int(round(s.close_seconds / DT))
    contact_steps = int(round(s.contact_seconds / DT))

    # 1. potential progress -- telescopes, so only the endpoints matter
    progress = PROGRESS_SCALE * s.progress_fraction
    disc_progress = None
    if pbrs != "raw":
        progress, disc_progress = pbrs_progress(s, steps, pbrs)

    # 2. proximity, paid only while the close-ray picture holds
    prox = prox_cost_per_step(s.close_rays, s.close_range_m, band_m) * close_steps

    # 3. contact ledger
    contact_dwell = V12_CONTACT_DWELL if v12 else CONTACT_DWELL
    contact = -(CONTACT_ENTRY * s.contacts) - (contact_dwell * contact_steps)

    # 4. terminal bonus. v12 drops the reward gate only; certification still
    # treats contacted arrivals as failures in the environment.
    reward_eligible = s.reached_goal and (v12 or s.contacts == 0)
    remaining = max(0.0, 1.0 - s.duration_s / HORIZON_S)
    goal = ((GOAL_ENTRY_BONUS + GOAL_TIME_BONUS * remaining)
            if reward_eligible else 0.0)

    # 5. v12 inherits v9's open-water tax. Scenario exposure is expressed as
    # seconds beyond the 12 m radius so the formula remains the environment's.
    open_water = (-V12_OPEN_WATER_SCALE * s.open_water_seconds
                  if v12 or open_water_tax else 0.0)

    # 6. behaviour costs
    reverse = -REVERSE_SCALE * DT * steps * s.reverse_fraction  # relu(-surge)^2 ~ 1
    idle = -SWIFT_SCALE * DT * steps * (
        1.0 - min(s.mean_speed_mps / SPEED_SCALE_MPS, 1.0)
    )

    # One-shot threading arc: paid only on a CLEAN passage, so any scenario
    # that touched something earns nothing here.
    arc = 0.0
    if threading > 0.0 and s.threads_gap and s.contacts == 0:
        arc = threading * math.cos(math.pi * s.thread_offset_u / 2.0)

    # The defect, for comparison: zeroing the potential at a TIMEOUT pays
    # +progress_scale * (d/D0) for ending far from the goal.
    timeout_spike = 0.0
    if pbrs_timeout_bug and not s.reached_goal:
        timeout_spike = PROGRESS_SCALE * (1.0 - s.progress_fraction)

    total = progress + prox + contact + goal + reverse + idle + arc + timeout_spike
    if v12 or open_water_tax:
        total += open_water

    # Discounted view: progress and costs accrue along the way, the bonus lands
    # at the end. This is what the agent's value function actually optimises.
    disc_all = discount_sum(steps) / max(steps, 1)
    if disc_progress is None:
        disc_progress = progress * disc_all
    disc_prox = prox * (discount_sum(close_steps) / max(close_steps, 1)) if close_steps else 0.0
    disc_contact = contact * (GAMMA ** (steps // 2))  # contact happens mid-run
    disc_goal = goal * (GAMMA**steps)
    disc_total = (disc_progress + disc_prox + disc_contact + disc_goal
                  + (reverse + idle) * disc_all
                  + (arc + timeout_spike) * (GAMMA ** steps))
    if v12 or open_water_tax:
        # The measured tax is distributed along the route, like progress and
        # other per-step behaviour costs in this behavioural ledger.
        disc_total += open_water * disc_all

    s.terms = {
        "progress": progress,
        "proximity": prox,
        "contact": contact,
        "goal_bonus": goal,
        "open_water": open_water,
        "reverse": reverse,
        "idle": idle,
        "arc": arc,
        "timeout_spike": timeout_spike,
        "TOTAL": total,
        "DISCOUNTED": disc_total,
    }
    return s


def build_scenarios(v12: bool = False, open_water_tax: bool = False) -> list[Scenario]:
    """The behaviours we want, and the ones we fear."""
    scenarios = [
        # --- what we WANT to be the best ------------------------------------
        Scenario(
            "A 理想:直穿缝心,快,零接触",
            duration_s=20.0, reached_goal=True,
            close_rays=6, close_range_m=1.10, close_seconds=6.0,
            mean_speed_mps=1.6, threads_gap=True, thread_offset_u=0.15,
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
            mean_speed_mps=1.3, threads_gap=True, thread_offset_u=0.35,
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
    if not v12 and not open_water_tax:
        return scenarios

    by = {scenario.name[0]: scenario for scenario in scenarios}
    thread_open_s = V12_THREAD_TAX / V12_OPEN_WATER_SCALE
    detour_open_s = V12_DETOUR_TAX / V12_OPEN_WATER_SCALE
    thread_s = CERTIFIED_THREAD_M / CERTIFIED_CRUISE_MPS
    detour_s = CERTIFIED_DETOUR_M / CERTIFIED_CRUISE_MPS

    if open_water_tax and not v12:
        for key in "ACDE":
            by[key] = replace(by[key], open_water_seconds=thread_open_s)
        for key in "BF":
            by[key] = replace(by[key], open_water_seconds=detour_open_s)
        return [by[scenario.name[0]] for scenario in scenarios]

    # B and D are pinned to the owner-certified path/time inputs used to
    # pre-register the thin anti-brush margin: D is the same nominal 32.9 m
    # thread as a clean shortcut, except for exactly one contact step.
    by["B"] = replace(
        by["B"], name="B measured taxed detour", duration_s=detour_s,
        close_rays=0, close_seconds=0.0, mean_speed_mps=CERTIFIED_CRUISE_MPS,
        open_water_seconds=detour_open_s,
    )
    by["D"] = replace(
        by["D"], name="D brush-a-pillar shortcut", duration_s=thread_s,
        close_rays=0, close_seconds=0.0, mean_speed_mps=CERTIFIED_CRUISE_MPS,
        open_water_seconds=thread_open_s,
    )

    # Thread-style arrivals pay the measured ~7 points. The cowardly outer
    # detour pays ~25. H no longer terminates at contact: it spends one second
    # in contact, recovers for five more seconds, then completes the nominal
    # thread, matching the reviewed recovery model.
    for key in "ACE":
        by[key] = replace(by[key], open_water_seconds=thread_open_s)
    by["F"] = replace(by["F"], open_water_seconds=detour_open_s)
    recovery_s = thread_s + 6.0
    by["H"] = replace(
        by["H"], name="H contact then recover and reach",
        duration_s=recovery_s, reached_goal=True, progress_fraction=1.0,
        close_rays=0, close_seconds=0.0, contact_seconds=1.0,
        mean_speed_mps=CERTIFIED_THREAD_M / recovery_s,
        open_water_seconds=thread_open_s,
    )
    return [by[scenario.name[0]] for scenario in scenarios]


def ordering_checks(rows: list[Scenario], v12: bool = False) -> list[tuple[str, bool]]:
    """Return the historical eight criteria plus v12's anti-brush criterion."""
    by = {s.name[0]: s.terms for s in rows}
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
        ("超时后走得少的必须差于走得多的(抓'离目标越远越划算')",
         by["F"]["TOTAL"] < by["I"]["TOTAL"]),
        ("最好的失败必须差于最差的成功",
         max(by[k]["TOTAL"] for k in "DEFGHI")
         < min(by[k]["TOTAL"] for k in "ABC")),
    ]
    if v12:
        checks.append(
            ("brush-a-pillar shortcut must NOT out-score the taxed detour",
             by["D"]["TOTAL"] <= by["B"]["TOTAL"])
        )
    return checks


def print_all_versions(band: float) -> None:
    """Print per-ledger winners/checks, followed by PPO's discounted matrix."""
    results = []
    print(f"全版本奖励对照 | 邻近带上限 {band:.2f} m | gamma={GAMMA} | 时限 {HORIZON_S:.0f}s")
    print("数值口径：折现列是 PPO 优化口径；PBRS 行按均匀进展投影，观测改动不在本台测量范围内。")
    print()

    for ledger in LEDGERS:
        overrides = ledger.overrides
        is_v12 = overrides.get("contact_terminates") is False
        has_tax = overrides.get("reward_open_water_scale", 0.0) > 0.0 and not is_v12
        if overrides.get("pbrs_shift_potential"):
            pbrs_mode = "shifted"
        elif overrides.get("pbrs_zero_at_terminal"):
            pbrs_mode = "negative_terminal"
        elif overrides.get("pbrs_correct"):
            pbrs_mode = "negative"
        else:
            pbrs_mode = "raw"
        rows = [
            score(
                scenario, band,
                threading=overrides.get("reward_threading_amplitude", 0.0),
                v12=is_v12,
                pbrs=pbrs_mode,
                open_water_tax=has_tax,
            )
            for scenario in build_scenarios(is_v12, has_tax)
        ]
        checks = ordering_checks(rows, is_v12)
        failed = [label for label, ok in checks if not ok]
        raw_winner = max(rows, key=lambda row: row.terms["TOTAL"])
        disc_winner = max(rows, key=lambda row: row.terms["DISCOUNTED"])
        override_text = ", ".join(
            f"{key}={value}" for key, value in overrides.items()
        ) or "无"

        print(f"[{ledger.name}] {ledger.description}")
        print(f"  参数覆盖: {override_text}")
        if ledger.fidelity_note:
            print(f"  能力边界: {ledger.fidelity_note}")
        print(f"  冠军(未折现): {raw_winner.name} ({raw_winner.terms['TOTAL']:.1f})")
        print(f"  冠军(折现后): {disc_winner.name} ({disc_winner.terms['DISCOUNTED']:.1f})")
        print(f"  判据: {len(checks) - len(failed)}/{len(checks)} 通过")
        print(f"  失败: {'；'.join(failed) if failed else '无'}")
        print()
        results.append((ledger, rows))

    print("折现合计矩阵（PPO 优化口径；列 A-I 对应既有九个场景）")
    print(f"{'账本':<8}" + "".join(f"{key:>10}" for key in "ABCDEFGHI"))
    print("-" * 98)
    for ledger, rows in results:
        by = {row.name[0]: row.terms["DISCOUNTED"] for row in rows}
        print(f"{ledger.name:<8}" + "".join(f"{by[key]:>10.1f}" for key in "ABCDEFGHI"))

    print()
    print("不可由本 CPU 账本忠实测量的版本差异:")
    print("  v5: 可行性池化只改变观测；奖励数字与 v1/v3 相同。")
    print("  v10: 池化部分只改变观测；奖励数字与 v9 相同。")
    print("  v4/v7/v8: 场景没有逐步距离轨迹，未折现 PBRS 不能精确重放；表中是明确标注的均匀进展投影。")
    print("  v8: 权威 cfg 是非负势平移且未启用超时归零；旧 --pbrs-timeout-bug 不代表当前注册 v8，故未套用。")
    print("  v12: 为保持现有 --v12 口径，B、D、H 使用已认证的测量/恢复版本，不能与其它行当作同轨迹反事实。")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--band", type=float, default=SAFE_CLEARANCE_M + HALF_BEAM_M,
                        help="proximity band cap in metres (default 1.35)")
    parser.add_argument("--threading", type=float, default=0.0,
                        help="amplitude of the one-shot half-sine arc (v6 uses 5)")
    parser.add_argument("--pbrs-timeout-bug", action="store_true",
                        help="reproduce the defect that cost 75.8 points: zero "
                             "the potential at TIMEOUTS as well as at true "
                             "terminals, paying +20*(d/D0) for running out of "
                             "time far from the goal")
    parser.add_argument("--v12", action="store_true",
                        help="apply the v12 open-water tax and softened contact ledger")
    parser.add_argument("--all-versions", action="store_true",
                        help="score every registered Task-A ledger and print a matrix")
    args = parser.parse_args()
    band = args.band

    if args.all_versions:
        print_all_versions(band)
        return

    rows = [score(s, band, args.threading, args.pbrs_timeout_bug, args.v12)
            for s in build_scenarios(args.v12)]

    print(f"奖励打分台 | 邻近带上限 {band:.2f} m | gamma={GAMMA} | 时限 {HORIZON_S:.0f}s")
    if args.v12:
        print(f"v12 | 接触不终止 | 终点奖励门已移除 | 驻留 {V12_CONTACT_DWELL}/步 | "
              f"开放水域 {V12_OPEN_WATER_SCALE}/秒 @ {V12_OPEN_WATER_RADIUS_M:.0f} m")
    table_width = 117 if args.v12 else 108
    print("=" * table_width)
    open_head = f"{'开放水域':>9}" if args.v12 else ""
    head = f"{'场景':<28}{'进展':>7}{'邻近':>8}{'接触':>8}{'终点奖':>9}{open_head}{'倒车':>7}{'迟缓':>8}{'合计':>9}{'折现后':>10}"
    print(head)
    print("-" * table_width)
    for s in rows:
        t = s.terms
        open_cell = f"{t['open_water']:>9.1f}" if args.v12 else ""
        print(
            f"{s.name:<28}{t['progress']:>7.1f}{t['proximity']:>8.2f}{t['contact']:>8.1f}"
            f"{t['goal_bonus']:>9.1f}{open_cell}{t['reverse']:>7.2f}{t['idle']:>8.2f}"
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
    print()
    print("必须成立的判据:")
    checks = ordering_checks(rows, args.v12)
    for label, ok in checks:
        print(f"  [{'通过' if ok else '不通过'}] {label}")


if __name__ == "__main__":
    main()
