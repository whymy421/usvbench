"""Independent contact-ledger cross-check for Task A (hazard_nav).

Reads constants directly from the source tree without importing Isaac Lab,
then evaluates every entry/termination/gate/dwell combination under one
explicit recovery model. Run from the repository root with:

    python scripts/codex_contact_ledger.py
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from itertools import product
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "tasks" / "hazard_nav" / "hazard_nav_env_cfg.py"
ENV_PATH = ROOT / "tasks" / "hazard_nav" / "hazard_nav_env.py"
OBS_PATH = ROOT / "tasks" / "_shared" / "obs_superset.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise KeyError(f"class {name!r} not found")


def _literal(node: ast.AST) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -float(_literal(node.operand))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return float(_literal(node.left)) / float(_literal(node.right))
    raise TypeError(f"not a supported literal: {ast.dump(node)}")


def _field(cls: ast.ClassDef, name: str) -> object:
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name and node.value is not None:
                return _literal(node.value)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return _literal(node.value)
    raise KeyError(f"field {name!r} not found in {cls.name}")


def _module_field(tree: ast.Module, name: str) -> object:
    pseudo_class = ast.ClassDef(name="module", bases=[], keywords=[], body=tree.body,
                                decorator_list=[])
    return _field(pseudo_class, name)


def _simulation_dt(cls: ast.ClassDef) -> float:
    for node in cls.body:
        value = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "sim":
                value = node.value
        if isinstance(value, ast.Call):
            for keyword in value.keywords:
                if keyword.arg == "dt":
                    return float(_literal(keyword.value))
    raise KeyError("SimulationCfg dt not found")


CFG_TREE = _tree(CFG_PATH)
BASE = _class(CFG_TREE, "HazardNavEnvCfg")
V9 = _class(CFG_TREE, "HazardOpenWaterTaxEnvCfg")
OBS_TREE = _tree(OBS_PATH)

DECIMATION = int(_field(BASE, "decimation"))
SIM_DT = _simulation_dt(BASE)
DT = SIM_DT * DECIMATION
STEPS_PER_S = 1.0 / DT
HORIZON_S = float(_field(BASE, "episode_length_s"))
PROGRESS_SCALE = float(_field(BASE, "reward_progress_scale"))
CLEARANCE_SCALE = float(_field(BASE, "reward_clearance_scale"))
PROX_SCALE = float(_field(BASE, "reward_prox_scale"))
PROX_FLOOR_M = float(_field(BASE, "prox_ray_floor_m"))
SAFE_CLEARANCE_M = float(_field(BASE, "safe_clearance_m"))
HALF_BEAM_M = float(_field(BASE, "half_beam_m"))
PROX_CAP_M = SAFE_CLEARANCE_M + HALF_BEAM_M
RAY_COUNT = int(_field(BASE, "ray_count"))
SOURCE_ENTRY = float(_field(BASE, "reward_contact_entry_penalty"))
SOURCE_DWELL = float(_field(BASE, "reward_contact_dwell_penalty"))
GOAL_ENTRY = float(_field(BASE, "reward_goal_entry_bonus"))
GOAL_TIME = float(_field(BASE, "reward_goal_time_bonus"))
REVERSE_SCALE = float(_field(BASE, "reward_reverse_action_scale"))
SWIFT_SCALE = float(_field(BASE, "reward_swift_scale"))
SPEED_SCALE_MPS = float(_module_field(OBS_TREE, "SPEED_SCALE_MPS"))
BASE_OPEN_SCALE = float(_field(BASE, "reward_open_water_scale"))
BASE_OPEN_RADIUS_M = float(_field(BASE, "open_water_radius_m"))
OPEN_GOAL_EXEMPT_M = float(_field(BASE, "open_water_goal_exempt_m"))
V9_OPEN_SCALE = float(_field(V9, "reward_open_water_scale"))
V9_OPEN_RADIUS_M = float(_field(V9, "open_water_radius_m"))
THREADING_AMPLITUDE = float(_field(BASE, "reward_threading_amplitude"))


# Certified measurements supplied by the owner; these are inputs, not fitted.
STRAIGHT_M = 29.7
DETOUR_M = 53.6
THREAD_M = 32.9
CERTIFIED_THREAD_Q = 0.84
CRUISE_RANGE_MPS = (1.6, 1.9)
CRUISE_MPS = sum(CRUISE_RANGE_MPS) / 2.0
TAX_DETOUR = 25.0
TAX_THREAD = 7.0

# Explicitly invented recovery model.
CRASH_ROUTE_FRACTION = 0.35
CRASH_PATH_M = CRASH_ROUTE_FRACTION * THREAD_M
# Invented geometry: the pre-gap segment points directly toward the goal, so
# source potential progress is travelled pre-gap path / certified straight line.
CRASH_PROGRESS_FRACTION = min(1.0, CRASH_PATH_M / STRAIGHT_M)
CONTACT_SECONDS_IF_RECOVERED = 1.0
RECOVERY_SECONDS_AFTER_CONTACT = 5.0
EXTRA_RECOVERY_SECONDS = CONTACT_SECONDS_IF_RECOVERED + RECOVERY_SECONDS_AFTER_CONTACT
TERMINAL_CONTACT_STEPS = 1
RECOVERED_CONTACT_STEPS = round(CONTACT_SECONDS_IF_RECOVERED * STEPS_PER_S)


def swift_cost(cruise_seconds: float, stopped_seconds: float = 0.0) -> float:
    cruise_fraction = 1.0 - min(CRUISE_MPS / SPEED_SCALE_MPS, 1.0)
    return SWIFT_SCALE * (cruise_seconds * cruise_fraction + stopped_seconds)


def goal_bonus(arrival_seconds: float) -> float:
    remaining = max(0.0, min(1.0, 1.0 - arrival_seconds / HORIZON_S))
    return GOAL_ENTRY + GOAL_TIME * remaining


THREAD_SECONDS = THREAD_M / CRUISE_MPS
DETOUR_SECONDS = DETOUR_M / CRUISE_MPS
CRASH_SECONDS = THREAD_SECONDS * CRASH_ROUTE_FRACTION
RECOVERED_SECONDS = THREAD_SECONDS + EXTRA_RECOVERY_SECONDS


def route_tax(kind: str, tax_on: bool) -> float:
    if not tax_on:
        return 0.0
    if kind == "detour":
        return TAX_DETOUR
    if kind == "thread":
        return TAX_THREAD
    if kind == "early_crash":
        return TAX_THREAD * CRASH_ROUTE_FRACTION
    raise ValueError(kind)


def detour_return(tax_on: bool) -> float:
    return (
        PROGRESS_SCALE
        + goal_bonus(DETOUR_SECONDS)
        - swift_cost(DETOUR_SECONDS)
        - route_tax("detour", tax_on)
    )


def clean_thread_return(tax_on: bool) -> float:
    return (
        PROGRESS_SCALE
        + goal_bonus(THREAD_SECONDS)
        - swift_cost(THREAD_SECONDS)
        - route_tax("thread", tax_on)
    )


@dataclass(frozen=True)
class Design:
    entry: float
    terminate: bool
    keep_gate: bool
    dwell: float


def failed_attempt_return(design: Design, tax_on: bool) -> float:
    if design.terminate:
        return (
            PROGRESS_SCALE * CRASH_PROGRESS_FRACTION
            - swift_cost(CRASH_SECONDS)
            - route_tax("early_crash", tax_on)
            - design.entry
            - design.dwell * TERMINAL_CONTACT_STEPS
        )

    delayed_goal = 0.0 if design.keep_gate else goal_bonus(RECOVERED_SECONDS)
    return (
        PROGRESS_SCALE
        + delayed_goal
        - swift_cost(THREAD_SECONDS, EXTRA_RECOVERY_SECONDS)
        - route_tax("thread", tax_on)
        - design.entry
        - design.dwell * RECOVERED_CONTACT_STEPS
    )


def expected_coefficients(design: Design, tax_on: bool) -> tuple[float, float]:
    failure = failed_attempt_return(design, tax_on)
    success = clean_thread_return(tax_on)
    return failure, success - failure


def threshold_q(design: Design, tax_on: bool) -> float:
    intercept, slope = expected_coefficients(design, tax_on)
    if slope <= 0.0:
        return math.inf
    return (detour_return(tax_on) - intercept) / slope


def one_ray_prox_cost(seconds: float) -> float:
    return PROX_SCALE * seconds * math.log(PROX_CAP_M / PROX_FLOOR_M) / RAY_COUNT


def max_prox_cost(seconds: float) -> float:
    return PROX_SCALE * seconds * math.log(PROX_CAP_M / PROX_FLOOR_M)


def graze_summary(design: Design) -> tuple[float, float, bool]:
    all_in = (
        design.entry
        + design.dwell * RECOVERED_CONTACT_STEPS
        + one_ray_prox_cost(CONTACT_SECONDS_IF_RECOVERED)
    )
    ratio = all_in / design.entry
    return all_in, ratio, ratio > 3.0


def brush_margin(design: Design, tax_on: bool) -> tuple[str, float | None]:
    if design.terminate:
        return "NO(term)", None

    # One-step contact, otherwise the measured threading path and arrival time.
    if design.keep_gate:
        brush = (
            PROGRESS_SCALE
            - swift_cost(THREAD_SECONDS)
            - route_tax("thread", tax_on)
            - design.entry
            - design.dwell
        )
    else:
        brush = clean_thread_return(tax_on) - design.entry - design.dwell
    margin = brush - detour_return(tax_on)
    return ("YES" if margin > 0.0 else "no"), margin


def post_crash_signal(design: Design) -> str:
    if design.terminate:
        return "none"
    remaining_progress = PROGRESS_SCALE * (1.0 - CRASH_PROGRESS_FRACTION)
    if design.keep_gate:
        return f"+{remaining_progress:.1f} prog"
    return f"+{remaining_progress + goal_bonus(RECOVERED_SECONDS):.1f} prog+goal"


DESIGNS = [
    Design(entry, terminate, keep_gate, dwell)
    for entry, terminate, keep_gate, dwell in product(
        (25.0, 10.0), (True, False), (True, False), (1.0, 0.1)
    )
]


def yn(value: bool) -> str:
    return "Y" if value else "N"


def gate(value: bool) -> str:
    return "keep" if value else "drop"


def q_text(value: float) -> str:
    if not math.isfinite(value):
        return "never"
    if value > 1.0:
        return f">1 ({value:.3f})"
    if value < 0.0:
        return f"always ({value:.3f})"
    return f"{value:.3f}"


def affine_text(intercept: float, slope: float) -> str:
    sign = "+" if slope >= 0.0 else "-"
    return f"{intercept:.2f} {sign} {abs(slope):.2f}q"


def print_source_ledger() -> None:
    print("INDEPENDENT TASK A CONTACT-LEDGER CROSS-CHECK")
    print("=" * 78)
    print("SOURCE-TRANSCRIBED LEDGER (HazardNavV3 inherits HazardNavEnvCfg)")
    print(f"control: sim dt={SIM_DT:.8f}s, decimation={DECIMATION}, "
          f"control dt={DT:.8f}s ({STEPS_PER_S:.0f} steps/s), horizon={HORIZON_S:.0f}s")
    print(f"progress: {PROGRESS_SCALE:g} * [Phi(s')-Phi(s)], "
          "Phi=-clamp(distance/D0,0,1); successful endpoint total = +20")
    print(f"goal: clean_entry * [{GOAL_ENTRY:g} + {GOAL_TIME:g} * "
          "clamp((N-episode_step)/N,0,1)]")
    print(f"swift (v3 active): -{SWIFT_SCALE:g} * dt * "
          f"[1-clamp(speed/{SPEED_SCALE_MPS:g},max=1)] until goal")
    print(f"per-ray proximity: -{PROX_SCALE:g} * dt * mean(ln({PROX_CAP_M:.2f}/"
          f"clamp(ray,{PROX_FLOOR_M:.2f},{PROX_CAP_M:.2f}))) over {RAY_COUNT} rays")
    print(f"quadratic clearance: scale={CLEARANCE_SCALE:g} (present but off)")
    print(f"contact: -{SOURCE_ENTRY:g} on false->true entry; "
          f"-{SOURCE_DWELL:g} per contact control step; contact terminates episode")
    print("clean-goal gate: any contact before goal suppresses the entire goal bonus")
    print(f"reverse: -{REVERSE_SCALE:g} * dt * relu(-surge_action)^2")
    print(f"one-shot threading bonus: amplitude={THREADING_AMPLITUDE:g} (off)")
    print(f"base open-water fields: scale={BASE_OPEN_SCALE:g}/s, "
          f"radius={BASE_OPEN_RADIUS_M:g}m, goal exemption={OPEN_GOAL_EXEMPT_M:g}m")
    print(f"v9 HazardOpenWaterTaxEnvCfg overrides: scale={V9_OPEN_SCALE:g}/s, "
          f"radius={V9_OPEN_RADIUS_M:g}m; goal exemption remains {OPEN_GOAL_EXEMPT_M:g}m")
    print("v9 tax predicate: clearance > 12m AND goal distance > 6m AND goal not yet reached")


def print_inputs_and_assumptions() -> None:
    print("\nCERTIFIED INPUTS USED")
    print(f"straight={STRAIGHT_M:.1f}m; detour=100% SR/{DETOUR_M:.1f}m median; "
          f"thread=~{CERTIFIED_THREAD_Q:.0%} SR/{THREAD_M:.1f}m; "
          f"cruise={CRUISE_RANGE_MPS[0]:.1f}-{CRUISE_RANGE_MPS[1]:.1f}m/s; "
          f"episode={HORIZON_S:.0f}s")
    print(f"v9 measured tax used in calculations: detour={TAX_DETOUR:.1f}, "
          f"thread={TAX_THREAD:.1f} per episode (owner-certified values)")
    print("\nINVENTED ASSUMPTIONS (complete list)")
    print(f"A1. Use cruise midpoint {CRUISE_MPS:.2f}m/s for both routes; "
          "arrival time = path/cruise.")
    print(f"A2. q means a clean early pass; a failed pass first contacts after "
          f"{CRASH_ROUTE_FRACTION:.0%} of the threading route ({CRASH_PATH_M:.2f}m). "
          "Invent the pre-gap segment as goal-aligned, so source potential progress "
          f"is {CRASH_PATH_M:.2f}/{STRAIGHT_M:.1f}={CRASH_PROGRESS_FRACTION:.3f}.")
    print("A3. Terminating failure receives one entry and one terminal contact step.")
    print(f"A4. Non-terminating failure has one entry, {CONTACT_SECONDS_IF_RECOVERED:.0f}s "
          f"({RECOVERED_CONTACT_STEPS} steps) in contact, then "
          f"{RECOVERY_SECONDS_AFTER_CONTACT:.0f}s additional zero-speed recovery; "
          "it follows the remaining nominal path and reaches the goal.")
    print("A5. Recovery adds no second contact entry and happens near the gap, so a "
          "recovered run pays the measured full thread tax (7), not extra open-water tax.")
    print(f"A6. An early terminated crash pays {CRASH_ROUTE_FRACTION:.0%} of the measured "
          "thread tax (2.45); this is a path-proportional approximation.")
    print("A7. Nominal route calculations set reverse-action, quadratic-clearance, "
          "threading-bonus, and ray-proximity costs to zero: no certified action/ray "
          "trace was supplied. This makes threading q* optimistic if it has more proximity cost.")
    print("A8. Gate=keep suppresses only the post-contact goal bonus in this model; "
          "progress to the reached goal remains. Gate=drop restores that bonus.")
    print("A9. Brush exploit = one contact step on the 32.9m shortcut, otherwise a "
          "nominal thread; profitability is versus the certified detour.")
    print("A10. One-second wall graze assumes one of 36 rays is clamped at 0.45m and "
          "the others are at the 1.35m cap; the all-rays-at-floor maximum is also reported.")


def print_table(tax_on: bool) -> None:
    title = "V9 TAX ON" if tax_on else "TAX OFF"
    print(f"\nDESIGN SPACE -- {title}")
    print(f"detour return={detour_return(tax_on):.2f}; clean-thread return="
          f"{clean_thread_return(tax_on):.2f}; q reference={CERTIFIED_THREAD_Q:.2f}")
    print(" E  term gate dwell | failedR | E_thread(q)       | E@.84  | q*     "
          "| brush vs detour | 1s graze all-in/entry | post-crash signal")
    print("-" * 139)
    for design in DESIGNS:
        intercept, slope = expected_coefficients(design, tax_on)
        expected_ref = intercept + CERTIFIED_THREAD_Q * slope
        brush, margin = brush_margin(design, tax_on)
        brush_cell = brush if margin is None else f"{brush} {margin:+.1f}"
        graze, ratio, flag = graze_summary(design)
        graze_cell = f"{graze:.1f}/{ratio:.2f}x" + (" !" if flag else "")
        print(
            f"{design.entry:>2.0f}   {yn(design.terminate):>1}   {gate(design.keep_gate):<4} "
            f" {design.dwell:>3.1f} | {intercept:>7.2f} | "
            f"{affine_text(intercept, slope):<17} | {expected_ref:>6.2f} | "
            f"{q_text(threshold_q(design, tax_on)):<6} | "
            f"{brush_cell:<15} | {graze_cell:<21} | {post_crash_signal(design)}"
        )


def print_exploit_notes() -> None:
    one_ray = one_ray_prox_cost(1.0)
    maximum = max_prox_cost(1.0)
    print("\nEXPLOIT-CHECK DETAILS")
    print(f"one-ray wall proximity costs {one_ray:.3f}/s; all-{RAY_COUNT}-ray "
          f"floor is the mathematical max {maximum:.3f}/s.")
    print("'1s graze' includes one entry + 60 dwell steps + the one-ray proximity "
          "cost. '!' means the all-in graze exceeds 3 entry penalties.")
    print("For scale, historical v2 dwell=-50/step would charge 3000 in one second "
          "before any entry penalty; the proposed 1.0 and 0.1 charge 60 and 6.")
    print("No terminating design can exploit a contact shortcut. No keep-gate design "
          "can do so here because it forfeits ~90 points of goal reward.")
    print("With tax ON, both entry=10 / no-termination / drop-gate designs make the "
          "one-step brush profitable (+13.2 versus detour); entry=25 keeps it "
          "slightly unprofitable (-1.8 or -2.7 depending on dwell).")
    remaining_progress = PROGRESS_SCALE * (1.0 - CRASH_PROGRESS_FRACTION)
    print("Post-crash signal is none under termination; without termination it is "
          f"+{remaining_progress:.1f} progress with the gate kept, or that progress "
          "plus the delayed goal bonus with the gate dropped.")


def print_recommendation() -> None:
    recommended = Design(25.0, False, False, 0.1)
    q0 = threshold_q(recommended, False)
    q9 = threshold_q(recommended, True)
    b9, m9 = expected_coefficients(recommended, True)
    e9 = b9 + CERTIFIED_THREAD_Q * m9
    advantage = e9 - detour_return(True)
    _, brush9 = brush_margin(recommended, True)
    print("\nRECOMMENDATION")
    print("entry=25, terminate-on-contact=NO, clean-goal gate=DROP, "
          "dwell=0.1/step, v9 open-water tax=ON.")
    print(f"Reason: it preserves a substantial entry consequence, removes the terminal "
          f"learning dead-end, keeps 1s grazing to {graze_summary(recommended)[0]:.1f} "
          f"({graze_summary(recommended)[1]:.2f} entries), and lets a recovered collision "
          "receive a learnable terminal bonus. In this model q* moves from "
          f"{q0:.3f} without tax to {q9:.3f} with tax; at q=.84 expected thread "
          f"return={e9:.2f}, {advantage:+.2f} over taxed detour.")
    print("Do NOT pair entry=10 with tax+drop-gate: a one-step collision shortcut is "
          "then directly profitable in this model.")
    print("\nSINGLE STRONGEST FAILURE MODE")
    print(f"The recommended anti-brush margin is only {-brush9:.2f} return points. "
          "Small errors in the measured tax split, route time, or a shorter-than-32.9m "
          "contact path can flip the sign and teach deliberate pillar clipping.")
    print("\nFALSIFIABLE TWO-SEED PREDICTION")
    print("At the same training budget and fixed evaluation tier, BOTH seeds will end "
          "with >=80% collision-free success and median successful path <=40m. "
          "Either seed missing either bound falsifies the prediction.")


def main() -> None:
    print_source_ledger()
    print_inputs_and_assumptions()
    print_table(tax_on=False)
    print_table(tax_on=True)
    print_exploit_notes()
    print_recommendation()


if __name__ == "__main__":
    main()
