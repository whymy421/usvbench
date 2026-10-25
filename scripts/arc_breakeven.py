"""How large must a threading bonus be to beat going around? (open-water Task A)

The advisor's question, verbatim: "rewarding the middle could also work, but you
need to compute how large a reward offsets the attraction of the open water."

This answers it from the shipped reward ledger in `hazard_nav_env._get_rewards`
plus certified measurements, and it also prices the OTHER mechanism the advisor
proposed -- a penalty on the surrounding open water -- so the two can be
compared on the same axis.

The headline is not the amplitude. It is that at q = 1 the ledger ALREADY pays
more for threading than for detouring, so the detour is not a reward-ranking
problem at all; it is a RISK problem. That changes what a fix has to do.

Run: python scripts/arc_breakeven.py
"""

from __future__ import annotations

import math

# --- reward ledger, transcribed from hazard_nav_env.py::_get_rewards ---------
DT = 2 / 120.0            # sim.dt * decimation
T_EPISODE_S = 120.0
PROGRESS_TOTAL = 20.0     # reward_progress_scale x potential 0->1, path-independent
GOAL_ENTRY = 50.0         # reward_goal_entry_bonus
GOAL_TIME = 50.0          # reward_goal_time_bonus x remaining_fraction
SWIFT_SCALE = 0.25        # per second, x (1 - speed/SPEED_SCALE_MPS)
SPEED_SCALE_MPS = 2.0     # tasks/_shared/obs_superset.py
PROX_SCALE = 4.1
PROX_CAP_M = 1.35         # safe_clearance 0.90 + half_beam 0.45
PROX_FLOOR_M = 0.45
CONTACT_ENTRY = 25.0
RAY_COUNT = 36

# --- certified measurements (do not re-derive) -------------------------------
STRAIGHT_M = 29.7         # median start->goal
DETOUR_M = 53.6           # v3 s43 certifies 100% with this median path
THREAD_M = 32.0           # a route that threads instead of circumventing
CRUISE_MPS = 1.6
GAP_W_M = 2.2             # tier 2-3 gap, 2.4-3 hull beams
GAPS_PER_RUN = 2
GAP_CORRIDOR_M = 3.0      # how far the hull stays inside a gap's proximity band
RAYS_IN_BAND = 6          # side rays that see a jamb within the 1.35 m band


def episode_return(path_m, n_gaps, gap_w_m, arc_amplitude=0.0, u_abs=0.0,
                   open_water_penalty_per_s=0.0, open_water_fraction=1.0):
    """Total return for one successful run along `path_m`."""
    t = path_m / CRUISE_MPS
    time_bonus = GOAL_TIME * max(0.0, 1.0 - t / T_EPISODE_S)
    swift = SWIFT_SCALE * t * (1.0 - min(CRUISE_MPS / SPEED_SCALE_MPS, 1.0))

    # Per-ray log proximity, only the rays that actually see a jamb.
    r_side = max(min(gap_w_m / 2.0, PROX_CAP_M), PROX_FLOOR_M)
    per_ray = math.log(PROX_CAP_M / r_side)
    per_step = PROX_SCALE * DT * (RAYS_IN_BAND / RAY_COUNT) * per_ray
    steps_in_gap = (GAP_CORRIDOR_M / CRUISE_MPS) / DT
    prox = n_gaps * per_step * steps_in_gap

    arc = arc_amplitude * math.cos(math.pi * u_abs / 2.0) * n_gaps
    open_water = open_water_penalty_per_s * t * open_water_fraction
    total = (PROGRESS_TOTAL + GOAL_ENTRY + time_bonus
             - swift - prox + arc - open_water)
    return total, dict(time_bonus=time_bonus, swift=swift, prox=prox,
                       arc=arc, open_water=open_water, t=t)


def crash_return(reached_m=10.0):
    """Contact at the gap mouth: partial progress, entry penalty, no goal bonus."""
    return PROGRESS_TOTAL * (reached_m / STRAIGHT_M) - CONTACT_ENTRY


def main() -> None:
    v_thread, d_t = episode_return(THREAD_M, GAPS_PER_RUN, GAP_W_M)
    v_detour, d_d = episode_return(DETOUR_M, 0, 0.0)
    r_crash = crash_return()

    print("=== deterministic ledger (both routes succeed) ===")
    print(f"{'':14}{'thread':>10}{'detour':>10}")
    print(f"{'path (m)':14}{THREAD_M:>10.1f}{DETOUR_M:>10.1f}")
    print(f"{'progress':14}{PROGRESS_TOTAL:>10.1f}{PROGRESS_TOTAL:>10.1f}")
    print(f"{'goal entry':14}{GOAL_ENTRY:>10.1f}{GOAL_ENTRY:>10.1f}")
    print(f"{'time bonus':14}{d_t['time_bonus']:>10.1f}{d_d['time_bonus']:>10.1f}")
    print(f"{'swift cost':14}{-d_t['swift']:>10.1f}{-d_d['swift']:>10.1f}")
    print(f"{'proximity':14}{-d_t['prox']:>10.2f}{-d_d['prox']:>10.2f}")
    print(f"{'TOTAL':14}{v_thread:>10.2f}{v_detour:>10.2f}")
    print(f"\n  threading - detour = {v_thread - v_detour:+.2f}")
    if v_thread > v_detour:
        print("  => The ledger ALREADY prefers threading. The detour is not")
        print("     bought by the reward ranking, so a bonus is aimed at the")
        print("     wrong target: what makes detouring rational is RISK.")
    print(f"\n  return if the attempt ends in contact: {r_crash:.1f}")

    print("\n=== what it costs to make threading rational under risk ===")
    print("  q = probability a threading attempt succeeds rather than colliding")
    print(f"{'q':>6}{'arc A needed':>16}{'open-water pen/s':>20}{'ratio':>8}")
    for q in (0.9, 0.7, 0.5, 0.3, 0.2, 0.1):
        # (a) bonus on the risky option: collected only when it works -> /q
        need_arc = ((v_detour - (1 - q) * r_crash) / q - v_thread) / GAPS_PER_RUN
        # (b) penalty on the safe option: lands with certainty
        gap = v_detour - (q * v_thread + (1 - q) * r_crash)
        need_pen = gap / d_d['t']
        ratio = (need_arc * GAPS_PER_RUN) / gap if gap > 0 else float('nan')
        print(f"{q:>6.1f}{max(0.0, need_arc):>16.1f}"
              f"{max(0.0, need_pen):>20.2f}{ratio:>8.2f}")

    print("\n  The arc must be ~1/q times the gap it closes, because it is only")
    print("  collected on the runs that succeed. The open-water penalty is")
    print("  collected with certainty, so it closes the same gap at face value.")
    print("  That asymmetry is why the advisor's penalty is the cheaper lever.")
    print(f"\n  For scale: the goal bonus is {GOAL_ENTRY:.0f}-"
          f"{GOAL_ENTRY + GOAL_TIME:.0f} and the whole progress budget is "
          f"{PROGRESS_TOTAL:.0f}.")
    print("  An arc of 100+ would dominate every other term in the ledger and")
    print("  become the thing the policy optimises. That is not a shaping term")
    print("  any more, it is a new objective.")


if __name__ == "__main__":
    main()
