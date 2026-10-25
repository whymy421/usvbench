"""G23c layer 2: candidate runtime rule S2 = d7_reveal_sync_v1 (arrival-synchronized REVEAL).

Owner decision pending (REV-G23b follow-up). S2 restores the documented dose semantics (DOSE_TTC_RATIOS,
dose_axis = TTC_over_t_avoid): the dose is the own ship's time to first contact when the intruder appears,
for EVERY own ship, not only for the nominal one at u_ref. Every pinned quantity of the admitted candidate is
unchanged (line, speed, radius, start/stop of the motion window, t_c, t_avoid, aim offset); D7_SUITE_VERSION and
the pools are unchanged.

Transit tasks.
  Before the reveal the intruder is HIDDEN: not detectable (visible False) and not collidable (physical False),
  so it is absent from rays, the swept-contact latch and every reward / clearance term (G23c F12). Its sample
  position is the pinned position(t) (bookkeeping only).
  At every update before the reveal, from the own ship's pose q = (x, y, psi) and speed estimate V (below), the
  constant-velocity own track q(tau) = q + V tau (cos psi, sin psi) with fixed heading (as visibility/TTC elsewhere)
  is intersected with the intruder's CERTIFIED line p(phi) = P0 + v phi (P0 = (x0, y0), v = (vx, vy)):
    body frame of the own ship: b = R(psi)^T v, w = b - (V, 0) (relative velocity), footprint centre c, half
    extents (hx, hy), intruder radius r, aim offset fraction a (the candidate's, encounter metadata).
    crossing (|b_y| >= COLLINEAR_SIN |b| and V >= F): the contact point E is the generator's entry point for the
      relative direction w: w_hat = (nx, ny), half_width = hx |ny| + hy |nx| + r, lateral = a * half_width,
      E = c + (-ny, nx) lateral + w_hat (entry - back) with entry = rounded_box_entry from back metres upstream
      (trajectory.generate_encounter's construction, so E equals the admitted contact point for the nominal);
      the contact phase psi_c and the time tau solve the linear system
        b psi_c - (V, 0) tau = E - R^T (P0 - q)      ->  psi_c = D_y / b_y,  tau = (b_x psi_c - D_x) / V;
    collinear (a purely geometric test at every update; the candidate's path_type is not used, so a head-on
      candidate approached from the side takes the crossing branch -- every pinned head-on candidate is exactly
      collinear, every pinned crossing one exactly at 90 deg, G23c probe): |b_y| < COLLINEAR_SIN |b| (heading
      along the line), or the footprint already straddles the line, i.e. the distance of the footprint centre
      from the line is below the lane half-width reach + |c| + r (a ship on the line is approached ALONG it,
      whatever its heading; the crossing system's determinant b_y V vanishes there in the limit and its tau
      would be the distance to a crossing the ship is already on -- e.g. a nominal own ship turning in place
      on a head-on line at the episode start): the line runs along the track and every phase
      meets it, so the contact phase is the CERTIFIED contact phase psi_c = t_c and tau is the time for the
      constant-velocity footprint to touch a disk at p(t_c) (1-D: the intruder's own motion cancels, see the G23c
      report); tau = inf when the disk is off the track (no reveal).
  Contact window: a contact whose location on the line lies more than CONTACT_WINDOW_M from the certified
  conflict point, |psi_c - t_c| |v| > CONTACT_WINDOW_M, is not a contact with THIS encounter (the straight
  extension of an earlier route leg crossing the same line far away -- G23c P1: path_following envs 7/17 crossed
  the line 123 m from the conflict point on their first leg -- or the leg before a bend): no reveal at that
  update. If the TTC was already within the dose while outside the window, the reveal happens at the first
  update inside the window and is stamped "late_contact_window" with the TTC obtained.
  The reveal phase is phi_r = psi_c - tau. The intruder is revealed at the first update at which
  tau <= T_d = ratio(dose) * t_avoid (frozen per-vessel t_avoid), the phases lie on the certified segment
  (phi_r >= lo, psi_c <= hi, below) and tau > 0. The reveal time is interpolated inside the update interval,
  t_r = t - (T_d - tau) (tau falls at 1 s/s under constant velocity), clamped to the previous update. From the
  reveal on the intruder moves on its line with its pinned speed: phase(t') = phi_r(t) + (t' - t) for t' >= t_r,
  held at the segment bounds, regardless of the own ship. |phi_r(t) - t| <= REVEAL_SNAP_S (the owner-approved
  0.25 ms unforced-window timing bound) snaps the phase to t: then the motion is the pinned one.
  Clamping: if tau <= T_d but phi_r < lo (the required start lies before the certified segment) the reveal waits
  until phi_r reaches lo and is stamped "clamped" with the TTC actually obtained; if the first update already has
  tau < T_d (the own ship starts inside the dose) the reveal happens at once, also "clamped". If psi_c > hi the
  contact would lie beyond the admitted stop point: no reveal at that update. Never revealed -> "not_revealed".
  Certified segment [lo, hi]: the admitted motion window [start_s, stop_s], extended backwards by at most
  EXTENSION_MAX_S (only when start_s = 0, i.e. not a static boundary) and forwards past the episode duration
  (when stop_s is None) as far as the line keeps STATIC_MARGIN_M (0.05 m, the margin the pools were admitted with,
  DynamicSuiteConfig.static_margin_m) from every static disk -- trajectory._forbidden_interval, the exact test of
  fit_motion_window.
  After the reveal: physical; detectable when LOS is clear and the surface range is <= min(sensor max range,
  max(detection range, surface range at the reveal update)) -- the pinned detection horizon, widened to the reveal
  range when the reveal happened farther out (fast own ships), capped by the sensor.
  Speed estimate (continuous): V = max(0, v_obs + min(k, 1) (v_nom(t*) - v_ref)), v_obs / v_ref the window means
  (0.5 s trapezoid) of the surge and of the nominal foot-point speed, v_nom(t*) the nominal speed at the foot point,
  k = max(v_obs, F) / max(v_ref, F). Where k <= 1 this is rule S's k * v_nom(t*) (release.py) exactly; it equals the
  nominal speed for the nominal own ship (k = 1; needed for the u_ref identity at D1, where the nominal may still
  accelerate within the window). The term min(k, 1) (v_nom - v_ref) is the nominal's window lag, weighted so that a
  stopped ship gets none and a faster ship no more than the nominal's own. Unlike rule S's F9 switch (window mean
  below NOMINAL_REST_FRACTION u_ref) it has no discontinuity: the switch made V jump 0.75 -> 1.42 m/s in one
  update for a BlueBoat starting from rest on a 3 m offset (G23c probe), revealing at TTC 2.96 t_avoid at D1.
  u_ref identity: for the nominal own ship the reveal falls on the unforced D1 leg (constant speed, straight), where
  the constant-velocity prediction is exact: E equals the admitted contact point, psi_c = t_c, tau = t_c - t, so
  t_r = reveal_s(dose), phi_r = t and every sample from the reveal on is the v2.5 one (positions and, the horizon
  being the pinned detection range there, visibility).
  occluder_exit candidates (none in the pinned pools): the same TTC reveal; the occluders keep blocking LOS after
  it. ACCELERATION_AWARE (False; sensitivity only) predicts the own track with the window acceleration (speed
  capped to [0, top speed]) instead of constant velocity.
Hold tasks (station keeping): the pinned schedule (as in rule S). Moving basin: excluded. D0: unchanged.
G23e: station keeping no longer meets the intruder during the hold: its encounter lies on the approach to the hold
circle (sk_approach.py) and is timed exactly like a transit encounter (plan kind "approach"); the stamp adds the
SK block release.sk_approach. The pinned-schedule branch (route None) now serves only pinned_v0.

G23c layer 3: rule d7_reveal_sync_v2 (HYBRID_RULE_VERSION) = everything above plus
  * hybrid predictor. While the own ship is in the CORRIDOR of the frozen nominal route -- foot point of rule S's
    monotone progress (end-normal rule) not beyond the conflict node, signed cross-track offset |e| <= w_c and
    heading within theta_c of the route tangent at the foot point -- its future path is the route shifted
    laterally by e, travelled at the speed estimate V: the time to the D1-leg start t_1 is d_pre / V (d_pre = the
    nominal route distance from the foot point to the t_1 node, rule S's arithmetic; 0 on the D1 leg), and the
    first contact on the (straight) offset D1 leg is S2's crossing / collinear solve from the virtual observer =
    the route pose at max(t*, t_1) shifted by e, heading = route tangent, speed V. tau = d_pre / V + tau_leg,
    psi_c from the leg solve. Outside the corridor S2's constant-velocity solve is used unchanged.
    Speed estimate of v2: V = max(0, v_obs + w (v_nom - v_ref)) with w = min(k, 1/k) (v1: min(k, 1)): the
    nominal's window lag is informative only as far as the ship keeps the nominal's pace (k = 1: full weight, the
    identity; a ship at three times the nominal pace gets a third, a stopped ship none); for k <= 1 it is still
    rule S's k v_nom.
    u_ref identity: for the nominal own ship e = 0 and the heading error 0 (up to the foot-point Newton, ~1e-13)
    and V = v_nom. On the D1 leg (constant speed u_1, straight) the virtual observer is the ship itself, so the
    prediction is S2's, exact. Before t_1: tau = d_pre / v_nom(t*) + (leg contact distance) / v_nom(t*); the
    pinned pools never have a nominal faster than its D1-leg speed (measured on every admitted key of the 10
    transit tasks x 2 vessels: max(v_nom - u_1) = 4.3e-9 m/s), so tau >= d_pre / u_1 + T_D1 (1 - 4.3e-9) > T_D1 >=
    T_d for d_pre >= u_1 / 120 (one update): no reveal before the D1 leg; the reveal falls on the leg at
    t* = t_c - T_d, exactly as in S2 / v2.5 (G23c P1-L3 checks it on 2236 cells).
  * vessel-scaled constants (no tuning on evaluation keys):
      w_c = r + hy + CORRIDOR_CLEARANCE_BEAMS * 2 hy  (intruder radius + half beam + the one-beam clearance that
            t_avoid opens: the half-width of the conflict lane);
      sin theta_c = min(1, w_c / (u_ref * DOSE_TTC_RATIOS[1] * t_avoid)) (a heading error that leaves the
            corridor within the longest (D1) prediction horizon at u_ref);
      contact window W = |v| DOSE_TTC_RATIOS[1] t_avoid + w_c (the line length the pinned intruder covers during
            the D1 interval plus the conflict-lane half-width) in place of the fixed 10 m of v1.
    W_C_SCALE, THETA_C_SCALE, WINDOW_SCALE (1.0) exist for the sensitivity study only.
  Disclosure (REV-G23c H4): the replacement of the nominal-paced corridor speed by route distance / V and the lag
  weight min(k, 1/k) were chosen after a smoke run on seed-42 envs 0-3 (development keys; seed 42 is a training
  candidate, never an evaluation seed). Layer 4 replaces min(k, 1/k) (below).

G23d layer 4 (REV-G23c fixes; rule version unchanged d7_reveal_sync_v2, constants block extended):
  * H1 head-on offset: inside the corridor, when the offset D1 leg is collinear with the line and the shifted
    footprint misses the disk at p(t_c) (|lateral| > hy + r, i.e. beyond the touch lane but within w_c), the reveal
    is timed by the time to the closest point of approach instead: tau_cpa = (q_x - c_x) / V from the virtual
    observer (q = p(t_c) in its body frame), psi_c = t_c; the intruder then passes abeam at the stamped lateral
    offset (inside the conflict lane, a near miss by construction). Cause "cpa_reveal"; release.cpa holds the
    offset. Outside the corridor (|e| > w_c, beyond one beam of clearance) a parallel offset is still never
    revealed.
  * H2 independent dose: at the reveal tick the stamp also records the instantaneous constant-velocity TTC
    (geometry.ttc_linear of the observer pose and surge against the revealed runtime object; for a cpa_reveal the
    constant-velocity time to the closest point of approach) and dose_achieved = |that / (dose x t_avoid) - 1| <=
    DOSE_ACHIEVED_TOL. ttc_at_reveal_s remains the rule's own (predicted) figure.
  * H3 labels: a v2 reveal whose first meeting with the line already had an independent TTC below
    LATE_MEETING_FRACTION x dose is "late_first_meeting"; a reveal whose finite independent TTC misses the dose
    while the own ship accelerates (|window acceleration| > ACCEL_GUARD_MPS2) is "clamped_accelerating"; both are
    clamped (dose not achievable for that ship). An infinite independent TTC (the instantaneous heading does not
    meet the intruder, e.g. a corridor reveal before a bend) leaves the cause and stamps dose_achieved false with
    ttc_cv_at_reveal_s null: the constant-velocity measure is undefined there, not missed.
  * speed estimate of v2: the finite-window bias correction V = v_obs + (v_now - v_first) / 2 (window-mean speed
    + window-mean acceleration x span / 2: exact for a linear speed ramp), calibrated by the same correction's
    residual on the nominal foot-point speed, r = v_ref + (vn_now - vn_first) / 2 - v_nom(t*) (zero on linear
    nominal stretches; it makes V equal v_nom for the nominal own ship, which the exact u_ref identity needs),
    clipped to [0, top speed]: V = clip(v_obs + (v_now - v_first) / 2 - r). It replaces min(k, 1/k) (no tuned
    number). SPEED_ESTIMATOR selects the alternatives for the comparison study only.
  * H9: the identity relies on the nominal never being faster than its D1-leg speed before t_1 (guaranteed by
    reference.build_transit_plan: desired speed = plan speed x cos(error) <= plan speed); asserted at
    construction (tolerance NOMINAL_LEG_SPEED_TOL_MPS) and stamped (release.nominal_max_excess_over_leg_speed_mps).
  Scope of the dose statement: the stamped dose is the rule's predicted TTC for a ship holding course and speed
  within the route corridor (constant velocity outside it); dose_achieved records the independent value; head-on
  encounters with a parallel offset beyond the corridor are not revealed.

G23e layer B (REV-G23d M1-M4, controller decision J7): rule d7_reveal_sync_v3 (V3_RULE_VERSION, the default);
d7_reveal_sync_v2 stays available unchanged as a diagnostic rule. Everything of v2 applies except:
  * M1/M2 speed estimate (V3_SPEED_ESTIMATOR = "mean_unless_trend_v1"): the measured surge only, never the nominal
    (no calibration term): the 0.5 s window mean (trapezoid, rule S's window) unless the window shows a
    statistically significant speed trend (least-squares slope over the window samples, |t| >= TREND_T_SIG = 3),
    then the latest measured surge; clipped to [0, top speed]. Rationale: the window mean is the low-noise choice
    for a ship holding its speed (REV-G23d E2: 0.96-1.00 within 5 % under white noise 0.05 m/s); it lags by about
    a x 0.25 s for a ship that changes speed, where the latest measured surge is unbiased (E2 accelerate-from-rest:
    instantaneous 1.00, window mean 0.33); the t test separates the two without a tuned acceleration threshold
    (for a noise-free trend the residual is zero and the latest sample is used). u_ref identity: on and near the
    D1 leg the nominal's estimate is its own measured speed (exactly when the window holds a trend, to rounding
    when it is flat), so the reveal stays at t_c - dose x t_avoid (checked on every admitted cell, G23e report).
    The alternatives "window_mean", "instantaneous" and "ls_endpoint" remain selectable for the comparison study
    once the GPU surge noise is measured (REV-G23d M2: UNVERIFIED today).
  * M3 independent dose: dose_achieved = |T_ind / (dose x t_avoid) - 1| <= DOSE_ACHIEVED_TOL with T_ind the
    acceleration-aware first-contact time of the observer at the reveal tick: straight along its heading from its
    measured surge (while accelerating the window's least-squares value at the latest sample, otherwise the mean of
    the last ACCEL_RECENT_S; INDEPENDENT_SPEED) with the window acceleration (the least-squares slope when significant and above
    ACCEL_GUARD_MPS2, else 0), speed held in [0, top speed], against the revealed intruder moving on its line;
    for a cpa_reveal the time of the closest point of approach of the same prediction. It reuses neither the
    rule's speed estimate nor its route. Undefined (no contact predicted within INDEPENDENT_HORIZON_DOSES x dose)
    is stamped dose_undefined = true, dose_achieved = false, and counted separately in the summary. The
    constant-velocity ttc_linear stays as a secondary field (ttc_cv_at_reveal_s). clamped_accelerating: the
    ship accelerates at the reveal (window acceleration as above) and T_ind misses the dose or is undefined.
    Limitation: the prediction uses the heading (Pose carries no course over ground), so a crabbing ship can
    give an undefined T_ind (REV-G23d E7).
  * M4 labels: after late_first_meeting and clamped_accelerating, any remaining ttc_reveal / cpa_reveal whose
    rule TTC (ttc_at_reveal) or defined T_ind is below LATE_MEETING_FRACTION x dose is "late_reveal" (clamped).
  * m1: release.cpa is reset at every update and stamped only for a cpa_reveal.
  * station keeping: the approach encounter of sk_approach.py (the v2 text's "hold tasks: pinned schedule" clause
    does not apply to v3).
"""

import math

from .geometry import point_clearance, rounded_box_entry, to_body
from .model import UNFORCED_TTC_TOLERANCE_S, DOSE_TTC_RATIOS, Pose
from .geometry import relative_body_motion, ttc_linear
from .release import ACCEL_GUARD_MPS2, ReleasedIntruder
from .release import (PACE_SPEED_FLOOR_MPS, PACE_WINDOW_S, ReleaseRoute, ReleaseState,
                      _Held)
from .trajectory import _forbidden_interval
from .sk_approach import stamp_block as sk_stamp_block
from .visibility import detectable, edge_range_and_occlusion

REVEAL_RULE_VERSION = "d7_reveal_sync_v1"
HYBRID_RULE_VERSION = "d7_reveal_sync_v2"   # G23c L3: S2 + hybrid corridor predictor + vessel-scaled window
CORRIDOR_CLEARANCE_BEAMS = 1.0  # the clearance t_avoid opens (one beam, model.py t_avoid definition)
W_C_SCALE = 1.0                 # sensitivity only (G23c L3 report); 1.0 in every run
THETA_C_SCALE = 1.0             # sensitivity only
WINDOW_SCALE = 1.0              # sensitivity only
# G23d L4
SPEED_ESTIMATOR = "bias_corrected_calibrated_v1"   # v2; alternatives for the comparison study only:
SPEED_ESTIMATORS = ("bias_corrected_calibrated_v1", "lag_weighted_v2", "window_mean_calibrated")
DOSE_ACHIEVED_TOL = 0.05        # |independent TTC / dose - 1| <= this: dose achieved (REV-G23c H2)
LATE_MEETING_FRACTION = 0.95    # first meeting below this x dose: late_first_meeting (REV-G23c H3)
NOMINAL_LEG_SPEED_TOL_MPS = 1.0e-6   # nominal speed before t_1 <= D1-leg speed + this (REV-G23c H9)
ZERO_SEGMENT_M2 = 1.0e-18        # squared chord of a zero-length (turn-in-place) route segment
V2_CAUSES = ("ttc_reveal", "clamped_start", "clamped_late", "late_contact_window", "cpa_reveal",
             "late_first_meeting", "clamped_accelerating")
CLAMPED_CAUSES = ("clamped_start", "clamped_late", "late_contact_window", "late_first_meeting",
                  "clamped_accelerating")
# G23e layer B (REV-G23d M1-M4): rule d7_reveal_sync_v3 (module docstring)
V3_RULE_VERSION = "d7_reveal_sync_v3"
V3_SPEED_ESTIMATOR = "mean_unless_trend_v1"      # named switch (REV-G23d M2); alternatives for the study only:
V3_SPEED_ESTIMATORS = ("mean_unless_trend_v1", "window_mean", "instantaneous", "ls_endpoint")
TREND_T_SIG = 3.0               # |t| of the window's least-squares speed slope that counts as a trend
ACCEL_RECENT_S = 0.125          # "still accelerating" test: besides the window's LS slope (significant, above the
                                # guard), the last ACCEL_RECENT_S must show a significant (|t| >= TREND_T_SIG) slope
                                # of the same sign above the guard; otherwise the speed has "levelled off"
INDEPENDENT_STEP_S = 0.02       # scan step of the acceleration-aware first-contact search (then bisection)
INDEPENDENT_SPEED = "ls_or_recent_mean_v1"   # its initial speed: the window LS value at the latest sample while
                                # accelerating, else the mean of the last ACCEL_RECENT_S; "instantaneous" = the first
                                # G23e version (single raw sample), kept for the comparison (outcome-informed change,
                                # disclosed in the G23e report)
INDEPENDENT_HORIZON_DOSES = 3.0  # search horizon in units of dose x t_avoid
V3_CAUSES = V2_CAUSES + ("late_reveal",)
V3_CLAMPED_CAUSES = CLAMPED_CAUSES + ("late_reveal",)
COLLINEAR_SIN = 0.05            # |sin| of the track / line angle below which the geometry is collinear
EXTENSION_MAX_S = 4.0           # backward extension of the certified segment (2 m at 0.5 m/s)
STATIC_MARGIN_M = 0.05          # = DynamicSuiteConfig.static_margin_m of the pinned pools
REVEAL_SNAP_S = UNFORCED_TTC_TOLERANCE_S
REVEAL_DECISION_TOL_S = 1.0e-6  # the reveal decision itself (tau <= T_d) admits only rounding, never the snap
ACCELERATION_AWARE = False      # sensitivity only (G23c report); never True in a run
CONTACT_WINDOW_M = 10.0         # contact location within this distance (along the line) of the conflict point
CAUSES = ("ttc_reveal", "clamped_start", "clamped_late", "late_contact_window")
RULE_TEXT = ("hidden (not detectable, not collidable) until the constant-velocity first-contact TTC of the own "
             "ship against the intruder placed on its certified line (aim offset kept; collinear geometry: the "
             "certified contact phase) reaches dose x t_avoid; then revealed at that phase, already moving, on its "
             "pinned line and speed regardless of the ship; clamped when the required phase lies before the "
             "certified segment or the ship starts inside the dose; hold tasks: pinned schedule; moving basin: "
             "native axis, excluded")


def rule_constants():
    return {"collinear_sin": COLLINEAR_SIN, "extension_max_s": EXTENSION_MAX_S, "static_margin_m": STATIC_MARGIN_M,
            "reveal_snap_s": REVEAL_SNAP_S, "reveal_decision_tol_s": REVEAL_DECISION_TOL_S,
            "pace_window_s": PACE_WINDOW_S,
            "pace_speed_floor_mps": PACE_SPEED_FLOOR_MPS, "speed_estimate": "lag_weighted_window_mean_v1",
            "acceleration_aware": ACCELERATION_AWARE, "contact_window_m": CONTACT_WINDOW_M,
            "dose_ttc_ratios": list(DOSE_TTC_RATIOS[1:])}


HYBRID_TEXT = ("as d7_reveal_sync_v1, except: inside the corridor of the frozen route (|cross-track| <= w_c = r + hy + "
               "2 hy, heading within theta_c, sin theta_c = w_c / (u_ref 4 t_avoid), monotone foot point before the "
               "conflict node) the first-contact time is predicted along the route shifted by the cross-track offset "
               "at the speed estimate (route distance to the D1 leg / V, then the S2 solve on the offset D1 leg); "
               "contact window |v| 4 t_avoid + w_c instead of 10 m; G23d: speed = window mean + window-mean "
               "acceleration x span / 2, calibrated on the nominal, clipped to [0, top]; head-on offset inside the "
               "corridor beyond the touch lane: reveal at time-to-CPA = dose (cpa_reveal); stamps the independent "
               "constant-velocity TTC and dose_achieved (5 %); late_first_meeting / clamped_accelerating clamped")


def hybrid_constants():
    out = rule_constants()
    out.pop("contact_window_m")
    out["speed_estimate"] = SPEED_ESTIMATOR
    out.update({"dose_achieved_tol": DOSE_ACHIEVED_TOL, "late_meeting_fraction": LATE_MEETING_FRACTION,
                "accel_guard_mps2": ACCEL_GUARD_MPS2, "nominal_leg_speed_tol_mps": NOMINAL_LEG_SPEED_TOL_MPS,
                "head_on_offset": "cpa_reveal inside the corridor beyond the touch lane"})
    out.update({"corridor_clearance_beams": CORRIDOR_CLEARANCE_BEAMS,
                "corridor_half_width": "r + hy + corridor_clearance_beams * 2 hy",
                "corridor_heading": "asin(min(1, w_c / (u_ref * dose_ttc_ratios[0] * t_avoid)))",
                "contact_window": "|v| * dose_ttc_ratios[0] * t_avoid + w_c",
                "w_c_scale": W_C_SCALE, "theta_c_scale": THETA_C_SCALE, "window_scale": WINDOW_SCALE})
    return out


V3_TEXT = ("as d7_reveal_sync_v2, except (G23e, REV-G23d M1-M4): speed = 0.5 s window mean of the measured surge "
           "unless the window's least-squares speed slope is significant (|t| >= 3), then the latest measured surge "
           "(no nominal calibration), clipped to [0, top]; dose_achieved from the acceleration-aware first-contact "
           "time along the heading at the reveal tick (window acceleration when significant and above the guard; "
           "time to CPA for a cpa_reveal), undefined counted apart, constant-velocity TTC kept as a secondary field; "
           "clamped_accelerating when the ship accelerates and that time misses the dose; late_reveal (clamped) when "
           "the rule's or the independent TTC is below 0.95 x dose; release.cpa only for a cpa_reveal; station "
           "keeping: approach encounter (sk_approach_crossing_v1) instead of the pinned schedule")


def v3_constants():
    out = hybrid_constants()
    out.update({"speed_estimate": V3_SPEED_ESTIMATOR, "trend_t_sig": TREND_T_SIG, "accel_recent_s": ACCEL_RECENT_S,
                "independent_measure": "acceleration-aware first contact along the heading (time to CPA for "
                                       "cpa_reveal)", "independent_step_s": INDEPENDENT_STEP_S,
                "independent_horizon_doses": INDEPENDENT_HORIZON_DOSES, "independent_speed": INDEPENDENT_SPEED,
                "late_reveal": "rule or independent TTC < late_meeting_fraction x dose"})
    return out


def rule_block(hybrid=False, v3=False):
    if v3:
        return {"version": V3_RULE_VERSION, "constants": v3_constants(), "causes": list(V3_CAUSES), "text": V3_TEXT}
    if hybrid:
        return {"version": HYBRID_RULE_VERSION, "constants": hybrid_constants(), "causes": list(V2_CAUSES),
                "text": HYBRID_TEXT}
    return {"version": REVEAL_RULE_VERSION, "constants": rule_constants(), "causes": list(CAUSES),
            "text": RULE_TEXT}


def certified_segment(intruder, static_obstacles, duration_s):
    """(lo, hi): the admitted motion window, extended as stated in the module docstring."""
    core_lo = float(intruder.start_s)
    core_hi = float(intruder.stop_s) if intruder.stop_s is not None else float(duration_s)
    lo = core_lo
    if core_lo == 0.0:
        lo = -EXTENSION_MAX_S
    hi = float(intruder.stop_s) if intruder.stop_s is not None else math.inf
    for circle in static_obstacles:
        window = _forbidden_interval(intruder, circle, STATIC_MARGIN_M)
        if window is None:
            continue
        low, high = window
        if high <= core_lo:
            lo = max(lo, high)
        elif low >= core_hi:
            hi = min(hi, low)
    return min(lo, core_lo), max(hi, core_hi)


class RevealState(ReleaseState):
    """Per env row and episode under S2 (shares rule S's route, pace window and stamps' interface)."""

    def __init__(self, candidate, plan_kind, hybrid=False, v3=False):
        super().__init__(candidate, plan_kind, rule="d7_release_sync_v1")
        self.v3 = bool(v3)                           # G23e layer B: d7_reveal_sync_v3 (hybrid + M1-M4)
        self.hybrid = bool(hybrid) or self.v3
        self.rule = V3_RULE_VERSION if self.v3 else (HYBRID_RULE_VERSION if self.hybrid else REVEAL_RULE_VERSION)
        self.trend = (0.0, 0.0, 0.0, 0.0)
        self.indep_v0 = None
        encounter = candidate.request.encounter
        envelope = candidate.request.envelope
        self.footprint = envelope.footprint
        self.t_avoid = float(envelope.t_avoid_s)
        self.dose = int(encounter.dose)
        self.ttc_dose = DOSE_TTC_RATIOS[self.dose] * self.t_avoid
        self.aim = float(encounter.meta("aim_offset_fraction", 0.0))
        self.collinear_kind = encounter.meta("path_type") in ("head_on", "overtaking")
        self.lo, self.hi = certified_segment(self.obj, candidate.request.static_obstacles, encounter.duration_s)
        self.sk = sk_stamp_block(encounter)          # G23e: None for every task but station keeping
        self.reveal_s = None
        self.reveal_tick_s = None
        self.phase_at_tick = None
        self.snapped = False
        self.cause = None
        self.reveal_edge = None
        self.horizon = None
        self.ttc_at_reveal = None
        self.snapshot = None
        self.evaluations = 0
        self.met_track = False
        self.pending_clamp = False
        self.pending_window = False
        self.outside_window_updates = 0
        self.line_speed = math.hypot(self.obj.vx, self.obj.vy)
        # G23c L3: vessel-scaled corridor and contact window (module docstring); v1 keeps the fixed 10 m window
        hy = float(self.footprint.half_extents[1])
        base_w = self.obj.radius_m + hy + CORRIDOR_CLEARANCE_BEAMS * 2.0 * hy
        horizon = self.u_ref * DOSE_TTC_RATIOS[1] * self.t_avoid
        self.w_c = W_C_SCALE * base_w
        self.theta_c = min(math.pi, THETA_C_SCALE * math.asin(min(1.0, base_w / horizon)))
        self.window_m = (WINDOW_SCALE * (self.line_speed * DOSE_TTC_RATIOS[1] * self.t_avoid + base_w)
                         if self.hybrid else CONTACT_WINDOW_M)
        self.hybrid_updates = 0
        self.corridor_state = None
        # G23d L4
        self.cpa = None
        self.had_meeting = False
        self.independent = None
        self.nominal_excess = None
        if self.route is not None:                 # H9: identity basis (reference.build_transit_plan)
            nodes = self.route.nodes
            leg_speed = nodes[-2].pose.speed
            self.nominal_excess = max(n.pose.speed for n in nodes[:-1]) - leg_speed
            if self.nominal_excess > NOMINAL_LEG_SPEED_TOL_MPS:
                raise ValueError("D7 candidate violates the nominal-speed property (nominal %.3g m/s faster than "
                                 "its D1-leg speed before t_1)" % self.nominal_excess)

    # ------------------------------------------------------------------ geometry --
    def _progress(self, q):
        """G23d (v2): rule S's monotone end-normal progress, first passing ZERO-LENGTH route segments (the nominal's
        turn-in-place at the start): their end normal points along the in-place heading, so a laterally offset own
        ship never passed it and stayed on segment 0 for the whole episode (head-on offset probe, iceberg / ring /
        forced_crossing). They carry no route distance; for the nominal own ship, which sits on them, nothing
        changes (it passes them at the same update)."""
        if self.hybrid and self.route is not None:
            nodes = self.route.nodes
            last = len(nodes) - 2
            while self.segment < last:
                a, b = nodes[self.segment].pose, nodes[self.segment + 1].pose
                if (b.x - a.x) ** 2 + (b.y - a.y) ** 2 > ZERO_SEGMENT_M2:
                    break
                self.segment += 1
        return super()._progress(q)

    def _speed(self, time_s, observer):
        """V = max(0, v_obs + min(k, 1) (v_nom(t*) - v_ref)) (module docstring; = k v_nom(t*) for k <= 1)."""
        index, tau_seg = self._progress(observer)
        before, after = self.route.nodes[index], self.route.nodes[index + 1]
        v_nom = before.pose.speed + after.accel_mps2 * tau_seg
        v_obs, v_ref, accel = self._push(time_s, observer.speed, v_nom)
        k = max(v_obs, PACE_SPEED_FLOOR_MPS) / max(v_ref, PACE_SPEED_FLOOR_MPS)
        if k <= 1.0 and v_ref >= PACE_SPEED_FLOOR_MPS and v_obs >= PACE_SPEED_FLOOR_MPS:
            return k * v_nom, accel                     # rule S's arithmetic (bit-identical where it applies)
        if self.hybrid and SPEED_ESTIMATOR == "lag_weighted_v2":            # G23c L3 (comparison only)
            return max(0.0, v_obs + min(k, 1.0 / k) * (v_nom - v_ref)), accel
        return max(0.0, v_obs + min(k, 1.0) * (v_nom - v_ref)), accel

    def _speed_v2(self, time_s, observer):
        """G23d L4: finite-window bias-corrected speed, calibrated on the nominal (module docstring)."""
        if SPEED_ESTIMATOR == "lag_weighted_v2":
            return self._speed(time_s, observer)
        index, tau_seg = self._progress(observer)
        before, after = self.route.nodes[index], self.route.nodes[index + 1]
        v_nom = before.pose.speed + after.accel_mps2 * tau_seg
        v_obs, v_ref, accel = self._push(time_s, observer.speed, v_nom)
        w = self.window
        if SPEED_ESTIMATOR == "window_mean_calibrated":
            corr = corr_nom = 0.0
        else:
            corr, corr_nom = 0.5 * (w[-1][1] - w[0][1]), 0.5 * (w[-1][2] - w[0][2])
        residual = v_ref + corr_nom - v_nom
        return min(max(v_obs + corr - residual, 0.0), max(self.top_speed, 0.0)), accel

    def _trend(self, span_s=None):
        """(slope, t) of the least-squares line through the window's measured surge samples (G23e layer B); with
        span_s only the samples of the last span_s seconds."""
        w = self.window if span_s is None else [s for s in self.window if s[0] >= self.window[-1][0] - span_s - 1e-12]
        n = len(w)
        if n < 3:
            return 0.0, 0.0, None
        tbar = sum(s[0] for s in w) / n
        vbar = sum(s[1] for s in w) / n
        sxx = sum((s[0] - tbar) ** 2 for s in w)
        if sxx <= 0.0:
            return 0.0, 0.0, None
        slope = sum((s[0] - tbar) * (s[1] - vbar) for s in w) / sxx
        resid = sum((s[1] - vbar - slope * (s[0] - tbar)) ** 2 for s in w)
        se = math.sqrt(resid / (n - 2) / sxx)
        if slope == 0.0:
            return 0.0, 0.0, se
        return slope, (math.copysign(math.inf, slope) if se == 0.0 else slope / se), se

    def _speed_v3(self, time_s, observer):
        """G23e layer B (REV-G23d M1/M2): measured surge only (module docstring); returns (V, accel or 0)."""
        index, tau_seg = self._progress(observer)       # route progress (the corridor predictor's state)
        before, after = self.route.nodes[index], self.route.nodes[index + 1]
        v_nom = before.pose.speed + after.accel_mps2 * tau_seg
        v_mean, _, _ = self._push(time_s, observer.speed, v_nom)
        slope, tstat, _ = self._trend()
        trend = slope != 0.0 and abs(tstat) >= TREND_T_SIG
        if V3_SPEED_ESTIMATOR == "window_mean":
            value = v_mean
        elif V3_SPEED_ESTIMATOR == "instantaneous":
            value = observer.speed
        elif V3_SPEED_ESTIMATOR == "ls_endpoint":
            w = self.window
            n = len(w)
            tbar = sum(s[0] for s in w) / n
            vbar = sum(s[1] for s in w) / n
            value = vbar + slope * (w[-1][0] - tbar)
        else:
            value = observer.speed if trend else v_mean
        accelerating = trend and abs(slope) > ACCEL_GUARD_MPS2
        recent, t_recent, se_recent = self._trend(ACCEL_RECENT_S)
        levelled = None
        if accelerating and se_recent is not None:     # has the speed just levelled off?
            # the last ACCEL_RECENT_S show no material, significant trend of the same sign
            levelled = not (recent * slope > 0.0 and abs(recent) > ACCEL_GUARD_MPS2 and abs(t_recent) >= TREND_T_SIG)
            accelerating = not levelled
        accel = slope if accelerating else 0.0
        self.trend = (slope, tstat, recent, levelled)
        # speed of the independent measure (INDEPENDENT_SPEED): while accelerating, the window's least-squares value
        # at the latest sample (exact on a ramp, 0.26 sigma under white noise); otherwise the mean of the last
        # ACCEL_RECENT_S (0.26 sigma at 120 Hz; follows a speed that has just levelled off or recovered from a dip)
        w = self.window
        if accelerating and len(w) >= 3:
            tbar = sum(s[0] for s in w) / len(w)
            self.indep_v0 = sum(s[1] for s in w) / len(w) + slope * (w[-1][0] - tbar)
        else:
            tail = [s[1] for s in w if s[0] >= w[-1][0] - ACCEL_RECENT_S - 1e-12]
            self.indep_v0 = sum(tail) / len(tail)
        return min(max(value, 0.0), max(self.top_speed, 0.0)), accel

    def _travel_time(self, distance, speed, accel):
        if distance < 0.0:
            return -1.0
        if not ACCELERATION_AWARE or accel is None or abs(accel) < 1e-9:
            return distance / speed if speed > PACE_SPEED_FLOOR_MPS else math.inf
        top = max(self.top_speed, speed)
        disc = speed * speed + 2.0 * accel * distance
        if accel < 0.0:
            return math.inf if disc < 0.0 else (-speed + math.sqrt(disc)) / accel
        ramp = (top - speed) / accel
        ramp_distance = speed * ramp + 0.5 * accel * ramp * ramp
        if distance <= ramp_distance:
            return (-speed + math.sqrt(disc)) / accel
        return ramp + (distance - ramp_distance) / top

    def solve(self, observer, speed, accel=None):
        """(psi_c, tau, branch): G23c L3 hybrid inside the corridor, else the constant-velocity solve."""
        if self.hybrid and self.route is not None:
            out = self._hybrid(observer, speed)
            if out is not None:
                return out
        self.corridor_state = None
        return self._solve_cv(observer, speed, accel)

    def _hybrid(self, observer, speed):
        """Route-corridor prediction (module docstring, layer 3); None outside the corridor."""
        route = self.route
        index, tau_seg = self._progress(observer)
        nodes = route.nodes
        before, after = nodes[index], nodes[index + 1]
        foot = route.segment_pose(index, tau_seg)
        c, s = math.cos(foot.yaw), math.sin(foot.yaw)
        dx, dy = observer.x - foot.x, observer.y - foot.y
        along, cross = dx * c + dy * s, -dx * s + dy * c
        end = nodes[-1].pose
        beyond = (observer.x - end.x) * math.cos(end.yaw) + (observer.y - end.y) * math.sin(end.yaw)
        heading = math.atan2(math.sin(observer.yaw - foot.yaw), math.cos(observer.yaw - foot.yaw))
        if beyond > 0.0 or abs(cross) > self.w_c or abs(heading) > self.theta_c:
            return None
        t_star = before.time_s + tau_seg
        self.hybrid_updates += 1
        leg = nodes[-2]                                   # the D1-leg start node (t_1); the leg is straight
        d_pre = max(0.0, route._dist[-2] - route.distance_in_segment(index, tau_seg)) if t_star < leg.time_s else 0.0
        self.corridor_state = {"cross_track_m": cross, "along_m": along, "heading_error_rad": heading,
                               "nominal_time_at_foot_s": t_star, "route_distance_to_leg_m": d_pre}
        if speed < PACE_SPEED_FLOOR_MPS:
            return self.t_c, math.inf, "hybrid"
        base = leg.pose if t_star < leg.time_s else foot
        virtual = Pose(base.x - cross * math.sin(base.yaw), base.y + cross * math.cos(base.yaw), base.yaw, speed)
        psi_c, tau, branch = self._solve_cv(virtual, speed, None)
        self.cpa = None
        if branch == "collinear" and not math.isfinite(tau):
            # G23d H1: parallel offset beyond the touch lane, inside the corridor: time to the closest point of
            # approach of the disk at p(t_c) (it then passes abeam at the lateral offset, a near miss)
            px, py = self._point(self.t_c)
            cb, sb = math.cos(base.yaw), math.sin(base.yaw)
            qx = (px - virtual.x) * cb + (py - virtual.y) * sb
            qy = -(px - virtual.x) * sb + (py - virtual.y) * cb
            cx, cy = self.footprint.center
            t_cpa = (qx - cx) / speed
            if t_cpa > 0.0:
                self.cpa = {"lateral_offset_m": qy - cy, "cross_track_m": cross}
                psi_c, tau, branch = self.t_c, t_cpa, "collinear_cpa"
        pre = d_pre / speed
        return psi_c, pre + tau, "hybrid_" + branch

    def _solve_cv(self, observer, speed, accel=None):
        """(psi_c, tau, branch) for the own track at this update; tau = inf when the track misses."""
        c, s = math.cos(observer.yaw), math.sin(observer.yaw)
        vx, vy = self.obj.vx, self.obj.vy
        bx, by = c * vx + s * vy, -s * vx + c * vy
        bnorm = math.hypot(bx, by)
        fp = self.footprint
        cx, cy = fp.center
        hx, hy = fp.half_extents
        r = self.obj.radius_m
        dx0, dy0 = self.obj.x0 - observer.x, self.obj.y0 - observer.y
        p0x, p0y = c * dx0 + s * dy0, -s * dx0 + c * dy0          # R^T (P0 - q)
        lane = fp.reach_m + math.hypot(cx, cy) + r
        on_line = bnorm > 0.0 and abs(bx * (cy - p0y) - by * (cx - p0x)) < lane * bnorm
        collinear = bnorm == 0.0 or abs(by) < COLLINEAR_SIN * bnorm or on_line
        if not collinear and speed >= PACE_SPEED_FLOOR_MPS:
            wx, wy = bx - speed, by
            wn = math.hypot(wx, wy)
            nx, ny = wx / wn, wy / wn
            half_width = hx * abs(ny) + hy * abs(nx) + r
            lateral = self.aim * half_width
            px, py = -ny * lateral, nx * lateral
            back = fp.reach_m + math.hypot(cx, cy) + r + abs(lateral) + 1.0
            entry = rounded_box_entry(px - nx * back, py - ny * back, nx, ny, hx, hy, r)
            if not math.isfinite(entry):
                return None, math.inf, "crossing"
            ex = cx + px - nx * back + nx * entry
            ey = cy + py - ny * back + ny * entry
            ddx, ddy = ex - p0x, ey - p0y
            psi_c = ddy / by
            distance = bx * psi_c - ddx                  # along-heading travel of the own ship to the contact
            return psi_c, self._travel_time(distance, speed, accel), "crossing"
        # collinear: certified contact phase; constant-velocity footprint against a disk at p(t_c)
        psi_c = self.t_c
        qx, qy = p0x + bx * psi_c - cx, p0y + by * psi_c - cy
        if speed < PACE_SPEED_FLOOR_MPS:
            return psi_c, math.inf, "collinear"
        hit = rounded_box_entry(qx, qy, -1.0, 0.0, hx, hy, r)   # distance along -x of the relative disk
        if not math.isfinite(hit):
            return psi_c, math.inf, "collinear"
        return psi_c, self._travel_time(hit, speed, accel), "collinear"

    # -------------------------------------------------------------------- motion --
    def phase(self, time_s):
        if self.route is None:
            return time_s
        if self.reveal_s is None or time_s < self.reveal_tick_s:
            return time_s                                # hidden: pinned position, bookkeeping only
        if self.snapped:
            raw = time_s
        else:
            raw = self.phase_at_tick + (time_s - self.reveal_tick_s)
        return min(max(raw, self.lo), self.hi)

    def released_at(self, time_s):
        if self.route is None:
            return time_s >= self.obj.start_s
        return self.reveal_s is not None and time_s >= self.reveal_tick_s

    def _point(self, phase):
        """Position on the certified line at phase (the pinned arithmetic inside the admitted window, the
        extended line on [lo, start_s) and (stop_s, hi])."""
        obj = self.obj
        if phase >= obj.start_s and (obj.stop_s is None or phase <= obj.stop_s):
            return obj.position(phase)
        return obj.x0 + obj.vx * phase, obj.y0 + obj.vy * phase

    def line_position(self, time_s):
        if self.route is None or not self.released_at(time_s):
            return self.obj.position(time_s)              # pinned schedule / hidden (bookkeeping only)
        return self._point(self.phase(time_s))

    def line_velocity(self, time_s):
        if self.route is None:
            return self.obj.velocity(time_s)
        if not self.released_at(time_s):
            return 0.0, 0.0
        if self.snapped:
            raw = time_s
        else:
            raw = self.phase_at_tick + (time_s - self.reveal_tick_s)
        if raw < self.lo or raw >= self.hi:
            return 0.0, 0.0
        return self.obj.vx, self.obj.vy

    def unreleased(self):
        return self.route is not None and self.reveal_s is None

    def unreleased_reason(self):
        return "not_revealed"

    def present(self, time_s):
        return self.route is None or self.released_at(time_s)

    def moving_since(self):
        return self.obj.start_s if self.route is None else self.reveal_s

    def designed_contact_s(self):
        if self.route is None:
            return self.t_c
        return None if self.reveal_s is None else self.reveal_s + self.ttc_at_reveal

    @property
    def conflict_pose(self):
        return None

    @property
    def release_s(self):                    # interface of ReleaseState (exposure, adapters)
        return self.reveal_s

    @release_s.setter
    def release_s(self, value):
        if value is not None:
            raise AttributeError("S2 sets reveal_s")

    def detect(self, observer, obj, time_s, evidence, sensor):
        if self.route is None:
            return detectable(observer, obj, time_s, evidence, sensor)
        if sensor is None or evidence is None or not self.released_at(time_s):
            return False
        edge, clearance = edge_range_and_occlusion(observer, obj, time_s, evidence.occluders)
        return edge <= self.horizon and clearance > 0.0

    def observe(self, time_s, observer, evidence=None, sensor=None):
        if self.route is None or self.reveal_s is not None:
            return False
        previous_time = self.last_time
        self.last_time = time_s
        if observer is None:
            return False
        if self.v3:
            self.cpa = None                              # G23e layer B (REV-G23d m1): no stale cpa block
            speed, accel = self._speed_v3(time_s, observer)
        else:
            speed, accel = self._speed_v2(time_s, observer) if self.hybrid else self._speed(time_s, observer)
        psi_c, tau, branch = self.solve(observer, speed, None if self.v3 else accel)
        self.evaluations += 1
        if psi_c is None or not math.isfinite(tau) or tau <= 0.0:
            self.updates_held += 1
            return False
        self.met_track = True
        phi_r = psi_c - tau
        within = tau <= self.ttc_dose + REVEAL_DECISION_TOL_S
        if abs(psi_c - self.t_c) * self.line_speed > self.window_m:
            self.updates_held += 1                  # a contact elsewhere on the line: not this encounter
            self.outside_window_updates += 1
            self.pending_window = self.pending_window or within
            return False
        prior_meeting = self.had_meeting
        if not within:
            self.pending_clamp = self.pending_window = False   # the ship left the dose: a fresh crossing later
            self.had_meeting = True                 # G23d H3: met the line with the TTC still above the dose
        if not within or psi_c > self.hi:
            self.updates_held += 1
            return False
        if phi_r < self.lo:
            self.updates_held += 1                  # the required start lies before the certified segment: wait
            self.pending_clamp = True
            return False
        first = previous_time is None
        if first and tau < self.ttc_dose - REVEAL_SNAP_S:
            cause, t_r = "clamped_start", time_s
        elif self.pending_clamp:
            cause, t_r = "clamped_late", time_s
        elif self.pending_window:
            cause, t_r = "late_contact_window", time_s
        else:
            cause = "ttc_reveal"
            lower = time_s if previous_time is None else previous_time
            t_r = max(lower, min(time_s, time_s - (self.ttc_dose - tau)))
        pinned = self.t_c - self.ttc_dose
        if (self.hybrid and abs(phi_r - time_s) <= REVEAL_SNAP_S and abs(t_r - pinned) <= REVEAL_SNAP_S
                and pinned > time_s):
            # G23d: a snapped (pinned) reveal time just after this tick (a tick-boundary tie, seen on held-out keys
            # path_hazard BlueBoat env 41 / WAM-V env 48: pinned 58.50000000000001 vs tick 58.5) is revealed at the
            # first update at or after it, as v2.5 shows the object (t >= reveal_s)
            self.updates_held += 1
            return False
        self.reveal_s = t_r
        self.reveal_tick_s = time_s
        self.phase_at_tick = phi_r
        self.snapped = abs(phi_r - time_s) <= REVEAL_SNAP_S
        if self.snapped and abs(t_r - (self.t_c - self.ttc_dose)) <= REVEAL_SNAP_S:
            self.reveal_s = self.t_c - self.ttc_dose
        self.ttc_at_reveal = tau + (time_s - t_r)
        if cause == "ttc_reveal" and self.hybrid and branch.endswith("_cpa"):
            cause = "cpa_reveal"
        if self.v3:                                      # G23e layer B (REV-G23d M3/M4)
            self.independent = self._independent_v3(time_s, observer, branch.endswith("_cpa"), accel)
            if cause in ("ttc_reveal", "cpa_reveal"):
                value = self.independent["value_s"]
                late = LATE_MEETING_FRACTION * self.ttc_dose
                if not prior_meeting and value is not None and value < late:
                    cause = "late_first_meeting"
                elif self.independent["accelerating"] and not self.independent["dose_achieved"]:
                    cause = "clamped_accelerating"
                elif self.ttc_at_reveal < late or (value is not None and value < late):
                    cause = "late_reveal"
        else:
            self.independent = self._independent(time_s, observer, branch.endswith("_cpa"))
        if self.hybrid and not self.v3 and cause in ("ttc_reveal", "cpa_reveal"):   # G23d H3
            value = self.independent["value_s"]
            if not prior_meeting and value is not None and value < LATE_MEETING_FRACTION * self.ttc_dose:
                cause = "late_first_meeting"
            elif (value is not None and not self.independent["dose_achieved"] and accel is not None
                  and abs(accel) > ACCEL_GUARD_MPS2):
                cause = "clamped_accelerating"
        self.cause = cause
        x, y = self._point(min(max(phi_r if not self.snapped else time_s, self.lo), self.hi))
        edge = max(0.0, math.hypot(x - observer.x, y - observer.y) - self.obj.radius_m)
        self.reveal_edge = edge
        detection = None if evidence is None else evidence.detection_range_m
        max_range = None if sensor is None else sensor.max_range_m
        if detection is not None and max_range is not None:
            self.horizon = min(max_range, max(detection, edge))
        self.snapshot = {"observer": {"x": observer.x, "y": observer.y, "yaw": observer.yaw, "speed": observer.speed},
                         "speed_estimate_mps": speed, "observer_accel_mps2": accel, "branch": branch,
                         "contact_phase_s": psi_c, "reveal_phase_s": phi_r, "ttc_at_update_s": tau,
                         "reveal_range_m": edge, "detection_horizon_m": self.horizon,
                         "sensor_limited": (detection is not None and max_range is not None and edge > max_range),
                         "corridor": self.corridor_state}
        return True

    def _independent(self, time_s, observer, cpa):
        """G23d H2: the instantaneous constant-velocity figure at the reveal tick, independent of the rule's speed
        estimate: geometry.ttc_linear (observer pose and surge, revealed runtime object), or for a cpa_reveal the
        constant-velocity time to the closest point of approach."""
        obj = ReleasedIntruder(self.obj, self)
        if cpa:
            bx, by, vx, vy = relative_body_motion(observer, obj, time_s)
            cx, cy = self.footprint.center
            v2 = vx * vx + vy * vy
            value = None if v2 <= 0.0 else -((bx - cx) * vx + (by - cy) * vy) / v2
            measure = "time_to_cpa_constant_velocity"
            value = value if value is not None and value > 0.0 else None
        else:
            ttc = ttc_linear(observer, self.footprint, obj, at_s=time_s)
            value = None if math.isinf(ttc) else ttc
            measure = "ttc_linear"
        ratio = None if value is None else value / self.ttc_dose
        return {"measure": measure, "value_s": value,
                "value_over_t_avoid": None if value is None else value / self.t_avoid,
                "ratio_to_dose": ratio,
                "dose_achieved": ratio is not None and abs(ratio - 1.0) <= DOSE_ACHIEVED_TOL}

    def _independent_v3(self, time_s, observer, cpa, accel):
        """G23e layer B (REV-G23d M3): acceleration-aware first contact (or CPA) of the observer at the reveal tick
        (module docstring); independent of the rule's speed estimate and route."""
        obj = ReleasedIntruder(self.obj, self)
        fp = self.footprint
        v0 = max(0.0, float(observer.speed if INDEPENDENT_SPEED == "instantaneous" or self.indep_v0 is None
                            else self.indep_v0))
        top = max(self.top_speed, v0)
        a = float(accel)
        c, s = math.cos(observer.yaw), math.sin(observer.yaw)

        def travelled(tau):
            if a > 0.0:
                ramp = (top - v0) / a
                if tau > ramp:
                    return v0 * ramp + 0.5 * a * ramp * ramp + top * (tau - ramp)
            elif a < 0.0:
                stop = v0 / -a
                if tau > stop:
                    return v0 * stop + 0.5 * a * stop * stop
            return v0 * tau + 0.5 * a * tau * tau

        def gap(tau):
            d = travelled(tau)
            x, y = obj.position(time_s + tau)
            return point_clearance(x, y, Pose(observer.x + d * c, observer.y + d * s, observer.yaw, v0), fp) \
                - self.obj.radius_m

        horizon = INDEPENDENT_HORIZON_DOSES * self.ttc_dose
        steps = max(1, int(math.ceil(horizon / INDEPENDENT_STEP_S)))
        value = None
        if cpa:
            best_t, best = 0.0, gap(0.0)
            for i in range(1, steps + 1):
                t = horizon * i / steps
                g = gap(t)
                if g < best:
                    best_t, best = t, g
            low, high = max(0.0, best_t - horizon / steps), min(horizon, best_t + horizon / steps)
            ratio = (math.sqrt(5.0) - 1.0) / 2.0
            for _ in range(60):
                p = high - ratio * (high - low)
                q = low + ratio * (high - low)
                if gap(p) <= gap(q):
                    high = q
                else:
                    low = p
            t = 0.5 * (low + high)
            value = t if t > 0.0 else None
            measure = "time_to_cpa_acceleration_aware_heading"
        else:
            measure = "ttc_acceleration_aware_heading"
            previous = 0.0
            if gap(0.0) <= 0.0:
                value = 0.0
            else:
                for i in range(1, steps + 1):
                    t = horizon * i / steps
                    if gap(t) <= 0.0:
                        low, high = previous, t
                        for _ in range(50):
                            mid = 0.5 * (low + high)
                            if gap(mid) <= 0.0:
                                high = mid
                            else:
                                low = mid
                        value = high
                        break
                    previous = t
        cv = ttc_linear(observer, fp, obj, at_s=time_s)
        ratio_d = None if value is None else value / self.ttc_dose
        return {"measure": measure, "value_s": value,
                "value_over_t_avoid": None if value is None else value / self.t_avoid,
                "ratio_to_dose": ratio_d, "undefined": value is None,
                "dose_achieved": ratio_d is not None and abs(ratio_d - 1.0) <= DOSE_ACHIEVED_TOL,
                "accelerating": a != 0.0, "accel_mps2": a, "speed_mps": v0,
                # null = an exact trend (zero residual: |t| infinite, not JSON-representable)
                "speed_trend_t": self.trend[1] if math.isfinite(self.trend[1]) else None,
                "recent_slope_mps2": self.trend[2], "levelled_off": self.trend[3],
                "cv_value_s": None if math.isinf(cv) else cv}

    def stamp(self, footprint):
        if self.route is None:
            return {"rule": self.rule, "plan_kind": self.plan_kind, "applies": False,
                    "reason": "stationary own ship: pinned schedule (no reveal timing)"}
        out = {"rule": self.rule, "plan_kind": self.plan_kind, "applies": True,
               "release_time_s": self.reveal_s, "release_tick_time_s": self.reveal_tick_s, "cause": self.cause,
               "released": self.reveal_s is not None, "dose_ttc_s": self.ttc_dose,
               "ttc_at_reveal_s": self.ttc_at_reveal,
               "ttc_at_reveal_over_t_avoid": None if self.ttc_at_reveal is None else self.ttc_at_reveal / self.t_avoid,
               "clamped": self.cause in (V3_CLAMPED_CAUSES if self.v3 else CLAMPED_CAUSES),
               "outside_window_updates": self.outside_window_updates,
               "snapped_to_pinned": self.snapped, "certified_segment": [self.lo, None if math.isinf(self.hi)
                                                                      else self.hi],
               "conflict_time_s": self.t_c, "route_digest": self.route.digest, "updates_held": self.updates_held,
               "track_met_line": self.met_track, "state_at_reveal": self.snapshot}
        ind = self.independent or {}
        if self.v3:                                      # G23e layer B (REV-G23d M3, m1, m5)
            out.update({"ttc_independent_at_reveal_s": ind.get("value_s"),
                        "ttc_independent_over_t_avoid": ind.get("value_over_t_avoid"),
                        "independent_measure": ind.get("measure"), "dose_achieved": ind.get("dose_achieved"),
                        "dose_undefined": ind.get("undefined"), "dose_achieved_tol": DOSE_ACHIEVED_TOL,
                        "accelerating_at_reveal": ind.get("accelerating"),
                        "accel_at_reveal_mps2": ind.get("accel_mps2"), "speed_trend_t": ind.get("speed_trend_t"),
                        "independent_speed_mps": ind.get("speed_mps"),
                        "ttc_cv_at_reveal_s": ind.get("cv_value_s"),
                        "cpa": self.cpa if self.cause == "cpa_reveal" else None,
                        "nominal_max_excess_over_leg_speed_mps": self.nominal_excess,
                        "corridor_half_width_m": self.w_c, "corridor_heading_rad": self.theta_c,
                        "contact_window_m": self.window_m, "hybrid_updates": self.hybrid_updates})
            if self.sk is not None:
                out["sk_approach"] = self.sk
            return out
        out.update({"ttc_cv_at_reveal_s": ind.get("value_s"),
                    "ttc_cv_at_reveal_over_t_avoid": ind.get("value_over_t_avoid"),
                    "independent_measure": ind.get("measure"), "dose_achieved": ind.get("dose_achieved"),
                    "dose_achieved_tol": DOSE_ACHIEVED_TOL, "cpa": self.cpa,
                    "nominal_max_excess_over_leg_speed_mps": self.nominal_excess})
        if self.hybrid:                                  # G23c L3
            out.update({"corridor_half_width_m": self.w_c, "corridor_heading_rad": self.theta_c,
                        "contact_window_m": self.window_m, "hybrid_updates": self.hybrid_updates})
        if self.sk is not None:                          # G23e: station keeping approach design
            out["sk_approach"] = self.sk
        return out


def reveal_state(candidate, plan_kind, hybrid=False, v3=False):
    return RevealState(candidate, plan_kind, hybrid, v3)


__all__ = ["REVEAL_RULE_VERSION", "HYBRID_RULE_VERSION", "V3_RULE_VERSION", "RevealState", "rule_block", "rule_constants",
           "certified_segment",
           "reveal_state", "point_clearance", "to_body", "_Held"]
