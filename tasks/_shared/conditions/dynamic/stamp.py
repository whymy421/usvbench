"""JSON-native detached dynamic-hazard episode stamps with a canonical SHA-256 digest.

The digest is an integrity check of the payload, not a signature. Full
admission records (witness trajectories, rejected candidates) belong in the
admission ledger (AdmissionPool / ledger JSON); an episode stamp carries the
compact assignment summary plus its candidate/witness/pool digests.
"""
from dataclasses import asdict, is_dataclass
import hashlib
import json

SCHEMA_VERSION = 3        # E4: assignment gains sensor_source
MODEL_VERSION = "d7-kinematic-3"   # E4: envelopes from D5's final registry (centreline arm)

SEMANTICS = {
    "denominator": "every_assigned_episode (intention to treat)",
    "schedule": "pre-generated from (eval_seed, env, episode, hazard_group); exogenous episode seconds",
    "dose": ("TTC/t_avoid at the first detection of the frozen nominal own ship at u_ref (D1..D4 = 4, 3, 2, "
             "1.25); D0 = no intruder. Under pinned_v0 and d7_release_sync_v1 the runtime dose is a detection "
             "DISTANCE, so the realized TTC at detection scales with the closing speed (G23c F11)"),
    "t_avoid": "frozen per vessel: reaction + min(brake, max-rate turn) time to displace one beam",
    "reveal": ("physical detectability through the task's range sensor (its existing sensor, or the "
               "journal 36-ray 30 m sensor when journal_ray_sensor_all_tasks); no time gate"),
    "admission": "finite-search kinematic witness, replay-validated; joint across doses by default",
    "feasibility": "finite_search_kinematic_witness_not_simulator_proof",
    "collision": "physical before and after reveal; swept contact latched by the env bridge",
    "moving_basin": "native motion ratio is the dose; no added intruder",
    "vessel_values": ("D5 final registry (vehicles_v2.py), centreline thrust arm; reference speed 1.0 m/s "
                      "and turning radius R = L are PENDING owner decisions (CONTEXT_R3 decision 14)"),
}


def plain(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def canonical(value):
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("ascii")).hexdigest()


# G7 item A (2026-10-01): identity digest of a value that is RECOMPUTED on the run machine (e.g. a
# candidate rebuilt from the env's layout) and compared with the admission machine's copy. digest()
# hashes the exact repr of every float, so its last ulp decides; identity_digest() first quantises
# every finite float with the declared quantum (same rule and version as
# journal_geometry_v2.geometry_digest: q = round(x / 1e-6), half to even, "f:<q>").
IDENTITY_CANON_VERSION = "usvbench.identity.fixed_quantum_1e-6/1"
IDENTITY_QUANTUM = 1.0e-6
_IDENTITY_SCALE = 1.0e6


def identity_plain(value):
    value = plain(value)
    if isinstance(value, dict):
        return {str(k): identity_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [identity_plain(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return "nonfinite:%r" % (value,)
        return "f:%d" % round(value * _IDENTITY_SCALE)
    raise TypeError("identity digest cannot canonicalise %r" % (type(value).__name__,))


def identity_digest(value):
    text = json.dumps({"canon": IDENTITY_CANON_VERSION, "value": identity_plain(value)},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("ascii")).hexdigest()


# G23b (owner decision 2026-10-03, NC 09, option S): semantics entries of an ADMITTED stamp under the
# arrival-synchronized runtime rule (release.py). Not-admitted stamps (no intruder) keep SEMANTICS as is.
SEMANTICS_RELEASE_SYNC = {
    "schedule": ("pinned assignment from (eval_seed, env, episode, hazard_group); runtime rule "
                 "d7_release_sync_v1: the intruder rests at its hold point on the pinned line and is released "
                 "when the own ship's constant-velocity time to the conflict along the frozen nominal route "
                 "(route distance to go / window-paced speed) <= t_c - t_hold (docking: also its accelerated "
                 "arrival; always when the resting object becomes detectable); stationary tasks keep the pinned "
                 "schedule; moving basin native"),
    "release": "runtime.release: hold point, release time and cause, state at release, predicted closest approach",
    "finite_ttc_at_first_detection": ("runtime.finite_ttc_at_first_detection: a perception-timing statistic "
                                      "(the constant-velocity TTC at the first detection is finite), NOT a "
                                      "conflict; runtime.exposure_record holds the realized exposure"),
}


# G23c L2: semantics of an ADMITTED stamp under the candidate rule S2 (reveal.py)
SEMANTICS_REVEAL_SYNC = {
    "schedule": ("pinned assignment from (eval_seed, env, episode, hazard_group); runtime rule d7_reveal_sync_v1: "
                 "the intruder is hidden (not detectable, not collidable) until the own ship's constant-velocity "
                 "first-contact TTC against the intruder on its certified line reaches dose x t_avoid; then it is "
                 "revealed at that phase, moving on its pinned line and speed regardless of the ship; station "
                 "keeping keeps the pinned schedule; moving basin native"),
    "dose": ("TTC/t_avoid of the own ship at the reveal (D1..D4 = 4, 3, 2, 1.25) for a ship holding its course and "
             "speed; runtime.release.ttc_at_reveal_over_t_avoid records the rule's predicted TTC (clamped reveals are "
             "stamped); G23d: runtime.release.dose_achieved / ttc_cv_at_reveal_s give the independent "
             "constant-velocity value; head-on encounters with a parallel offset beyond the conflict lane are not "
             "revealed"),
    "clearance": ("G23c F12: in journal mode the D7 term of the env clearance (rewards AND certificate metrics "
                  "min_clearance_m, contact_before_goal / last_gate / dock) uses the DETECTABLE D7 objects only, so "
                  "an undetected object never shapes a reward or a clearance metric; collisions still reach the "
                  "contract through the bridge's swept-contact latch; under S2 / v2 a revealed object is detectable, "
                  "under the diagnostic rules (S, pinned_v0) the metric differs from a physical-mask clearance"),
    "collision": "physical from the reveal on (hidden before); swept contact latched by the env bridge",
    "release": ("runtime.release: reveal time and cause (ttc_reveal / clamped_start / clamped_late / "
                "late_contact_window), TTC at the reveal, "
                "certified segment, state at the reveal"),
    "finite_ttc_at_first_detection": SEMANTICS_RELEASE_SYNC["finite_ttc_at_first_detection"],
}


# G23c L3: rule d7_reveal_sync_v2 (S2 + hybrid corridor predictor)
SEMANTICS_REVEAL_HYBRID = dict(
    SEMANTICS_REVEAL_SYNC,
    schedule=("pinned assignment from (eval_seed, env, episode, hazard_group); runtime rule d7_reveal_sync_v2: as "
              "d7_reveal_sync_v1, with the own ship's first-contact time predicted along the frozen route shifted by "
              "its cross-track offset while it is in the route corridor (constant velocity outside); hidden until "
              "that time reaches dose x t_avoid, then revealed moving on its pinned line and speed; station keeping "
              "keeps the pinned schedule; moving basin native"),
    dose=("TTC/t_avoid of the own ship at the reveal (D1..D4 = 4, 3, 2, 1.25) for a ship holding its course and "
          "speed within the route corridor (predicted along the corridor, constant velocity outside it); "
          "runtime.release.ttc_at_reveal_over_t_avoid records the rule's predicted TTC; runtime.release.dose_achieved "
          "(independent constant-velocity TTC at the reveal tick within 5 %) the independent value; a head-on "
          "parallel offset inside the corridor but beyond the touch lane is revealed at time-to-CPA = dose "
          "(cause cpa_reveal, release.cpa = lateral offset; a near miss by construction); beyond the corridor it is "
          "not revealed; late_first_meeting / clamped_accelerating mark reveals whose dose the ship made "
          "unachievable"))


# G23e layer B: rule d7_reveal_sync_v3 (REV-G23d M1-M4)
SEMANTICS_REVEAL_V3 = dict(
    SEMANTICS_REVEAL_HYBRID,
    schedule=("pinned assignment from (eval_seed, env, episode, hazard_group); runtime rule d7_reveal_sync_v3: as "
              "d7_reveal_sync_v2 (first-contact time predicted along the route corridor, constant velocity outside), "
              "with the speed estimated from the measured surge only (0.5 s window mean, the latest sample when the "
              "window shows a significant trend); hidden until that time reaches dose x t_avoid, then revealed "
              "moving on its pinned line and speed; station keeping: the encounter lies on the approach to the hold "
              "circle (sk_approach_crossing_v1); moving basin native"),
    dose=("TTC/t_avoid of the own ship at the reveal (D1..D4 = 4, 3, 2, 1.25) for a ship holding its course and "
          "speed within the route corridor; runtime.release.ttc_at_reveal_over_t_avoid records the rule's predicted "
          "TTC; runtime.release.dose_achieved the independent acceleration-aware first-contact time along the "
          "heading at the reveal tick within 5 % (time to CPA for a cpa_reveal; dose_undefined when none is "
          "predicted; ttc_cv_at_reveal_s the constant-velocity value); late_first_meeting / clamped_accelerating / "
          "late_reveal (rule or independent TTC below 0.95 x dose) mark reveals whose dose the ship made "
          "unachievable"))


def semantics_for(rule):
    """Semantics entries an ADMITTED stamp adds for its runtime rule (G23c); None keeps SEMANTICS."""
    if rule == "d7_reveal_sync_v3":
        return SEMANTICS_REVEAL_V3
    if rule == "d7_reveal_sync_v2":
        return SEMANTICS_REVEAL_HYBRID
    if rule == "d7_reveal_sync_v1":
        return SEMANTICS_REVEAL_SYNC
    if rule == "d7_release_sync_v1":
        return SEMANTICS_RELEASE_SYNC
    return None


def make_stamp(assignment, runtime, semantics=None):
    """semantics: optional entries added to / replacing SEMANTICS in this stamp (G23b)."""
    payload = {"schema_version": SCHEMA_VERSION, "model_version": MODEL_VERSION,
               "assignment": plain(assignment), "runtime": plain(runtime),
               "semantics": dict(SEMANTICS, **(semantics or {}))}
    return json.loads(canonical({"payload": payload, "sha256": digest(payload)}))


def read_stamp(value):
    """Verify and round-trip JSON without importing or executing class names."""
    value = json.loads(value) if isinstance(value, str) else json.loads(canonical(value))
    if set(value) != {"payload", "sha256"}:
        raise ValueError("invalid dynamic stamp envelope")
    if digest(value["payload"]) != value["sha256"]:
        raise ValueError("dynamic stamp digest mismatch")
    payload = value["payload"]
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("model_version") != MODEL_VERSION:
        raise ValueError("unsupported dynamic stamp version")
    for field in ("assignment", "runtime", "semantics"):
        if field not in payload:
            raise ValueError("incomplete dynamic stamp")
    return value
