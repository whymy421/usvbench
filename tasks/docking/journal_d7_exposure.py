"""D16 not-admitted keys at runtime: the family-side D7 bridge wrapper (pure Python; no torch).

CONTEXT_R4 D16: a key that E4 rejected keeps its E5 scenario and runs the
dynamic suite at D0 - no intruder, an EMPTY assignment - at every requested
dose. E4's HazardBridge only accepts admitted candidates, and F0's mixin
hands every reset row to ``bridge.reset_rows(ids, {env: pool.assignment(key,
dose)})``; the docking pool returns a ``NotAdmittedAssignment`` for a
rejected key (journal_d7_pool.py), and ``ExposureBridge`` (installed by the
docking env over the mixin's bridge right after ``journal_setup``) routes:

  * admitted rows -> the E4 HazardBridge, unchanged (same samples, swept
    contact and E4 stamp, byte for byte);
  * not-admitted rows -> an empty row: no D7 object (empty samples, the D7
    slot stays unpublished), no contact, the tick clock checked exactly as E4
    does, and at ``finish`` a D7 stamp envelope {"payload", "sha256"} (E4
    schema/model version, E4 canonical digest) whose payload carries

        payload["exposure"] = "not_admitted"
        payload["not_admitted"] = {"reasons": [{dose, reason, sampled_states}],
                                   "top_reason", "requested_dose", "record_digest",
                                   "pool_digest", "rule"}
        payload["assignment"] = {key, task, dose: 0, exposure, requested_dose,
                                 candidate_digest: record_digest, witness_digest:
                                 None, geometry_id, vessel_id, intruders: [], ...}
        payload["runtime"]["non_exposure_reason"] = "not_admitted"
        payload["runtime"]["assigned"] = False, ["itt_count"] = 0

    so ``pairing.d7_primitive_hash`` (candidate digest, task, dose, key) is
    the same at every requested dose and equals the D0 run of every other
    suite: the record is unexposed for the dynamic treatment and pairs for
    every other suite. Admitted records carry no "exposure" field (E4 stamp
    unchanged); a reader treats a missing field as admitted.

CONTEXT_R4 D24: a key whose episode index is beyond the pool's offered grid
(env inside it, the pool's seed and group) arrives here as the same
``NotAdmittedAssignment`` with the single reason ``beyond_offered_grid``
(journal_d7_pool.BEYOND_GRID_REASONS); it runs the same empty D0 row and
stamps the same envelope, ``not_admitted.reasons`` =
[{"dose": 0, "reason": "beyond_offered_grid", "sampled_states": 0}] and
``not_admitted.rule`` = journal_d7_pool.BEYOND_GRID_RULE.
"""
from __future__ import annotations

from .journal_d7_pool import BEYOND_GRID_RULE, EXPOSURE_NOT_ADMITTED, NotAdmittedAssignment

EXPOSURE_RULE = ("CONTEXT_R4 D16: E4 rejected this key; the E5 scenario is kept (no redraw) and the dynamic "
                 "suite runs at D0 (no intruder, empty assignment) at every requested dose")


def _dynamic():
    from .._shared.conditions import dynamic
    return dynamic


def _plain_key(key):
    return {"eval_seed": key.eval_seed, "env": key.env, "episode": key.episode, "hazard_group": key.hazard_group}


def not_admitted_stamp(record, env, termination_reason, runtime_row, physics_dt_s):
    """D7 stamp envelope of one finished not-admitted episode (E4 digest rule)."""
    from .._shared.conditions.dynamic import stamp as e4_stamp
    reasons = [{"dose": int(d), "reason": r, "sampled_states": int(n)} for d, r, n in record.reasons]
    assignment = {
        "key": _plain_key(record.key), "task": "docking", "dose": 0, "exposure": EXPOSURE_NOT_ADMITTED,
        "requested_dose": record.requested_dose, "geometry_id": record.geometry_id, "vessel_id": record.vessel,
        "intruders": [], "candidate_digest": record.record_digest, "witness_digest": None,
        "not_admitted_reasons": reasons, "pool_digest": record.pool_digest,
    }
    runtime = {
        "assigned": False, "itt_count": 0, "env": int(env), "termination_reason": termination_reason,
        "last_time_s": max(0.0, runtime_row["last_tick"] * physics_dt_s),
        "first_detection_s": None, "realized_ttc_at_first_detection_s": None,
        "detected_samples": 0, "observer_updates": runtime_row["observer_updates"],
        "min_sampled_surface_clearance_m": None, "sampled_contact": False,
        "non_exposure_reason": EXPOSURE_NOT_ADMITTED, "exposure": EXPOSURE_NOT_ADMITTED,
        "bridge": {"swept_contact_latched": False, "first_swept_contact_interval_start_s": None,
                   "swept_intervals": runtime_row["swept_intervals"], "last_tick": runtime_row["last_tick"],
                   "physics_dt_s": physics_dt_s, "empty_assignment": True},
    }
    payload = {"schema_version": e4_stamp.SCHEMA_VERSION, "model_version": e4_stamp.MODEL_VERSION,
               "assignment": e4_stamp.plain(assignment), "runtime": e4_stamp.plain(runtime),
               "semantics": dict(e4_stamp.SEMANTICS), "exposure": EXPOSURE_NOT_ADMITTED,
               "not_admitted": {"reasons": reasons, "top_reason": record.top_reason,
                                "requested_dose": record.requested_dose, "record_digest": record.record_digest,
                                "pool_digest": record.pool_digest,
                                "rule": BEYOND_GRID_RULE if getattr(record, "beyond_grid", False) else EXPOSURE_RULE}}
    import json
    return json.loads(e4_stamp.canonical({"payload": payload, "sha256": e4_stamp.digest(payload)}))


class ExposureBridge:
    """HazardBridge wrapper: admitted rows -> E4 bridge; not-admitted rows -> empty D0 rows (D16)."""

    def __init__(self, inner):
        self.inner = inner
        self.adapter = inner.adapter
        self.cfg = inner.cfg
        self.dt = inner.dt
        self.own_speed_bound = inner.own_speed_bound
        self._empty = {}          # env -> row dict of the CURRENT not-admitted episode
        self._mode = {}           # env -> "admitted" | "not_admitted" (current episode)
        self.completed_not_admitted = []

    # rows view: the CURRENT episode's row per env (E4 row dicts are the same objects)
    @property
    def rows(self):
        out = {}
        for env, mode in self._mode.items():
            if mode == EXPOSURE_NOT_ADMITTED:
                out[env] = self._empty[env]
            elif env in self.inner.rows:
                out[env] = self.inner.rows[env]
        return out

    def exposure(self, env):
        return self._mode.get(int(env))

    def reset_rows(self, env_ids, assignments):
        ids = tuple(int(i) for i in env_ids)
        for env in ids:
            if env in self._mode and not self.rows[env]["finished"]:
                raise ValueError("finish and stamp every assigned episode before reset")
        admitted = {e: assignments[e] for e in ids if not isinstance(assignments[e], NotAdmittedAssignment)}
        empty = {e: assignments[e] for e in ids if isinstance(assignments[e], NotAdmittedAssignment)}
        for env, record in empty.items():
            if int(record.key.env) != env:
                raise ValueError("wrong environment for a not-admitted record")
        if admitted:
            self.inner.reset_rows(tuple(e for e in ids if e in admitted), admitted)
        for env in ids:
            if env in empty:
                self._empty[env] = {"record": empty[env], "last_tick": -1, "prev": None, "cache": None,
                                    "contact": False, "first_contact_s": None, "swept_intervals": 0,
                                    "observer_updates": 0, "finished": False, "stamp": None}
                self._mode[env] = EXPOSURE_NOT_ADMITTED
            else:
                self._mode[env] = "admitted"

    def step(self, ticks, poses):
        Pose = _dynamic().Pose
        admitted = {e: t for e, t in ticks.items() if self._mode.get(e) != EXPOSURE_NOT_ADMITTED}
        out = self.inner.step(admitted, poses) if admitted else {}
        for env, tick in ticks.items():
            if env in admitted:
                continue
            row = self._empty[env]
            if row["finished"]:
                raise ValueError("row finished; reset it before stepping")
            if type(tick) is not int or tick < 0:
                raise ValueError("tick must be a nonnegative integer")
            if tick == row["last_tick"]:
                out[env] = row["cache"]
                continue
            if tick < row["last_tick"]:
                raise ValueError("physics tick went backwards without a reset")
            if not isinstance(poses[env], Pose):
                raise TypeError("pose must be a Pose")
            if row["last_tick"] >= 0:
                row["swept_intervals"] += 1
            row["observer_updates"] += 1
            row["last_tick"] = tick
            row["cache"] = ()
            out[env] = ()
        return out

    def contact(self, env):
        if self._mode.get(env) == EXPOSURE_NOT_ADMITTED:
            return False
        return self.inner.contact(env)

    def finish(self, env, termination_reason):
        if self._mode.get(env) != EXPOSURE_NOT_ADMITTED:
            return self.inner.finish(env, termination_reason)
        if termination_reason not in _dynamic().TERMINATION_REASONS:
            raise ValueError("explicit terminal reason required: " + repr(termination_reason))
        row = self._empty[env]
        if row["finished"]:
            return row["stamp"]
        row["stamp"] = not_admitted_stamp(row["record"], env, termination_reason, row, self.dt)
        row["finished"] = True
        self.completed_not_admitted.append(row["stamp"])
        return row["stamp"]


def record_exposure(envelope):
    """"not_admitted" for a D16 record, "admitted" for an E4 stamp (reader helper, verifies the digest)."""
    from .._shared.conditions.certificate import verify_dynamic_hazard_stamp
    value = verify_dynamic_hazard_stamp(envelope)
    return value["payload"].get("exposure", "admitted")


__all__ = ["ExposureBridge", "not_admitted_stamp", "record_exposure", "EXPOSURE_RULE"]
