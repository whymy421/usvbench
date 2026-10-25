"""Docking level tables: which one is in force, snapshot and override refusal (D2 item 14).

REVIEW F14 (D2): under physics v2 the legacy env reads ``eval_level_distances_v2``
(docking_env.py:84-86) and an override of ``eval_level_distances`` was silently
ignored. E1 refuses a ``--set`` of the table that is not in force
(scripts/eval_protocol.py); this module lets the cfg itself expose the table in
force (``DockingEnvCfg.docking_level_table_in_force()``), refuse such an
override in ``validate_after_overrides`` and stamp the choice. Journal cfgs
(contract 2) read NO level table: the geometry is the frozen journal plan.
Import-light on purpose (no numpy/torch): the legacy cfg module imports it.
"""
from __future__ import annotations

# D2 item 14 / REVIEW F14: which docking level table is in force.
LEVEL_TABLE_V1 = "eval_level_distances"
LEVEL_TABLE_V2 = "eval_level_distances_v2"
LEVEL_TABLE_JOURNAL = "journal_plan"          # neither table: geometry from the frozen journal plan
LEVEL_FIELDS = ("eval_level", "eval_level_distances", "eval_level_distances_v2",
                "spawn_distance_override_m", "curriculum_frozen", "curriculum_start_distance_m",
                "curriculum_distance_increment_m", "curriculum_max_distance_m")


def level_table_in_force(cfg):
    """D2 item 14: name of the docking level table the env reads for this cfg.

    journal cfgs (contract 2) read no level table (the frozen journal plan);
    legacy cfgs read eval_level_distances_v2 under physics v2 and
    eval_level_distances otherwise (docking_env.py:84-86).
    """
    if getattr(cfg, "journal_contract_version", None) == 2:
        return LEVEL_TABLE_JOURNAL
    physics = getattr(cfg, "underwater_physics_cfg", None)
    version = getattr(physics, "physics_version", 1)
    return LEVEL_TABLE_V2 if int(version) == 2 else LEVEL_TABLE_V1


def level_snapshot(cfg):
    """Values of every level/curriculum field (compared after overrides)."""
    out = {}
    for name in LEVEL_FIELDS:
        value = getattr(cfg, name, None)
        out[name] = tuple(value) if isinstance(value, (list, tuple)) else value
    return out


def check_level_overrides(cfg, snapshot):
    """Refuse an override of a level table that is not in force (D2 F14; E1 refusal).

    Legacy: the inactive table must keep its construction value. Journal: no
    level/curriculum field is in force, so none may change (the geometry is
    the frozen journal plan).
    """
    table = level_table_in_force(cfg)
    now = level_snapshot(cfg)
    if table == LEVEL_TABLE_JOURNAL:
        changed = sorted(k for k in LEVEL_FIELDS if now[k] != snapshot.get(k))
        if changed:
            raise ValueError("journal docking reads no level table (docking_level_table_in_force = %r); "
                             "overrides of %s have no effect and are refused" % (table, changed))
        return table
    inactive = LEVEL_TABLE_V1 if table == LEVEL_TABLE_V2 else LEVEL_TABLE_V2
    if now[inactive] != snapshot.get(inactive):
        raise ValueError("--set %s has no effect: the docking level table in force is %s (physics_version "
                         "decides; D2 item 14)" % (inactive, table))
    return table


def level_table_stamp(cfg, realised_distance_m=None):
    """Stamp block: the table in force and (legacy) the realised spawn distance."""
    table = level_table_in_force(cfg)
    stamp = {"docking_level_table_in_force": table}
    if table == LEVEL_TABLE_JOURNAL:
        plan = getattr(cfg, "journal_plan", None)
        params = plan.params if plan is not None else {}
        stamp.update(journal_geometry_tier=getattr(plan, "geometry_tier", None),
                     approach_distance_m=params.get("approach_distance_m"),
                     lateral_offset_max_m=params.get("lateral_offset_max_m"))
        return stamp
    values = [float(v) for v in getattr(cfg, table)]
    level = int(getattr(cfg, "eval_level", 0))
    index = min(max(level, 0), len(values) - 1)
    stamp.update(table_m=values, eval_level=level, table_index=index,
                 spawn_distance_override_m=float(getattr(cfg, "spawn_distance_override_m", 0.0) or 0.0),
                 realised_distance_m=realised_distance_m)
    return stamp


__all__ = ["LEVEL_TABLE_V1", "LEVEL_TABLE_V2", "LEVEL_TABLE_JOURNAL", "LEVEL_FIELDS", "level_table_in_force",
           "level_snapshot", "check_level_overrides", "level_table_stamp"]
