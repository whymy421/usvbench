"""Journal environment mixin: the shared round-F glue (CONTEXT_R4, task F0).

A family env composes ``JournalEnvMixin`` and calls it ONLY inside its
journal branches (``if self.cfg.journal_contract_version == 2:``; legacy
paths stay byte-identical, D15). No Isaac Lab import happens here; the
mixin touches the env only through ``self.cfg``, ``self.num_envs``,
``self.device``, ``self.extras``, ``self.robot.data`` and
``self.scene.env_origins`` (standard DirectRLEnv members).

EXACT CALL ORDER (every family follows it verbatim)
---------------------------------------------------
__init__, after ``super().__init__`` (the scene exists):
    self.journal_setup()                      # vessel, runtime, conditions, D7 bridge, latches

_pre_physics_step(actions)                    # once per control step
    self.actions = self.journal_pre_physics(actions)          # clamp to [-1, 1] ONCE (D13)

_apply_action()                               # once per PHYSICS substep (decimation = 2)
    forces, torques = self.journal_substep(self.robot.data, self.scene.env_origins,
                                           water_surface_z)   # scalar or [N], env-local
    apply_body_wrench(self.robot, forces, torques)   # tasks/_shared/body_wrench.py (G6)
  inside journal_substep, in this order (DESIGN_E3 section 3 + D8 + D7 B06/B07):
    1. self._journal_observe_physics(robot_data, env_origins, water_level_local)
       [family hook: contract observe_physics on the state of the PREVIOUS
       substep; water_level_local = water_surface_z + the elevation of the
       previous RuntimeStep (the buoyancy elevation, D8); the constant
       surface on the first substep of an episode and in calm water]
    2. D7 bridge.step at the row tick (COM pose, env-local) + publication
       (self._journal_d7_samples; optional family hook _journal_publish_d7)
    3. inputs = conditions.physics_inputs(root_com_vel_w, root_link_quat)
    4. step = runtime.step_from_root_link(actions, root_link_pose_w,
       root_com_vel_w, env_origins, mean_water_level_local=water_surface_z,
       **inputs) ; stored as self.journal_last_step ; _d7_tick += 1
    5. returns (force_actor[:, None, :], torque_actor[:, None, :])
    (G10: with waves in the condition stack, conditions/green_water.py observes the
    deck-edge points right after step 4 with the SAME pose, clock and waves; read-only;
    G18: its read-only diagnostics are latched beside the rule/1 record)

_get_dones()                                  # once per control step
    view = self.journal_pre_dones(self.robot.data, self.scene.env_origins, water_surface_z)
        # D7 final bridge step (B06) + publication; view.water_level_local (D8),
        # view.d7_contact (swept-contact latch, merge into the collision ledger)
    codes = <family tensor oracle>(..., collision |= view.d7_contact, ...)   # 0 running, 1-9 E5
    terminated, truncated = self.journal_dones(codes)
        # D9 publisher (replaces publish_termination_reason) + completed-episode
        # latch BEFORE any reset: episode_conditions[i], episode_dynamic_hazards[i]
        # (bridge.finish), episode_scenario_hashes[i] (D11)
    return terminated, truncated

_reset_idx(env_ids)                           # after task layout, spawn and ROOT WRITE
    self.journal_reset(env_ids, reference_axis_rad, task_episode_ids, outage_start_s=None,
                       reference_route_length_m=<nominal route length per row>)
        # runtime.reset -> conditions.reset_runtime (episode ids mandatory, D7)
        # G28: outage start = critical phase (station keeping: outage_start_s) or f x route length / cruise
        # -> conditions.configure_runtime -> bridge.reset_rows -> tick-0 bridge step

_get_observations()                           # once per control step
    est = self.journal_estimates(true_com_xy_local, body_rates, rays)
        # D5: est.position / est.velocity / est.rays; EVERY position-derived
        # observation (goal vector, cross-track, lookahead, berth pose, phase
        # targets) is computed from est.position; MPPI / classical controllers
        # read self.journal_estimate (the same object). Exactly one call per
        # control step (enforced by the condition layer, checked at the stamp).

Family hooks (implemented by the family env):
    _journal_observe_physics(robot_data, env_origins, water_level_local) -> None
    _journal_family_scenario_hashes(env_id) -> {primitive: sha256 hex}
        (at least pairing.FAMILY_PRIMITIVES[family], e.g. layout, spawn_pose)
    _journal_publish_d7(samples) -> None          (optional; B07 storage write)

Rewards, termination, metrics and certificates use TRUTH; only the
observation/controller path reads the estimates. Waves, current and wind
are never observed.

HAZARD CHANNELS (G3, 2026-10-01)
--------------------------------
A journal env carries the D7 channel (moving intruder: suite config, pool,
bridge, ``dynamic_hazard_header``, per-episode stamp) ALWAYS, and the
appearing channel (``conditions/appearing``: a static cylinder that appears
when the own ship first comes within the dose gap) ONLY when the resolved
condition stack names the ``appearing`` suite (``appearing=D0..D4``). With the
appearing suite absent nothing below changes (no attribute read, no call, no
header block, no primitive): the D7 behaviour is byte-identical. With it
present, each D7 call site is followed by the appearing one:
    journal_setup      appearing suite, pool (``appearing_pool`` or
                       ``cfg.journal_appearing_pool`` or the family's pinned
                       file), AppearingBridge, ``appearing_hazard_header``,
                       pairing inventory + ``appearing_assignment``
    journal_substep    after the D7 bridge step: appearing step (swept contact)
    journal_pre_dones  after the D7 final step: appearing step WITH the
                       control-rate trigger check (physical + visible at once)
    journal_dones      after bridge.finish: appearing finish (stamp) and the
                       appearing primitive in episode_scenario_hashes[i]
    journal_reset      after the D7 rows and tick-0 step: appearing rows + tick 0
Additional family hooks (required only with the appearing suite present):
    _journal_publish_appearing(samples) -> None   (slot write, physical/visible)
    _journal_appearing_geometry(env_id) -> geometry id of the env's E5 layout
and the class attribute ``journal_appearing_tasks`` (tasks the family wired).
``journal_appearing_contact()`` returns the appearing swept-contact latch
([N] bool) for families whose collision ledger has no static-obstacle path.
"""
from __future__ import annotations

from collections import namedtuple
from dataclasses import asdict
import math

import torch

from . import journal_cfg_v2
from .conditions import build_conditions
from .conditions import green_water
from .conditions import wave_spawn
from .conditions import rollpitch_override
from .conditions import pairing
from .conditions.journal_config import JournalConditions, check_journal_cfg
from .hydro_v2_runtime import HydroV2Runtime
from .journal_vehicle import envelope_spec, resolve_journal_vehicle
from .termination_reason import (publish_termination_codes, termination_code_name,
                                 termination_masks_from_codes)

MIXIN_VERSION = "usvbench.journal_env_mixin/1"
JournalEstimates = namedtuple("JournalEstimates", ("position", "velocity", "rays"))
JournalPreDones = namedtuple("JournalPreDones", ("water_level_local", "d7_contact"))


def _dynamic():
    from .conditions import dynamic
    return dynamic


def _d7_group(task):
    """E4 hazard group of a journal task: "d7_" + the ADAPTER task name (harbour_mission -> harbour).

    E4 build_candidate requires hazard_group == "d7_" + TaskContext.task, and the harbour
    adapter task is "harbour" (adapters.TASK_ALIASES), so admitted harbour keys are "d7_harbour".
    """
    from .conditions.dynamic import adapters
    return "d7_" + adapters.TASK_ALIASES.get(task, task)


def _appearing():
    from .conditions.appearing import config, pool, runtime
    return config, pool, runtime


def _yaw_from_quat(quat):
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def host_copies(tensors):
    """G11 (direct helper of JournalEnvMixin._journal_host_state): ``[t.detach().to("cpu") for t in
    tensors]`` with ONE device-to-host copy per device.

    The bytes travel verbatim: the tensors of one device are made contiguous there (as a per-tensor
    copy does), viewed as uint8 and concatenated (one kernel) in descending element size so every
    host view is aligned; the returned host tensors have exactly the dtype, shape and bits of the
    per-tensor copies (no conversion happens anywhere). CPU tensors are copied the same way.
    """
    tensors = [t.detach() for t in tensors]
    out = [None] * len(tensors)
    for device in {t.device for t in tensors}:
        group = [i for i, t in enumerate(tensors) if t.device == device]
        _blob_copy(tensors, group, "cpu", out)
    return out


def device_copies(tensors, device):
    """G11: ``[t.to(device) for t in tensors]`` with ONE host-to-device copy.

    Each tensor must already carry the dtype of the per-tensor copy (a blocking ``.to(device, dtype)``
    converts on the host, so callers convert on the host first). Tensors already on ``device``
    (other than the CPU) pass through as ``.to(device)`` returns them; all others travel as verbatim
    bytes (descending element size, aligned views): the results have exactly the dtype, shape and
    bits of the per-tensor copies.
    """
    tensors = [t.detach() for t in tensors]
    device = torch.device(device)
    out = [None] * len(tensors)
    moved = []
    for i, t in enumerate(tensors):
        if t.device == device and device.type != "cpu":
            out[i] = t
        else:
            moved.append(i)
    for source in {tensors[i].device for i in moved}:
        _blob_copy(tensors, [i for i in moved if tensors[i].device == source], device, out)
    return out


def _bytes(t):
    """The tensor's elements as contiguous uint8 bytes on its device (a 1-element tensor may be
    'contiguous' with a 0 stride; it is then copied so the byte view is legal)."""
    flat = t.reshape(-1)
    if flat.numel() and flat.stride(0) != 1:
        flat = flat.clone(memory_format=torch.contiguous_format)
    return flat.view(torch.uint8)


def _blob_copy(tensors, group, device, out):
    group = sorted(group, key=lambda i: -tensors[i].element_size())
    if not group:
        return
    blob = torch.cat([_bytes(tensors[i]) for i in group]).to(device)
    pos = 0
    for i in group:
        t = tensors[i]
        size = t.numel() * t.element_size()
        out[i] = blob[pos:pos + size].view(t.dtype).reshape(t.shape)
        pos += size


class JournalEnvMixin:
    """Shared journal glue; see the module docstring for the call order."""

    journal_trace = None          # set to a list to record the call sequence (tests)

    # ------------------------------------------------------------------ helpers --
    def _jtrace(self, event):
        if self.journal_trace is not None:
            self.journal_trace.append(event)

    def _journal_hook(self, name):
        hook = getattr(self, name, None)
        if hook is None:
            raise NotImplementedError(f"journal family env must implement {name} (see journal_env_mixin)")
        return hook

    # -------------------------------------------------------------------- setup --
    def journal_setup(self, *, d7_pool=None, appearing_pool=None):
        """Build vessel, HydroV2Runtime, condition layer, D7 bridge and latches.

        Requirements (fail closed): a journal cfg (contract 2, hydro_v2,
        physics 2, dt 1/120 s, decimation 2), an integer ``cfg.seed`` (D7),
        ``cfg.sim.gravity`` and the asset's
        ``cfg.robot_cfg.spawn.rigid_props.disable_gravity`` (both checked by the
        runtime, D13), and the admitted D7 assignment pool (``d7_pool`` or
        ``cfg.journal_d7_pool``; any object with ``assignment(key, dose)`` and
        ``pool_digest``, e.g. E4 ``AdmissionPool``; D7 B02/B05).
        """
        cfg = self.cfg
        family = check_journal_cfg(cfg)
        declared = getattr(self, "journal_family", None)
        if declared is not None and declared != family:
            raise ValueError(f"env family {declared!r} cannot carry journal task {cfg.journal_task!r} "
                             f"(carrier {family!r})")
        seed = getattr(cfg, "seed", None)
        if seed is None or isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("journal cfgs must carry an integer evaluation seed (cfg.seed); a missing seed "
                             "is an error, never silently 0 (CONTEXT_R4 D7)")
        n, device = int(self.num_envs), self.device
        # T25 (owner decision T1.35b): a diagnostic run's hydro.rollpitch_zeta override (cfg.journal_rollpitch_zeta,
        # set only then) selects the profile option rollpitch_damping = "zeta:<x>"; without it the options are
        # exactly dict(cfg.journal_vehicle_options or {})
        options = rollpitch_override.vehicle_options(cfg)
        vehicle = resolve_journal_vehicle(cfg.vehicle, **options)
        gravity = getattr(cfg.sim, "gravity", None)
        if gravity is None:
            raise ValueError("journal_setup needs cfg.sim.gravity (checked against the vessel profile, D13)")
        try:
            disable_gravity = cfg.robot_cfg.spawn.rigid_props.disable_gravity
        except AttributeError:
            raise ValueError("journal_setup needs cfg.robot_cfg.spawn.rigid_props.disable_gravity (D13)") from None
        if not isinstance(disable_gravity, bool):
            raise ValueError("robot_cfg.spawn.rigid_props.disable_gravity must be a bool")
        runtime = HydroV2Runtime(vehicle.hydro_profile, n, device, float(cfg.sim.dt),
                                 engine_gravity=tuple(float(g) for g in gravity),
                                 rigid_body_disable_gravity=disable_gravity, com_source="root_link")
        conditions = build_conditions(cfg, n, device, family=family)
        if not getattr(conditions.contract, "journal", False):
            raise ValueError(f"journal envs use a journal observation contract, got "
                             f"{conditions.contract.family}/{conditions.contract.contract}")
        conditions.bind_runtime(runtime)
        # ---- D7 (E4 bridge; D7 B02-B06)
        dynamic = _dynamic()
        resolved = getattr(cfg, "journal_conditions", None)
        if resolved is not None and not isinstance(resolved, JournalConditions):
            raise ValueError("cfg.journal_conditions must come from resolve_journal_conditions")
        if resolved is not None and resolved.conditions != conditions.cfg:
            raise ValueError("cfg.journal_conditions and cfg.conditions disagree (use apply_journal_conditions)")
        suite = (resolved.dynamic_config(cfg.vehicle) if resolved is not None
                 else dynamic.DynamicSuiteConfig(vessel_id=cfg.vehicle))
        pool = d7_pool if d7_pool is not None else getattr(cfg, "journal_d7_pool", None)
        if pool is None or not callable(getattr(pool, "assignment", None)) or not hasattr(pool, "pool_digest"):
            raise ValueError("journal envs need the admitted D7 assignment pool (d7_pool or cfg.journal_d7_pool "
                             "with assignment(key, dose) and pool_digest; D7 B02/B05): every episode, D0 "
                             "included, runs an admitted assignment")
        record = journal_cfg_v2.vessel_record(cfg.vehicle)
        # G4-A5: the registry record with this vehicle's own sway/yaw damping (journal_vehicle.envelope_spec);
        # the envelope fields do not depend on it, its steady-spin check does.
        envelope, _ = dynamic.envelope_from_spec_v2(envelope_spec(vehicle),
                                                    reference_speed_mps=record.cruise_speed_m_s,
                                                    turning_radius_m=record.turning_radius_design_m)
        runtime_cfg = dynamic.DynamicRuntimeConfig(enabled=bool(suite.enabled or suite.dose > 0),
                                                   verify_witness_on_reset=suite.verify_witness_on_reset)
        bridge = dynamic.HazardBridge(dynamic.make_adapter(cfg.journal_task), runtime_cfg, float(cfg.sim.dt),
                                      envelope.top_speed_mps)
        # ---- state
        self.journal_vehicle = vehicle
        self._hydro_v2 = runtime
        self._conditions = conditions
        self._journal_family = family
        self._journal_decimation = int(cfg.decimation)
        self._d7_suite = suite
        self._d7_pool = pool
        self._d7_bridge = bridge
        self._d7_tick = torch.zeros(n, dtype=torch.long, device=device)
        self._journal_d7_samples = {}
        self._journal_actions = None
        self._journal_substeps = None
        self._journal_elevation = torch.zeros(n, dtype=runtime.dtype, device=device)
        self.journal_last_step = None
        self.journal_estimate = None
        self.episode_conditions = [{} for _ in range(n)]
        self.episode_dynamic_hazards = [None for _ in range(n)]
        self.episode_scenario_hashes = [None for _ in range(n)]
        self.journal_aborted_dynamic_hazards = []
        self.dynamic_hazard_header = {
            "pool_digest": getattr(pool, "pool_digest", None),
            "per_dose_counts": [list(x) for x in getattr(pool, "per_dose_counts", ())],
            "keys_offered": getattr(pool, "keys_offered", None),
            "keys_admitted": getattr(pool, "keys_admitted", None),
            "joint": getattr(pool, "joint", None),
            "suite": asdict(suite), "hazard_group": _d7_group(cfg.journal_task),
            "envelope_top_speed_mps": envelope.top_speed_mps,
            # G7 item B: the E4 unforced-window rule that admitted the pool, with its declared error bound
            "unforced_window": {"rule": dynamic.model.UNFORCED_WINDOW_RULE,
                                "yaw_rate_tol_rps": dynamic.model.UNFORCED_YAW_RATE_TOL_RPS,
                                "accel_tol_mps2": dynamic.model.UNFORCED_ACCEL_TOL_MPS2,
                                "bound": dynamic.model.UNFORCED_WINDOW_BOUND},
            # G23c (REV-G23b F7): the runtime rule that times the D7 object (release.py), constants included
            "runtime_rule": dict(dynamic.release.rule_block_for(runtime_cfg.motion_rule),
                                 motion_rule=runtime_cfg.motion_rule),
        }
        inventory = (list(pairing.CONDITION_PRIMITIVES) + [pairing.D7_PRIMITIVE]
                     + list(pairing.FAMILY_PRIMITIVES.get(family, ())))
        self.journal_pairing_inventory = tuple(sorted(set(inventory)))
        # ---- G10: read-only green-water detection (only when the stack includes waves; else None)
        self._green_water = green_water.monitor_for(conditions, runtime)
        self.episode_green_water = [None for _ in range(n)]
        self.episode_green_water_diagnostics = [None for _ in range(n)]
        # T25 (owner decision T1.35d): report-only point sets beside the declared set (WAM-V deck edge; {} else)
        self._green_water_report = green_water.report_monitors_for(self._green_water)
        self.episode_green_water_report = [None for _ in range(n)]
        # G18 option, ON by default since T25 (owner decision T1.35a, wave_spawn.JOURNAL_SPAWN_ON_WAVES_DEFAULT):
        # place the hull on the footprint-filtered wave surface at every reset (conditions/wave_spawn.py)
        self._journal_spawn_on_waves = wave_spawn.enabled(cfg)
        # ---- appearing channel (G3): only when the stack names the appearing suite
        self._appearing_bridge = None
        if resolved is not None and getattr(resolved, "appearing_enabled", False):
            self._journal_appearing_setup(resolved, appearing_pool, envelope, family)
        self._jtrace("setup")

    def _journal_appearing_setup(self, resolved, appearing_pool, envelope, family):
        """Appearing suite config, pool, bridge, header block and pairing primitive (G3 rule 8)."""
        cfg = self.cfg
        aconfig, apool, aruntime = _appearing()
        task = cfg.journal_task
        if task not in tuple(getattr(self, "journal_appearing_tasks", ())):
            raise ValueError(f"the appearing suite is not wired on journal task {task!r} of family {family!r} "
                             f"(phase-1 prototype: {', '.join(aconfig.WIRED_TASKS)})")
        for hook in ("_journal_publish_appearing", "_journal_appearing_geometry"):
            self._journal_hook(hook)
        suite = resolved.appearing_config(cfg.vehicle)
        pool = appearing_pool if appearing_pool is not None else getattr(cfg, "journal_appearing_pool", None)
        record = None
        if pool is None:
            # G3c: the pool FILE comes from the runtime pool directory (the bundle) when one is set,
            # else from the in-repo folder; the in-repo manifest pin is checked either way
            runtime_dir, runtime_source = apool.runtime_pool_dir(cfg)
            pool, record = apool.load_pinned_pool(family, task, cfg.vehicle, int(cfg.seed),
                                                  int(getattr(cfg, "journal_geometry_tier", 2)),
                                                  suite_identity=suite.identity(), runtime_dir=runtime_dir,
                                                  runtime_source=runtime_source)
        if not callable(getattr(pool, "assignment", None)) or not hasattr(pool, "pool_digest"):
            raise ValueError("the appearing suite needs its admitted pool (assignment(key, dose), pool_digest)")
        identity = getattr(pool, "suite_identity", None)
        if identity is not None and dict(identity) != suite.identity():
            raise ValueError("the appearing pool was admitted with a different suite configuration")
        if getattr(pool, "task", task) != task or getattr(pool, "vessel_id", cfg.vehicle) != cfg.vehicle \
                or getattr(pool, "seed", int(cfg.seed)) != int(cfg.seed):
            raise ValueError("the appearing pool belongs to another task, vessel or seed")
        from . import journal_cfg_v2 as plans
        base_max, reserve = int(plans.STATIC_CAPACITY[task]), int(plans.D7_RESERVED_SLOTS[task])
        slot = aconfig.appearing_slot(base_max, reserve)
        bridge = aruntime.AppearingBridge(task, slot, float(cfg.sim.dt), envelope.top_speed_mps,
                                          envelope.footprint, envelope.t_avoid_s,
                                          gap_override_m=suite.gap_override_m)     # G20b diagnostic token
        n = int(self.num_envs)
        self._appearing_suite = suite
        self._appearing_pool = pool
        self._appearing_bridge = bridge
        self._appearing_slot = slot
        self._journal_appearing_samples = {}
        self.episode_appearing_hazards = [None for _ in range(n)]
        self.journal_aborted_appearing_hazards = []
        header = {"schema": "usvbench.appearing_header/1", "suite": asdict(suite), "dose": suite.dose,
                  "suite_version": suite.version, "hazard_group": aconfig.hazard_group(task), "slot": slot,
                  "reference_speed_mps": aconfig.REFERENCE_SPEED_MPS, "t_avoid_s": envelope.t_avoid_s,
                  **aconfig.ladder_header(envelope),          # G20b: gaps in metres + equivalent k per dose
                  "footprint": [envelope.footprint.x_min_m, envelope.footprint.x_max_m,
                                envelope.footprint.y_min_m, envelope.footprint.y_max_m],
                  "trigger_rule": ("once per control step after the physics substeps; first gap (D7 footprint to "
                                   "cylinder surface) <= d_k -> physical and visible at once"),
                  "exposure_rule": apool.D16_RULE, "beyond_grid_rule": apool.D24_RULE,
                  "pool_pinned": record is not None}
        if suite.gap_override_m is not None:            # G20b: diagnostic only (refused in official runs)
            header["gap_override_m"] = suite.gap_override_m
        block = getattr(pool, "header_block", None)
        if callable(block):
            header.update(block())
        else:
            header["pool_digest"] = pool.pool_digest
        if record is not None:
            header.update(pool_file=record["file"], pool_sha256=record["sha256"],
                          pool_manifest=record["manifest"], pool_manifest_sha256=record["manifest_sha256"],
                          pool_source=record["source"], pool_source_detail=record["source_detail"])
        self.appearing_hazard_header = header
        self.journal_pairing_inventory = tuple(sorted(set(self.journal_pairing_inventory)
                                                      | {pairing.APPEARING_PRIMITIVE}))

    # ------------------------------------------------------------- pre-physics --
    def journal_pre_physics(self, actions):
        """Clamp to [-1, 1] once (D13); start a control step; return the actions."""
        if self._journal_substeps not in (None, self._journal_decimation):
            raise RuntimeError(f"previous control step ran {self._journal_substeps} journal substeps, "
                               f"expected decimation {self._journal_decimation}")
        acts = self._conditions.action_in(actions.clamp(-1.0, 1.0))
        self._journal_actions = acts
        self._journal_substeps = 0
        self._jtrace("pre_physics")
        return acts

    # ----------------------------------------------------------------- substep --
    def _journal_host_state(self, robot_data, env_origins, ticks=None):
        """G11: the host copies the per-quantity transfers made, in ONE device-to-host copy.

        Returns host tensors (xy, quat, v[, ticks]) with the bits of
        (pose[:, :2] - env_origins[:, :2]).detach().to("cpu"), pose[:, 3:7].detach().to("cpu"),
        vel[:, :2].detach().to("cpu") (and ticks.detach().to("cpu")); the float64 widening
        then runs on the host, where the blocking .to("cpu", torch.float64) copies ran it."""
        pose = robot_data.root_link_pose_w
        vel = robot_data.root_com_vel_w
        parts = [(pose[:, :2] - env_origins[:, :2]).detach(), pose[:, 3:7].detach(), vel[:, :2].detach()]
        if ticks is not None:
            parts.append(ticks.detach())
        return host_copies(parts)

    def _journal_poses(self, robot_data, env_origins, host=None):
        if host is None:
            host = self._journal_host_state(robot_data, env_origins)
        xy = host[0].to("cpu", torch.float64)
        yaw = _yaw_from_quat(host[1].to("cpu", torch.float64))
        v = host[2].to("cpu", torch.float64)
        surge = v[:, 0] * torch.cos(yaw) + v[:, 1] * torch.sin(yaw)
        Pose = _dynamic().Pose
        rows = torch.stack((xy[:, 0], xy[:, 1], yaw, surge), -1).tolist()
        # A non-finite row (engine blow-up) must reach the family contract as code 9
        # (non_finite_state): it is left out of the D7 step instead of raising in E4 Pose.
        return {i: Pose(float(r[0]), float(r[1]), float(r[2]), float(r[3])) for i, r in enumerate(rows)
                if all(math.isfinite(v) for v in r)}

    def _journal_d7_step(self, robot_data, env_origins):
        host = self._journal_host_state(robot_data, env_origins, ticks=self._d7_tick)   # G11: one transfer
        ticks = [int(t) for t in host[3].tolist()]
        poses = self._journal_poses(robot_data, env_origins, host=host)
        samples = self._d7_bridge.step({i: t for i, t in enumerate(ticks) if i in poses}, poses)
        self._journal_d7_samples = samples
        publish = getattr(self, "_journal_publish_d7", None)
        if publish is not None:
            publish(samples)
        return samples

    def _journal_appearing_step(self, robot_data, env_origins, check_trigger):
        """Appearing bridge at the row tick (G3); the trigger check only at the control rate."""
        ticks = [int(t) for t in self._d7_tick.detach().to("cpu").tolist()]
        poses = self._journal_poses(robot_data, env_origins)
        samples = self._appearing_bridge.step({i: t for i, t in enumerate(ticks) if i in poses}, poses,
                                              check_trigger=check_trigger)
        self._journal_appearing_samples = samples
        self._journal_hook("_journal_publish_appearing")(samples)
        return samples

    def journal_appearing_contact(self):
        """[N] bool swept-contact latch of the appearing cylinder (all False without the suite)."""
        n = int(self.num_envs)
        bridge = getattr(self, "_appearing_bridge", None)
        values = [False] * n if bridge is None else [bridge.contact(i) for i in range(n)]
        return torch.tensor(values, dtype=torch.bool, device=self._d7_tick.device)

    def journal_substep(self, robot_data, env_origins, water_surface_z):
        """One physics substep (DESIGN_E3 section 3); returns [N,1,3] force and torque."""
        if self._journal_actions is None:
            raise RuntimeError("journal_pre_physics must run before journal_substep")
        if self._journal_substeps >= self._journal_decimation:
            raise RuntimeError("more journal substeps than the decimation in one control step")
        water_level = water_surface_z + self._journal_elevation
        self._journal_hook("_journal_observe_physics")(robot_data, env_origins, water_level)
        self._journal_d7_step(robot_data, env_origins)
        if getattr(self, "_appearing_bridge", None) is not None:
            self._journal_appearing_step(robot_data, env_origins, check_trigger=False)
        pose = robot_data.root_link_pose_w
        vel = robot_data.root_com_vel_w
        inputs = self._conditions.physics_inputs(vel, pose[:, 3:7])
        step = self._hydro_v2.step_from_root_link(self._journal_actions, pose, vel, env_origins,
                                                  mean_water_level_local=water_surface_z,
                                                  **inputs.as_kwargs())
        self.journal_last_step = step
        if getattr(self, "_green_water", None) is not None:      # G10: same pose, clock and waves (read-only)
            self._green_water.observe(pose, env_origins, water_surface_z, step.time_s, inputs.waves)
            for monitor in getattr(self, "_green_water_report", {}).values():     # T25 (T1.35d), read-only
                monitor.observe(pose, env_origins, water_surface_z, step.time_s, inputs.waves)
        self._journal_elevation = step.waves.elevation.detach().clone()
        self._d7_tick += 1
        self._journal_substeps += 1
        self._jtrace("substep")
        return step.force_actor.unsqueeze(1), step.torque_actor.unsqueeze(1)

    # ------------------------------------------------------------------- dones --
    def journal_pre_dones(self, robot_data, env_origins, water_surface_z):
        """D7 final bridge step (B06) + D8 water level for the contract step()."""
        if self._journal_substeps != self._journal_decimation:
            raise RuntimeError(f"journal_pre_dones after {self._journal_substeps} substeps "
                               f"(decimation {self._journal_decimation})")
        self._journal_d7_step(robot_data, env_origins)
        if getattr(self, "_appearing_bridge", None) is not None:
            self._journal_appearing_step(robot_data, env_origins, check_trigger=True)
        contact = torch.tensor([bool(self._d7_bridge.contact(i)) for i in range(int(self.num_envs))],
                               dtype=torch.bool, device=self._d7_tick.device)
        self._jtrace("pre_dones")
        return JournalPreDones(water_surface_z + self._journal_elevation, contact)

    def journal_dones(self, contract_codes):
        """D9 publisher + completed-episode latch (BEFORE reset); returns (terminated, truncated)."""
        terminated, truncated = termination_masks_from_codes(contract_codes)
        publish_termination_codes(self, contract_codes, terminated=terminated, truncated=truncated)
        codes = [int(c) for c in contract_codes.detach().to("cpu").tolist()]
        family_hashes = self._journal_hook("_journal_family_scenario_hashes")
        for i, code in enumerate(codes):
            if code == 0:
                continue
            stamp = self._conditions.episode_stamp(i)
            envelope = self._d7_bridge.finish(i, termination_code_name(code))
            hashes = pairing.episode_scenario_hashes(stamp, envelope, family_hashes(i),
                                                     family=self._journal_family)
            if getattr(self, "_appearing_bridge", None) is not None:
                appearing = self._appearing_bridge.finish(i, termination_code_name(code))
                hashes[pairing.APPEARING_PRIMITIVE] = pairing.appearing_primitive_hash(appearing)
                hashes = dict(sorted(hashes.items()))
                self.episode_appearing_hazards[i] = appearing
            self.episode_conditions[i] = stamp
            self.episode_dynamic_hazards[i] = envelope
            self.episode_scenario_hashes[i] = hashes
            if getattr(self, "_green_water", None) is not None:   # G10 latch (before the reset)
                self.episode_green_water[i] = self._green_water.episode_record(i, stamp["runtime"]["runtime_ticks"])
                self.episode_green_water_diagnostics[i] = self._green_water.episode_diagnostics(i)   # G18
                if getattr(self, "_green_water_report", None):                             # T25 (T1.35d)
                    self.episode_green_water_report[i] = green_water.report_episode(
                        self._green_water_report, i, stamp["runtime"]["runtime_ticks"])
        self._jtrace("dones")
        return terminated, truncated

    # ------------------------------------------------------------------- reset --
    def journal_reset(self, env_ids, reference_axis_rad, task_episode_ids, outage_start_s=None,
                      reference_route_length_m=None):
        """runtime.reset -> conditions.reset_runtime -> configure_runtime -> bridge rows + tick 0.

        G28: ``reference_route_length_m`` = the reset rows' nominal (witness) route lengths; divided by the
        vessel's reference (cruise) speed it is the T_ref of the perception outage start rule
        (perception_v3.OUTAGE_ROUTE_RULE). With the outage axis active a reset must name a start rule
        (``outage_start_s`` = the task critical phase, or the route length); there is no silent fallback.
        """
        ids = torch.as_tensor(env_ids, dtype=torch.long).detach().to("cpu").flatten()
        if ids.numel() == 0:
            return None
        reference_duration_s = None
        if reference_route_length_m is not None:
            cruise = float(journal_cfg_v2.vessel_record(self.cfg.vehicle).cruise_speed_m_s)
            reference_duration_s = (torch.as_tensor(reference_route_length_m, dtype=torch.float64)
                                    .detach().to("cpu").flatten() / cruise)
        perception = getattr(self._conditions, "perception", None)
        if outage_start_s is None and reference_duration_s is None and perception is not None \
                and bool(getattr(perception, "axes", {}).get("outage", False)):
            raise ValueError("perception outage cell: journal_reset needs outage_start_s (task critical phase) or "
                             "reference_route_length_m (G28 reference-route rule); no fallback window")
        if task_episode_ids is None:
            raise ValueError("task_episode_ids is mandatory at every journal reset (CONTEXT_R4 D7)")
        episodes = [int(e) for e in torch.as_tensor(task_episode_ids).detach().to("cpu").flatten().tolist()]
        if len(episodes) != ids.numel():
            raise ValueError("one task episode index per reset env is required")
        rows = self._d7_bridge.rows
        for i in ids.tolist():
            if i in rows and not rows[i]["finished"]:
                # reset without a completed latch (e.g. a manual env.reset): stamp as aborted,
                # never as a certificate record
                self.journal_aborted_dynamic_hazards.append(self._d7_bridge.finish(i, "aborted"))
        device_ids = ids.to(self._d7_tick.device)
        self._hydro_v2.reset(device_ids)
        if getattr(self, "_green_water", None) is not None:      # G10: reset rows only
            self._green_water.reset(device_ids)
            for monitor in getattr(self, "_green_water_report", {}).values():     # T25 (T1.35d)
                monitor.reset(device_ids)
        extra = {} if reference_duration_s is None else {"reference_duration_s": reference_duration_s}   # G28
        self._conditions.reset_runtime(ids, reference_axis_rad=reference_axis_rad,
                                       task_episode_ids=torch.as_tensor(episodes, dtype=torch.long),
                                       outage_start_s=outage_start_s, **extra)
        self._conditions.configure_runtime(self._hydro_v2, ids)
        if getattr(self, "_journal_spawn_on_waves", False):     # G18 option (on by default since T1.35a)
            self._journal_wave_spawn(device_ids)
        dynamic = _dynamic()
        seed, task = int(self.cfg.seed), self.cfg.journal_task
        assignments = {}
        for i, episode in zip(ids.tolist(), episodes):
            key = dynamic.EpisodeKey(seed, int(i), int(episode), _d7_group(task))
            try:
                assignments[int(i)] = self._d7_pool.assignment(key, int(self._d7_suite.dose))
            except KeyError:
                raise ValueError(f"no admitted D7 assignment for {key} at dose D{self._d7_suite.dose} "
                                 "(preflight error, never an in-episode redraw; D7 B02)") from None
        self._d7_bridge.reset_rows(tuple(int(i) for i in ids.tolist()), assignments)
        self._d7_tick[device_ids] = 0
        self._journal_elevation[device_ids] = 0.0
        poses = self._journal_poses(self.robot.data, self.scene.env_origins)
        samples = self._d7_bridge.step({int(i): 0 for i in ids.tolist()},
                                       {int(i): poses[int(i)] for i in ids.tolist()})
        self._journal_d7_samples.update(samples)
        publish = getattr(self, "_journal_publish_d7", None)
        if publish is not None:
            publish(samples)
        if getattr(self, "_appearing_bridge", None) is not None:
            self._journal_appearing_reset(ids, episodes, poses)
        self._jtrace("reset")
        return None

    def _journal_wave_spawn(self, device_ids):
        """G18 option: rewrite the reset rows' root state onto the filtered wave surface (wave_spawn.py)."""
        # T25 (T1.35a, on by default): only the just-configured reset rows must be ready (a first partial reset of a
        # fresh env is legal); same realisation object as wave_source() once every env has been reset
        waves = self._conditions.wave_source_for_reset(device_ids)
        if waves is None:
            return
        rows = getattr(self._conditions, "training_wave_rows", None)   # G19b: a training mixture's rows with
        if rows is not None:                                            # the waves gate off keep the calm spawn
            device_ids = rows(device_ids)                               # (same object without a mixture)
            if device_ids.numel() == 0:
                return
        data = self.robot.data
        state = wave_spawn.wave_following_root_state(
            waves, self._hydro_v2.profile, data.root_link_pose_w, data.root_com_vel_w, self.scene.env_origins,
            device_ids)
        self.robot.write_root_state_to_sim(state, device_ids)

    def _journal_appearing_reset(self, ids, episodes, poses):
        """Appearing rows (D16 not-admitted / D24 beyond-grid keys run D0) + the tick-0 step (G3)."""
        aconfig, _, _ = _appearing()
        bridge = self._appearing_bridge
        rows = bridge.rows
        for i in ids.tolist():
            if i in rows and not rows[i]["finished"]:
                self.journal_aborted_appearing_hazards.append(bridge.finish(i, "aborted"))
        seed, task = int(self.cfg.seed), self.cfg.journal_task
        dose = int(self._appearing_suite.dose)
        geometry = self._journal_hook("_journal_appearing_geometry")
        assignments = {}
        for i, episode in zip(ids.tolist(), episodes):
            key = _dynamic().EpisodeKey(seed, int(i), int(episode), aconfig.hazard_group(task))
            try:
                assignment = self._appearing_pool.assignment(key, dose)
            except KeyError:
                raise ValueError(f"no appearing assignment for {key} at dose D{dose} (preflight error: the key "
                                 "lies outside the appearing pool's grid, seed or group)") from None
            if not assignment.beyond_offered_grid and assignment.geometry_id != geometry(int(i)):
                raise ValueError(f"appearing assignment {key} was admitted on layout {assignment.geometry_id}, "
                                 f"the env built {geometry(int(i))} (D16 never redraws)")
            assignments[int(i)] = assignment
        bridge.reset_rows(tuple(int(i) for i in ids.tolist()), assignments)
        samples = bridge.step({int(i): 0 for i in ids.tolist()}, {int(i): poses[int(i)] for i in ids.tolist()})
        self._journal_appearing_samples.update(samples)
        self._journal_hook("_journal_publish_appearing")(samples)

    # --------------------------------------------------------------- estimates --
    def journal_estimates(self, true_com_xy_local, body_rates, rays):
        """D5 own-state estimates (each group once per control step, counted)."""
        if rays is None:
            raise ValueError("journal contracts carry the 36-ray block: pass the physical ranges (D14)")
        est = JournalEstimates(self._conditions.estimate_position(true_com_xy_local),
                               self._conditions.estimate_velocity(body_rates),
                               self._conditions.estimate_rays(rays))
        self.journal_estimate = est
        self._jtrace("estimates")
        return est

    # ----------------------------------------------------------- certificates --
    def journal_episode_record(self, env_id):
        """Latched finished-episode record {conditions, dynamic_hazards, scenario_hashes[, appearing_hazards]
        [, green_water (G10, waves in the stack)]}."""
        i = int(env_id)
        if not self.episode_conditions[i]:
            raise RuntimeError(f"env {i} has no completed journal episode latched")
        record = {"conditions": self.episode_conditions[i],
                  "dynamic_hazards": self.episode_dynamic_hazards[i],
                  "scenario_hashes": dict(self.episode_scenario_hashes[i])}
        if getattr(self, "_appearing_bridge", None) is not None:
            record["appearing_hazards"] = self.episode_appearing_hazards[i]
        if getattr(self, "_green_water", None) is not None:      # G10 (waves in the stack only)
            record["green_water"] = self.episode_green_water[i]
            record["green_water_diagnostics"] = self.episode_green_water_diagnostics[i]   # G18 (read-only)
            if getattr(self, "_green_water_report", None):                          # T25 (T1.35d, report-only)
                record["green_water_report"] = self.episode_green_water_report[i]
        return record

    def journal_rollpitch_damping_stamp(self):
        """T25 (T1.35b) header block ``rollpitch_damping``: journal-default vs used roll/pitch damping.

        Read by cert_v3 (the header blocks of journal_certificate_header stay unchanged). ``pinned`` is the
        profile of cfg.journal_vehicle_options without the diagnostic override, ``used`` the profile the
        HydroV2Runtime runs (cross-checked)."""
        cfg = self.cfg
        if self._hydro_v2.profile is not self.journal_vehicle.hydro_profile:
            raise RuntimeError("the runtime profile is not the journal vehicle's profile")
        pinned = resolve_journal_vehicle(cfg.vehicle, **dict(getattr(cfg, "journal_vehicle_options", None) or {}))
        return rollpitch_override.stamp(cfg.vehicle, rollpitch_override.profile_values(pinned.hydro_profile),
                                        rollpitch_override.profile_values(self._hydro_v2.profile),
                                        rollpitch_override.cfg_override(cfg))

    def journal_certificate_header(self, treatment=None):
        """Header blocks: schema-3 stamp, D7 header, D11 pairing declaration[, appearing header]
        [, green_water_header (G10, waves in the stack)]."""
        header = {"conditions": self._conditions.certificate_stamp(),
                  "dynamic_hazards_header": dict(self.dynamic_hazard_header),
                  "pairing": pairing.pairing_declaration(self.journal_pairing_inventory, treatment),
                  "journal_mixin": MIXIN_VERSION}
        if getattr(self, "_appearing_bridge", None) is not None:
            header["appearing_hazards_header"] = dict(self.appearing_hazard_header)
        if getattr(self, "_green_water", None) is not None:      # G10 (waves in the stack only)
            header["green_water_header"] = self._green_water.header()
            if getattr(self, "_green_water_report", None):                          # T25 (T1.35d, report-only)
                header["green_water_header"]["report_points"] = green_water.report_header(self._green_water_report)
            if getattr(self, "_journal_spawn_on_waves", False):   # G18 option: stamped when on (default, T1.35a)
                header["green_water_header"]["spawn"] = wave_spawn.stamp()
        return header


__all__ = ["JournalEnvMixin", "JournalEstimates", "JournalPreDones", "MIXIN_VERSION"]
