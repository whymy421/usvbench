"""Condition layer for ``hydrodynamics_model="hydro_v2"`` (schema 3, E3).

The layer turns the schema-3 doses into the INPUTS of D3's HydroV2Runtime
(INTEGRATION_SPEC.md sections 2 and 6). Per physics substep it exposes
exactly:

  (a) current_velocity_world() -> [N,3] world water velocity (XY from the
      current schedule evaluated at HydroV2Runtime.time(), Z = 0; exact zeros
      when off or at zero dose);
  (b) wave_source() -> a SpectralWaves (or None for calm / Hs = 0) built from
      a C2 SeaState(physics_version=2) keyed by (seed, env, scenario episode),
      band and component count resolved against the vessel (check_wave_band
      enforced);
  (c) wind_wrench_world(com_velocity_w, actor_quat_wxyz) -> C3 Fossen wind
      force/torque (world, about the COM; D5 windage via to_c3_windage_kwargs,
      U10 converted with vessel_wind_speed_from_u10), or (None, None) when off;
  (d) configure_runtime(runtime, env_ids): actuator doses latched once per
      episode into ThrusterDegradation (cap, one-sided gain balanced by
      scenario-ordinal pairs, additional lag) and resistance_scale;
  (e) centred-mass / eccentric-payload doses do not exist in schema 3
      (rejected by config_v3.resolve_v3).

The layer owns NO clock: waves and current are evaluated at the runtime's
tick clock. The wind OU state needs one step per physics tick; its step
counter is a host-side mirror of HydroV2Runtime.ticks advanced by
wind_wrench_world (one call per runtime.step) and cross-checked against the
runtime at every reset/configure/stamp (fail closed). The layer never calls
SeaState.forces, a current wrench, an actuator ``dynamics`` surrogate or
``scale_drag``; those legacy hooks raise here.

Nothing here is observed by a controller: waves, current and wind have no
observation channel. Perception acts only on declared measurement groups.

F0 (round F, CONTEXT_R4) additions:
  * D7: the factory requires an integer evaluation seed on the env cfg (a
    missing seed is an error, never 0); ``reset_runtime`` requires
    ``task_episode_ids`` for every reset; the wave stamp field ``band_hz`` is
    renamed ``component_frequency_range_hz`` (first/last realised component)
    and the true band edges are stamped beside it (``band_edges_hz``).
  * D5: ``estimate_position(true_com_xy_local)``, ``estimate_velocity(body
    rates)`` and ``estimate_rays(ranges)`` degrade the OWN-state estimate
    (perception_v3.estimate). Each group may be estimated at most once per
    control step: every call checks estimates x decimation == runtime ticks
    for every env, and ``episode_stamp`` re-checks it for every estimate
    group of a journal contract (fail closed). ``observation_out`` (the E3
    task-relative entry point) is refused on journal contracts.
  * D4: per-suite relative direction offsets (``direction_offset_rad``;
    perception ``bias_direction_offset_rad``) added to the shared stratum.
  * D11: ``scenario_hashes(env_id)`` = per-primitive hashes derived from the
    episode stamp (conditions/pairing.py, recomputable by any reader).
  * D14: journal contracts take the ray sensor (count, range) from the task
    plan (journal_cfg_v2) and refuse a cfg that declares a different one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math

import numpy as np
import torch

from ..physics_v2 import rotation_matrix
from ..scenario_rng import SCENARIO_PROTOCOL_VERSION, ScenarioRNG
from ..sea_state import SeaState, SeaStateCfg
from ..wind import RNG_PROTOCOL_V2, VesselWindage, Wind, WindCfg
from .config_v3 import JournalConditionConfig, resolve_v3, steepness
from .observation_groups import (JOURNAL_RAY_BLOCK, admit_perception, derive_contract,
                                 effective_contract, lookup)
from .perception_v3 import PerceptionV3Degrader
from . import strata as strata_module
from .stamp import stamp_conditions_v3
from . import pairing as pairing_module
from . import wind_coefficients as wind_coefficients_module

LAYER_KIND = "hydro_v2"
HYDRO_V2_FAMILIES = ("station_keeping", "hazard_nav", "blueboat_calm_nav", "path_following",
                     "docking", "path_hazard", "harbor_mission")

PENDING_OWNER_DECISIONS = {
    "wave_ladder": {"status": "DECIDED (owner 2026-10-01 00:0x, B1; 13:4x 'A + rule B'; NC 09 decision record)",
                    "default": "A: Tp 4.0 s, Hs 0/0.125/0.25/0.5/1.0 m",
                    "alternative": "B: CEM U10 3/6/9/12 m/s, 25 km fetch (supplement only)",
                    "source": "CONTEXT_R3 decision 8; D9 D-1"},
    "wave_steepness_admission": {"status": "PENDING",
                                 "default": "DNV-RP-C205 3.5.4 Sp <= 1/15 and Ss <= 1/10; 1/7 guard",
                                 "source": "CONTEXT_R3 decision 8; D9 D-2"},
    "wave_spreading": {"status": "PENDING", "default": "uniform +/-45 deg", "source": "D9 D-5"},
    "wave_band": {"status": "PENDING", "default": "[0.2, 4.0] fp raised to the vessel "
                  "requirement; >= 1024 components; group repeat >= 480 s", "source": "decision 8; D9 D-6"},
    "wind_dose_axis": {"status": "PENDING", "default": "U10 with neutral log profile z0 = 2e-4 m to "
                       "the lateral-centroid height", "source": "CONTEXT_R2 decision 2, R3 decision 9"},
    "wind_gusts": {"status": "PENDING", "default": "OU SD 15 % of the vessel-height mean, tau 2 s",
                   "source": "C7 STRATEGY.md:180 (proposed test parameters)"},
    "direction_strata": {"status": "PENDING", "default": "compass8_v1 relative to the task axis",
                         "source": "C7 STRATEGY.md:175; E3 choice of the exact set"},
    "perception_outage_start": {"status": "DECIDED (owner 2026-10-04: replace the 5-20 s window by a rule; "
                                          "rule G28)",
                                "default": "station keeping: the task critical phase (scored late-interval "
                                           "start); every other task: f x T_ref, T_ref = reference route length "
                                           "/ reference speed, f keyed uniform in [0.40, 0.60] "
                                           "(perception_v3.OUTAGE_ROUTE_RULE)",
                                "source": "C7 STRATEGY.md:217; NC 39 defaults list section 2 item 4"},
    "rollpitch_damping_blueboat": {"status": "DECIDED (owner 2026-10-01 00:0x, A1; kept after the G1 sweep)",
                                   "default": "zeta = 0.1 per axis",
                                   "source": "CONTEXT_R3 decision 2 (value read from the runtime profile)"},
    "wamv_profile": {"status": "DECIDED (owner 2026-10-01 00:0x, A4); WAM-V model audit G27 done (controller decisions J1-J3)",
                     "default": "wamv_vrx_v2_opt1", "source": "CONTEXT_R3 decision 5"},
    "thrust_line_pitch_coupling": {"status": "PENDING", "default": "off (mixer_mount_z_m = 0)",
                                   "source": "CONTEXT_R3 decision 6"},
    "froude_krylov_horizontal": {"status": "REQUIRED (decision 7); implemented by E2",
                                 "source": "CONTEXT_R3 decision 7"},
}


# G16: columns of the diagnostic heading statistics (override runs with wind only)
HEADING_STAT_COLUMNS = ("ticks", "gamma", "abs_gamma", "abs_offset_from_beam", "cos_2gamma", "nonfinite_ticks")


def _shared_module(name):
    """E2-owned hydro_v2 modules, imported lazily (legacy paths never load them)."""
    import importlib
    return importlib.import_module(__package__.rsplit(".", 1)[0] + "." + name)


@dataclass(frozen=True)
class PhysicsInputs:
    """Keyword arguments for HydroV2Runtime.step (one physics substep)."""
    current_velocity_w: torch.Tensor
    waves: object
    wind_force_w: torch.Tensor | None
    wind_torque_w: torch.Tensor | None

    def as_kwargs(self):
        return {"current_velocity_w": self.current_velocity_w, "waves": self.waves,
                "wind_force_w": self.wind_force_w, "wind_torque_w": self.wind_torque_w}


def _float32_digest(*tensors) -> str:
    h = hashlib.sha256()
    for tensor in tensors:
        h.update(np.ascontiguousarray(tensor.detach().cpu().to(torch.float32).numpy()).astype("<f4").tobytes())
    return h.hexdigest()


class HydroV2ConditionLayer:
    layer_kind = LAYER_KIND

    def __init__(self, cfg, num_envs, device, *, family, vehicle, contract, physics_dt, control_dt,
                 ray_count=None, ray_max_range_m=None, context=None, wind_coefficients=None):
        cfg = resolve_v3(overrides=cfg)
        # G16: diagnostic-only coefficient override (None = the vessel's pinned values)
        self.wind_coefficient_override = wind_coefficients_module.validate_override(wind_coefficients)
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be positive")
        if family not in HYDRO_V2_FAMILIES:
            raise ValueError(f"family {family!r} is not a hydro_v2 condition family")
        self.cfg: JournalConditionConfig = cfg
        self.family = family
        self.num_envs, self.device = int(num_envs), torch.device(device)
        self.physics_dt, self.control_dt = float(physics_dt), float(control_dt)
        for value, name in ((self.physics_dt, "physics dt"), (self.control_dt, "control period")):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be positive and finite")
        ratio = self.control_dt / self.physics_dt
        self.decimation = int(round(ratio))
        if self.decimation < 1 or abs(ratio - self.decimation) > 1e-9:
            raise ValueError("control period must be an integer multiple of the physics dt")
        vehicles = _shared_module("vehicles_v2")
        self.spec = vehicles.get_vehicle_v2(vehicle)
        self._vehicles = vehicles
        self.contract = contract
        admit_perception(contract, cfg.perception)
        if contract.sea_state_observed:
            if cfg.waves.enabled:
                raise ValueError("observation contract observes the sea state while condition waves "
                                 "are enabled (D2 item 18): rejected")
            raise ValueError("hydro_v2 journal layers never observe waves/current/wind")
        if cfg.wind.enabled and cfg.seed < 0:
            raise ValueError("enabled wind needs a nonnegative evaluation seed")
        self.scenario = ScenarioRNG(num_envs=self.num_envs, device=self.device, eval_seed=cfg.seed)
        n = self.num_envs
        self.episode_index = np.full(n, -1, dtype=np.int64)
        self.ordinal = np.full(n, -1, dtype=np.int64)
        self.stratum = np.zeros(n, dtype=np.int64)
        self.reference_axis = torch.zeros(n, dtype=torch.float64)
        self.stratum_world = torch.zeros(n, dtype=torch.float64)
        self.rotation = strata_module.strata_rotation(cfg.seed)
        self.perception = PerceptionV3Degrader(cfg.perception, n, self.device, cfg.seed, contract,
                                               self.control_dt, ray_count=ray_count,
                                               ray_max_range_m=ray_max_range_m)
        self.runtime = None
        self.sea = None
        self.sea_cfg = None
        self._spectral = None
        self.wind = None
        self._wind_heading_stats = None                    # G16: override runs with wind only
        self._ticks = np.zeros(n, dtype=np.int64)          # host mirror of runtime ticks
        self._pending_configure = np.zeros(n, dtype=bool)
        self._ever_reset = np.zeros(n, dtype=bool)
        self._actuator = None
        self._band_over_fp = None
        # F0 / D5: estimate bookkeeping (per group, per env, per episode)
        self._estimate_calls = {g: np.zeros(n, dtype=np.int64) for g in ("pos", "vel", "ray")}
        self._group_mode = {}            # group -> "estimate" | "task_relative" (never both)
        # G19 (training only; rebased on v2.5 by G19b): per-env, per-episode suite gates of an attached
        # training mixture (conditions/training_mixture.py). None = no mixture = unchanged behaviour.
        self._mixture = None
        self._gates = None
        self._component = None
        self._mixture_counts = None
        self.context = dict(context or {})
        self.context.update({
            "layer_active": True, "layer": LAYER_KIND, "hydrodynamics_model": "hydro_v2",
            "family": family, "vehicle": self.spec.name,
            "observation_contract": {"family": contract.family, "contract": contract.contract,
                                     "groups": list(contract.groups), "hooks_wired": contract.hooks_wired},
            "physics_dt_s": self.physics_dt, "control_dt_s": self.control_dt,
            "grid_width": n,
            "clock": "HydroV2Runtime integer tick clock; the layer owns none (wind step counter is a "
                     "checked host mirror)",
            "observed": {"waves": False, "current": False, "wind": False},
            "direction_strata": dict(strata_module.describe(), rotation=self.rotation),
            "direction_offsets_rad": {"waves": cfg.waves.direction_offset_rad,
                                      "current": cfg.current.direction_offset_rad,
                                      "wind": cfg.wind.direction_offset_rad,
                                      "perception_bias": cfg.perception.bias_direction_offset_rad},
            "estimates": {"journal_contract": bool(getattr(contract, "journal", False)),
                          "estimate_groups": list(getattr(contract, "estimate_groups", ()) or ()),
                          "decimation": self.decimation,
                          "rule": "estimates per group x decimation == runtime ticks (D5, checked "
                                  "at every call and at episode_stamp)"},
            "dynamic_hazards": "D7 package conditions/dynamic owns them; C4 progress trigger inert (B01)",
            "pending_owner_decisions": PENDING_OWNER_DECISIONS,
            "scenario_protocol_version": SCENARIO_PROTOCOL_VERSION,
        })

    # ----------------------------------------------------- training mixture --
    LAYER_GATES = ("waves", "wind", "current", "actuator.cap", "actuator.gain", "actuator.lag",
                   "actuator.resistance", "perception.noise", "perception.bias", "perception.latency",
                   "perception.dropout")

    def attach_training_mixture(self, mixture) -> None:
        """G19 (training only): switch every suite axis per env and per episode (see training_mixture.py).

        The layer must have been built with the mixture's D1 stack; attach before the first reset.
        With a mixture attached, certificate_stamp() refuses (training is never evaluation evidence).
        """
        from . import training_mixture as tm
        if self._mixture is not None:
            raise RuntimeError("a training mixture is already attached")
        if bool(self._ever_reset.any()):
            raise RuntimeError("attach the training mixture before the first reset")
        if not isinstance(mixture, tm.TrainingMixture):
            raise TypeError("attach_training_mixture needs a training_mixture.TrainingMixture")
        if int(mixture.seed) != int(self.cfg.seed):
            raise tm.MixtureError(f"mixture seed {mixture.seed} differs from the layer seed {self.cfg.seed}")
        if self.wind_coefficient_override is not None:      # G19b: G16 override is diagnostic only
            raise tm.MixtureError("a layer with a wind-coefficient override (diagnostic only) cannot train")
        tm.check_layer_config(mixture, self.cfg)
        n = self.num_envs
        self._mixture = mixture
        self._gates = {g: torch.zeros(n, dtype=torch.bool, device=self.device) for g in self.LAYER_GATES}
        self._component = ["calm"] * n
        self._mixture_counts = {name: 0 for name in tm.COMPONENTS}
        self.perception.attach_training_gates()
        self.context["training_mixture"] = mixture.stamp()
        self.context["certificate_eligible"] = False

    @property
    def training_mixture(self):
        return self._mixture

    def training_component(self, env_id):
        """Component of the current episode of ``env_id`` (None without a mixture)."""
        return None if self._component is None else self._component[int(env_id)]

    def training_mixture_counts(self):
        """Episode starts per component since the mixture was attached."""
        return None if self._mixture_counts is None else dict(self._mixture_counts)

    def _apply_training_gates(self, ids, episodes):
        gates = self._mixture.gates(ids.numpy(), episodes.numpy())
        ids_d = ids.to(self.device)
        for name in self.LAYER_GATES:
            self._gates[name][ids_d] = torch.as_tensor(gates[name], dtype=torch.bool).to(self.device)
        names = self._mixture.components(ids.numpy(), episodes.numpy())
        for i, name in zip(ids.tolist(), names):
            self._component[int(i)] = name
            self._mixture_counts[name] += 1
        self.perception.set_training_gates(
            ids, {axis: gates["perception." + axis] for axis in ("noise", "bias", "latency", "dropout")})

    def _gate(self, name):
        return self._gates[name]

    def training_wave_rows(self, env_ids):
        """G19b (P1 on v2.5): the rows of ``env_ids`` whose wave-following spawn must run.

        Without a mixture: ``env_ids`` itself (the same object; every row spawns on the waves, T1.35a). With a
        mixture: only the rows whose waves gate is on; a gated-off row has zero wave amplitude and keeps the
        family's calm spawn exactly (it is never rewritten, as in a calm env)."""
        if self._mixture is None:
            return env_ids
        ids = torch.as_tensor(env_ids, dtype=torch.long)
        on = self._gate("waves")[ids.to(self.device)].to(ids.device)
        return ids[on]

    # ------------------------------------------------------------ runtime --
    def bind_runtime(self, runtime) -> None:
        """Attach the env's HydroV2Runtime (once, before the first reset)."""
        if self.runtime is not None:
            raise RuntimeError("a runtime is already bound")
        if int(runtime.num_envs) != self.num_envs:
            raise ValueError("runtime num_envs differs from the condition layer")
        if float(runtime.dt) != self.physics_dt:
            raise ValueError("runtime physics dt differs from the condition layer")
        profile = runtime.profile
        hydro = profile.hydro
        if not math.isclose(float(hydro.mass_kg), float(self.spec.mass_kg), rel_tol=1e-9):
            raise ValueError(f"runtime profile mass {hydro.mass_kg} kg is not the task vessel "
                             f"{self.spec.name} ({self.spec.mass_kg} kg)")
        self.runtime = runtime
        dtype = runtime.dtype
        self.basis = torch.as_tensor(profile.actor_to_hull, dtype=torch.float64).to(
            device=self.device, dtype=dtype)
        resolved = {}
        # ---- waves
        w = self.cfg.waves
        resolved["waves"] = {"enabled": w.enabled, "calm": (not w.enabled) or w.hs_m == 0.0}
        if w.enabled and w.hs_m > 0.0:
            hydro_v2 = _shared_module("hydro_v2")
            check_wave_band = _shared_module("hydro_v2_runtime").check_wave_band
            fp = 1.0 / w.tp_s
            required = hydro_v2.required_high_frequency_hz(hydro, fp) * w.tp_s
            hi = float(w.band_max_over_fp)
            if w.band_policy == "raise_to_vessel" and hi < required:
                hi = math.ceil(required * 100.0 - 1e-9) / 100.0
            lo = float(w.band_min_over_fp)
            n_components = max(int(w.n_components),
                               int(math.ceil(w.recurrence_min_s * (hi - lo) / w.tp_s - 1e-9)))
            self.sea_cfg = SeaStateCfg(
                enable=True, hs_range=(w.hs_m, w.hs_m), tp_range=(w.tp_s, w.tp_s),
                gamma_range=(w.gamma, w.gamma), direction_spread_rad=w.spreading_rad,
                n_components=n_components, v2_f_min_over_fp=lo, v2_f_max_over_fp=hi,
                calm_water_increment=True)
            check_wave_band(profile, self.sea_cfg)           # fail closed
            with torch.random.fork_rng(devices=[]):
                sea = SeaState(self.sea_cfg, self.num_envs, torch.device("cpu"), physics_version=2)
            sea.device = self.device
            for name, value in vars(sea).copy().items():
                if isinstance(value, torch.Tensor):
                    setattr(sea, name, value.to(self.device))
            self.sea = sea
            self._band_over_fp = (lo, hi)
            self._spectral_cls = hydro_v2.SpectralWaves
            realised_n = int(sea.amplitude.shape[1])
            df = (hi - lo) / realised_n
            resolved["waves"].update(
                hs_m=w.hs_m, tp_s=w.tp_s, gamma=w.gamma, spreading_rad=w.spreading_rad,
                band_min_over_fp=lo, band_max_over_fp=hi, band_requested_max_over_fp=w.band_max_over_fp,
                band_required_max_over_fp=required, band_policy=w.band_policy,
                n_components=realised_n, group_repeat_s=w.tp_s / df,
                grid_rule="C2 SeaState v2 midpoint grid over [lo, hi] * fp, per realisation",
                steepness=steepness(w.hs_m, w.tp_s, w.gamma), steepness_rule=w.steepness_rule,
                spectrum="JONSWAP, Hs renormalised per realisation (C2)",
                source="SpectralWaves.from_sea_state(C2 SeaState v2); never SeaState.forces",
                direction_mode=w.direction_mode, direction_rad=w.direction_rad)
        # ---- wind
        wi = self.cfg.wind
        resolved["wind"] = {"enabled": wi.enabled}
        if wi.enabled:
            windage = self.spec.windage
            applied = (self._vehicles.vessel_wind_speed_from_u10(self.spec, wi.u10_mps)
                       if wi.reference == "u10" else float(wi.u10_mps))
            kwargs = self._vehicles.to_c3_windage_kwargs(self.spec)
            if self.wind_coefficient_override is not None:      # G16 (diagnostic runs only)
                kwargs = dict(kwargs, **wind_coefficients_module.coefficient_override(self.wind_coefficient_override))
            wind_cfg = WindCfg(
                enable=True, mean_speed_mps=applied,
                reference_height_m=float(windage.lateral_centroid_height_above_water_m),
                direction_rad=0.0, direction_drift_rad_s=0.0,
                gust_intensity=float(wi.gust_intensity),
                gust_correlation_time_s=float(wi.gust_correlation_time_s),
                air_density_kg_m3=float(windage.air_density_kg_m3),
                windage=VesselWindage(**kwargs))
            offset = wind_coefficients_module.offset_value(self.wind_coefficient_override)
            if offset is None:
                self.wind = Wind(wind_cfg, self.num_envs, self.device, eval_seed=int(self.cfg.seed),
                                 dtype=dtype, rng_protocol=RNG_PROTOCOL_V2, check_finite=False)
            else:                                               # G16 diagnostic centroid-offset yaw term
                model = wind_coefficients_module.OffsetFossenCoefficients(
                    kwargs["c_x0"], kwargs["c_y0"], kwargs["c_n0"], offset)
                self.wind = Wind(wind_cfg, self.num_envs, self.device, eval_seed=int(self.cfg.seed),
                                 dtype=dtype, rng_protocol=RNG_PROTOCOL_V2, check_finite=False,
                                 coefficient_model=model)
            resolved["wind"].update(
                u10_mps=wi.u10_mps, reference=wi.reference,
                applied_vessel_height_speed_mps=applied,
                reference_height_m=float(windage.lateral_centroid_height_above_water_m),
                profile_law=("neutral_log U(z) = U10 ln(z/z0)/ln(10/z0), z0 = %g m" % windage.roughness_length_m
                             if wi.reference == "u10" else "none (vessel-height dose)"),
                gust_sigma_mps=applied * wi.gust_intensity, gust_intensity=wi.gust_intensity,
                gust_correlation_time_s=wi.gust_correlation_time_s,
                windage=dict(kwargs, air_density_kg_m3=windage.air_density_kg_m3,
                             status=windage.status),
                heel_lever="roll_lever_com_m (lateral centroid z minus COM z; D2 item 16)",
                rng_protocol=RNG_PROTOCOL_V2, backend="C3 tasks/_shared/wind.py Wind (Fossen first harmonic)")
            if self.wind_coefficient_override is not None:      # G16: absent at the pinned values
                resolved["wind"]["coefficient_override"] = dict(self.wind_coefficient_override)
                # diagnostic heading statistics (episode stamp wind.heading_relative_wind), override runs only
                self._wind_heading_stats = torch.zeros((self.num_envs, len(HEADING_STAT_COLUMNS)),
                                                       dtype=torch.float64, device=self.device)
        # ---- current / actuator / perception
        c = self.cfg.current
        resolved["current"] = {"enabled": c.enabled, "speed_mps": c.speed_mps,
                               "speed_table": [list(r) for r in c.speed_table],
                               "law": "uniform water velocity into the single water-relative hull law",
                               "direction_mode": c.direction_mode}
        a = self.cfg.actuator
        resolved["actuator"] = {"enabled": a.enabled, "thrust_cap": a.thrust_cap,
                                "motor_gain_one_sided": a.motor_gain_one_sided,
                                "additional_lag_s": a.additional_lag_s,
                                "resistance_scale": a.resistance_scale,
                                "side_balance": "thruster_mixer.one_sided_motor_gains over scenario "
                                                "ordinal pairs (2j, 2j+1), balance_seed = seed",
                                "mass_payload": "rejected under hydro_v2 (decision 11)"}
        p = self.perception
        resolved["perception"] = {"enabled": self.cfg.perception.enabled, "active_axes": dict(p.axes),
                                  "latency_frames": p.latency, "outage_frames": p.outage,
                                  "pos_noise_ar1_coefficient": p.a, "groups": sorted(p.groups)}
        self.context["resolved"] = resolved
        self.context["vessel_profile"] = {
            "registry_name": self.spec.name, "runtime_profile_name": getattr(profile, "name", None),
            "runtime_profile_status": getattr(profile, "status", None), "mass_kg": float(hydro.mass_kg),
            "asset_sha256": self.spec.asset_sha256, "windage_status": self.spec.windage.status,
        }
        self.context["rotational_damping"] = {
            "reference": ("hull rate minus filtered surface-normal rate"
                          if getattr(hydro, "rotational_relative_motion", False) else "inertial rate"),
            "rotational_relative_motion": bool(getattr(hydro, "rotational_relative_motion", False)),
            "angular_linear_damping": list(hydro.angular_linear_damping),
            "angular_quadratic_damping": list(hydro.angular_quadratic_damping)}
        fk = getattr(hydro, "froude_krylov_horizontal", None)
        self.context["froude_krylov_horizontal"] = (
            fk if fk is not None else "absent_in_runtime_kernel (pre-E2 HydroSpec)")
        added_mass = getattr(hydro, "added_mass_coefficient_horizontal", None)
        self.context["added_mass_coefficient_horizontal"] = (
            added_mass if added_mass is not None else "absent_in_runtime_kernel (pre-E2 HydroSpec)")
        self.context["saturation_policy"] = getattr(runtime, "saturation", None)
        self.context["footprint_filter"] = {
            "kind": "Gaussian hull average, RMS extents L/sqrt(12), B/sqrt(12) (hydro_v2.footprint_transfer)",
            "length_m": float(hydro.length_m), "beam_m": float(hydro.beam_m)}

    def _require_runtime(self):
        if self.runtime is None:
            raise RuntimeError("bind_runtime(runtime) must be called before use")
        return self.runtime

    def _require_configured(self):
        if bool(self._pending_configure.any()):
            raise RuntimeError("configure_runtime(runtime, env_ids) was not called after reset")
        if not bool(self._ever_reset.all()):
            raise RuntimeError("every env must be reset before the first physics substep")

    # -------------------------------------------------------------- reset --
    def reset_runtime(self, env_ids=None, progress=None, *, reference_axis_rad=None,
                      task_episode_ids=None, outage_start_s=None, reference_duration_s=None):
        """Start new episodes for ``env_ids`` (call AFTER runtime.reset(env_ids)).

        G28: ``reference_duration_s`` (reference route length / reference speed per reset row) feeds the
        perception outage start rule (perception_v3.OUTAGE_ROUTE_RULE); it is read only when the outage
        axis is active (every other cell is unchanged).

        ``env_ids=None`` is the legacy per-step progress call: under hydro_v2
        the C4 progress trigger is inert (D7 B01) and ``progress`` is ignored.
        F0 / D7: ``task_episode_ids`` (the task scenario counter of the reset
        rows) is mandatory for every reset.
        """
        if env_ids is None:
            return
        self._require_runtime()
        ids = torch.as_tensor(env_ids, dtype=torch.long).detach().cpu().flatten()
        if ids.numel() == 0:
            return
        if ids.unique().numel() != ids.numel() or bool(((ids < 0) | (ids >= self.num_envs)).any()):
            raise ValueError("reset ids must be unique and in bounds")
        if task_episode_ids is None:
            raise ValueError("task_episode_ids is mandatory at every condition reset (CONTEXT_R4 D7): "
                             "pass the task scenario counter of the reset rows")
        supplied = torch.as_tensor(task_episode_ids).detach().cpu().to(torch.long).flatten()
        if supplied.numel() != ids.numel():
            raise ValueError("task_episode_ids needs one episode index per reset env")
        expected = self.scenario.episode_indices()[ids] + 1
        if not torch.equal(expected, supplied):
            raise ValueError("condition episode counter diverged from the task scenario counter")
        self.scenario.reset_idx(ids)
        episodes = self.scenario.episode_indices()[ids]
        needs_axis = any((
            self.cfg.waves.enabled and self.cfg.waves.direction_mode == "strata" and self.sea is not None,
            self.cfg.current.enabled and self.cfg.current.direction_mode == "strata",
            self.cfg.wind.enabled and self.cfg.wind.direction_mode == "strata",
            self.perception.axes["pos_bias"]))
        if reference_axis_rad is None:
            if needs_axis:
                raise ValueError("stratified directions need the task reference axis at reset")
            axis = torch.zeros(ids.numel(), dtype=torch.float64)
        else:
            axis = torch.as_tensor(reference_axis_rad, dtype=torch.float64).detach().cpu().flatten()
            if axis.numel() != ids.numel() or not bool(torch.isfinite(axis).all()):
                raise ValueError("reference_axis_rad must be finite, one value per reset env")
        ordinals = strata_module.scenario_ordinals(ids, episodes, self.num_envs)
        strata = strata_module.stratum_indices(ordinals, self.cfg.seed)
        world = axis + strata_module.relative_angles(strata)
        ids_np = ids.numpy()
        self.episode_index[ids_np] = episodes.numpy()
        self.ordinal[ids_np] = ordinals.numpy()
        self.stratum[ids_np] = strata.numpy()
        self.reference_axis[ids] = axis
        self.stratum_world[ids] = world
        if self._mixture is not None:                       # G19: gates of the new episodes
            self._apply_training_gates(ids, episodes)
        if self.sea is not None:
            w = self.cfg.waves
            mean = (_offset(world, w.direction_offset_rad) if w.direction_mode == "strata"
                    else torch.full_like(world, w.direction_rad))
            self.sea.resample(ids.to(self.device), self.scenario, mean_direction_rad=mean)
            if self._mixture is not None:
                # G19: a gated-off row is calm water (zero amplitude reproduces calm exactly, hydro_v2.py doc)
                off = ids.to(self.device)[~self._gate("waves")[ids.to(self.device)]]
                if off.numel():
                    self.sea.amplitude[off] = 0.0
            self._spectral = self._spectral_cls.from_sea_state(self.sea)   # validates the tensors
        if self.wind is not None:
            wi = self.cfg.wind
            direction = (_offset(world, wi.direction_offset_rad) if wi.direction_mode == "strata"
                         else torch.full_like(world, wi.direction_rad))
            self.wind.reset(ids, episodes, direction_rad=direction)
            if self._wind_heading_stats is not None:           # G16 diagnostic statistics
                self._wind_heading_stats[ids.to(self.device)] = 0.0
        self.perception.reset(ids, episodes, strata,
                              _offset(axis, self.cfg.perception.bias_direction_offset_rad),
                              outage_start_s=outage_start_s, reference_duration_s=reference_duration_s)
        for calls in self._estimate_calls.values():
            calls[ids_np] = 0
        self._ticks[ids_np] = 0
        self._pending_configure[ids_np] = True
        self._ever_reset[ids_np] = True
        self._current_unit = None

    def _current_direction_world(self):
        c = self.cfg.current
        if c.direction_mode == "fixed":
            return torch.full((self.num_envs,), float(c.direction_rad), dtype=torch.float64)
        return _offset(self.stratum_world.clone(), c.direction_offset_rad)

    def configure_runtime(self, runtime, env_ids) -> None:
        """Latch the actuator doses of the new episodes into the runtime."""
        if runtime is not self._require_runtime():
            raise ValueError("configure_runtime must receive the bound runtime")
        ids = torch.as_tensor(env_ids, dtype=torch.long).detach().cpu().flatten()
        if ids.numel() == 0:
            return
        if bool((self.episode_index[ids.numpy()] < 0).any()):
            raise RuntimeError("reset_runtime must precede configure_runtime")
        ticks = runtime.ticks.detach().to("cpu")[ids]
        if bool((ticks != 0).any()):
            raise RuntimeError("runtime.reset(env_ids) must precede the condition reset/configure")
        thruster_mixer = _shared_module("thruster_mixer")
        dtype, device = runtime.dtype, self.device
        a = self.cfg.actuator
        if self._actuator is None:
            self._actuator = {
                "cap": torch.ones((self.num_envs, 1), dtype=dtype, device=device),
                "gain": torch.ones((self.num_envs, 2), dtype=dtype, device=device),
                "lag": torch.zeros((self.num_envs, 1), dtype=dtype, device=device),
                "resistance": torch.ones(self.num_envs, dtype=dtype, device=device),
                "side": np.full(self.num_envs, -1, dtype=np.int64),
            }
        state = self._actuator
        ids_d = ids.to(device)
        if a.enabled and self._mixture is not None:
            # G19: each actuator sub-axis at its D1 value only where its gate is on, else neutral
            ordinals = torch.as_tensor(self.ordinal[ids.numpy()], dtype=torch.int64)
            gains = thruster_mixer.one_sided_motor_gains(float(a.motor_gain_one_sided), ordinals,
                                                         int(self.cfg.seed), dtype=torch.float64)
            sides = thruster_mixer.balanced_failure_sides(ordinals, int(self.cfg.seed))
            g = {k: self._gate("actuator." + k)[ids_d] for k in ("cap", "gain", "lag", "resistance")}
            one = torch.ones(ids.numel(), dtype=dtype, device=device)
            state["cap"][ids_d] = torch.where(g["cap"], one * float(a.thrust_cap), one).unsqueeze(-1)
            state["gain"][ids_d] = torch.where(g["gain"].unsqueeze(-1), gains.to(dtype=dtype, device=device),
                                               torch.ones((ids.numel(), 2), dtype=dtype, device=device))
            state["lag"][ids_d] = torch.where(g["lag"], one * float(a.additional_lag_s),
                                              torch.zeros_like(one)).unsqueeze(-1)
            state["resistance"][ids_d] = torch.where(g["resistance"], one * float(a.resistance_scale), one)
            state["side"][ids.numpy()] = np.where(g["gain"].cpu().numpy(), sides.numpy(), -1)
        elif a.enabled:
            ordinals = torch.as_tensor(self.ordinal[ids.numpy()], dtype=torch.int64)
            gains = thruster_mixer.one_sided_motor_gains(float(a.motor_gain_one_sided), ordinals,
                                                         int(self.cfg.seed), dtype=torch.float64)
            sides = thruster_mixer.balanced_failure_sides(ordinals, int(self.cfg.seed))
            state["cap"][ids_d] = float(a.thrust_cap)
            state["gain"][ids_d] = gains.to(dtype=dtype, device=device)
            state["lag"][ids_d] = float(a.additional_lag_s)
            state["resistance"][ids_d] = float(a.resistance_scale)
            state["side"][ids.numpy()] = sides.numpy()
        degradation = thruster_mixer.ThrusterDegradation(
            thrust_cap=state["cap"].clone(), motor_gain=state["gain"].clone(),
            additional_lag_s=state["lag"].clone())
        runtime.configure(degradation=degradation, resistance_scale=state["resistance"].clone())
        self._pending_configure[ids.numpy()] = False

    # ------------------------------------------------------ physics hooks --
    def current_velocity_world(self) -> torch.Tensor:
        """(a) world water velocity [N,3] at the runtime tick clock; Z = 0."""
        runtime = self._require_runtime()
        self._require_configured()
        out = torch.zeros((self.num_envs, 3), dtype=runtime.dtype, device=self.device)
        c = self.cfg.current
        if not c.enabled or (not c.speed_table and c.speed_mps == 0.0):
            return out
        if self._current_unit is None:
            direction = self._current_direction_world()
            self._current_unit = torch.stack((torch.cos(direction), torch.sin(direction)), -1).to(
                device=self.device, dtype=runtime.dtype)
        if c.speed_table:
            t = runtime.time().to(runtime.dtype)
            table = c.speed_table
            speed = torch.full_like(t, table[0][1])
            for (start, lo), (end, hi) in zip(table, table[1:]):
                fraction = ((t - start) / (end - start)).clamp(0, 1)
                speed = torch.where(t >= start, lo + (hi - lo) * fraction, speed)
            out[:, :2] = speed[:, None] * self._current_unit
        else:
            out[:, :2] = float(c.speed_mps) * self._current_unit
        if self._mixture is not None:                       # G19: still water where the gate is off
            out = torch.where(self._gate("current").unsqueeze(-1), out, torch.zeros_like(out))
        return out

    def wave_source(self):
        """(b) SpectralWaves of the current episodes, or None (calm / Hs = 0)."""
        self._require_runtime()
        self._require_configured()
        return self._spectral

    def wave_source_for_reset(self, env_ids):
        """T25 (owner decision T1.35a, wave-following spawn on by default): wave_source() for the spawn of the rows
        ``env_ids`` right after their configure_runtime. Only those rows must be configured; wave_source() requires
        EVERY env to have been reset, which a family's first partial reset (e.g. rows 3 and 1 of a fresh env) does
        not satisfy, while the spawn reads the reset rows only. Same object as wave_source() otherwise."""
        self._require_runtime()
        ids = torch.as_tensor(env_ids).detach().to("cpu").flatten().numpy().astype(np.int64)
        if bool(self._pending_configure[ids].any()) or not bool(self._ever_reset[ids].all()):
            raise RuntimeError("configure_runtime(runtime, env_ids) must run before the wave spawn of those rows")
        return self._spectral

    def heading_rad(self, actor_quat_wxyz) -> torch.Tensor:
        """Physical bow azimuth, computed exactly as HydroV2Runtime.step does."""
        runtime = self._require_runtime()
        rotation = rotation_matrix(actor_quat_wxyz.to(runtime.dtype)) @ self.basis
        return torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])

    def wind_wrench_world(self, com_velocity_w, actor_quat_wxyz):
        """(c) world wind force/torque about the COM at this tick, then one OU step."""
        runtime = self._require_runtime()
        self._require_configured()
        if self.wind is None:
            return None, None
        velocity = com_velocity_w[:, :2].to(device=self.device, dtype=runtime.dtype)
        heading = self.heading_rad(actor_quat_wxyz)
        force, torque = self.wind.forces(velocity, heading)
        if self._wind_heading_stats is not None:               # G16 diagnostic statistics
            self._accumulate_wind_heading(heading)
        self.wind.step(self.physics_dt)
        self._ticks += 1
        if self._mixture is not None:                       # G19: no wind load where the gate is off
            on = self._gate("wind").unsqueeze(-1)
            force = torch.where(on, force, torch.zeros_like(force))
            torque = torch.where(on, torque, torch.zeros_like(torque))
        return force, torque

    def _accumulate_wind_heading(self, heading):
        """G16: per tick, the heading relative to the TRUE wind, gamma = wrap(psi - psi_from) with
        psi_from = flow-to direction + pi (code convention: 0 = bow into the wind, +pi/2 = wind from
        starboard; a CCW bow rotation increases gamma). Accumulates the HEADING_STAT_COLUMNS sums; a
        non-finite heading (a code-9 episode) is counted in nonfinite_ticks and left out of the means."""
        psi_from = (self.wind.direction_rad + self.wind.cfg.direction_drift_rad_s * self.wind.time_s + math.pi)
        rel = heading.to(torch.float64) - psi_from.to(device=heading.device)
        gamma = torch.atan2(torch.sin(rel), torch.cos(rel))
        finite = torch.isfinite(gamma)
        gamma = torch.where(finite, gamma, torch.zeros_like(gamma))
        weight = finite.to(torch.float64)
        absolute = gamma.abs()
        self._wind_heading_stats += torch.stack(
            (weight, weight * gamma, weight * absolute, weight * (absolute - 0.5 * math.pi).abs(),
             weight * torch.cos(2.0 * gamma), 1.0 - weight), -1)

    def _wind_heading_record(self, i):
        sums = self._wind_heading_stats[i].detach().cpu().tolist()
        n = sums[0]
        nonfinite = int(sums[-1])
        if n <= 0:
            return {"ticks": 0, "nonfinite_ticks": nonfinite}
        mean = {name: value / n for name, value in zip(HEADING_STAT_COLUMNS[1:-1], sums[1:-1])}
        return {"ticks": int(n), "nonfinite_ticks": nonfinite,
                "definition": ("gamma = wrap(psi - psi_from) per physics tick, psi = physical bow azimuth, psi_from = "
                               "world flow-to direction + pi (true wind); 0 = bow into the wind, +pi/2 = wind from "
                               "starboard (code convention, tasks/_shared/wind.py)"),
                "mean_gamma_rad": mean["gamma"], "mean_abs_gamma_rad": mean["abs_gamma"],
                "mean_abs_offset_from_beam_rad": mean["abs_offset_from_beam"],
                "mean_cos_2gamma": mean["cos_2gamma"],
                # > 0: closer to beam-on than a uniform heading (pi/4); < 0: closer to the wind axis
                "beam_tendency_rad": 0.25 * math.pi - mean["abs_offset_from_beam"]}

    def physics_inputs(self, com_velocity_w, actor_quat_wxyz) -> PhysicsInputs:
        """(a)+(b)+(c) for ONE runtime.step call; call exactly once per substep."""
        current = self.current_velocity_world()
        waves = self.wave_source()
        force, torque = self.wind_wrench_world(com_velocity_w, actor_quat_wxyz)
        return PhysicsInputs(current, waves, force, torque)

    # -------------------------------------------------- legacy hooks: no --
    def _legacy(self, name):
        raise RuntimeError(f"{name} belongs to the legacy composition; under hydro_v2 the env calls "
                           "physics_inputs()/HydroV2Runtime.step instead (never SeaState.forces, "
                           "current wrenches, dynamics or scale_drag)")

    def command(self, *args, **kwargs):
        self._legacy("command")

    def scale_drag(self, *args, **kwargs):
        self._legacy("scale_drag")

    def environmental_wrench(self, *args, **kwargs):
        self._legacy("environmental_wrench")

    def dynamics(self, *args, **kwargs):
        self._legacy("dynamics")

    def hull_damping_scales(self, *args, **kwargs):
        self._legacy("hull_damping_scales")

    def action_in(self, actions):
        """Identity: the bounded action goes to the runtime mixer unchanged."""
        return actions

    # -------------------------------------------------------- perception --
    def wants(self, name):
        if name not in ("pos", "vel", "ray"):
            raise KeyError(name)
        return self.perception.wants(name)

    def _claim_group(self, name, mode):
        previous = self._group_mode.setdefault(name, mode)
        if previous != mode:
            raise RuntimeError(f"group {name!r} is already driven through the {previous} entry point; "
                               "estimate_* and observation_out share internals and cannot be mixed")

    def observation_out(self, name, measurement):
        """E3 task-relative entry point (legacy contracts only; refused on journal contracts)."""
        if getattr(self.contract, "journal", False):
            raise RuntimeError("journal observation contracts degrade the own-state estimate: use "
                               "estimate_position/estimate_velocity/estimate_rays (CONTEXT_R4 D5)")
        self._claim_group(name, "task_relative")
        return self.perception.apply(name, measurement)

    apply = observation_out

    # ---------------------------------------------- F0 / D5 own-state estimates --
    def _estimate(self, name, measurement):
        runtime = self._require_runtime()
        self._require_configured()
        self._claim_group(name, "estimate")
        ticks = runtime.ticks.detach().to("cpu").numpy()
        calls = self._estimate_calls[name]
        wrong = calls * self.decimation != ticks
        if bool(wrong.any()):
            i = int(np.flatnonzero(wrong)[0])
            raise RuntimeError(
                f"estimate_{_GROUP_WORD[name]}: env {i} already has {int(calls[i])} estimate(s) at runtime "
                f"tick {int(ticks[i])} (decimation {self.decimation}); each group is estimated exactly once "
                "per control step (CONTEXT_R4 D5)")
        out = self.perception.estimate(name, measurement)
        calls += 1
        return out

    def estimate_position(self, true_com_xy_local):
        """[N,2] estimated env-local COM xy (truth + e_n + e_b); once per control step."""
        return self._estimate("pos", true_com_xy_local)

    def estimate_velocity(self, body_rates):
        """[N,3] estimated (surge m/s, sway m/s, yaw rate rad/s); once per control step."""
        return self._estimate("vel", body_rates)

    def estimate_rays(self, ranges_m):
        """[N,rays] estimated ranges (m), clamped to [0, max range]; once per control step."""
        return self._estimate("ray", ranges_m)

    def estimate_counts(self, env_id):
        i = int(env_id)
        return {g: int(c[i]) for g, c in self._estimate_calls.items()}

    def _check_estimates(self, i, runtime_ticks):
        groups = list(getattr(self.contract, "estimate_groups", ()) or ())
        groups += [g for g, mode in self._group_mode.items() if mode == "estimate" and g not in groups]
        for g in groups:
            calls = int(self._estimate_calls[g][i])
            if calls * self.decimation != runtime_ticks:
                raise RuntimeError(
                    f"env {i}: group {g!r} was estimated {calls} times over {runtime_ticks} runtime ticks "
                    f"(decimation {self.decimation}): estimates x decimation must equal runtime ticks "
                    "(exactly one estimate per group per control step, CONTEXT_R4 D5)")
        return groups

    def measurement_age_s(self, name):
        return self.perception.measurement_age_s(name)

    # ------------------------------------------------------------ stamps --
    def certificate_stamp(self):
        self._require_runtime()
        if self._mixture is not None:
            raise RuntimeError("a training mixture is attached: this layer never produces certificate evidence "
                               "(G19; training is not evaluation)")
        return stamp_conditions_v3(self.cfg, context=self.context)

    def wind_coefficient_stamp(self):
        """G16 header block: pinned vs used wind coefficients (usvbench.wind_coefficients/1)."""
        pinned = wind_coefficients_module.pinned_values(self.spec.windage)
        block = wind_coefficients_module.stamp(self.spec.name, pinned, self.wind_coefficient_override,
                                               self.cfg.wind.enabled)
        if self.wind is not None:
            model = self.wind.coefficient_model
            applied = {name: float(getattr(model, name)) for name in wind_coefficients_module.NAMES}
            if applied != block["used"]:
                raise RuntimeError(f"wind backend coefficients {applied} differ from the stamp {block['used']}")
            offset = getattr(model, "offset_sl", None)
            if offset != block["centroid_offset"]["offset_sl"]:
                raise RuntimeError(f"wind backend centroid offset {offset} differs from the stamp "
                                   f"{block['centroid_offset']['offset_sl']}")
        return block

    def _check_ticks(self, i):
        if self.wind is None:
            return None
        runtime_ticks = int(self.runtime.ticks[i])
        if runtime_ticks != int(self._ticks[i]) or self.wind.physics_steps(i) != int(self._ticks[i]):
            raise RuntimeError(f"env {i}: wind steps {self.wind.physics_steps(i)} / mirror "
                               f"{int(self._ticks[i])} differ from runtime ticks {runtime_ticks}; "
                               "wind_wrench_world must be called once per runtime.step")
        return runtime_ticks

    def episode_stamp(self, env_id):
        """Latched by the env for the COMPLETED episode, before its reset."""
        runtime = self._require_runtime()
        i = int(env_id)
        if self.episode_index[i] < 0:
            raise RuntimeError(f"env {i} has no episode to stamp")
        if bool(self._pending_configure[i]):
            raise RuntimeError(f"env {i}: configure_runtime was never called for this episode")
        self._check_ticks(i)
        estimate_groups = self._check_estimates(i, int(runtime.ticks[i]))
        index = int(self.stratum[i])
        record = {
            "layer_active": True, "layer": LAYER_KIND,
            "env_index": i, "episode_index": int(self.episode_index[i]),
            "scenario_ordinal": int(self.ordinal[i]),
            "runtime_ticks": int(runtime.ticks[i]),
            "elapsed_physics_s": float(int(runtime.ticks[i]) * self.physics_dt),
            "direction_stratum": {"index": index, "name": strata_module.stratum_name(index),
                                  "class": strata_module.STRATA_CLASSES[strata_module.stratum_name(index)],
                                  "relative_flow_to_rad": float(strata_module.STRATA_RAD[index]),
                                  "world_flow_to_rad": float(self.stratum_world[i]),
                                  "reference_axis_rad": float(self.reference_axis[i])},
            "tensor_device_type": self.device.type,
            "direction_offsets_rad": {"waves": self.cfg.waves.direction_offset_rad,
                                      "current": self.cfg.current.direction_offset_rad,
                                      "wind": self.cfg.wind.direction_offset_rad,
                                      "perception_bias": self.cfg.perception.bias_direction_offset_rad},
            "estimates": {"decimation": self.decimation, "checked_groups": sorted(estimate_groups),
                          "counts": self.estimate_counts(i)},
        }
        c = self.cfg.current
        if c.enabled:
            record["current"] = {"world_flow_to_rad": float(self._current_direction_world()[i]),
                                 "speed_mps": c.speed_mps, "speed_table": [list(r) for r in c.speed_table]}
        if self.sea is not None:
            fp = 1.0 / float(self.sea.tp[i])
            lo, hi = self._band_over_fp
            record["waves"] = {
                "hs_m": float(self.sea.hs[i]), "tp_s": float(self.sea.tp[i]), "gamma": float(self.sea.gamma[i]),
                "mean_direction_rad": float(self.sea.mean_direction[i]),
                "n_components": int(self.sea.amplitude.shape[1]),
                # D7 rename: first/last realised component (midpoint grid), and the true band edges
                "component_frequency_range_hz": [float(self.sea.freqs[i, 0]), float(self.sea.freqs[i, -1])],
                "band_edges_hz": [lo * fp, hi * fp],
                "band_edges_over_fp": [lo, hi],
                "realisation_key": {"eval_seed": int(self.cfg.seed), "env_index": i,
                                    "episode_index": int(self.episode_index[i]), "group": "wave"},
                "realisation_sha256": _float32_digest(self.sea.amplitude[i], self.sea.phase[i],
                                                      self.sea.direction[i], self.sea.freqs[i]),
            }
        if self.wind is not None:
            record["wind"] = {
                "rng_protocol": RNG_PROTOCOL_V2, "physics_steps": self.wind.physics_steps(i),
                "episode_index": self.wind.episode_id(i),
                "world_flow_to_rad": float(self.wind.direction_rad[i]),
                "applied_vessel_height_speed_mps": float(self.wind.cfg.mean_speed_mps),
                "gust_mps_at_stamp": float(self.wind.gust_mps[i]),
            }
            if self._wind_heading_stats is not None:           # G16: override runs only
                record["wind"]["heading_relative_wind"] = self._wind_heading_record(i)
        if self._actuator is not None and self.cfg.actuator.enabled:
            side = int(self._actuator["side"][i])
            record["actuator"] = {
                "thrust_cap": float(self._actuator["cap"][i, 0]),
                "motor_gain_port_starboard": [float(v) for v in self._actuator["gain"][i]],
                "failing_side": None if (self.cfg.actuator.motor_gain_one_sided == 1.0 or side < 0) else
                ("port" if side == 0 else "starboard"),     # side -1: G19 training row with actuator.gain off
                "additional_lag_s": float(self._actuator["lag"][i, 0]),
                "resistance_scale": float(self._actuator["resistance"][i]),
            }
        record["perception"] = self.perception.episode_parameters(i)
        if self._mixture is not None:                       # G19: the episode's training component
            record["training_mixture"] = {
                "component": self._component[i],
                "gates_on": sorted(g for g in self.LAYER_GATES if bool(self._gates[g][i])),
                "mixture_sha256": self.context["training_mixture"]["mixture_sha256"],
                "certificate_eligible": False}
        return stamp_conditions_v3(self.cfg, runtime=record, context=self.context)

    def scenario_hashes(self, env_id, stamp=None):
        """D11: per-primitive SHA-256 hashes of the COMPLETED episode (pairing.py).

        Derived from the episode stamp only, so a certificate reader can
        recompute them (``pairing.condition_primitive_hashes``). Pass the
        stamp already latched for this episode to avoid recomputing it.
        """
        stamp = self.episode_stamp(env_id) if stamp is None else stamp
        return pairing_module.condition_primitive_hashes(stamp)


def build_hydro_v2_conditions(env_cfg, num_envs, device, *, family):
    """Factory for ``hydrodynamics_model="hydro_v2"``: always returns a layer."""
    from .layer import validate_no_legacy_overlap
    physics = getattr(env_cfg, "underwater_physics_cfg", None)
    if int(getattr(physics, "physics_version", 1)) != 2:
        raise ValueError("hydrodynamics_model='hydro_v2' requires physics_version=2")
    if family not in HYDRO_V2_FAMILIES:
        raise ValueError(f"family {family!r} is not a hydro_v2 condition family")
    raw = getattr(env_cfg, "conditions", None)
    presets = tuple(getattr(env_cfg, "condition_presets", ()) or ())
    seed = getattr(env_cfg, "seed", None)
    if seed is None or isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("hydro_v2 configurations must carry an integer evaluation seed (cfg.seed); a "
                         "missing seed is an error, never silently 0 (CONTEXT_R4 D7)")
    cfg = resolve_v3(presets, raw, seed=seed)
    if cfg.seed != seed:
        raise ValueError(f"condition seed {cfg.seed} differs from the env evaluation seed {seed}")
    validate_no_legacy_overlap(env_cfg, physics)
    contract = effective_contract(lookup(family, derive_contract(family, env_cfg)), env_cfg)
    if getattr(env_cfg, "obs_sea_state", False) or contract.sea_state_observed:
        if cfg.waves.enabled:
            raise ValueError("'waves observed' plus condition waves is rejected (D2 item 18)")
        raise ValueError("hydro_v2 journal configurations never observe the sea state")
    vehicle = getattr(env_cfg, "vehicle", None)
    if vehicle in ("blueboat", "wamv", "rov"):
        raise ValueError(f"vehicle {vehicle!r} is a legacy registry name; hydro_v2 needs a D5 v2 "
                         "registry name (e.g. 'blueboat_v2', 'wamv_vrx_v2_opt1')")
    ray_count, ray_range = _ray_sensor(env_cfg, contract)
    context = {"journal_mode": bool(getattr(env_cfg, "journal_mode", False)),
               "physics_version": 2,
               "legacy_current_enabled": bool(getattr(physics, "enable_current", False)),
               "legacy_waves_enabled": bool(getattr(getattr(env_cfg, "sea_state", None), "enable", False))}
    layer = HydroV2ConditionLayer(cfg, num_envs, device, family=family, vehicle=vehicle, contract=contract,
                                  physics_dt=float(env_cfg.sim.dt),
                                  control_dt=float(env_cfg.sim.dt) * int(env_cfg.decimation),
                                  ray_count=ray_count, ray_max_range_m=ray_range, context=context,
                                  wind_coefficients=getattr(env_cfg, "journal_wind_coefficients", None))
    mixture = getattr(env_cfg, "journal_training_mixture", None)
    if mixture is not None:                                  # G19: training-only mixture (cfg field set by
        from .training_mixture import MixtureError, check_stack    # the training entry point; journal_* is
        for field in ("journal_wind_coefficients", "journal_rollpitch_zeta"):  # protected from --set, D12)
            if getattr(env_cfg, field, None) is not None:    # G19b: diagnostic-only overrides never train
                raise MixtureError(f"cfg.{field} is a diagnostic-only override; a training run never sets it")
        resolved = getattr(env_cfg, "journal_conditions", None)
        if resolved is not None:
            check_stack(mixture, resolved)
        layer.attach_training_mixture(mixture)
    return layer


def _ray_sensor(env_cfg, contract):
    """(ray_count, max_range_m) of the contract's ray group, or (None, None).

    Journal contracts (D14): the task plan's ray sensor is authoritative
    (journal_cfg_v2 resolve: 36 rays, 30 m x vessel length scale); a cfg that
    declares a different sensor is refused (fail closed, the clamp range and
    the env's ray casts must agree).
    """
    if "ray" not in contract.groups:
        return None, None
    count = getattr(env_cfg, "ray_count", None)
    rng = getattr(env_cfg, "ray_max_range_m", None)
    if not getattr(contract, "journal", False):
        return count, rng
    plan = getattr(env_cfg, "journal_plan", None)
    if plan is None:
        if count is None or rng is None:
            raise ValueError("journal contract needs cfg.journal_plan or an explicit ray sensor "
                             "(ray_count, ray_max_range_m)")
        if int(count) != JOURNAL_RAY_BLOCK:
            raise ValueError("journal contracts carry the 36-ray block (D14)")
        return int(count), float(rng)
    params = plan.params
    want = (int(params["ray_sensor_ray_count"]), float(params["ray_sensor_max_range_m"]))
    if count is None and rng is None:
        return want
    if count is None or rng is None or int(count) != want[0] or not math.isclose(float(rng), want[1],
                                                                                 rel_tol=1e-12):
        raise ValueError(f"journal cfg ray sensor ({count}, {rng}) differs from the task plan's "
                         f"{want} (journal_cfg_v2: 36 rays, 30 m x length scale); set cfg.ray_count / "
                         "cfg.ray_max_range_m from cfg.journal_plan.params")
    return want


def _offset(angles, offset_rad):
    """Shared-stratum direction plus a per-suite offset (exact identity at 0)."""
    return angles if offset_rad == 0.0 else angles + float(offset_rad)


_GROUP_WORD = {"pos": "position", "vel": "velocity", "ray": "rays"}

__all__ = ["HydroV2ConditionLayer", "PhysicsInputs", "build_hydro_v2_conditions",
           "HYDRO_V2_FAMILIES", "PENDING_OWNER_DECISIONS", "LAYER_KIND"]
