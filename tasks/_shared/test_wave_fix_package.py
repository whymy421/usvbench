"""Revert-proof CPU tests for the 2026-08-26 wave fix package (F1-F7).

Three-way verification froze seven fixes into tasks/_shared/sea_state.py,
scripts/wave_acceptance.py and the six new Wave variant cfgs.  Each test here
is PROVEN to fail when its fix is reverted (scratch-copy protocol; failures
recorded in the fix-package report):

* F1 band: SeaStateCfg defaults 104 components over [0.04, 1.60] Hz
  (df = 0.0150 Hz); worst-corner (Tp=1.5, gamma=1) analytic capture 96.30%.
* F2 steepness admission: any (hs_range, tp_range) box whose steepest corner
  exceeds Sp = Hs/(g*Tp^2/(2*pi)) = 1/7 (DNV-RP-C205) is refused at
  construction; the certified StationKeep-BlueBoat-Wave box is grandfathered
  BY VALUE with a RuntimeWarning, never silently.
* F3 hs_norm: the sea-intensity observation channel divides by the FROZEN
  cfg.hs_norm_ref (0.60 m), not the per-config hs_range[1]; bit-identical on
  the certified path, and a pinned rung no longer reads 1.0.
* F4 dual calm reference: zero_amplitude_drag_force quantifies the quadratic
  hull drag forces() applies at Hs=0 (physics deliberately unchanged).
* F5 analytic capture gate: analytic_spectrum_capture measures band capture
  against the ANALYTIC spectrum, which the Hs renormalisation can never show
  (renorm scale 1.024 at Tp=2.25/gamma=3.3 on the old band while 37% of the
  variance was missing); scripts/wave_acceptance.py gates on it at 95%.
* F6 frozen evaluation box for the six new Wave variants: every corner of
  (Hs 0.30-0.60, Tp 2.0-2.5, gamma 1-5) passes F2 and captures >= 95%.
  (The cfg-side pin itself is machine-checked by test_wave_variants.py.)
* F7 slope-channel disclosure: the band extension roughly doubles slope RMS
  at unchanged labels; the disclosure text is pinned into sea_state.py and
  wave_acceptance.py so pre/post-band moment channels are never read as
  comparable.

Run: python tasks/_shared/test_wave_fix_package.py
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import torch

try:
    from . import sea_state
    from .sea_state import SeaState, SeaStateCfg
except ImportError:  # direct execution
    import sea_state
    from sea_state import SeaState, SeaStateCfg

SHARED = Path(__file__).resolve().parent
REPO = SHARED.parent.parent
WAVE_ACCEPTANCE = REPO / "scripts" / "wave_acceptance.py"

DEVICE = torch.device("cpu")

# The certified StationKeep-BlueBoat-Wave box -- frozen history, grandfathered.
CERTIFIED_HS = (0.30, 0.60)
CERTIFIED_TP = (1.5, 3.0)
# The frozen evaluation box for the six new Wave variants (F6).
FROZEN_HS = (0.30, 0.60)
FROZEN_TP = (2.0, 2.5)


def _certified_cfg(**overrides) -> SeaStateCfg:
    kwargs = dict(enable=True, hs_range=CERTIFIED_HS, tp_range=CERTIFIED_TP)
    kwargs.update(overrides)
    return SeaStateCfg(**kwargs)


def _quiet_sea(cfg: SeaStateCfg, num_envs: int = 8) -> SeaState:
    """Construct a SeaState with grandfather warnings silenced (asserted
    separately in test_f2)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SeaState(cfg, num_envs, DEVICE)


# ---------------------------------------------------------------------------
# F1 -- band defaults and the measured worst-corner capture basis.
# ---------------------------------------------------------------------------
def test_f1_band_defaults_are_the_frozen_2026_08_26_band():
    cfg = SeaStateCfg()
    assert cfg.n_components == 104, cfg.n_components
    assert cfg.f_min == 0.04, cfg.f_min
    assert cfg.f_max == 1.60, cfg.f_max
    sea = _quiet_sea(SeaStateCfg(), num_envs=2)
    # df must keep the historical resolution: (1.60 - 0.04) / 104 = 0.0150 Hz.
    assert abs(float(sea.df) - 0.0150) < 1.0e-12, float(sea.df)
    # Cell-centred grid spans the band symmetrically.
    freqs = sea.freqs
    assert abs(float(freqs[0]) - (0.04 + 0.0150 / 2.0)) < 1.0e-6
    assert abs(float(freqs[-1]) - (1.60 - 0.0150 / 2.0)) < 1.0e-6


def test_f1_worst_corner_capture_is_the_frozen_96_30_percent():
    # Measured basis of the band decision: 104 @ [0.04, 1.60], worst corner
    # Tp = 1.5 s / gamma = 1 (P-M tail, fattest spectrum), any Hs (capture is
    # Hs-independent: numerator and denominator both scale with Hs^2).
    captured = sea_state.analytic_spectrum_capture(
        104, 0.04, 1.60, 0.60, 1.5, 1.0
    )
    assert abs(captured - 0.9630) < 5.0e-4, captured
    # The old band was catastrophic at the same corner -- the reason F1 exists.
    old = sea_state.analytic_spectrum_capture(30, 0.04, 0.50, 0.60, 1.5, 1.0)
    assert old < 0.05, old


# ---------------------------------------------------------------------------
# F2 -- steepness admission box (DNV-RP-C205), grandfathered certified box.
# ---------------------------------------------------------------------------
def test_f2_steepness_formula():
    # Sp = Hs / (g * Tp^2 / (2*pi)); certified worst corner 0.60 m @ 1.5 s.
    sp = sea_state.significant_steepness(0.60, 1.5)
    assert abs(sp - 0.60 / (9.81 * 1.5**2 / (2.0 * math.pi))) < 1.0e-12
    assert abs(1.0 / sp - 5.855) < 0.01, 1.0 / sp  # the "~1/5.9" corner
    assert sp > sea_state.STEEPNESS_LIMIT


def test_f2_new_steep_box_is_refused_at_construction():
    # Not the certified box (tp floor 1.4 != 1.5) and beyond 1/7: must raise.
    steep = SeaStateCfg(enable=True, hs_range=(0.30, 0.60), tp_range=(1.4, 3.0))
    raised = False
    try:
        SeaState(steep, 4, DEVICE)
    except ValueError as exc:
        raised = True
        assert "1/7" in str(exc) and "REFUSED" in str(exc), str(exc)
    assert raised, "steep non-certified box constructed without a ValueError"

    # A steep PINNED rung is a box too and must be refused the same way.
    pinned = SeaStateCfg(enable=True, hs_range=(0.55, 0.55), tp_range=(1.5, 1.5))
    raised = False
    try:
        SeaState(pinned, 4, DEVICE)
    except ValueError:
        raised = True
    assert raised, "steep pinned rung constructed without a ValueError"


def test_f2_certified_box_is_grandfathered_loudly_not_silently():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SeaState(_certified_cfg(), 4, DEVICE)
    grandfather = [
        w for w in caught
        if issubclass(w.category, RuntimeWarning)
        and "grandfathered" in str(w.message)
    ]
    assert grandfather, (
        "certified box constructed silently; the grandfather RuntimeWarning "
        "must fire on every construction"
    )


def test_f2_frozen_evaluation_box_passes_without_any_warning():
    cfg = SeaStateCfg(enable=True, hs_range=FROZEN_HS, tp_range=FROZEN_TP)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SeaState(cfg, 4, DEVICE)
    assert not caught, [str(w.message) for w in caught]
    # Worst corner of the frozen box: Sp ~ 1/10.4, comfortably inside 1/7.
    sp = sea_state.significant_steepness(max(FROZEN_HS), min(FROZEN_TP))
    assert sp <= sea_state.STEEPNESS_LIMIT, sp
    assert abs(1.0 / sp - 10.41) < 0.05, 1.0 / sp


# ---------------------------------------------------------------------------
# F3 -- hs_norm divides by the frozen reference, bit-identical when certified.
# ---------------------------------------------------------------------------
def test_f3_hs_norm_ref_field_defaults_to_the_frozen_reference():
    assert SeaStateCfg().hs_norm_ref == 0.60


def test_f3_certified_path_is_bit_identical_to_the_old_denominator():
    # The certified StationKeep Wave id has hs_range[1] == 0.60 == the default
    # hs_norm_ref, so the observation must be BIT-IDENTICAL to the historical
    # hs / max(hs_range[1], 1e-6) formula -- torch.equal, not allclose.
    torch.manual_seed(1234)
    sea = _quiet_sea(_certified_cfg(), num_envs=64)
    forward = torch.zeros((64, 2))
    forward[:, 0] = 1.0
    obs = sea.observation(forward)
    historical = sea.hs / max(sea.cfg.hs_range[1], 1.0e-6)
    assert torch.equal(obs[:, 2], historical), (
        "certified-path hs_norm is not bit-identical to the historical "
        "hs / hs_range[1] formula"
    )


def test_f3_pinned_rung_reads_its_true_intensity_not_1():
    # The defect F3 fixes: a pinned rung hs_range=(x, x) used to read exactly
    # 1.0 at EVERY rung.  Against the frozen reference it must read x / 0.60.
    forward = torch.zeros((8, 2))
    forward[:, 0] = 1.0
    for rung in (0.30, 0.45, 0.60):
        cfg = SeaStateCfg(
            enable=True, hs_range=(rung, rung), tp_range=FROZEN_TP
        )
        sea = _quiet_sea(cfg)
        channel = sea.observation(forward)[:, 2]
        expected = rung / 0.60
        assert torch.allclose(
            channel, torch.full_like(channel, expected), atol=1.0e-6
        ), (rung, channel)
        if rung != 0.60:
            assert not torch.allclose(
                channel, torch.ones_like(channel), atol=1.0e-3
            ), f"rung {rung} still reads 1.0 -- F3 denominator reverted"


# ---------------------------------------------------------------------------
# F4 -- dual calm reference: the zero-amplitude force floor is quantified.
# ---------------------------------------------------------------------------
def test_f4_zero_amplitude_drag_helper_quantifies_the_floor():
    assert abs(sea_state.zero_amplitude_drag_force(0.1) - 0.08) < 1.0e-12
    assert abs(sea_state.zero_amplitude_drag_force(1.0) - 8.0) < 1.0e-12
    assert abs(sea_state.zero_amplitude_drag_force(3.0) - 72.0) < 1.0e-12


def test_f4_helper_matches_the_shipped_forces_at_zero_amplitude():
    # Hs = 0: orbital velocity vanishes, so the hull drag is exactly the
    # helper's value, opposing the hull's motion.  Physics unchanged -- this
    # pins the equality so the docstring numbers can be cited.
    sea = _quiet_sea(SeaStateCfg(enable=True, hs_range=(0.0, 0.0)))
    xy = torch.zeros((8, 2))
    for speed in (0.1, 1.0, 3.0):
        vel = torch.zeros((8, 2))
        vel[:, 0] = speed
        force, torque = sea.forces(xy, vel, 2.0)
        magnitude = float(force[:, :2].norm(dim=-1).max())
        expected = sea_state.zero_amplitude_drag_force(speed)
        assert abs(magnitude - expected) < 1.0e-4 * max(expected, 1.0), (
            speed, magnitude, expected
        )
        assert float(force[:, 0].max()) < 0.0, "drag must oppose the motion"
        assert float(torque.abs().max()) < 1.0e-9
    # enable=True with Hs -> 0 is NOT enable=False: the floor is nonzero at
    # speed -- the very fact the dose ladder must carry both calm references.
    assert sea_state.zero_amplitude_drag_force(1.0) > 0.0


def test_f4_dual_calm_reference_is_documented():
    assert "DUAL CALM REFERENCE" in (sea_state.__doc__ or "")
    for rel in (
        "tasks/hazard_nav/hazard_nav_env_cfg.py",
        "tasks/path_following/path_following_env_cfg.py",
        "tasks/path_hazard/path_hazard_env_cfg.py",
        "tasks/docking/docking_env_cfg.py",
        "tasks/harbor_mission/harbor_mission_env_cfg.py",
    ):
        text = (REPO / rel).read_text(encoding="utf-8")
        assert "DUAL CALM REFERENCE" in text, rel


# ---------------------------------------------------------------------------
# F5 -- analytic capture: the only check renormalisation cannot blind.
# ---------------------------------------------------------------------------
def test_f5_renormalisation_is_blind_to_truncation_but_the_gate_is_not():
    # Old band at Tp=2.25/gamma=3.3: the discretised, renormalised amplitudes
    # reproduce Hs EXACTLY (that is what resample enforces), while the band
    # holds only ~63% of the analytic variance.  An amplitude-based check
    # passes; the analytic gate fails.  This is the measured 1.024/37% pair.
    hs, tp, gamma = 0.45, 2.25, 3.3
    cfg = SeaStateCfg(
        enable=True,
        hs_range=(hs, hs),
        tp_range=(tp, tp),
        gamma_range=(gamma, gamma),
        n_components=30,
        f_max=0.50,
    )
    torch.manual_seed(7)
    sea = _quiet_sea(cfg, num_envs=16)
    m0_realised = float((sea.amplitude[0] ** 2 / 2.0).sum())
    hs_realised = 4.0 * math.sqrt(m0_realised)
    assert abs(hs_realised - hs) < 1.0e-5, hs_realised  # amplitude check blind

    # Pre-normalisation scale factor on this band: sqrt(target_m0 / grid_m0).
    grid_m0 = float(
        sea_state.jonswap_spectrum(sea.freqs.double(), hs, tp, gamma).sum()
    ) * float(sea.df)
    scale = math.sqrt((hs / 4.0) ** 2 / grid_m0)
    assert abs(scale - 1.024) < 2.0e-3, scale  # the measured near-unity scale

    captured = sea_state.analytic_spectrum_capture(30, 0.04, 0.50, hs, tp, gamma)
    assert abs(captured - 0.6255) < 2.0e-3, captured
    assert captured < 0.95, "the analytic gate must fail the old band here"

    # Same point on the frozen band: comfortably above the gate.
    new = sea_state.analytic_spectrum_capture(104, 0.04, 1.60, hs, tp, gamma)
    assert new >= 0.99, new


def test_f5_module_formula_matches_the_class_spectrum():
    # analytic_spectrum_capture must measure the SAME spectrum the field
    # realises: pin jonswap_spectrum to SeaState._jonswap on the class grid.
    hs, tp, gamma = 0.60, 2.25, 3.3
    sea = _quiet_sea(
        SeaStateCfg(
            enable=True, hs_range=(hs, hs), tp_range=(tp, tp),
            gamma_range=(gamma, gamma),
        )
    )
    class_row = sea._jonswap(
        torch.tensor([hs]), torch.tensor([tp]), torch.tensor([gamma])
    )[0]
    module_row = sea_state.jonswap_spectrum(sea.freqs, hs, tp, gamma)
    assert torch.allclose(class_row, module_row, rtol=1.0e-5, atol=1.0e-10)


def test_f5_gamma1_quadrature_matches_the_closed_form_m0():
    # For gamma = 1 (Pierson-Moskowitz) m0 = Hs^2 / 16 in closed form, so a
    # full-band capture must approach 1 and the quadrature must agree with
    # the closed form through it.
    hs, tp = 0.60, 5.0
    wide = sea_state.analytic_spectrum_capture(4000, 0.001, 40.0, hs, tp, 1.0)
    assert abs(wide - 1.0) < 5.0e-3, wide


def test_f5_wave_acceptance_carries_the_capture_gate():
    text = WAVE_ACCEPTANCE.read_text(encoding="utf-8")
    assert "analytic_spectrum_capture" in text, "capture gate call missing"
    assert "CAPTURE_THRESHOLD = 0.95" in text, "95% threshold missing"
    assert "overall_pass = overall_pass and capture_ok" in text, (
        "capture verdict is not wired into the acceptance verdict"
    )
    assert "analytic_capture_pass" in text, "capture verdict missing from JSON"
    # The gate must be pre-normalisation and analytic -- the docstring carries
    # the reason (renorm scale 1.024 cancels truncation) as a permanent note.
    assert "1.024" in text, "the renorm-blindness rationale was dropped"


# ---------------------------------------------------------------------------
# F6 -- the frozen evaluation box passes F2 and F1 capture at every corner.
# ---------------------------------------------------------------------------
def test_f6_frozen_box_every_corner_admissible_and_captured():
    cfg = SeaStateCfg()  # gamma_range default (1.0, 5.0) is what variants use
    worst = 1.0
    for hs in FROZEN_HS:
        for tp in FROZEN_TP:
            sp = sea_state.significant_steepness(hs, tp)
            assert sp <= sea_state.STEEPNESS_LIMIT, (hs, tp, sp)
            for gamma in cfg.gamma_range:
                captured = sea_state.analytic_spectrum_capture(
                    cfg.n_components, cfg.f_min, cfg.f_max, hs, tp, gamma
                )
                worst = min(worst, captured)
                assert captured >= 0.95, (hs, tp, gamma, captured)
    # The measured worst corner of the frozen box: Tp=2.0, gamma=1 -> 98.8%.
    assert abs(worst - 0.9882) < 2.0e-3, worst


# ---------------------------------------------------------------------------
# F7 -- slope-channel disclosure: doubled slope RMS is documented, not hidden.
# ---------------------------------------------------------------------------
def test_f7_band_extension_roughly_doubles_slope_rms_at_fixed_labels():
    # The roll/pitch moment integrand ~ f^4 S(f) does not converge with
    # f_max: quantify the renormalised in-line slope RMS sqrt(sum(a^2 k^2/2))
    # on the old and new grids at the same (Hs, Tp, gamma) label.
    hs, tp, gamma = 0.45, 2.25, 3.3

    def slope_rms(n, f_min, f_max):
        df = (f_max - f_min) / n
        f = torch.linspace(f_min + df / 2.0, f_max - df / 2.0, n,
                           dtype=torch.float64)
        spectrum = sea_state.jonswap_spectrum(f, hs, tp, gamma)
        amplitude = torch.sqrt(2.0 * spectrum * df)
        m0 = float((amplitude**2 / 2.0).sum())
        scale = math.sqrt((hs / 4.0) ** 2 / m0)
        k = (2.0 * math.pi * f) ** 2 / sea_state.GRAVITY
        return math.sqrt(float(((amplitude * scale) ** 2 * k**2 / 2.0).sum()))

    old = slope_rms(30, 0.04, 0.50)
    new = slope_rms(104, 0.04, 1.60)
    assert abs(old - 0.088) < 2.0e-3, old
    assert abs(new - 0.178) < 2.0e-3, new
    assert 1.8 <= new / old <= 2.3, (old, new)


def test_f7_disclosure_is_pinned_in_module_and_acceptance_report():
    assert "SLOPE-CHANNEL DISCLOSURE" in (sea_state.__doc__ or "")
    text = WAVE_ACCEPTANCE.read_text(encoding="utf-8")
    assert "SLOPE-CHANNEL DISCLOSURE" in text
    assert "NOT comparable" in text


# ---------------------------------------------------------------------------
# Hygiene -- every file this package touched stays pure ASCII.
# ---------------------------------------------------------------------------
def test_sources_are_pure_ascii_without_control_characters():
    files = [
        SHARED / "sea_state.py",
        SHARED / "test_wave_fix_package.py",
        SHARED / "test_wave_variants.py",
        WAVE_ACCEPTANCE,
        REPO / "tasks/hazard_nav/hazard_nav_env_cfg.py",
        REPO / "tasks/path_following/path_following_env_cfg.py",
        REPO / "tasks/path_hazard/path_hazard_env_cfg.py",
        REPO / "tasks/docking/docking_env_cfg.py",
        REPO / "tasks/harbor_mission/harbor_mission_env_cfg.py",
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        for i, ch in enumerate(text):
            code = ord(ch)
            assert code < 128 and (code >= 32 or ch in "\n\r"), (
                f"{path.name}: non-ASCII or control char {code!r} at offset {i}"
            )


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # a reverted fix may raise, not assert
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failures else 0)
