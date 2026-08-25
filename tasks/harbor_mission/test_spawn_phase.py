"""Guards for the staged-spawn option that isolates the dock phase.

Source-level checks, deliberately free of any isaaclab import so this runs on a
machine with no simulator (the repo's other admission tests do the same).

Two things must hold. First, IDENTITY: spawn_phase defaults to 0 and no
pre-existing config overrides it, so every certified HarborStage / HarborMission
number keeps meaning exactly what it meant. Second, COMPLETENESS: the reset path
has to use the staged spawn point everywhere it previously used the env origin --
a half-wired version would place the hull at the field exit while still measuring
clearance, path length and gate arming from the harbor mouth.

Run: python tasks/harbor_mission/test_spawn_phase.py
"""
from __future__ import annotations

from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
CFG = (HERE / "harbor_mission_env_cfg.py").read_text(encoding="utf-8")
ENV = (HERE / "harbor_mission_env.py").read_text(encoding="utf-8")
INIT = (HERE / "__init__.py").read_text(encoding="utf-8")


def config_bodies(text):
    """Map class name -> source body, for every @configclass in the file."""
    bodies = {}
    matches = list(re.finditer(r"^class (\w+)\(", text, flags=re.M))
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        bodies[match.group(1)] = text[match.end():stop]
    return bodies


def test_default_is_the_harbor_mouth():
    base = config_bodies(CFG)["HarborMissionEnvCfg"]
    assert re.search(r"^\s{4}spawn_phase: int = 0$", base, flags=re.M), \
        "spawn_phase must default to 0; a non-zero default silently moves every certified number"
    assert re.search(r"^\s{4}spawn_phase_jitter_m: float = 0\.0$", base, flags=re.M), \
        "jitter must default to 0.0 so the default reset draws no extra RNG"


def test_only_the_dock_phase_config_opts_in():
    optedin = {
        name for name, body in config_bodies(CFG).items()
        if re.search(r"^\s{4}spawn_phase(: int)? = [1-9]", body, flags=re.M)
    }
    assert optedin == {"HarborDockPhaseEnvCfg"}, (
        "exactly one config may opt into a staged spawn; found " + repr(sorted(optedin))
    )


def test_dock_phase_config_shape():
    body = config_bodies(CFG)["HarborDockPhaseEnvCfg"]
    assert re.search(r"spawn_phase: int = 2", body)
    assert re.search(r"spawn_phase_jitter_m: float = 2\.0", body)
    assert re.search(r"mission_depth: int = 3", body), \
        "the dock phase still scores the ordered mission's M3"
    assert re.search(r"observation_space = 49", body), \
        "must speak the 49-D staged contract so bridged policies keep velocity channels"


def test_validation_rejects_unsupported_stages():
    assert "spawn_phase must be 0 (harbor mouth) or 2 (field exit)" in CFG
    assert "spawn_phase_jitter_m must be non-negative" in CFG


def test_reset_uses_the_staged_spawn_everywhere():
    reset = ENV[ENV.index("def _reset_idx"):]
    # The four consumers that previously read origins_xy directly.
    assert "root_state[:, :2] = spawn_xy - com_offset_world[:, :2]" in reset, "hull placement"
    assert "self._previous_xy[env_ids] = spawn_xy" in reset, "path-length origin"
    assert re.search(r"analytic_min_clearance\(\s*spawn_xy,", reset), "initial clearance"
    arming = reset[reset.index("arm_gate_approach("):]
    assert re.match(r"arm_gate_approach\([^;]*?\n\s+spawn_xy,\n\s+gate1,", arming), "gate arming"
    # And the latches that must move with it, or phase 2 would re-run phase 0.
    assert "self.phase[env_ids] = start_phase" in reset
    assert "self._max_phase[env_ids] = start_phase" in reset
    assert "self._phase_at_step_start[env_ids] = start_phase" in reset
    assert "self._m1[env_ids] = start_phase >= 1" in reset
    assert "self._m2[env_ids] = start_phase >= 2" in reset
    assert "self._exit_gate_progress[env_ids] = 2 if start_phase >= 1 else 0" in reset


def test_spawn_point_is_the_field_exit():
    reset = ENV[ENV.index("def _reset_idx"):]
    assert "local_exits_np[row] = route.field_exit" in reset, \
        "the hand-off point is the field exit, which is where stage 2 begins"
    assert re.search(r"spawn_xy = origins_xy \+ torch\.as_tensor\(\s*local_exits_np \+ offsets",
                     reset), "staged spawn must be env origin + field exit + jitter"
    assert "else:\n            spawn_xy = origins_xy" in reset, \
        "the default branch must still be the bare env origin"


def test_registered():
    assert 'id="Isaac-USV-HarborDockPhase-Direct-v1"' in INIT
    assert "HarborDockPhaseEnvCfg" in INIT


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    print("OK" if not failures else f"{failures} FAILED")
    raise SystemExit(1 if failures else 0)
