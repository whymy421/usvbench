"""No-Isaac validation of scripts/cfg_override.py.

The module exists to let a queue pin one wave-ladder rung, so the tests are
written against that job: a cfg shaped like the real one (a nested
``sea_state`` holding two-element ranges) must accept a pinned rung, and must
REFUSE every way of getting it subtly wrong -- a typo'd path, a half-specified
range, a value that cannot cast. A silent no-op there would file the default
band's numbers under a rung label, which is the one failure mode that would
not show up in any downstream check.

Run:  python scripts/test_cfg_override.py
  or: python -m pytest scripts/test_cfg_override.py -v
"""

import ast
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cfg_override import (  # noqa: E402
    OverrideError,
    apply_overrides,
    coerce_like,
    resolve_owner,
    split_assignment,
    validate_assignments,
)


class _SeaStateCfg:
    def __init__(self):
        self.enable = False
        self.hs_range = (0.3, 0.6)
        self.tp_range = (1.5, 3.0)
        self.gamma_range = (1.0, 5.0)
        self.n_components = 104
        self.f_max = 1.60
        self.hs_norm_ref = 0.60


class _EnvCfg:
    def __init__(self):
        self.sea_state = _SeaStateCfg()
        self.observation_space = 9
        self.thrust_cap_scale = 1.0
        self.curriculum_frozen = False
        self.label = "calm"
        self.missing_type = None
        self.disabled_sea = None


def _expect_refusal(fn, needle):
    try:
        fn()
    except OverrideError as exc:
        assert needle in str(exc), f"wrong message: {exc}"
        return
    raise AssertionError(f"expected OverrideError containing {needle!r}")


def test_source_hygiene():
    """The repo bans control characters and non-ASCII in scripts."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "cfg_override.py")
    raw = io.open(path, "rb").read()
    ast.parse(raw.decode("utf-8"))
    assert not [b for b in raw if b < 32 and b not in (9, 10, 13)]
    assert not [b for b in raw if b > 127]


def test_split_assignment():
    assert split_assignment("a=1") == ("a", "1")
    assert split_assignment(" a.b = 1 ") == ("a.b", "1")
    # An empty value is legal: it is how a string field is cleared.
    assert split_assignment("a=") == ("a", "")
    _expect_refusal(lambda: split_assignment("no_equals"), "FIELD=VALUE")
    _expect_refusal(lambda: split_assignment("=3"), "FIELD=VALUE")


def test_resolve_owner_nested():
    cfg = _EnvCfg()
    owner, leaf = resolve_owner(cfg, "sea_state.hs_range")
    assert owner is cfg.sea_state and leaf == "hs_range"
    owner, leaf = resolve_owner(cfg, "thrust_cap_scale")
    assert owner is cfg and leaf == "thrust_cap_scale"


def test_resolve_owner_refuses_typos():
    cfg = _EnvCfg()
    _expect_refusal(lambda: resolve_owner(cfg, "sea_state.hs_rnge"),
                    "is not a cfg field")
    _expect_refusal(lambda: resolve_owner(cfg, "seastate.hs_range"),
                    "does not exist")
    _expect_refusal(lambda: resolve_owner(cfg, "disabled_sea.hs_range"),
                    "is None")


def test_coerce_scalars():
    assert coerce_like("0.75", 1.0) == 0.75
    assert coerce_like("15", 104) == 15
    assert coerce_like("hello", "calm") == "hello"
    for token in ("1", "true", "TRUE", "yes", "on"):
        assert coerce_like(token, False) is True
    for token in ("0", "false", "no", "off"):
        assert coerce_like(token, True) is False
    _expect_refusal(lambda: coerce_like("maybe", False), "not a boolean")
    _expect_refusal(lambda: coerce_like("1.5", 104), "not an int")
    _expect_refusal(lambda: coerce_like("abc", 1.0), "not a float")
    _expect_refusal(lambda: coerce_like("1", None), "cannot infer a cast")


def test_coerce_sequences():
    assert coerce_like("0.6,0.6", (0.3, 0.6)) == (0.6, 0.6)
    assert coerce_like("(0.6, 0.6)", (0.3, 0.6)) == (0.6, 0.6)
    assert coerce_like("[0.6,0.6]", (0.3, 0.6)) == (0.6, 0.6)
    assert coerce_like("1,2", [3, 4]) == [1, 2]
    assert isinstance(coerce_like("0.6,0.6", (0.3, 0.6)), tuple)
    # Arity is enforced, never broadcast: a one-value "range" is the exact
    # mistake that would mislabel a rung.
    _expect_refusal(lambda: coerce_like("0.6", (0.3, 0.6)), "expected 2")
    _expect_refusal(lambda: coerce_like("0.6,0.6,0.6", (0.3, 0.6)), "expected 2")


def test_apply_pins_a_wave_rung():
    cfg = _EnvCfg()
    applied = apply_overrides(cfg, [
        "sea_state.enable=true",
        "sea_state.hs_range=0.6,0.6",
        "sea_state.tp_range=2.25,2.25",
    ], echo=False)
    assert cfg.sea_state.enable is True
    assert cfg.sea_state.hs_range == (0.6, 0.6)
    assert cfg.sea_state.tp_range == (2.25, 2.25)
    assert applied == [
        ("sea_state.enable", True),
        ("sea_state.hs_range", (0.6, 0.6)),
        ("sea_state.tp_range", (2.25, 2.25)),
    ]
    # Untouched neighbours stay exactly as they were.
    assert cfg.sea_state.gamma_range == (1.0, 5.0)
    assert cfg.sea_state.n_components == 104
    assert cfg.sea_state.hs_norm_ref == 0.60


def test_empty_is_a_no_op():
    """The bit-identity contract for every pre-existing certified command."""
    cfg = _EnvCfg()
    before = (cfg.sea_state.hs_range, cfg.sea_state.tp_range,
              cfg.observation_space, cfg.thrust_cap_scale, cfg.label)
    assert apply_overrides(cfg, [], echo=False) == []
    assert apply_overrides(cfg, None, echo=False) == []
    after = (cfg.sea_state.hs_range, cfg.sea_state.tp_range,
             cfg.observation_space, cfg.thrust_cap_scale, cfg.label)
    assert before == after


def test_apply_refuses_and_stops():
    cfg = _EnvCfg()
    _expect_refusal(
        lambda: apply_overrides(cfg, ["sea_state.hs_rnge=0.6,0.6"], echo=False),
        "is not a cfg field")
    assert cfg.sea_state.hs_range == (0.3, 0.6)
    _expect_refusal(
        lambda: apply_overrides(cfg, ["missing_type=3"], echo=False),
        "current value is None")


def test_validate_assignments_uses_the_caller_sink():
    seen = []
    validate_assignments(["a=1", "b.c=2"], seen.append)
    assert seen == []
    validate_assignments(["oops"], seen.append)
    assert len(seen) == 1 and "FIELD=VALUE" in seen[0]


WIRED_CALLERS = (
    "classical_baseline.py",
    "eval_mppi.py",
    "eval_v6_frozen.py",
    "eval_obs_bridge.py",
)


def test_callers_import_the_shared_module():
    """Every wired evaluator must not keep a private copy of this logic.

    ``caster = type(current)`` is the fingerprint of the copied loop this
    module replaced; four scripts each had one, and each reached top-level
    fields only. If it reappears, a wave rung silently stops pinning.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for name in WIRED_CALLERS:
        src = io.open(os.path.join(here, name), encoding="utf-8").read()
        assert "from cfg_override import" in src, name
        assert "apply_overrides(env_cfg" in src, name
        assert "caster = type(current)" not in src, (
            f"{name} still has an inline copy of the override loop"
        )


def test_v6_still_stamps_applied_overrides():
    """eval_v6_frozen echoes what it applied into the output JSON.

    That stamp is how a gcert record proves which dose it actually carried,
    so the refactor must not drop it.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    src = io.open(os.path.join(here, "eval_v6_frozen.py"), encoding="utf-8").read()
    assert "applied_overrides = dict(apply_overrides(" in src
    assert "applied_overrides" in src.split("applied_overrides = dict(", 1)[1]


if __name__ == "__main__":
    failures = 0
    for key, value in sorted(dict(globals()).items()):
        if key.startswith("test_") and callable(value):
            try:
                value()
                print(f"PASS {key}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {key}: {type(exc).__name__}: {exc}")
    print(f"\n{'ALL PASS' if not failures else str(failures) + ' FAILED'}")
    sys.exit(1 if failures else 0)
