"""Tests for scripts/check_scenario_independence.py (pure python, no Isaac).

Builds synthetic certificate JSONs in a temp directory -- one directory per
scenario so each case is audited in isolation -- and asserts the verdict for
EVERY classification branch plus the process exit code that makes the auditor
usable as a gate.

    python scripts/test_check_scenario_independence.py
"""

from __future__ import annotations

import ast
import io
import json
import os
import random
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_scenario_independence as csi  # noqa: E402

# Every test takes a `root` scratch directory. Standalone execution supplies it
# from main(); under pytest this fixture does, so `pytest scripts/` collects the
# file instead of erroring on an unknown argument. Module scope because the
# later gate tests re-audit fixture directories the earlier ones wrote, and
# pytest preserves definition order (which ORDER at the bottom mirrors).
try:
    import pytest

    @pytest.fixture(scope="module")
    def root(tmp_path_factory):
        return str(tmp_path_factory.mktemp("scenario_audit"))
except ImportError:  # pytest is not required to run this file
    pass


# ---------------------------------------------------------------------------
# Fixture construction
# ---------------------------------------------------------------------------

TASK = "Isaac-USV-StationKeep-BlueBoat-Direct-v1"
CHECKPOINT = "C:/usvb/fixture.pt"

# What a run ON the protocol stamps: the header block of
# tasks/_shared/scenario_draws.py:scenario_protocol_header().
PROTOCOL_HEADER = {"version": 2, "hash": "blake2b",
                   "note": "per (eval_seed, env, episode, group) blake2b "
                           "stream; skrl-proof"}
# What a run OFF the protocol stamps. Pinned against the shared definition by
# test_off_protocol_literal_matches_scenario_draws.
PROTOCOL_OFF = csi.SCENARIO_PROTOCOL_OFF


def write_cert(directory, name, seed, records, **meta):
    """Write one certificate with the real top-level schema.

    Field names copied from the ``json.dump`` at the end of
    scripts/eval_v6_frozen.py: task / level / seed / checkpoint / records.
    """
    payload = {"task": TASK, "level": 0, "seed": seed,
               "checkpoint": CHECKPOINT}
    payload.update(meta)
    payload["records"] = records
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "{}_e{}.json".format(name, seed))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    return path


def episodes(n, successes, tts, paths, d0=None, hashes=None, first_env=0):
    """Build n per-episode records in the real record schema."""
    out = []
    for i in range(n):
        record = {
            "env": first_env + i,
            "ep": 0,
            "success": bool(successes[i]),
            "tts_s": tts[i],
            "path_length_m": paths[i],
        }
        if d0 is not None:
            record["d0_m"] = d0[i]
        if hashes is not None:
            record["scenario_hashes"] = hashes[i]
        out.append(record)
    return out


def floats(rng, n, low=10.0, high=90.0):
    """n distinct floats, so the channel carries real per-episode entropy."""
    seen, out = set(), []
    while len(out) < n:
        value = round(rng.uniform(low, high), 9)
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def alternating(n):
    return [i % 2 == 0 for i in range(n)]


def hexes(tag, n, primitives):
    return [{p: "{}-{}-{:04d}".format(tag, p, i) for p in primitives}
            for i in range(n)]


# ---------------------------------------------------------------------------
# Driving the auditor
# ---------------------------------------------------------------------------

def audit_dir(directory, float_tol=0.0, chance_threshold=1e-2,
              require_hashes=False):
    paths = csi.expand_inputs([directory], False)
    certificates, skipped = csi.load_certificates(paths)
    pairs, unpaired, blocked = csi.build_pairs(certificates)
    results = [csi.audit_pair(arm, a, b, float_tol, chance_threshold,
                              require_hashes)
               for arm, a, b in pairs]
    return results, unpaired, blocked, skipped


def verdicts(results):
    return {result.arm: result.verdict for result in results}


def run_main(argv):
    stream = io.StringIO()
    code = csi.main(argv, stream=stream)
    return code, stream.getvalue()


def channel(result, name):
    for item in result.channels:
        if item.name == name:
            return item
    raise AssertionError("no channel {!r}; have {}".format(
        name, [c.name for c in result.channels]))


# ---------------------------------------------------------------------------
# Branch: CONFIRMED_DUPLICATE
# ---------------------------------------------------------------------------

def test_duplicate_numeric(root):
    """Same scenarios twice: every continuous fingerprint is byte-identical."""
    directory = os.path.join(root, "dup_numeric")
    rng = random.Random(1)
    n = 64
    success = alternating(n)
    tts = floats(rng, n, 5.0, 40.0)
    path = floats(rng, n)
    records = episodes(n, success, tts, path)
    write_cert(directory, "dup", 42, records)
    write_cert(directory, "dup", 123, json.loads(json.dumps(records)))

    results, _, _, _ = audit_dir(directory)
    assert len(results) == 1, results
    result = results[0]
    assert result.verdict == "CONFIRMED_DUPLICATE", result.verdict
    assert result.n_common == n, result.n_common
    assert channel(result, "success").status == "IDENTICAL"
    assert channel(result, "path_length_m").status == "IDENTICAL"
    assert csi.success_cell(result) == "yes"


def test_duplicate_from_hashes_alone(root):
    """All-fail pair with nothing but scenario hashes: still a duplicate.

    This is the case a success-pattern census must call undecidable.
    """
    directory = os.path.join(root, "dup_hash")
    n = 64
    hashes = hexes("shared", n, ("spawn", "wave", "current"))
    records = episodes(n, [False] * n, [None] * n, [0.0] * n, hashes=hashes)
    for record in records:
        del record["path_length_m"]
    write_cert(directory, "duph", 42, records)
    write_cert(directory, "duph", 123, json.loads(json.dumps(records)))

    results, _, _, _ = audit_dir(directory)
    result = results[0]
    assert result.verdict == "CONFIRMED_DUPLICATE", result.verdict
    assert channel(result, "success").status == "UNINFORMATIVE_MATCH"
    for primitive in ("spawn", "wave", "current"):
        assert channel(result, "hash:" + primitive).status == "IDENTICAL"
    assert csi.hash_cell(result) == "3/3 id", csi.hash_cell(result)
    assert csi.success_cell(result) == "yes*"
    assert "per-primitive hashes (3)" == result.coverage()


def test_duplicate_survives_shuffled_record_order(root):
    """Alignment is by (env, ep), not by list position.

    Records are appended in episode-completion order, so a positional zip
    compares unrelated episodes and would miss this duplicate.
    """
    directory = os.path.join(root, "dup_shuffled")
    rng = random.Random(2)
    n = 64
    records = episodes(n, alternating(n), floats(rng, n, 5.0, 40.0),
                       floats(rng, n))
    shuffled = json.loads(json.dumps(records))
    random.Random(9).shuffle(shuffled)
    assert [r["env"] for r in shuffled] != [r["env"] for r in records]
    write_cert(directory, "shuf", 42, records)
    write_cert(directory, "shuf", 123, shuffled)

    results, _, _, _ = audit_dir(directory)
    assert results[0].verdict == "CONFIRMED_DUPLICATE", results[0].verdict
    assert results[0].n_common == n


def test_duplicate_on_partial_episode_overlap(root):
    """Only the shared (env, ep) keys are compared, and that is enough."""
    directory = os.path.join(root, "dup_overlap")
    rng = random.Random(3)
    n = 64
    success = alternating(n)
    tts = floats(rng, n, 5.0, 40.0)
    path = floats(rng, n)
    a = episodes(n, success, tts, path, first_env=0)
    b = json.loads(json.dumps(a))[8:]          # seed 123 lost the first 8
    b.extend(episodes(8, [True] * 8, floats(rng, 8, 5.0, 40.0),
                      floats(rng, 8), first_env=64))
    write_cert(directory, "ovl", 42, a)
    write_cert(directory, "ovl", 123, b)

    results, _, _, _ = audit_dir(directory)
    result = results[0]
    assert result.verdict == "CONFIRMED_DUPLICATE", result.verdict
    assert result.n_common == n - 8, result.n_common
    assert any("episode keys not shared" in w for w in result.warnings), \
        result.warnings


# ---------------------------------------------------------------------------
# Branch: DEGENERATE_UNDECIDABLE
# ---------------------------------------------------------------------------

def test_degenerate_all_fail(root):
    """All-fail with a constant path length: nothing is proved either way."""
    directory = os.path.join(root, "degenerate")
    n = 64
    records = episodes(n, [False] * n, [None] * n, [5.0] * n)
    write_cert(directory, "deg", 42, records)
    write_cert(directory, "deg", 123, json.loads(json.dumps(records)))

    results, _, _, _ = audit_dir(directory)
    result = results[0]
    assert result.verdict == "DEGENERATE_UNDECIDABLE", result.verdict
    assert channel(result, "success").status == "UNINFORMATIVE_MATCH"
    assert channel(result, "path_length_m").status == "UNINFORMATIVE_MATCH"
    assert channel(result, "tts_s").status == "SKIPPED"
    assert "free" in result.reason


def test_degenerate_when_no_episode_key_is_shared(root):
    directory = os.path.join(root, "degenerate_disjoint")
    rng = random.Random(4)
    a = episodes(16, alternating(16), floats(rng, 16, 5.0, 40.0),
                 floats(rng, 16), first_env=0)
    b = episodes(16, alternating(16), floats(rng, 16, 5.0, 40.0),
                 floats(rng, 16), first_env=100)
    write_cert(directory, "disj", 42, a)
    write_cert(directory, "disj", 123, b)

    results, _, _, _ = audit_dir(directory)
    result = results[0]
    assert result.verdict == "DEGENERATE_UNDECIDABLE", result.verdict
    assert result.n_common == 0
    assert "share no (env, ep)" in result.reason


# ---------------------------------------------------------------------------
# Branch: CONFIRMED_INDEPENDENT
# ---------------------------------------------------------------------------

def test_independent_numeric(root):
    directory = os.path.join(root, "independent")
    rng_a, rng_b = random.Random(5), random.Random(6)
    n = 64
    a = episodes(n, alternating(n), floats(rng_a, n, 5.0, 40.0),
                 floats(rng_a, n), d0=floats(rng_a, n, 15.0, 35.0))
    b = episodes(n, [not v for v in alternating(n)],
                 floats(rng_b, n, 5.0, 40.0), floats(rng_b, n),
                 d0=floats(rng_b, n, 15.0, 35.0))
    write_cert(directory, "ind", 42, a)
    write_cert(directory, "ind", 123, b)

    results, _, _, _ = audit_dir(directory)
    result = results[0]
    assert result.verdict == "CONFIRMED_INDEPENDENT", result.verdict
    assert channel(result, "d0_m").status == "DIFFERING"
    assert result.coverage() == "spawn/goal geometry only"
    assert csi.success_cell(result) == "no"


def test_independent_by_hashes(root):
    """The healthy case: hashes differ AND both sides ran on the protocol."""
    directory = os.path.join(root, "independent_hash")
    rng_a, rng_b = random.Random(7), random.Random(8)
    n = 64
    primitives = ("spawn", "wave")
    a = episodes(n, alternating(n), floats(rng_a, n, 5.0, 40.0),
                 floats(rng_a, n), hashes=hexes("s42", n, primitives))
    b = episodes(n, [not v for v in alternating(n)],
                 floats(rng_b, n, 5.0, 40.0), floats(rng_b, n),
                 hashes=hexes("s123", n, primitives))
    write_cert(directory, "indh", 42, a, scenario_protocol=PROTOCOL_HEADER)
    write_cert(directory, "indh", 123, b, scenario_protocol=PROTOCOL_HEADER)

    results, _, _, _ = audit_dir(directory)
    result = results[0]
    assert result.verdict == "CONFIRMED_INDEPENDENT", result.verdict
    assert csi.hash_cell(result) == "0/2 id", csi.hash_cell(result)
    assert result.coverage() == "per-primitive hashes (2)"
    assert result.protocol_status() == csi.PROTOCOL_ON
    assert csi.protocol_cell(result) == "on"
    assert not any("OFF PROTOCOL" in w for w in result.warnings), \
        result.warnings


# ---------------------------------------------------------------------------
# Branch: PARTIAL_SHORT_CIRCUIT
# ---------------------------------------------------------------------------

def test_partial_short_circuit_by_hashes(root):
    """Spawn follows the eval seed, the wave field does not."""
    directory = os.path.join(root, "partial_hash")
    rng_a, rng_b = random.Random(10), random.Random(11)
    n = 64
    shared_wave = ["wave-{:04d}".format(i) for i in range(n)]
    a = episodes(n, alternating(n), floats(rng_a, n, 5.0, 40.0),
                 floats(rng_a, n),
                 hashes=[{"spawn": "s42-{:04d}".format(i),
                          "wave": shared_wave[i]} for i in range(n)])
    b = episodes(n, [not v for v in alternating(n)],
                 floats(rng_b, n, 5.0, 40.0), floats(rng_b, n),
                 hashes=[{"spawn": "s123-{:04d}".format(i),
                          "wave": shared_wave[i]} for i in range(n)])
    write_cert(directory, "parh", 42, a)
    write_cert(directory, "parh", 123, b)

    results, _, _, _ = audit_dir(directory)
    result = results[0]
    assert result.verdict == "PARTIAL_SHORT_CIRCUIT", result.verdict
    assert channel(result, "hash:wave").status == "IDENTICAL"
    assert channel(result, "hash:spawn").status == "DIFFERING"
    assert csi.hash_cell(result) == "1/2 id", csi.hash_cell(result)
    assert "hash:wave" in result.reason


def test_partial_short_circuit_pinned_scenario_field(root):
    """No hashes: spawn geometry is pinned while the outcomes move.

    The success patterns differ, so a success-pattern census sees nothing.
    """
    directory = os.path.join(root, "partial_numeric")
    rng_a, rng_b = random.Random(12), random.Random(13)
    n = 64
    shared_d0 = floats(random.Random(14), n, 15.0, 35.0)
    a = episodes(n, alternating(n), floats(rng_a, n, 5.0, 40.0),
                 floats(rng_a, n), d0=list(shared_d0))
    b = episodes(n, [not v for v in alternating(n)],
                 floats(rng_b, n, 5.0, 40.0), floats(rng_b, n),
                 d0=list(shared_d0))
    write_cert(directory, "parn", 42, a)
    write_cert(directory, "parn", 123, b)

    results, _, _, _ = audit_dir(directory)
    result = results[0]
    assert result.verdict == "PARTIAL_SHORT_CIRCUIT", result.verdict
    assert channel(result, "d0_m").status == "IDENTICAL"
    assert channel(result, "path_length_m").status == "DIFFERING"
    assert csi.success_cell(result) == "no"
    assert "d0_m is pinned" in result.reason, result.reason


def test_partial_short_circuit_anomalous_fraction(root):
    """Some episodes share a scenario, most do not: no channel is total."""
    directory = os.path.join(root, "partial_anomalous")
    rng_a, rng_b = random.Random(15), random.Random(16)
    n = 64
    path_a = floats(rng_a, n)
    path_b = floats(rng_b, n)
    for i in range(40):                       # 40/64 episodes coincide
        path_b[i] = path_a[i]
    a = episodes(n, alternating(n), floats(rng_a, n, 5.0, 40.0), path_a)
    b = episodes(n, [not v for v in alternating(n)],
                 floats(rng_b, n, 5.0, 40.0), path_b)
    write_cert(directory, "para", 42, a)
    write_cert(directory, "para", 123, b)

    results, _, _, _ = audit_dir(directory)
    result = results[0]
    assert result.verdict == "PARTIAL_SHORT_CIRCUIT", result.verdict
    path_channel = channel(result, "path_length_m")
    assert path_channel.status == "ANOMALOUS", path_channel.describe()
    assert path_channel.n_match == 40, path_channel.n_match
    assert not result.by_status("IDENTICAL"), result.by_status("IDENTICAL")
    assert "40/64" in result.reason, result.reason


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

def test_training_seed_token_is_not_stripped(root):
    """cert_x_s42_e123 keeps s42: the training seed is a different ARM."""
    directory = os.path.join(root, "trainseed")
    rng = random.Random(17)
    n = 16
    for train in ("s42", "s43"):
        for seed in (42, 123):
            records = episodes(n, alternating(n), floats(rng, n, 5.0, 40.0),
                               floats(rng, n))
            write_cert(directory, "arm_" + train, seed, records)

    results, _, _, _ = audit_dir(directory)
    assert sorted(r.arm for r in results) == ["arm_s42", "arm_s43"], \
        [r.arm for r in results]
    assert csi.strip_eval_seed_token("cert_ringseal_s42_e123", 123) == \
        "cert_ringseal_s42"
    assert csi.strip_eval_seed_token("cert_ringseal_s42_e42", 42) == \
        "cert_ringseal_s42"


def test_identity_mismatch_blocks_pairing(root):
    """A different checkpoint is a different experiment, not a seed pair."""
    directory = os.path.join(root, "identity")
    rng = random.Random(18)
    n = 16
    write_cert(directory, "idm", 42,
               episodes(n, alternating(n), floats(rng, n, 5.0, 40.0),
                        floats(rng, n)))
    write_cert(directory, "idm", 123,
               episodes(n, alternating(n), floats(rng, n, 5.0, 40.0),
                        floats(rng, n)),
               checkpoint="C:/usvb/other.pt")

    results, _, blocked, _ = audit_dir(directory)
    assert results == [], results
    assert len(blocked) == 1
    assert "checkpoint" in blocked[0][2], blocked[0][2]


def test_run_counters_do_not_block_pairing(root):
    """planner.straight_fallbacks drifts between runs of the same arm."""
    directory = os.path.join(root, "counters")
    rng = random.Random(19)
    n = 16
    records = episodes(n, alternating(n), floats(rng, n, 5.0, 40.0),
                       floats(rng, n))
    write_cert(directory, "cnt", 42, records,
               planner={"controller": "planner_los_pid", "kp": 2.0,
                        "straight_fallbacks": 114, "legs_planned": 1597})
    write_cert(directory, "cnt", 123, json.loads(json.dumps(records)),
               planner={"controller": "planner_los_pid", "kp": 2.0,
                        "straight_fallbacks": 109, "legs_planned": 1649})

    results, _, blocked, _ = audit_dir(directory)
    assert blocked == [], blocked
    assert len(results) == 1
    assert results[0].verdict == "CONFIRMED_DUPLICATE"
    assert not any("metadata also differs" in w for w in results[0].warnings), \
        results[0].warnings


def test_unpaired_and_unreadable_inputs(root):
    directory = os.path.join(root, "loose")
    os.makedirs(directory, exist_ok=True)
    rng = random.Random(20)
    write_cert(directory, "lonely", 7,
               episodes(8, alternating(8), floats(rng, 8, 5.0, 40.0),
                        floats(rng, 8)))
    with open(os.path.join(directory, "scores.json"), "w",
              encoding="utf-8") as handle:
        json.dump([1, 2, 3], handle)
    with open(os.path.join(directory, "broken.json"), "w",
              encoding="utf-8") as handle:
        handle.write("{not json")

    results, unpaired, _, skipped = audit_dir(directory)
    assert results == []
    assert len(unpaired) == 1 and unpaired[0][0].stem == "lonely_e7"
    reasons = sorted(why.split(":")[0] for _, why in skipped)
    assert reasons == ["top level is list, not a certificate object",
                       "unreadable"], reasons


# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------

def test_float_tolerance_flips_the_verdict(root):
    """Bit-exact by default; --float-tol catches near-identical replays."""
    directory = os.path.join(root, "tolerance")
    rng = random.Random(21)
    n = 64
    path_a = floats(rng, n)
    tts_a = floats(rng, n, 5.0, 40.0)
    a = episodes(n, alternating(n), tts_a, path_a)
    b = episodes(n, alternating(n), [v + 1e-9 for v in tts_a],
                 [v + 1e-9 for v in path_a])
    write_cert(directory, "tol", 42, a)
    write_cert(directory, "tol", 123, b)

    strict, _, _, _ = audit_dir(directory, float_tol=0.0)
    assert strict[0].verdict == "PARTIAL_SHORT_CIRCUIT", strict[0].verdict
    assert channel(strict[0], "success").status == "IDENTICAL"

    loose, _, _, _ = audit_dir(directory, float_tol=1e-6)
    assert loose[0].verdict == "CONFIRMED_DUPLICATE", loose[0].verdict


def test_chance_threshold_controls_the_free_match_guard(root):
    """A 3-value channel matching everywhere is luck at n=8, not at n=64."""
    directory = os.path.join(root, "chance")
    n = 8
    values = [float(i % 3) for i in range(n)]
    records = episodes(n, alternating(n), [None] * n, values)
    write_cert(directory, "chc", 42, records)
    write_cert(directory, "chc", 123, json.loads(json.dumps(records)))

    strict, _, _, _ = audit_dir(directory, chance_threshold=1e-6)
    assert strict[0].verdict == "DEGENERATE_UNDECIDABLE", strict[0].verdict
    assert channel(strict[0], "path_length_m").status == "UNINFORMATIVE_MATCH"

    lenient, _, _, _ = audit_dir(directory, chance_threshold=0.5)
    assert lenient[0].verdict == "CONFIRMED_DUPLICATE", lenient[0].verdict


# ---------------------------------------------------------------------------
# Scenario-hash schema variants
# ---------------------------------------------------------------------------

def test_scenario_hash_schema_variants(root):
    n = 4
    keys = [(i, 0) for i in range(n)]

    per_record_dict = {"records": [
        {"env": i, "ep": 0, "scenario_hashes": {"spawn": "a%d" % i}}
        for i in range(n)]}
    parsed = csi.extract_scenario_hashes(
        per_record_dict, per_record_dict["records"], {})
    assert parsed == {"spawn": {k: "a%d" % k[0] for k in keys}}, parsed

    per_record_str = {"records": [
        {"env": i, "ep": 0, "scenario_hash": "b%d" % i} for i in range(n)]}
    parsed = csi.extract_scenario_hashes(
        per_record_str, per_record_str["records"], {})
    assert parsed == {"scenario": {k: "b%d" % k[0] for k in keys}}, parsed

    records = [{"env": i, "ep": 0} for i in range(n)]
    by_key = {(i, 0): (i, records[i]) for i in range(n)}
    top_list = {"scenario_hashes": {"wave": ["c%d" % i for i in range(n)]},
                "records": records}
    parsed = csi.extract_scenario_hashes(top_list, records, by_key)
    assert parsed == {"wave": {k: "c%d" % k[0] for k in keys}}, parsed

    top_map = {"scenario_hashes":
               {"wave": {"{}:0".format(i): "d%d" % i for i in range(n)}},
               "records": records}
    parsed = csi.extract_scenario_hashes(top_map, records, by_key)
    assert parsed == {"wave": {k: "d%d" % k[0] for k in keys}}, parsed

    assert csi.extract_scenario_hashes({"records": records}, records,
                                       by_key) == {}


def test_one_sided_hashes_are_not_evidence(root):
    """A new-format cert paired with an old one must not read as independent.

    Comparing a stamped hash against a missing one would manufacture a
    DIFFERING channel out of a schema gap.
    """
    directory = os.path.join(root, "hash_gap")
    n = 64
    hashed = episodes(n, [False] * n, [None] * n, [5.0] * n,
                      hashes=hexes("s42", n, ("spawn", "wave")))
    plain = episodes(n, [False] * n, [None] * n, [5.0] * n)
    write_cert(directory, "gap", 42, hashed)
    write_cert(directory, "gap", 123, plain)

    results, _, _, _ = audit_dir(directory)
    result = results[0]
    assert channel(result, "hash:spawn").status == "SKIPPED"
    assert channel(result, "hash:wave").status == "SKIPPED"
    assert result.hash_channels() == [], result.hash_channels()
    assert csi.hash_cell(result) == "-"
    assert result.verdict == "DEGENERATE_UNDECIDABLE", result.verdict
    assert any("one side only" in w for w in result.warnings), result.warnings


# ---------------------------------------------------------------------------
# The off-protocol marker (an outcome, not a footnote)
# ---------------------------------------------------------------------------

def _off_protocol_pair(root, name, protocol_a, protocol_b, tag="off"):
    """Two certificates whose hashes differ, stamped as asked.

    Hashes differ on every episode, so the numeric verdict is
    CONFIRMED_INDEPENDENT: exactly the shape that used to exit 0 with the
    off-protocol marker appearing nowhere in the report.
    """
    directory = os.path.join(root, name)
    rng_a, rng_b = random.Random(30), random.Random(31)
    n = 32
    primitives = ("spawn", "wave")
    a = episodes(n, alternating(n), floats(rng_a, n, 5.0, 40.0),
                 floats(rng_a, n), hashes=hexes("s42", n, primitives))
    b = episodes(n, [not v for v in alternating(n)],
                 floats(rng_b, n, 5.0, 40.0), floats(rng_b, n),
                 hashes=hexes("s123", n, primitives))
    meta_a = {} if protocol_a is None else {"scenario_protocol": protocol_a}
    meta_b = {} if protocol_b is None else {"scenario_protocol": protocol_b}
    write_cert(directory, tag, 42, a, **meta_a)
    write_cert(directory, tag, 123, b, **meta_b)
    return directory


def test_off_protocol_pair_is_reported(root):
    """Both sides off protocol: said out loud, per pair, in every rendering."""
    directory = _off_protocol_pair(root, "proto_off", PROTOCOL_OFF,
                                   PROTOCOL_OFF)
    results, _, _, _ = audit_dir(directory)
    result = results[0]
    # The numeric evidence is unchanged -- the marker is orthogonal to it.
    assert result.verdict == "CONFIRMED_INDEPENDENT", result.verdict
    assert result.protocol_status() == csi.PROTOCOL_OFF
    assert csi.protocol_cell(result) == "OFF"
    assert [c.stem for c in result.off_protocol_sides()] == \
        ["off_e42", "off_e123"], result.off_protocol_sides()
    assert any("OFF PROTOCOL" in w for w in result.warnings), result.warnings
    # Both sides carry the same marker, so the older "protocol differs" warning
    # cannot fire: this pair is precisely the one that used to pass in silence.
    assert not any("scenario_protocol differs" in w for w in result.warnings)

    code, text = run_main([directory])
    assert code == 0, text                    # still not a FAILING verdict
    assert PROTOCOL_OFF in text, text         # but no longer invisible
    assert "PROTO" in text and "OFF" in text, text
    assert "1 off" in text, text              # summary line, even without -v

    quiet_code, quiet_text = run_main([directory, "--quiet"])
    assert quiet_code == 0
    assert "1 off" in quiet_text, quiet_text


def test_off_protocol_pair_fails_the_require_hashes_gate(root):
    """--require-scenario-hashes cannot do its job on an off-protocol pair."""
    directory = _off_protocol_pair(root, "proto_off_gate", PROTOCOL_OFF,
                                   PROTOCOL_OFF, tag="offgate")
    assert run_main([directory, "--quiet"])[0] == 0
    code, text = run_main([directory, "--quiet", "--require-scenario-hashes"])
    assert code == 1, text
    assert "OFF_PROTOCOL (--require-scenario-hashes)" in text, text
    # The pair HAS hashes on both sides, so it is not the missing-hash rule
    # that fired.
    assert "NO_SCENARIO_HASHES" not in text, text


def test_mixed_protocol_pair_fails_the_require_hashes_gate(root):
    directory = _off_protocol_pair(root, "proto_mixed", PROTOCOL_HEADER,
                                   PROTOCOL_OFF, tag="mixed")
    results, _, _, _ = audit_dir(directory)
    result = results[0]
    assert result.protocol_status() == csi.PROTOCOL_MIXED
    assert [c.stem for c in result.off_protocol_sides()] == ["mixed_e123"]
    assert any("scenario_protocol differs" in w for w in result.warnings)
    assert any("OFF PROTOCOL" in w for w in result.warnings)

    assert run_main([directory, "--quiet"])[0] == 0
    code, text = run_main([directory, "--quiet", "--require-scenario-hashes"])
    assert code == 1, text
    assert "MIXED_PROTOCOL (--require-scenario-hashes)" in text, text


def test_unstamped_protocol_pair_fails_the_require_hashes_gate(root):
    """No scenario_protocol key at all is UNKNOWN, not a quiet pass."""
    directory = _off_protocol_pair(root, "proto_unstamped", None, None,
                                   tag="unstamped")
    results, _, _, _ = audit_dir(directory)
    result = results[0]
    assert result.protocol_status() == csi.PROTOCOL_UNSTAMPED
    assert result.off_protocol_sides() == []
    assert len(result.unstamped_sides()) == 2
    assert any("no scenario_protocol field" in w for w in result.warnings), \
        result.warnings

    assert run_main([directory, "--quiet"])[0] == 0
    code, text = run_main([directory, "--quiet", "--require-scenario-hashes"])
    assert code == 1, text
    assert "UNSTAMPED_PROTOCOL (--require-scenario-hashes)" in text, text


def test_off_protocol_reaches_both_report_formats(root):
    directory = _off_protocol_pair(root, "proto_report", PROTOCOL_OFF,
                                   PROTOCOL_OFF, tag="rep")
    csv_path = os.path.join(root, "protocol_report.csv")
    md_path = os.path.join(root, "protocol_report.md")
    assert run_main([directory, "--quiet", "--report", csv_path])[0] == 0
    assert run_main([directory, "--quiet", "--report", md_path])[0] == 0

    with open(csv_path, encoding="utf-8") as handle:
        csv_text = handle.read()
    assert "protocol_status" in csv_text.splitlines()[0]
    assert PROTOCOL_OFF in csv_text, csv_text

    with open(md_path, encoding="utf-8") as handle:
        md_text = handle.read()
    assert "| protocol |" in md_text, md_text
    assert "OFF" in md_text, md_text


def test_off_protocol_literal_matches_scenario_draws(root):
    """The auditor's copy of the marker vs the shared definition.

    The auditor is stdlib-only on purpose (it has to run as a CI gate on a
    machine with no torch), so it keeps its own copy of the literal. Read the
    definition out of tasks/_shared/scenario_draws.py by AST -- not by
    importing it -- and pin the two equal: a drifted marker would make the
    auditor blind to exactly the certificates it is meant to catch.
    """
    shared = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tasks", "_shared", "scenario_draws.py")
    tree = ast.parse(open(shared, encoding="utf-8").read(), filename=shared)
    literals = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and any(isinstance(target, ast.Name)
                and target.id == "SCENARIO_PROTOCOL_OFF"
                for target in node.targets)
    ]
    assert literals == [csi.SCENARIO_PROTOCOL_OFF], (
        "scenario_draws.py defines SCENARIO_PROTOCOL_OFF as {!r} but the "
        "auditor looks for {!r}".format(literals, csi.SCENARIO_PROTOCOL_OFF))


def test_describe_protocol_renders_each_state(root):
    class Fake:
        def __init__(self, protocol, off, unstamped):
            self.protocol = protocol
            self.off_protocol = off
            self.protocol_unstamped = unstamped

    assert csi.describe_protocol(
        Fake(PROTOCOL_HEADER, False, False)) == "v2/blake2b"
    assert csi.describe_protocol(
        Fake(PROTOCOL_OFF, True, False)) == PROTOCOL_OFF
    assert csi.describe_protocol(Fake(None, False, True)) == "unstamped"


def test_protocol_version_mismatch_is_reported(root):
    directory = os.path.join(root, "protocol")
    rng = random.Random(22)
    n = 16
    hashes = hexes("shared", n, ("spawn",))
    records = episodes(n, alternating(n), floats(rng, n, 5.0, 40.0),
                       floats(rng, n), hashes=hashes)
    write_cert(directory, "pro", 42, records,
               scenario_protocol={"version": 1, "hash": "blake2b"})
    write_cert(directory, "pro", 123, json.loads(json.dumps(records)),
               scenario_protocol={"version": 2, "hash": "blake2b"})

    results, _, _, _ = audit_dir(directory)
    assert any("scenario_protocol differs" in w for w in results[0].warnings), \
        results[0].warnings


# ---------------------------------------------------------------------------
# The protocol gate validates the header's SHAPE and CONTENT
#
# Presence of a scenario_protocol key used to be read as compliance: the
# auditor only ever compared the value against the off-protocol literal, so
# null, an arbitrary string, {} and -- worst of all -- an explicit
# {"version": 1} header (the historical global-torch-RNG behaviour the whole
# protocol exists to eliminate) ALL classified as protocol "on" and ALL passed
# --require-scenario-hashes with exit 0.
#
# Each test below writes a numerically CLEAN pair (differing hashes on both
# sides -> CONFIRMED_INDEPENDENT), so the only thing that can fail it is the
# header. That is the exact shape that used to sail through.
# ---------------------------------------------------------------------------

def _stamped_pair(root, name, protocol_a, protocol_b, tag):
    """Two certificates that both CARRY a scenario_protocol key.

    Unlike ``_off_protocol_pair``, ``None`` here means the JSON value ``null``
    rather than "omit the field": "the key is present but its value is junk"
    is precisely the shape under test, and it must not be confused with the
    unstamped state.
    """
    directory = os.path.join(root, name)
    rng_a, rng_b = random.Random(40), random.Random(41)
    n = 24
    primitives = ("spawn", "wave")
    a = episodes(n, alternating(n), floats(rng_a, n, 5.0, 40.0),
                 floats(rng_a, n), hashes=hexes("s42", n, primitives))
    b = episodes(n, [not v for v in alternating(n)],
                 floats(rng_b, n, 5.0, 40.0), floats(rng_b, n),
                 hashes=hexes("s123", n, primitives))
    write_cert(directory, tag, 42, a, scenario_protocol=protocol_a)
    write_cert(directory, tag, 123, b, scenario_protocol=protocol_b)
    return directory


def _assert_header_rejected(directory, expected_status, reason_fragment):
    """The full contract for a rejected header, in one place.

    Status, table cell, warning, summary counts, gate exit code and the fail
    reason that names WHAT was wrong -- a status the report does not surface
    is a status nobody acts on.
    """
    results, _, _, _ = audit_dir(directory)
    result = results[0]
    # The numeric evidence is clean: nothing but the header can fail this pair.
    assert result.verdict == "CONFIRMED_INDEPENDENT", result.verdict
    assert result.protocol_status() == expected_status, (
        result.protocol_status(), csi.describe_protocol(result.a))
    assert csi.protocol_cell(result) == expected_status.upper(), \
        csi.protocol_cell(result)
    assert any("PROTOCOL HEADER REJECTED" in w for w in result.warnings), \
        result.warnings
    assert reason_fragment in result.protocol_problem_text(), \
        result.protocol_problem_text()
    assert len(result.rejected_protocol_sides()) == 2, \
        result.rejected_protocol_sides()

    # Without the flag a junk header is loud but not fatal, exactly like the
    # off-protocol marker.
    code, text = run_main([directory])
    assert code == 0, text
    assert expected_status.upper() in text, text
    # Summary counts pairs, and this directory holds exactly one pair.
    assert "1 {}".format(expected_status) in text, text
    assert "rejected header" in text, text

    # With the flag it MUST fail, and the log must name the defect.
    code, text = run_main([directory, "--quiet", "--require-scenario-hashes"])
    assert code == 1, text
    assert reason_fragment in text, text
    # Specifically in the FAIL block, not merely somewhere in the output: the
    # fail reason is the line a gate log is read for, and "INVALID_PROTOCOL"
    # on its own sends the reader back to the certificate to guess which of
    # half a dozen defects fired.
    head, marker, fail_block = text.partition("\nFAIL: ")
    assert marker, text
    assert "{}_PROTOCOL (--require-scenario-hashes)".format(
        expected_status.upper()) in fail_block, fail_block
    assert reason_fragment in fail_block, fail_block
    # The pair carries hashes on both sides, so it is not the missing-hash
    # rule that fired.
    assert "NO_SCENARIO_HASHES" not in text, text

    # The same reason has to survive into the CSV a CI job keeps as its
    # artefact, which means it must be computed before the report is written.
    csv_path = os.path.join(directory, "fail_reasons.csv")
    code, _text = run_main([directory, "--quiet", "--require-scenario-hashes",
                            "--report", csv_path])
    assert code == 1
    with open(csv_path, encoding="utf-8") as handle:
        csv_text = handle.read()
    assert "{}_PROTOCOL".format(expected_status.upper()) in csv_text, csv_text
    assert reason_fragment in csv_text, csv_text
    return result


def test_null_protocol_header_is_invalid_not_on(root):
    """``"scenario_protocol": null`` is a present key with no claim in it."""
    directory = _stamped_pair(root, "proto_null", None, None, "null")
    result = _assert_header_rejected(directory, csi.PROTOCOL_INVALID,
                                     "scenario_protocol is null")
    # Not the unstamped state: the key IS there, which is why "present" can
    # never be the test.
    assert result.unstamped_sides() == [], result.unstamped_sides()
    assert result.off_protocol_sides() == []
    assert csi.describe_protocol(result.a) == "null (invalid)", \
        csi.describe_protocol(result.a)


def test_arbitrary_string_protocol_header_is_invalid_not_on(root):
    """Any string other than the off marker is junk, not compliance."""
    directory = _stamped_pair(root, "proto_string", "totally-bogus",
                              "totally-bogus", "str")
    result = _assert_header_rejected(directory, csi.PROTOCOL_INVALID,
                                     "not the header block")
    assert result.off_protocol_sides() == [], result.off_protocol_sides()
    assert "invalid" in csi.describe_protocol(result.a), \
        csi.describe_protocol(result.a)


def test_empty_object_protocol_header_is_invalid_not_on(root):
    """A mapping is necessary but not sufficient: it must carry a version."""
    directory = _stamped_pair(root, "proto_empty", {}, {}, "empty")
    result = _assert_header_rejected(directory, csi.PROTOCOL_INVALID,
                                     "carries no 'version' key")
    assert csi.describe_protocol(result.a) == "no-version (invalid)", \
        csi.describe_protocol(result.a)


def test_non_integer_protocol_version_is_invalid(root):
    """A version must be an integer -- "2", 2.0 and true are not."""
    for label, version, fragment in (
            ("str", "2", "version is string"),
            ("float", 2.0, "version is float"),
            ("bool", True, "version is bool"),
            ("zero", 0, "is not a positive integer")):
        directory = _stamped_pair(
            root, "proto_ver_" + label,
            {"version": version, "hash": "blake2b"},
            {"version": version, "hash": "blake2b"}, "v" + label)
        _assert_header_rejected(directory, csi.PROTOCOL_INVALID, fragment)


def test_stale_version_one_header_never_passes_the_gate(root):
    """The one header that must never pass: a declaration of protocol 1.

    Version 1 IS the historical behaviour -- scenarios drawn from the global
    torch RNG, which skrl's Runner reseeds to a constant after the env is
    built -- that the scenario protocol exists to eliminate. A certificate
    that says so out loud, in a perfectly well-formed header, used to be read
    as "on protocol" and exited 0 under --require-scenario-hashes.
    """
    header = {"version": 1, "hash": "blake2b",
              "note": "global torch RNG; eval seed clobbered by skrl Runner"}
    directory = _stamped_pair(root, "proto_stale_v1", header, header, "stale1")
    result = _assert_header_rejected(
        directory, csi.PROTOCOL_STALE,
        "declares version 1 but the current protocol is version {}".format(
            csi.current_protocol_version()))
    assert "historical global-torch-RNG" in result.protocol_problem_text(), \
        result.protocol_problem_text()
    assert csi.describe_protocol(result.a) == "v1/blake2b (stale)", \
        csi.describe_protocol(result.a)
    # Stale is its own status, distinct from invalid and from the marker
    # states, so a gate log can tell "wrong protocol" from "broken stamp".
    assert result.protocol_status() != csi.PROTOCOL_INVALID
    assert result.protocol_status() != csi.PROTOCOL_OFF


def test_one_bad_side_is_enough_to_reject_the_pair(root):
    """A current header opposite a stale one is not a protocol-paired run."""
    directory = _stamped_pair(root, "proto_half_stale", PROTOCOL_HEADER,
                              {"version": 1, "hash": "blake2b"}, "half")
    results, _, _, _ = audit_dir(directory)
    result = results[0]
    assert result.protocol_status() == csi.PROTOCOL_STALE, \
        result.protocol_status()
    assert len(result.rejected_protocol_sides()) == 1, \
        result.rejected_protocol_sides()
    assert "half_e123.json" in result.protocol_problem_text(), \
        result.protocol_problem_text()
    assert run_main([directory, "--quiet"])[0] == 0
    code, text = run_main([directory, "--quiet", "--require-scenario-hashes"])
    assert code == 1, text
    assert "STALE_PROTOCOL (--require-scenario-hashes)" in text, text


def test_rejected_headers_reach_both_report_formats(root):
    """A status only in stdout is a status the CI artefact does not carry."""
    invalid_dir = _stamped_pair(root, "proto_rep_invalid", {}, {}, "repinv")
    stale_dir = _stamped_pair(root, "proto_rep_stale",
                              {"version": 1, "hash": "blake2b"},
                              {"version": 1, "hash": "blake2b"}, "repstale")
    for directory, status, cell in ((invalid_dir, "invalid", "no-version"),
                                    (stale_dir, "stale", "v1/blake2b")):
        csv_path = os.path.join(root, "proto_{}.csv".format(status))
        md_path = os.path.join(root, "proto_{}.md".format(status))
        assert run_main([directory, "--quiet", "--report", csv_path])[0] == 0
        assert run_main([directory, "--quiet", "--report", md_path])[0] == 0

        with open(csv_path, encoding="utf-8") as handle:
            csv_text = handle.read()
        header_row = csv_text.splitlines()[0]
        assert "protocol_status" in header_row, header_row
        assert status in csv_text, csv_text
        assert cell in csv_text, csv_text
        assert "PROTOCOL HEADER REJECTED" in csv_text, csv_text

        with open(md_path, encoding="utf-8") as handle:
            md_text = handle.read()
        assert "| protocol |" in md_text, md_text
        assert status.upper() in md_text, md_text
        assert "PROTOCOL HEADER REJECTED" in md_text, md_text


def test_protocol_version_is_read_from_the_source_of_truth(root):
    """The required version is scenario_rng.py's, not a literal in here.

    Same AST-reading precedent as test_off_protocol_literal_matches_scenario
    _draws: read the constant out of the module by text, independently of the
    auditor's own reader, and pin the two equal. Hard-coding the version in
    the auditor would mean a protocol bump silently leaves this gate accepting
    certificates from the retired protocol.
    """
    source = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tasks", "_shared", "scenario_rng.py")
    tree = ast.parse(open(source, encoding="utf-8").read(), filename=source)
    declared = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and any(isinstance(target, ast.Name)
                and target.id == "SCENARIO_PROTOCOL_VERSION"
                for target in node.targets)
    ]
    assert declared == [csi.current_protocol_version()], (
        "scenario_rng.py declares SCENARIO_PROTOCOL_VERSION={!r} but the "
        "auditor requires {!r}".format(declared,
                                       csi.current_protocol_version()))
    assert os.path.abspath(csi.SCENARIO_RNG_SOURCE) == os.path.abspath(source)
    # It is version 1 that must never be current: the gate's whole point.
    assert csi.current_protocol_version() != 1


def test_protocol_version_bump_is_picked_up_without_editing_the_auditor(root):
    """Point the auditor at a bumped source: the accepted version follows.

    Proves the value is genuinely READ rather than defaulted -- a hard-coded
    2 would keep calling {"version": 2} "on" no matter what the source says.
    """
    bumped = os.path.join(root, "bumped_scenario_rng.py")
    with open(bumped, "w", encoding="utf-8", newline="\n") as handle:
        handle.write('"""Stand-in source of truth."""\n'
                     "SCENARIO_PROTOCOL_VERSION = 7\n")
    real = csi.SCENARIO_RNG_SOURCE
    try:
        csi.SCENARIO_RNG_SOURCE = bumped
        csi._PROTOCOL_VERSION_CACHE.pop(bumped, None)
        assert csi.current_protocol_version() == 7
        assert csi.classify_protocol_field(
            True, {"version": 7, "hash": "blake2b"},
            csi.current_protocol_version())[0] == csi.PROTOCOL_ON
        # The version that is current for the REAL source is now stale.
        state, why = csi.classify_protocol_field(
            True, PROTOCOL_HEADER, csi.current_protocol_version())
        assert state == csi.PROTOCOL_STALE, (state, why)
        assert "is version 7" in why, why
    finally:
        csi.SCENARIO_RNG_SOURCE = real
        csi._PROTOCOL_VERSION_CACHE.pop(bumped, None)
    assert csi.current_protocol_version() != 7


def test_unreadable_protocol_source_fails_closed(root):
    """No source of truth -> exit 2, never a silent pass on a guessed version.

    An auditor that cannot tell which protocol is current cannot tell a
    compliant header from a retired one; guessing would wave both through.
    """
    missing = os.path.join(root, "no_such_scenario_rng.py")
    empty = os.path.join(root, "versionless_scenario_rng.py")
    with open(empty, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("OTHER_CONSTANT = 3\n")
    directory = _stamped_pair(root, "proto_failclosed", PROTOCOL_HEADER,
                              PROTOCOL_HEADER, "failclosed")
    real = csi.SCENARIO_RNG_SOURCE
    for bad in (missing, empty):
        try:
            csi.SCENARIO_RNG_SOURCE = bad
            csi._PROTOCOL_VERSION_CACHE.pop(bad, None)
            try:
                csi.read_protocol_version(bad)
            except csi.ProtocolVersionUnavailable as error:
                assert bad in str(error), str(error)
            else:
                raise AssertionError(
                    "read_protocol_version({!r}) did not raise".format(bad))
            code, text = run_main([directory, "--quiet"])
            assert code == 2, (code, text)
            assert "cannot determine the current scenario protocol version" \
                in text, text
        finally:
            csi.SCENARIO_RNG_SOURCE = real
            csi._PROTOCOL_VERSION_CACHE.pop(bad, None)
    # Restored: a well-formed current header is on protocol again.
    assert run_main([directory, "--quiet",
                     "--require-scenario-hashes"])[0] == 0


def test_existing_protocol_states_survive_the_stricter_gate(root):
    """The four original statuses still mean what they meant.

    A stricter validator that reclassified the off marker, or the unstamped
    state, or a genuine current header, would have traded one blind spot for
    another.
    """
    good = _stamped_pair(root, "proto_still_on", PROTOCOL_HEADER,
                         PROTOCOL_HEADER, "stillon")
    results, _, _, _ = audit_dir(good)
    assert results[0].protocol_status() == csi.PROTOCOL_ON
    assert results[0].rejected_protocol_sides() == []
    assert csi.protocol_cell(results[0]) == "on"
    assert run_main([good, "--quiet", "--require-scenario-hashes"])[0] == 0

    for name, a, b, expected in (
            ("still_off", PROTOCOL_OFF, PROTOCOL_OFF, csi.PROTOCOL_OFF),
            ("still_mixed", PROTOCOL_HEADER, PROTOCOL_OFF,
             csi.PROTOCOL_MIXED)):
        directory = _stamped_pair(root, "proto_" + name, a, b, name)
        results, _, _, _ = audit_dir(directory)
        assert results[0].protocol_status() == expected, \
            (name, results[0].protocol_status())
        assert results[0].rejected_protocol_sides() == [], name

    unstamped = _off_protocol_pair(root, "proto_still_unstamped", None, None,
                                   tag="stillun")
    results, _, _, _ = audit_dir(unstamped)
    assert results[0].protocol_status() == csi.PROTOCOL_UNSTAMPED
    assert results[0].rejected_protocol_sides() == []


# ---------------------------------------------------------------------------
# Helper units
# ---------------------------------------------------------------------------

def test_helper_units(root):
    assert csi.filename_seed("gcert_sk_e123") == 123
    assert csi.filename_seed("wf_wvpid_kp2ki0") is None
    assert csi.parse_episode_key("12:3") == (12, 3)
    assert csi.parse_episode_key([12, 3]) == (12, 3)
    assert csi.parse_episode_key("nonsense") is None

    assert csi.values_equal(None, None, 0.0)
    assert not csi.values_equal(None, 1.0, 0.0)
    assert csi.values_equal(1.0, 1.0, 0.0)
    assert not csi.values_equal(1.0, 1.0 + 1e-9, 0.0)
    assert csi.values_equal(1.0, 1.0 + 1e-9, 1e-6)
    assert csi.values_equal(float("nan"), float("nan"), 0.0)
    assert csi.values_equal(True, 1, 0.0)

    assert csi.collision_probability([1.0, 2.0, 3.0, 4.0]) == 0.25
    assert csi.collision_probability([7.0] * 10) == 1.0
    assert csi.collision_probability([None, None]) == 1.0

    assert csi.short_task("Isaac-USV-HazardCross-Direct-v1") == "HazardCross"


def test_uninformative_match_on_all_fail_success(root):
    all_fail = [False] * 64
    result = csi.compare_channel("success", csi.LAYER_BINARY, all_fail,
                                 list(all_fail), 0.0, 1e-2)
    assert result.n_match == 64
    assert result.q == 1.0
    assert result.status == "UNINFORMATIVE_MATCH"


# ---------------------------------------------------------------------------
# Exit codes and reports (the gate contract)
# ---------------------------------------------------------------------------

def test_exit_code_zero_when_clean(root):
    code, text = run_main([os.path.join(root, "independent"), "--quiet"])
    assert code == 0, text
    assert "PASS:" in text


def test_exit_code_one_on_duplicate(root):
    code, text = run_main([os.path.join(root, "dup_numeric"), "--quiet"])
    assert code == 1, text
    assert "CONFIRMED_DUPLICATE" in text
    assert "FAIL:" in text


def test_exit_code_one_on_partial(root):
    for name in ("partial_hash", "partial_numeric", "partial_anomalous"):
        code, text = run_main([os.path.join(root, name), "--quiet"])
        assert code == 1, (name, text)
        assert "PARTIAL_SHORT_CIRCUIT" in text, name


def test_degenerate_does_not_fail_unless_asked(root):
    directory = os.path.join(root, "degenerate")
    code, text = run_main([directory, "--quiet"])
    assert code == 0, text
    assert "DEGENERATE_UNDECIDABLE" in text
    code, text = run_main([directory, "--quiet", "--fail-on-degenerate"])
    assert code == 1, text


def test_require_scenario_hashes_flag(root):
    directory = os.path.join(root, "independent")
    assert run_main([directory, "--quiet"])[0] == 0
    code, text = run_main([directory, "--quiet", "--require-scenario-hashes"])
    assert code == 1, text
    assert "NO_SCENARIO_HASHES (--require-scenario-hashes)" in text, text
    # Hashes on both sides AND both sides on the protocol: the only shape the
    # gate accepts, because the gate's premise is that an (env, ep) key names
    # the same scenario in both certificates.
    hashed = os.path.join(root, "independent_hash")
    assert run_main([hashed, "--quiet", "--require-scenario-hashes"])[0] == 0


def test_missing_input_exits_two(root):
    code, text = run_main([os.path.join(root, "no_such_dir", "*.json")])
    assert code == 2, text
    assert "no JSON files matched" in text


def test_reports_are_written(root):
    directory = os.path.join(root, "dup_numeric")
    csv_path = os.path.join(root, "report.csv")
    md_path = os.path.join(root, "report.md")
    assert run_main([directory, "--quiet", "--report", csv_path])[0] == 1
    assert run_main([directory, "--quiet", "--report", md_path])[0] == 1

    with open(csv_path, encoding="utf-8") as handle:
        csv_text = handle.read()
    assert "verdict" in csv_text.splitlines()[0]
    assert "CONFIRMED_DUPLICATE" in csv_text
    assert "success;path_length_m" in csv_text or \
           "path_length_m" in csv_text, csv_text

    with open(md_path, encoding="utf-8") as handle:
        md_text = handle.read()
    assert "# Scenario independence audit" in md_text
    assert "**CONFIRMED_DUPLICATE**" in md_text
    assert "## Per-channel detail" in md_text


def test_table_and_legend_render(root):
    code, text = run_main([os.path.join(root, "partial_numeric")])
    assert code == 1
    assert "VERDICT RULES" in text
    assert "ARM" in text and "VERDICT" in text
    assert "coverage : spawn/goal geometry only" in text
    assert all(ord(ch) >= 32 or ch in "\n\r" for ch in text), \
        "control characters in auditor output"


def test_arm_names_are_disambiguated_across_directories(root):
    left = os.path.join(root, "twodirs", "a")
    right = os.path.join(root, "twodirs", "b")
    rng = random.Random(23)
    flipped = [not v for v in alternating(16)]
    for directory in (left, right):
        for seed in (42, 123):
            write_cert(directory, "same_name", seed,
                       episodes(16,
                                alternating(16) if seed == 42 else flipped,
                                floats(rng, 16, 5.0, 40.0),
                                floats(rng, 16)))
    code, text = run_main([left, right, "--quiet"])
    assert code == 0, text
    assert "a/same_name" in text and "b/same_name" in text, text
    assert "CONFIRMED_INDEPENDENT" in text, text


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ORDER = [
    "test_duplicate_numeric",
    "test_duplicate_from_hashes_alone",
    "test_duplicate_survives_shuffled_record_order",
    "test_duplicate_on_partial_episode_overlap",
    "test_degenerate_all_fail",
    "test_degenerate_when_no_episode_key_is_shared",
    "test_independent_numeric",
    "test_independent_by_hashes",
    "test_partial_short_circuit_by_hashes",
    "test_partial_short_circuit_pinned_scenario_field",
    "test_partial_short_circuit_anomalous_fraction",
    "test_training_seed_token_is_not_stripped",
    "test_identity_mismatch_blocks_pairing",
    "test_run_counters_do_not_block_pairing",
    "test_unpaired_and_unreadable_inputs",
    "test_float_tolerance_flips_the_verdict",
    "test_chance_threshold_controls_the_free_match_guard",
    "test_scenario_hash_schema_variants",
    "test_one_sided_hashes_are_not_evidence",
    "test_off_protocol_pair_is_reported",
    "test_off_protocol_pair_fails_the_require_hashes_gate",
    "test_mixed_protocol_pair_fails_the_require_hashes_gate",
    "test_unstamped_protocol_pair_fails_the_require_hashes_gate",
    "test_off_protocol_reaches_both_report_formats",
    "test_off_protocol_literal_matches_scenario_draws",
    "test_describe_protocol_renders_each_state",
    "test_protocol_version_mismatch_is_reported",
    "test_null_protocol_header_is_invalid_not_on",
    "test_arbitrary_string_protocol_header_is_invalid_not_on",
    "test_empty_object_protocol_header_is_invalid_not_on",
    "test_non_integer_protocol_version_is_invalid",
    "test_stale_version_one_header_never_passes_the_gate",
    "test_one_bad_side_is_enough_to_reject_the_pair",
    "test_rejected_headers_reach_both_report_formats",
    "test_protocol_version_is_read_from_the_source_of_truth",
    "test_protocol_version_bump_is_picked_up_without_editing_the_auditor",
    "test_unreadable_protocol_source_fails_closed",
    "test_existing_protocol_states_survive_the_stricter_gate",
    "test_helper_units",
    "test_uninformative_match_on_all_fail_success",
    "test_exit_code_zero_when_clean",
    "test_exit_code_one_on_duplicate",
    "test_exit_code_one_on_partial",
    "test_degenerate_does_not_fail_unless_asked",
    "test_require_scenario_hashes_flag",
    "test_missing_input_exits_two",
    "test_reports_are_written",
    "test_table_and_legend_render",
    "test_arm_names_are_disambiguated_across_directories",
]


def main() -> int:
    passed, failed = 0, []
    with tempfile.TemporaryDirectory(prefix="scenario_audit_") as root:
        for name in ORDER:
            function = globals()[name]
            try:
                function(root)
            except Exception:                       # noqa: BLE001
                failed.append(name)
                print("FAIL {}".format(name))
                traceback.print_exc()
            else:
                passed += 1
                print("ok   {}".format(name))
    print("\n{} passed, {} failed".format(passed, len(failed)))
    if failed:
        print("failures: " + ", ".join(failed))
        return 1
    print("PASS: scenario-independence auditor covers every classification "
          "branch and both gate exit codes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
