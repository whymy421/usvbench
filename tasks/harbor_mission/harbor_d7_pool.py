"""Harbour-mission D7 admission pool on disk: typed JSON, verified on load (fail closed).

Pure Python (json, gzip, hashlib); no torch, no Isaac Lab. The pool is built
OFFLINE by the admission tool (work/tools_harbour_mission/admit_d7_pool.py,
task F6) with ONE E4 ``admit_grid`` pass over the REAL E5 layouts of the
grid keys (harbor_journal.d7_task_context; attempt 0 only, no redraw, no
re-attempt, CONTEXT_R4 D16) and written as a gzip-compressed JSON document
plus a sidecar ``<file>.sha256``. The document stores, per offered key,
either the admitted assignment ladder (all five doses, joint) or an entry of
the EXPLICIT ``not_admitted`` list (E5 geometry id, strata stage, every E4
rejection record of the key). The environment loads it BY PATH (see
``resolve_pool_source``; every variable is a RUNTIME variable, D19) and every
check below must pass before a single assignment is handed to the bridge:

  1. the file SHA-256 equals the sidecar digest (and an optional expected value);
  2. schema, task and vessel match the env cfg; if the pinned manifest
     ``tasks/harbor_mission/d7_pools/POOL_MANIFEST.json`` lists the pool's
     (vessel, seed, tier, grid), the file SHA-256 and pool digest equal the
     pinned values (D19 re-pin);
  3. every assignment's candidate digest and witness digest recompute
     (``stamp.digest`` over the reconstructed dataclasses, bit exact), and
     every assignment carries the pool digest;
  4. the pool digest recomputes from the stored manifest (offered keys in
     order, accepted digests, rejections, joint flag) exactly as E4
     ``adapters._assemble`` computes it;
  5. the not-admitted list equals the E4 outcome (``verify_not_admitted``).

D16 at run time: ``HarbourPoolView.assignment`` returns a ``NotAdmitted``
record for a rejected key and ``HarbourD7Bridge`` (wrapping the E4 bridge)
runs such a row with no D7 object and stamps it ``exposure = "not_admitted"``
(see the class docstrings). Never an exception, never a redraw, never a
skipped scenario.

D24 (episodes beyond the offered grid): the evaluators keep stepping every
env until the slowest one reaches its per-env quota, so faster envs reset
into task episodes the pool never offered. The pool's offered grid
(``offered_grid``: the offered keys, which must form envs 0..N-1 x episodes
0..E-1 of one eval seed and one hazard group, and agree with a recorded
``meta["grid"]``) defines them: a key with the grid's eval seed and hazard
group, 0 <= env < N and episode >= E gets a ``NotAdmitted`` record with
``beyond_grid=True`` and the single reason (0, "beyond_offered_grid", 0)
(dose 0: admitted at no dose, runs at D0, so the entry is identical at every
requested dose). It carries no pool geometry (geometry id and
stage None: the pool has no record of the key); the env still builds the
key's E5 layout from the keyed stream and skips the pool geometry check for
it. Any other key outside the pool (env >= N, another seed or group) stays a
KeyError, i.e. a preflight error in the mixin. ``per_dose_counts`` never
include beyond-grid keys (they are not pool keys); the header publishes
``grid = {"num_envs": N, "episodes": E}``.

Typed JSON: a dataclass instance is {"__dc__": class name, "f": {init fields}},
a tuple {"__t__": [...]}, a mapping {"__d__": [[key, value], ...]}; floats are
written with Python's repr (exact round trip). Classes are resolved only from
the E4 dynamic package (a closed registry built at load time), never from the
file: the file cannot name an arbitrary class.

Hazard group: E4 admits harbour keys under ``"d7_harbour"`` (``"d7_" +``
adapter task); F-base v2's mixin asks for that group directly (F6 proposal,
D18). ``HarbourPoolView.admitted_key`` still maps the pre-v2 group
``"d7_harbour_mission"`` for old callers.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path

POOL_SCHEMA = "usvbench.harbour_d7_pool/2"          # /2: explicit not_admitted list (round 4b, D16)
PINNED_SCHEMA = "usvbench.harbour_d7_pool_manifest/1"
FAMILY_DIR = "harbor_mission"
PINNED_DIR = Path(__file__).resolve().parent / "d7_pools"
PINNED_MANIFEST = PINNED_DIR / "POOL_MANIFEST.json"
MIXIN_GROUP = "d7_harbour_mission"      # "d7_" + journal task (pre-v2 F0 journal_env_mixin)
ADMITTED_GROUP = "d7_harbour"           # "d7_" + E4 adapter task (adapters.TASK_ALIASES)
POOL_PATH_ENV = "USVBENCH_HARBOUR_D7_POOL"
POOL_DIR_ENV = "USVBENCH_D7_POOL_DIR"
NOT_ADMITTED = "not_admitted"
JOINT_MEMBER = "joint_ladder_member_rejected"
BEYOND_GRID = "beyond_offered_grid"     # CONTEXT_R4 D24: the single reason of a beyond-grid key
_DYNAMIC_MODULES = ("model", "feasibility", "trajectory", "reference", "native", "visibility",
                    "capacity", "adapters", "vessels", "config", "geometry", "kinematics", "sensors")


class PoolIntegrityError(ValueError):
    """The pool file failed a fail-closed integrity check."""


def _registry(dynamic):
    base = dynamic.__name__
    out = {}
    for name in _DYNAMIC_MODULES:
        try:
            module = importlib.import_module(base + "." + name)
        except ImportError:
            continue
        for value in vars(module).values():
            if isinstance(value, type) and is_dataclass(value) and value.__module__ == module.__name__:
                if value.__name__ in out and out[value.__name__] is not value:
                    raise PoolIntegrityError("ambiguous dataclass name " + value.__name__)
                out[value.__name__] = value
    return out


def encode(value):
    """Typed JSON form of a pool value (dataclasses, tuples, mappings, scalars)."""
    if is_dataclass(value) and not isinstance(value, type):
        return {"__dc__": type(value).__name__,
                "f": {f.name: encode(getattr(value, f.name)) for f in fields(value) if f.init}}
    if isinstance(value, tuple):
        return {"__t__": [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {"__d__": [[encode(k), encode(v)] for k, v in value.items()]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("cannot encode %r in a D7 pool" % (type(value).__name__,))


def decode(value, registry):
    if isinstance(value, list):
        return [decode(v, registry) for v in value]
    if isinstance(value, dict):
        if set(value) == {"__t__"}:
            return tuple(decode(v, registry) for v in value["__t__"])
        if set(value) == {"__d__"}:
            return {_hashable(decode(k, registry)): decode(v, registry) for k, v in value["__d__"]}
        if set(value) == {"__dc__", "f"}:
            name = value["__dc__"]
            if name not in registry:
                raise PoolIntegrityError("pool names a class outside the E4 dynamic package: %r" % (name,))
            return registry[name](**{k: decode(v, registry) for k, v in value["f"].items()})
        raise PoolIntegrityError("untyped mapping in a D7 pool document")
    return value


def _hashable(value):
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pool_document(pool, meta):
    """JSON document of an AdmissionPool plus admission metadata (offered keys + explicit not-admitted list)."""
    offered = meta.get("offered_keys")
    if offered is None:
        raise ValueError("meta must carry offered_keys (EpisodeKey list in admission order)")
    if "not_admitted" not in meta:
        raise ValueError("meta must carry the explicit not_admitted list (D16; harbor_journal.pool_meta)")
    body = dict(meta)
    body["offered_keys"] = tuple(offered)
    return {"schema": POOL_SCHEMA, "meta": encode(body), "pool": encode(pool)}


def write_pool(path, pool, meta):
    """Write ``path`` (gzip JSON, mtime 0 for reproducible bytes) and ``path.sha256``; return the digest."""
    verify_not_admitted(pool, meta)
    document = pool_document(pool, meta)
    raw = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("ascii")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0, compresslevel=6) as gz:
            gz.write(raw)
    digest = file_sha256(path)
    Path(str(path) + ".sha256").write_text(digest + "  " + path.name + "\n", encoding="ascii")
    return digest


def _manifest_digest(pool, offered, dynamic):
    adapters = importlib.import_module(dynamic.__name__ + ".adapters")
    stamp = importlib.import_module(dynamic.__name__ + ".stamp")
    accepted = sorted([[k.eval_seed, k.env, k.episode, k.hazard_group, d, stamp.digest(a.candidate),
                        adapters._witness_digest(a.witness)] for (k, d), a in pool.assignments.items()])
    manifest = {"offered": list(offered), "accepted": accepted, "rejections": list(pool.rejections),
                "joint": pool.joint}
    return stamp.digest(manifest)


def verify_pool(pool, offered, dynamic):
    """Checks 3 and 4 of the module docstring; raises PoolIntegrityError."""
    adapters = importlib.import_module(dynamic.__name__ + ".adapters")
    stamp = importlib.import_module(dynamic.__name__ + ".stamp")
    if not isinstance(pool, adapters.AdmissionPool):
        raise PoolIntegrityError("document does not hold an E4 AdmissionPool")
    for (key, dose), assignment in pool.assignments.items():
        if assignment.pool_digest != pool.pool_digest:
            raise PoolIntegrityError("assignment %r D%d carries another pool digest" % (key, dose))
        if stamp.digest(assignment.candidate) != assignment.candidate_digest:
            raise PoolIntegrityError("candidate digest mismatch for %r D%d" % (key, dose))
        if adapters._witness_digest(assignment.witness) != assignment.witness_digest:
            raise PoolIntegrityError("witness digest mismatch for %r D%d" % (key, dose))
        if assignment.candidate.request.encounter.key != key or assignment.candidate.request.encounter.dose != dose:
            raise PoolIntegrityError("assignment stored under another key/dose")
    if _manifest_digest(pool, offered, dynamic) != pool.pool_digest:
        raise PoolIntegrityError("pool digest does not recompute from the stored manifest")
    return pool


def _reason_tuple(entry):
    dose, reason, sampled = entry
    return int(dose), str(reason), int(sampled)


def verify_not_admitted(pool, meta):
    """Check 5: the explicit not-admitted list equals the E4 outcome of the offered keys.

    Joint pools only (a key is admitted at all five doses or not at all).
    Listed keys, in offered order, are exactly the offered keys without an
    assignment; each entry's reasons equal the key's E4 rejection records
    (``pool.rejections`` order); every entry names a sha256 geometry id.
    Returns {EpisodeKey: (geometry_id, stage, reasons)}.
    """
    if not pool.joint:
        raise PoolIntegrityError("harbour D7 pools are joint across D0..D4 (D16 records whole keys)")
    offered = list(meta["offered_keys"])
    doses = {}
    for key, dose in pool.assignments:
        doses.setdefault(key, set()).add(dose)
    if any(value != {0, 1, 2, 3, 4} for value in doses.values()):
        raise PoolIntegrityError("an admitted key lacks one of the doses D0..D4")
    expected = [key for key in offered if key not in doses]
    listed = list(meta.get("not_admitted") or [])
    if len(listed) != len(expected):
        raise PoolIntegrityError("not_admitted list has %d keys, the pool leaves %d offered keys unadmitted"
                                 % (len(listed), len(expected)))
    rejections = {}
    for r in pool.rejections:
        rejections.setdefault(r.key, []).append((int(r.dose), str(r.reason), int(r.sampled_states)))
    out = {}
    for key, entry in zip(expected, listed):
        env, episode, geometry_id, stage, reasons = entry
        if (int(env), int(episode)) != (key.env, key.episode):
            raise PoolIntegrityError("not_admitted entry (%s, %s) is not the next unadmitted key %r"
                                     % (env, episode, key))
        if not (isinstance(geometry_id, str) and len(geometry_id) == 64):
            raise PoolIntegrityError("not_admitted entry %r lacks its E5 geometry id" % (key,))
        reasons = tuple(_reason_tuple(r) for r in reasons)
        if not reasons or reasons != tuple(rejections.get(key, ())):
            raise PoolIntegrityError("not_admitted reasons of %r differ from the E4 rejection records" % (key,))
        out[key] = (geometry_id, stage, reasons)
    return out


def offered_grid(pool, meta=None):
    """The pool's offered grid (CONTEXT_R4 D24); fails closed.

    Returns {"num_envs": N, "episodes": E, "eval_seed": s, "hazard_group": g}.
    The offered keys are ``meta["offered_keys"]`` (pool files, pool_meta) or,
    for a raw E4 pool, every key the pool admitted or rejected (a joint pool
    records a rejection for every key it does not admit); their count must
    equal ``pool.keys_offered``. They must be the full rectangle envs
    0..N-1 x episodes 0..E-1 of ONE eval seed and ONE hazard group, and a
    recorded ``meta["grid"]`` mapping must carry the same "envs" and
    "episodes".
    """
    meta = meta or {}
    if meta.get("offered_keys") is not None:
        keys = list(meta["offered_keys"])
    else:
        keys = list(dict.fromkeys([k for k, _d in pool.assignments] + [r.key for r in pool.rejections]))
    if not keys or len(set(keys)) != len(keys) or len(keys) != int(pool.keys_offered):
        raise PoolIntegrityError("the pool's offered keys (%d distinct of %d listed) do not match keys_offered %s"
                                 % (len(set(keys)), len(keys), pool.keys_offered))
    seeds = {k.eval_seed for k in keys}
    groups = {k.hazard_group for k in keys}
    if len(seeds) != 1 or len(groups) != 1:
        raise PoolIntegrityError("the pool offers keys of several eval seeds %s or hazard groups %s"
                                 % (sorted(seeds), sorted(groups)))
    num_envs = max(k.env for k in keys) + 1
    episodes = max(k.episode for k in keys) + 1
    if {(k.env, k.episode) for k in keys} != {(e, k) for e in range(num_envs) for k in range(episodes)}:
        raise PoolIntegrityError("the pool's offered keys are not the full grid envs 0..%d x episodes 0..%d"
                                 % (num_envs - 1, episodes - 1))
    recorded = meta.get("grid")
    if isinstance(recorded, dict) and (recorded.get("envs"), recorded.get("episodes")) != (num_envs, episodes):
        raise PoolIntegrityError("pool meta grid %r differs from the offered keys (%d envs x %d episodes)"
                                 % (recorded, num_envs, episodes))
    return {"num_envs": num_envs, "episodes": episodes, "eval_seed": seeds.pop(), "hazard_group": groups.pop()}


def _pinned_manifest(path):
    path = Path(path) if path is not None else None
    if path is None or not path.is_file():
        return None, None
    raw = path.read_bytes()
    manifest = json.loads(raw.decode("ascii"))
    if manifest.get("schema") != PINNED_SCHEMA:
        raise PoolIntegrityError("unsupported pinned pool manifest schema %r" % (manifest.get("schema"),))
    return manifest, hashlib.sha256(raw).hexdigest()


def pinned_entry(manifest, meta):
    """The pinned entry for the pool's (vessel, seed, tier, grid), or None."""
    if not manifest:
        return None
    for entry in manifest.get("pools", ()):
        if (entry.get("vessel"), entry.get("seed"), entry.get("tier"), entry.get("grid")) == (
                meta.get("vessel"), meta.get("seed"), meta.get("tier"), meta.get("grid")):
            return entry
    return None


def load_pool(path, dynamic, *, vessel=None, expected_sha256=None, manifest_path=PINNED_MANIFEST):
    """Load, verify and return (AdmissionPool, meta dict). Fails closed on any mismatch.

    ``meta`` gains file_sha256, path, not_admitted_map ({EpisodeKey:
    (geometry_id, stage, reasons)}), offered_grid (``offered_grid``, D24), pinned (the pinned manifest lists this
    pool's vessel/seed/tier/grid and the file matches it) and
    pool_manifest_sha256 (SHA-256 of the pinned manifest file, or None).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("harbour D7 pool %s is missing (admit it offline with "
                                "tools_harbour_mission/admit_d7_pool.py)" % (path,))
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        raise PoolIntegrityError("pool digest file %s is missing" % (sidecar,))
    declared = sidecar.read_text(encoding="ascii").split()[0].strip().lower()
    actual = file_sha256(path)
    if actual != declared:
        raise PoolIntegrityError("pool file SHA-256 %s differs from %s (%s)" % (actual, sidecar.name, declared))
    if expected_sha256 is not None and actual != str(expected_sha256).lower():
        raise PoolIntegrityError("pool file SHA-256 %s differs from the expected %s" % (actual, expected_sha256))
    with gzip.open(path, "rb") as handle:
        document = json.loads(handle.read().decode("ascii"))
    if document.get("schema") != POOL_SCHEMA:
        raise PoolIntegrityError("unsupported pool schema %r (round-4b pools are %s; re-admit older pools)"
                                 % (document.get("schema"), POOL_SCHEMA))
    registry = _registry(dynamic)
    meta = decode(document["meta"], registry)
    if meta.get("task") != "harbour_mission":
        raise PoolIntegrityError("pool was admitted for task %r" % (meta.get("task"),))
    if vessel is not None and meta.get("vessel") != vessel:
        raise PoolIntegrityError("pool was admitted for vessel %r, env runs %r" % (meta.get("vessel"), vessel))
    manifest, manifest_sha = _pinned_manifest(manifest_path)
    entry = pinned_entry(manifest, meta)
    if entry is not None and entry.get("file_sha256") != actual:
        raise PoolIntegrityError("pool %s (vessel %s seed %s grid %s) differs from the pinned manifest: file "
                                 "SHA-256 %s, pinned %s" % (path.name, meta.get("vessel"), meta.get("seed"),
                                                            meta.get("grid"), actual, entry.get("file_sha256")))
    pool = decode(document["pool"], registry)
    verify_pool(pool, meta["offered_keys"], dynamic)
    if entry is not None and entry.get("pool_digest") != pool.pool_digest:
        raise PoolIntegrityError("pool digest %s differs from the pinned %s" % (pool.pool_digest,
                                                                              entry.get("pool_digest")))
    meta = dict(meta)
    meta["not_admitted_map"] = verify_not_admitted(pool, meta)
    meta["offered_grid"] = offered_grid(pool, meta)
    meta["file_sha256"] = actual
    meta["path"] = str(path)
    meta["pinned"] = entry is not None
    meta["pool_manifest_sha256"] = manifest_sha
    return pool, meta


# ------------------------------------------------------------ not admitted --
@dataclass(frozen=True)
class NotAdmitted:
    """D16 record handed to the bridge for a key E4 did not admit (runs at D0, no intruder).

    D24: also the record of a key beyond the offered grid (``beyond_grid``
    True): geometry id and stage None (the pool has no record of the key),
    reasons ((0, "beyond_offered_grid", 0),): dose 0 because the key is
    admitted at no dose and runs at D0, so the entry is the same at every
    requested dose (``requested_dose`` keeps the run's level).
    """

    key: object                 # E4 EpisodeKey (admitted group d7_harbour)
    requested_dose: int
    geometry_id: object         # E5 geometry id of the key's layout (sha256 hex) or None (raw pool / D24)
    stage: object               # the key's strata stage (recorded, never exposed); None for D24 keys
    reasons: tuple              # ((dose, reason, sampled_states), ...) = E4 rejection records (D24: one entry)
    pool_digest: str
    vessel_id: object = None
    exposure: str = NOT_ADMITTED
    beyond_grid: bool = False   # D24: episode index >= the offered grid's episodes per env

    def failing_reasons(self):
        return sorted({r for _d, r, _s in self.reasons if r != JOINT_MEMBER})

    def stamp_assignment(self):
        """JSON-native ``payload.assignment`` of the D7 episode stamp (pairing reads key/task/dose/digest)."""
        k = self.key
        return {"exposure": NOT_ADMITTED,
                "key": {"eval_seed": k.eval_seed, "env": k.env, "episode": k.episode,
                        "hazard_group": k.hazard_group},
                "task": "harbour", "dose": 0, "requested_dose": int(self.requested_dose),
                "site": None, "stage": None, "declared_stage": self.stage, "geometry_id": self.geometry_id,
                "vessel_id": self.vessel_id, "intruders": [], "perceivable": False,
                "candidate_digest": None, "witness_digest": None, "pool_digest": self.pool_digest,
                "not_admitted_reasons": [{"dose": d, "reason": r, "sampled_states": s} for d, r, s in self.reasons],
                "not_admitted_failing_reasons": self.failing_reasons(),
                "rule": (BEYOND_GRID_RULE if self.beyond_grid else
                         "D16: E4 rejected this key; its E5 layout is kept and the dynamic suite runs at D0 "
                         "(no intruder); never redrawn, never skipped")}


BEYOND_GRID_RULE = ("D24: episode index beyond the offered grid (not a pool key); the key's E5 layout is built "
                    "and the dynamic suite runs at D0 (no intruder); never counted")


def stamp_exposure(envelope):
    """"not_admitted" for a D16 record, "admitted" for an E4 assignment stamp.

    Reads ``payload.runtime.exposure`` (absent in E4 stamps = admitted); the
    caller verifies the digest first (E4 ``read_stamp`` / F0
    ``verify_dynamic_hazard_stamp``).
    """
    return envelope["payload"]["runtime"].get("exposure", "admitted")


class HarbourPoolView:
    """The object handed to the F0 mixin (``assignment(key, dose)`` + ``pool_digest`` + counts).

    ``assignment`` returns the E4 Assignment of an admitted key, a
    ``NotAdmitted`` record for an offered key E4 rejected (D16) or for a
    key beyond the offered grid (D24: the grid's eval seed and hazard group,
    0 <= env < num_envs, episode >= episodes), and raises KeyError for any
    other key (env outside the grid, another seed or group: preflight error
    in the mixin). ``per_dose_counts`` rows are [offered, admitted,
    not_admitted] over the offered keys only; ``grid`` is the offered grid
    {"num_envs", "episodes"}.
    """

    def __init__(self, pool, meta=None):
        self.pool = pool
        self.meta = dict(meta or {})
        self.pool_digest = pool.pool_digest
        self.keys_offered = pool.keys_offered
        self.keys_admitted = pool.keys_admitted
        self.joint = pool.joint
        self.file_sha256 = self.meta.get("file_sha256")
        if not pool.joint:
            raise ValueError("harbour D7 pools are joint across D0..D4")
        known = self.meta.get("not_admitted_map")
        if known is None and "not_admitted" in self.meta and "offered_keys" in self.meta:
            known = verify_not_admitted(pool, self.meta)
        if known is None:
            # raw E4 pool (tests): reasons from the rejections, geometry unknown (the env refuses such a row)
            admitted = {k for k, _d in pool.assignments}
            derived = {}
            for r in pool.rejections:
                if r.key not in admitted:
                    derived.setdefault(r.key, []).append((int(r.dose), str(r.reason), int(r.sampled_states)))
            known = {k: (None, None, tuple(v)) for k, v in derived.items()}
        self.not_admitted = dict(known)
        self.keys_not_admitted = len(self.not_admitted)
        self.offered_grid = offered_grid(pool, self.meta)
        self.grid = {"num_envs": self.offered_grid["num_envs"], "episodes": self.offered_grid["episodes"]}
        self.per_dose_counts = tuple((int(o), int(a), int(o) - int(a)) for o, a in pool.per_dose_counts)

    @staticmethod
    def admitted_key(key):
        """Map a ``d7_harbour_mission`` group (pre-v2 F0 mixin) onto the admitted ``d7_harbour`` key."""
        if key.hazard_group == MIXIN_GROUP:
            return type(key)(key.eval_seed, key.env, key.episode, ADMITTED_GROUP)
        return key

    def assignment(self, key, dose):
        akey = self.admitted_key(key)
        found = self.pool.assignments.get((akey, int(dose)))
        if found is not None:
            return found
        if akey in self.not_admitted:
            geometry_id, stage, reasons = self.not_admitted[akey]
            return NotAdmitted(akey, int(dose), geometry_id, stage, reasons, self.pool_digest,
                               self.meta.get("vessel"))
        if self.is_beyond_grid(akey):
            return NotAdmitted(akey, int(dose), None, None, ((0, BEYOND_GRID, 0),), self.pool_digest,
                               self.meta.get("vessel"), beyond_grid=True)
        raise KeyError((key, dose))

    def is_beyond_grid(self, key):
        """D24: the grid's eval seed and hazard group, env inside the grid, episode >= its episodes per env."""
        akey = self.admitted_key(key)
        grid = self.offered_grid
        return (akey.eval_seed == grid["eval_seed"] and akey.hazard_group == grid["hazard_group"]
                and 0 <= akey.env < grid["num_envs"] and akey.episode >= grid["episodes"])

    def has(self, key, dose):
        return (self.admitted_key(key), dose) in self.pool.assignments

    def is_not_admitted(self, key):
        return self.admitted_key(key) in self.not_admitted

    def header_fields(self):
        """Extra ``dynamic_hazards_header`` fields (D16 counts, pool provenance, D19 pin)."""
        from collections import Counter

        failing = Counter()
        for _geometry, _stage, reasons in self.not_admitted.values():
            failing.update({r for _d, r, _s in reasons if r != JOINT_MEMBER})
        return {"per_dose_counts_fields": ["offered", "admitted", "not_admitted"],
                "keys_not_admitted": self.keys_not_admitted,
                "not_admitted_keys": sorted([k.env, k.episode] for k in self.not_admitted),
                "not_admitted_failing_reason_counts": dict(sorted(failing.items())),
                "not_admitted_rule": "D16: rejected keys keep their E5 layout and run the dynamic suite at D0; "
                                     "records carry exposure=not_admitted",
                "pool_schema": POOL_SCHEMA, "pool_file_sha256": self.file_sha256,
                "pool_pinned": bool(self.meta.get("pinned", False)),
                "pool_manifest_sha256": self.meta.get("pool_manifest_sha256"),
                "grid": dict(self.grid),
                "beyond_grid_rule": BEYOND_GRID_RULE}


class HarbourD7Bridge:
    """E4 HazardBridge plus D16 not-admitted rows (same interface: rows, reset_rows, step, contact, finish).

    Admitted rows go to the wrapped E4 bridge unchanged. A ``NotAdmitted``
    row carries no D7 object: ``step`` returns no samples (the env's D7 slot
    stays empty), ``contact`` is False, and ``finish`` returns an E4-format
    stamp (``stamp.make_stamp``: {payload, sha256}, readable by E4
    ``read_stamp`` and F0 ``verify_dynamic_hazard_stamp``) whose
    ``payload.assignment`` is ``NotAdmitted.stamp_assignment()`` (effective
    dose 0, candidate digest None, the E4 rejection records) and whose
    ``payload.runtime`` has exposure "not_admitted", ``assigned`` False and
    ``itt_count`` 0 (CONTEXT_R4 D27: a not-admitted key, D16 or D24, is never
    assigned and is not in the intention-to-treat denominator). The D11 d7_assignment
    hash of such a record depends only on (key, task, effective dose 0,
    None), so it is identical at every requested dose and pairs with the same
    key's D0 record of every other suite.
    """

    def __init__(self, inner, dynamic):
        self.inner = inner
        self.adapter = inner.adapter
        self.cfg = inner.cfg
        self.dt = inner.dt
        adapters = importlib.import_module(dynamic.__name__ + ".adapters")
        self._stamp = importlib.import_module(dynamic.__name__ + ".stamp")
        self._reasons = adapters.TERMINATION_REASONS
        self._pose = dynamic.Pose
        self._na = {}

    @property
    def rows(self):
        out = {env: row for env, row in self.inner.rows.items() if env not in self._na}
        out.update(self._na)
        return out

    def is_not_admitted(self, env):
        return int(env) in self._na

    def row_identity(self, env):
        """{exposure, geometry_id, stage, requested_dose, reasons, beyond_offered_grid} of the row's episode."""
        env = int(env)
        if env in self._na:
            rec = self._na[env]["record"]
            return {"exposure": NOT_ADMITTED, "geometry_id": rec.geometry_id, "stage": rec.stage,
                    "requested_dose": rec.requested_dose, "reasons": rec.reasons,
                    "beyond_offered_grid": bool(rec.beyond_grid)}
        candidate = self.adapter.episodes[env].assignment.candidate
        return {"exposure": "admitted", "geometry_id": candidate.geometry_id, "stage": candidate.stage,
                "requested_dose": candidate.request.encounter.dose, "reasons": (), "beyond_offered_grid": False}

    def reset_rows(self, env_ids, assignments):
        ids = tuple(int(i) for i in env_ids)
        rows = self.rows
        for env in ids:
            if env in rows and not rows[env]["finished"]:
                raise ValueError("finish and stamp every assigned episode before reset")
        admitted = {}
        for env in ids:
            value = assignments[env]
            if isinstance(value, NotAdmitted):
                if value.key.env != env:
                    raise ValueError("not-admitted record of another environment row")
            else:
                admitted[env] = value
        if admitted:
            self.inner.reset_rows(tuple(admitted), admitted)
        for env in ids:
            if env in admitted:
                self._na.pop(env, None)
            else:
                self._na[env] = {"record": assignments[env], "last_tick": -1, "last_time_s": 0.0,
                                 "observer_updates": 0, "swept_intervals": 0, "contact": False,
                                 "finished": False, "stamp": None}

    def step(self, ticks, poses):
        inner = {env: tick for env, tick in ticks.items() if env not in self._na}
        out = self.inner.step(inner, poses) if inner else {}
        for env, tick in ticks.items():
            if env not in self._na:
                continue
            row = self._na[env]
            if row["finished"]:
                raise ValueError("row finished; reset it before stepping")
            if type(tick) is not int or tick < 0:
                raise ValueError("tick must be a nonnegative integer")
            if tick == row["last_tick"]:
                out[env] = ()
                continue
            if tick < row["last_tick"]:
                raise ValueError("physics tick went backwards without a reset")
            if not isinstance(poses[env], self._pose):
                raise TypeError("pose must be a Pose")
            if row["last_tick"] >= 0:
                row["swept_intervals"] += 1
            row["observer_updates"] += 1
            row["last_tick"] = tick
            row["last_time_s"] = tick * self.dt
            out[env] = ()
        return out

    def contact(self, env):
        if int(env) in self._na:
            return False
        return self.inner.contact(env)

    def finish(self, env, termination_reason):
        env = int(env)
        if env not in self._na:
            return self.inner.finish(env, termination_reason)
        if termination_reason not in self._reasons:
            raise ValueError("explicit terminal reason required: " + repr(termination_reason))
        row = self._na[env]
        if row["finished"]:
            return row["stamp"]
        rec = row["record"]
        runtime = {
            # D27: never assigned, so outside the intention-to-treat denominator (E4 admitted stamps: True, 1)
            "exposure": NOT_ADMITTED, "assigned": False, "itt_count": 0, "env": env,
            "termination_reason": termination_reason, "requested_dose": int(rec.requested_dose),
            "effective_dose": 0, "last_time_s": max(0.0, row["last_time_s"]),
            "first_detection_s": None, "realized_ttc_at_first_detection_s": None,
            "detected_samples": 0, "observer_updates": row["observer_updates"],
            "min_sampled_surface_clearance_m": None, "sampled_contact": False,
            "non_exposure_reason": NOT_ADMITTED,
            "bridge": {"swept_contact_latched": False, "first_swept_contact_interval_start_s": None,
                       "swept_intervals": row["swept_intervals"], "last_tick": row["last_tick"],
                       "physics_dt_s": self.dt,
                       "swept_contact_semantics": (
                           "no D7 object: key beyond the offered grid (D24), dynamic suite at D0" if rec.beyond_grid
                           else "no D7 object: key not admitted (D16), dynamic suite at D0")},
        }
        stamp = self._stamp.make_stamp(rec.stamp_assignment(), runtime)
        row["finished"] = True
        row["stamp"] = stamp
        return stamp


# ------------------------------------------------------------------- paths --
def pool_file_name(vessel, seed):
    return "harbour_mission_%s_seed%d.json.gz" % (vessel, int(seed))


def resolve_pool_source(cfg):
    """(path, source) of the harbour pool, or (None, None). Order (all RUNTIME variables, D19):

    1. ``cfg.journal_d7_pool_path`` (file or directory);
    2. ``$USVBENCH_HARBOUR_D7_POOL`` (file or directory);
    3. ``$USVBENCH_D7_POOL_DIR`` (directory: ``<dir>/harbor_mission/<file>`` or ``<dir>/<file>``);
    4. the pinned folder ``tasks/harbor_mission/d7_pools/<file>`` when the file is there.
    ``<file>`` = ``harbour_mission_<vessel>_seed<seed>.json.gz``.
    """
    name = pool_file_name(cfg.vehicle, int(cfg.seed))
    for source, value in (("cfg.journal_d7_pool_path", getattr(cfg, "journal_d7_pool_path", None)),
                          (POOL_PATH_ENV, os.environ.get(POOL_PATH_ENV))):
        if value:
            path = Path(value)
            return (path / name if path.is_dir() else path), source
    shared = os.environ.get(POOL_DIR_ENV)
    if shared:
        candidates = (Path(shared) / FAMILY_DIR / name, Path(shared) / name)
        for path in candidates:
            if path.is_file():
                return path, POOL_DIR_ENV
        return candidates[0], POOL_DIR_ENV
    pinned = PINNED_DIR / name
    if pinned.is_file():
        return pinned, "pinned folder tasks/harbor_mission/d7_pools"
    return None, None


def resolve_pool_path(cfg):
    """The pool path of ``resolve_pool_source`` (None when nothing is configured)."""
    return resolve_pool_source(cfg)[0]


__all__ = ["POOL_SCHEMA", "PINNED_SCHEMA", "PINNED_DIR", "PINNED_MANIFEST", "MIXIN_GROUP", "ADMITTED_GROUP",
           "POOL_PATH_ENV", "POOL_DIR_ENV", "NOT_ADMITTED", "BEYOND_GRID", "BEYOND_GRID_RULE", "PoolIntegrityError",
           "encode", "decode", "file_sha256", "pool_document", "write_pool", "verify_pool", "verify_not_admitted",
           "offered_grid", "pinned_entry",
           "load_pool", "NotAdmitted", "HarbourPoolView", "HarbourD7Bridge", "stamp_exposure",
           "resolve_pool_source", "resolve_pool_path", "pool_file_name"]
