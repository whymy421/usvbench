"""E1: usvbench-code-tree-v1 equality between the evaluator and the rental kit.

The certificate writer (scripts/code_tree.py) and the release builder
(tools/rental_kit/usvb_kit.py build-release) must compute the SAME digest for
the same commit, whatever the checkout's line endings. Proven here on a
synthetic git repository (CPU only; git is the only external program).
"""
import importlib.util
import io
import contextlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
KIT_PATH = REPO / "tools" / "rental_kit" / "usvb_kit.py"
sys.path.insert(0, str(SCRIPTS))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


code_tree = load("code_tree", SCRIPTS / "code_tree.py")
protocol = load("eval_protocol", SCRIPTS / "eval_protocol.py")
kit = load("e1_usvb_kit", KIT_PATH)

FILES = {
    "tasks/_shared/__init__.py": b"",
    "tasks/fam/__init__.py": b'gym.register(id="Isaac-USV-Fam-Direct-v1")\n',
    "tasks/fam/fam_env.py": b"VALUE = 1\r\nOTHER = 2\r\n",          # CRLF blob
    "tasks/fam/agents/skrl_ppo_cfg.yaml": b"seed: 42\r\n",
    "tasks/fam/cfg.toml": b"[a]\nb = 1\n",
    "tasks/fam/checkpoints/agent_s42.pt": b"\x00\x01weights\r\n",   # excluded
    "tasks/fam/notes.md": b"# not code\n",                          # excluded
    "scripts/eval.py": b"import sys\r\nprint(sys.argv)\r\n",         # CRLF blob
    "scripts/data.json": b'{"a": 1}\n',
    "scripts/tool.cfg": b"[x]\n",
    "scripts/extra.yml": b"k: v\n",
    "scripts/run_queue.sh": b"echo hi\n",                           # excluded
    "docs/README.md": b"docs\n",                                    # excluded
    "assets/boat.usd": bytes(range(256)),                           # asset tree
}
EXPECTED_MEMBERS = sorted(p for p in FILES if p.endswith((".py", ".yaml", ".toml", ".json", ".cfg", ".yml")))


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=e1", "-c", "user.email=e1@example.invalid",
                           "-c", "core.autocrlf=false", "-c", "core.safecrlf=false"] + list(args),
                          check=True, capture_output=True)


def make_repo(root):
    repo = root / "repo"
    for rel, data in FILES.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "synthetic")
    return repo


def run_kit(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = kit.main(list(argv))
    return rc, out.getvalue()


def build(repo, out):
    out.mkdir()
    rc, text = run_kit("build-release", "--repo", str(repo), "--out-dir", str(out))
    assert rc == 0, text
    info = dict(line.split("=", 1) for line in text.splitlines() if "=" in line and " " not in line)
    rid = info["RELEASE_ID"]
    return SimpleNamespace(rid=rid, code_tree=info["CODE_TREE_SHA256"],
                           manifest=out / ("usvbench-%s.manifest" % rid),
                           descriptor=out / ("usvbench-%s.release.json" % rid),
                           artifact=out / ("usvbench-%s.tar.gz" % rid))


def test_inclusion_rules_are_one_definition():
    assert kit.INCLUSION_SPEC == code_tree.INCLUSION_SPEC
    assert kit.MANIFEST_MAGIC == code_tree.MANIFEST_MAGIC and kit.CODE_FILE_MODE == code_tree.FILE_MODE
    samples = list(FILES) + ["tasks/x/__pycache__/m.py", "tasks/.cache/m.py", "scripts/a.PY", "scripts/b.pyc",
                             "tasks/x.py", "scripts/sub/deep/c.json", "tools/rental_kit/usvb_kit.py", "x.py",
                             "tasks/fam/model.usd", "scripts/.hidden.json"]
    for path in samples:
        assert kit.is_code_member(path) == code_tree.is_included(path), path
    assert code_tree.normalize_content(b"a\r\nb\rc\n") == kit.normalize_content(b"a\r\nb\rc\n") == b"a\nb\rc\n"


def test_build_release_digest_equals_working_tree_digest(tmp_path):
    repo = make_repo(tmp_path)
    release = build(repo, tmp_path / "out")
    manifest, records, digest = code_tree.manifest_from_tree(repo)
    assert digest == release.code_tree
    assert release.manifest.read_bytes() == manifest  # byte-identical manifests
    assert sorted(records) == EXPECTED_MEMBERS
    assert all(mode == "100644" for mode, _, _ in records.values())
    # the CRLF blob is hashed LF-normalised on both sides
    assert records["scripts/eval.py"][2] == code_tree.sha256_bytes(b"import sys\nprint(sys.argv)\n")
    descriptor = json.loads(release.descriptor.read_text(encoding="ascii"))
    assert descriptor["inclusion"] == code_tree.INCLUSION_SPEC
    assert descriptor["excluded_by_suffix"] == {".md": 2, ".pt": 1, ".sh": 1}
    assert descriptor["assets_file_count"] == 1
    loaded = code_tree.load_release(release.descriptor)
    assert loaded["records"] == records and loaded["code_tree_sha256"] == digest


def test_crlf_checkout_and_untracked_caches_give_the_same_digest(tmp_path):
    repo = make_repo(tmp_path)
    release = build(repo, tmp_path / "out")
    windows = tmp_path / "windows_checkout"
    shutil.copytree(repo, windows, ignore=shutil.ignore_patterns(".git"))
    for path in windows.rglob("*"):
        if path.is_file() and code_tree.is_included(path.relative_to(windows).as_posix()):
            data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")  # autocrlf=true
            path.write_bytes(data)
    (windows / "scripts" / "__pycache__").mkdir()
    (windows / "scripts" / "__pycache__" / "eval.cpython-313.pyc").write_bytes(b"cache")
    (windows / "tasks" / ".pytest_cache").mkdir()
    (windows / "tasks" / ".pytest_cache" / "v.json").write_text("{}", encoding="ascii")
    (windows / "tasks" / "fam" / "checkpoints" / "agent_s43.pt").write_bytes(b"new weights")
    (windows / "tasks" / "fam" / "play.mp4").write_bytes(b"video")
    assert code_tree.manifest_from_tree(windows)[2] == release.code_tree


def test_source_changes_and_untracked_members_change_the_digest(tmp_path):
    repo = make_repo(tmp_path)
    base = code_tree.manifest_from_tree(repo)[2]
    (repo / "tasks" / "fam" / "fam_env.py").write_bytes(b"VALUE = 2\r\nOTHER = 2\r\n")
    changed = code_tree.manifest_from_tree(repo)[2]
    assert changed != base
    (repo / "scripts" / "stray_result.json").write_text("{}", encoding="ascii")
    assert code_tree.manifest_from_tree(repo)[2] not in (base, changed)


def test_release_tree_preflight_accepts_release_and_refuses_drift(tmp_path):
    repo = make_repo(tmp_path)
    release = build(repo, tmp_path / "out")
    dest = tmp_path / "releases" / release.rid
    dest.parent.mkdir()
    rc, text = run_kit("extract", "--artifact", str(release.artifact), "--manifest", str(release.manifest),
                       "--expect-code-tree", release.code_tree, "--dest", str(dest))
    assert rc == 0, text
    assert (dest / "assets" / "boat.usd").is_file() and not (dest / "docs").exists()
    args = SimpleNamespace(release_descriptor=str(release.descriptor))
    loaded = protocol.resolve_release(args, environ={}, code_root=dest)
    assert args.release_id == release.rid and loaded["code_tree_sha256"] == release.code_tree
    env_args = SimpleNamespace(release_descriptor=None)
    protocol.resolve_release(env_args, environ={"USVBENCH_RELEASE_DESCRIPTOR": str(release.descriptor)},
                             code_root=dest)
    assert env_args.release_id == release.rid
    (dest / "scripts" / "eval.py").write_bytes(b"import sys\nprint('patched')\n")
    with pytest.raises(ValueError, match="CHANGED scripts/eval.py"):
        protocol.resolve_release(SimpleNamespace(release_descriptor=str(release.descriptor)), environ={},
                                 code_root=dest)
    other = tmp_path / "other.release.json"
    other.write_bytes(release.descriptor.read_bytes())
    with pytest.raises(ValueError, match="different descriptors"):
        protocol.resolve_release(SimpleNamespace(release_descriptor=str(release.descriptor)),
                                 environ={"USVBENCH_RELEASE_DESCRIPTOR": str(other)}, code_root=dest)
