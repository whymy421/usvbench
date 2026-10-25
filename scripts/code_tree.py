"""usvbench-code-tree-v1: the release code identity. Standard library only.

One definition is shared by the certificate writer (cert_v3.py), the certificate
validator (validate_cert_v3.py), the evaluator preflight (eval_protocol.py) and
the rental kit (tools/rental_kit/usvb_kit.py build-release, which carries a
byte-for-byte equivalent copy of the functions below because it must run
without this repository; scripts/test_e1_code_tree.py proves the two agree on a
synthetic git repository).

Inclusion specification ``usvbench-release-inclusion-v1`` (C6 RECONCILE.md
section 6.2; D10 FINAL_REPORT decisions 5 and 7; CONTEXT_R3 decision 16):

* members: files under ``tasks/`` and ``scripts/`` whose name ends in one of
  ``.py .yaml .yml .json .toml .cfg`` (case-sensitive); never checkpoints
  (``*.pt``), media, ``__pycache__``/bytecode, or any path component that
  starts with a dot (tool caches such as ``.pytest_cache``);
* content: raw bytes with every CRLF pair replaced by LF (a Windows CRLF
  checkout and a Linux LF release of the same commit give the same digest);
* mode: every member is recorded as ``100644``. Executable bits carry no code
  semantics for interpreted source and cannot be read from a Windows checkout
  (C6: "do not hash Windows filesystem permission bits").

Manifest format (identical to D10 usvb_kit.py, C6 section 6.2): the manifest is
the exact byte string that is hashed. Line 1 is ``usvbench-code-tree-v1``; then
one canonical JSON array ``[path, "file", "100644", size, sha256]`` per line
(separators ``,`` and ``:``), sorted by UTF-8 path bytes, LF terminated.
``code_tree_sha256 = SHA-256(manifest bytes)``; size and sha256 are those of
the LF-normalised content.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

MANIFEST_MAGIC = "usvbench-code-tree-v1"
HASH_ALGORITHM = MANIFEST_MAGIC
DESCRIPTOR_FORMAT = "usvbench-release-v1"
INCLUSION_VERSION = "usvbench-release-inclusion-v1"
INCLUDE_ROOTS = ("scripts", "tasks")
INCLUDE_SUFFIXES = (".cfg", ".json", ".py", ".toml", ".yaml", ".yml")
EXCLUDED_PARTS = ("__pycache__",)
FILE_MODE = "100644"
ALLOWED_MODES = ("100644", "100755")
INCLUSION_SPEC = {
    "version": INCLUSION_VERSION,
    "roots": list(INCLUDE_ROOTS),
    "suffixes": list(INCLUDE_SUFFIXES),
    "excluded_parts": list(EXCLUDED_PARTS),
    "excluded_prefix": ".",
    "line_endings": "CRLF->LF",
    "file_mode": FILE_MODE,
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
GIT_OID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
RELEASE_ID = re.compile(r"^[0-9a-f]{12}-[0-9a-f]{12}$")
RELEASE_DESCRIPTOR_ENV = "USVBENCH_RELEASE_DESCRIPTOR"


class CodeTreeError(ValueError):
    """The code tree, manifest or release descriptor is not acceptable."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def normalize_content(data):
    """The one line-ending rule: every CRLF pair becomes LF; nothing else changes."""
    return bytes(data).replace(b"\r\n", b"\n")


def check_rel_path(path):
    if not isinstance(path, str) or not path:
        raise CodeTreeError("empty or non-string path")
    if path.startswith("/") or "\\" in path or "\x00" in path:
        raise CodeTreeError("unsafe path (absolute, backslash or NUL): %r" % path)
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in path):
        raise CodeTreeError("control character in path: %r" % path)
    for part in path.split("/"):
        if part in ("", ".", ".."):
            raise CodeTreeError("unsafe path component in %r" % path)


def is_included(path):
    """True iff the POSIX relative path is a member under the inclusion spec."""
    parts = path.split("/")
    if len(parts) < 2 or parts[0] not in INCLUDE_ROOTS:
        return False
    if any(part in EXCLUDED_PARTS or part.startswith(".") for part in parts):
        return False
    return parts[-1].endswith(INCLUDE_SUFFIXES)


def path_sort_key(path):
    return path.encode("utf-8")


def record_line(path, mode, size, digest):
    return json.dumps([path, "file", mode, size, digest], separators=(",", ":"), ensure_ascii=False)


def _check_case_collisions(paths):
    seen = {}
    for path in paths:
        parts = path.split("/")
        for depth in range(1, len(parts) + 1):
            prefix = "/".join(parts[:depth])
            other = seen.setdefault(prefix.casefold(), prefix)
            if other != prefix:
                raise CodeTreeError("case-fold collision: %r vs %r" % (other, prefix))


def render_manifest(entries):
    """entries: iterable of (path, size, sha256) of NORMALISED content -> bytes."""
    ordered = sorted(entries, key=lambda item: path_sort_key(item[0]))
    lines = [MANIFEST_MAGIC]
    previous = None
    for path, size, digest in ordered:
        check_rel_path(path)
        if not is_included(path):
            raise CodeTreeError("path is outside the inclusion specification: %r" % path)
        if previous is not None and path_sort_key(previous) >= path_sort_key(path):
            raise CodeTreeError("duplicate path in manifest: %r" % path)
        if not HEX64.match(digest):
            raise CodeTreeError("bad sha256 for %r" % path)
        lines.append(record_line(path, FILE_MODE, int(size), digest))
        previous = path
    if previous is None:
        raise CodeTreeError("the code tree has no member files")
    _check_case_collisions([item[0] for item in ordered])
    return ("\n".join(lines) + "\n").encode("utf-8")


def parse_manifest(raw):
    """Parse and canonically re-validate a manifest -> ({path: (mode, size, sha)}, digest)."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CodeTreeError("manifest is not UTF-8: %s" % exc)
    if not text.endswith("\n") or "\r" in text:
        raise CodeTreeError("manifest must be LF terminated without CR bytes")
    lines = text[:-1].split("\n")
    if lines[0] != MANIFEST_MAGIC:
        raise CodeTreeError("manifest magic line is %r" % lines[0])
    records, previous = {}, None
    for number, line in enumerate(lines[1:], start=2):
        try:
            item = json.loads(line)
        except ValueError as exc:
            raise CodeTreeError("manifest line %d is not JSON: %s" % (number, exc))
        if not (isinstance(item, list) and len(item) == 5):
            raise CodeTreeError("manifest line %d is not a 5-element array" % number)
        rel, kind, mode, size, digest = item
        if kind != "file" or mode not in ALLOWED_MODES:
            raise CodeTreeError("manifest line %d: unsupported kind/mode" % number)
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise CodeTreeError("manifest line %d: bad size" % number)
        if not isinstance(digest, str) or not HEX64.match(digest):
            raise CodeTreeError("manifest line %d: bad sha256" % number)
        check_rel_path(rel)
        if record_line(rel, mode, size, digest) != line:
            raise CodeTreeError("manifest line %d is not canonical" % number)
        if previous is not None and path_sort_key(previous) >= path_sort_key(rel):
            raise CodeTreeError("manifest line %d is out of order or duplicated" % number)
        previous = rel
        records[rel] = (mode, size, digest)
    if not records:
        raise CodeTreeError("manifest lists no files")
    _check_case_collisions(list(records))
    return records, sha256_bytes(raw)


def tree_entries(root):
    """(path, size, sha256) of every member file under root/tasks and root/scripts."""
    root = Path(root)
    entries = []
    for top in INCLUDE_ROOTS:
        base = root / top
        if not base.is_dir() or base.is_symlink():
            continue
        for current, dirnames, filenames in os.walk(base, followlinks=False):
            rel_dir = Path(current).relative_to(root).as_posix()
            keep = []
            for name in sorted(dirnames):
                full = os.path.join(current, name)
                if name in EXCLUDED_PARTS or name.startswith("."):
                    continue
                if os.path.islink(full):
                    raise CodeTreeError("symbolic link in the code tree: %s/%s" % (rel_dir, name))
                keep.append(name)
            dirnames[:] = keep
            for name in sorted(filenames):
                rel = rel_dir + "/" + name
                if not is_included(rel):
                    continue
                full = os.path.join(current, name)
                info = os.lstat(full)
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                    raise CodeTreeError("member is not a regular file: %s" % rel)
                with open(full, "rb") as stream:
                    content = normalize_content(stream.read())
                entries.append((rel, len(content), sha256_bytes(content)))
    return entries


def manifest_from_tree(root):
    """Return (manifest bytes, records, code_tree_sha256) for a checkout or release root."""
    manifest = render_manifest(tree_entries(root))
    records, digest = parse_manifest(manifest)
    return manifest, records, digest


def compare_tree(root, records):
    """Exact comparison of the member files under root with manifest records."""
    actual = {path: (FILE_MODE, size, digest) for path, size, digest in tree_entries(root)}
    errors = []
    for path in sorted(set(records) | set(actual), key=path_sort_key):
        if path not in actual:
            errors.append("MISSING " + path)
        elif path not in records:
            errors.append("EXTRA " + path)
        elif actual[path][1:] != tuple(records[path][1:]):
            errors.append("CHANGED " + path)
    return errors


def load_release(descriptor_path):
    """Load a release descriptor and its manifest (next to it); verify identities.

    Returns a dict with the descriptor fields plus ``records``, ``manifest_path``,
    ``descriptor_path`` and ``descriptor_sha256``. The manifest must have been
    built with the inclusion specification above.
    """
    path = Path(descriptor_path).resolve()
    raw = path.read_bytes()
    try:
        data = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CodeTreeError("release descriptor is not ASCII JSON: %s" % exc)
    if not isinstance(data, dict):
        raise CodeTreeError("release descriptor is not a JSON object")
    for key in ("format", "release_id", "git_commit", "git_tree", "code_tree_sha256", "manifest_file"):
        if key not in data:
            raise CodeTreeError("release descriptor lacks " + key)
    if data["format"] != DESCRIPTOR_FORMAT:
        raise CodeTreeError("release descriptor format %r" % (data["format"],))
    for key in ("git_commit", "git_tree"):
        if not GIT_OID.match(str(data[key])):
            raise CodeTreeError("release descriptor %s malformed" % key)
    if not HEX64.match(str(data["code_tree_sha256"])):
        raise CodeTreeError("release descriptor code_tree_sha256 malformed")
    if not RELEASE_ID.match(str(data["release_id"])) or data["release_id"] != "%s-%s" % (
            data["git_commit"][:12], data["code_tree_sha256"][:12]):
        raise CodeTreeError("release_id does not match the commit and code tree")
    inclusion = data.get("inclusion")
    if inclusion != INCLUSION_SPEC:
        raise CodeTreeError("release was not built with the %s inclusion specification" % INCLUSION_VERSION)
    name = str(data["manifest_file"])
    if "/" in name or "\\" in name or name in ("", ".", ".."):
        raise CodeTreeError("manifest_file must be a bare file name next to the descriptor")
    manifest_path = path.parent / name
    records, digest = parse_manifest(manifest_path.read_bytes())
    if digest != data["code_tree_sha256"]:
        raise CodeTreeError("manifest SHA-256 %s differs from the descriptor %s" % (digest, data["code_tree_sha256"]))
    outside = [rel for rel in records if not is_included(rel)]
    if outside:
        raise CodeTreeError("manifest lists files outside the inclusion specification: %s" % outside[:5])
    if any(mode != FILE_MODE for mode, _, _ in records.values()):
        raise CodeTreeError("inclusion specification records every member as %s" % FILE_MODE)
    result = dict(data)
    result.update(records=records, manifest_path=str(manifest_path), descriptor_path=str(path),
                  descriptor_sha256=sha256_bytes(raw))
    return result


def families(records):
    """Task packages of a manifest: tasks/<name>/__init__.py present."""
    return sorted({rel.split("/")[1] for rel in records
                   if rel.startswith("tasks/") and rel.count("/") == 2 and rel.endswith("/__init__.py")})


def _under(path, root):
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def module_member_path(file, scripts_root, task_roots, member_families):
    """Map a loaded source file to its manifest path, or None if it is not ours.

    ``task_roots``: directories whose children are task families, e.g. the
    repository ``tasks/`` folder or Isaac Lab's staged ``.../direct/`` folder
    (which also holds upstream Isaac Lab tasks: only member families map).
    """
    path = Path(file).resolve()
    relative = _under(path, Path(scripts_root).resolve())
    if relative is not None:
        return "scripts/" + relative.as_posix()
    for root in task_roots:
        relative = _under(path, Path(root).resolve())
        if relative is not None and relative.parts and relative.parts[0] in member_families:
            return "tasks/" + relative.as_posix()
    return None


def verify_loaded_modules(records, scripts_root, task_roots, modules):
    """Hash every loaded USVBench source module against the manifest records."""
    member_families = set(families(records))
    checked, mismatches, seen = 0, [], set()
    for name in sorted(modules):
        module = modules.get(name)
        file = getattr(module, "__file__", None)
        if not isinstance(file, str) or not file.endswith(".py"):
            continue
        rel = module_member_path(file, scripts_root, task_roots, member_families)
        if rel is None or rel in seen:
            continue
        seen.add(rel)
        checked += 1
        record = records.get(rel)
        try:
            content = normalize_content(Path(file).read_bytes())
        except OSError:
            mismatches.append("UNREADABLE " + rel)
            continue
        if record is None:
            mismatches.append("NOT-IN-MANIFEST " + rel)
        elif (len(content), sha256_bytes(content)) != (record[1], record[2]):
            mismatches.append("CHANGED " + rel)
    return {"checked": checked, "verified": not mismatches, "mismatches": mismatches[:50],
            "mismatch_count": len(mismatches)}
