#!/usr/bin/env python3
"""Check that a tree holds exactly the files of a USVBench release manifest, byte for byte.

usage:
  python releases/verify_release_tree.py --release releases/v2.6.6 [--root .]
  python releases/verify_release_tree.py --release releases/v2.6.6 --rev v2.6.6

--release  folder holding usvbench-<release_id>.manifest and usvbench-<release_id>.assets.manifest
--root     check files on disk under this folder (default: the repository root)
--rev      check the blobs committed at this git revision instead (reads them with git cat-file,
           so local line-ending settings cannot change the result)

The manifest format is usvbench-code-tree-v1 / usvbench-asset-tree-v1: one JSON array per line,
[path, "file", mode, size, sha256]. The SHA-256 of the code manifest file itself is the
code_tree_sha256 (the second half of the release id). Only the manifest roots (assets_v2,
scripts, tasks) are compared; other top-level folders of the repository are not part of the
release. Prints VERIFY-OK and exits 0 on an exact match.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def read_manifests(release_dir):
    expected, digests = {}, {}
    files = sorted(release_dir.glob("usvbench-*.manifest"))
    if len(files) != 2:
        sys.exit("expected one code manifest and one assets manifest in %s, found %d" % (release_dir, len(files)))
    for m in files:
        raw = m.read_bytes()
        digests[m.name] = hashlib.sha256(raw).hexdigest()
        lines = raw.decode("utf-8").splitlines()
        if lines[0] not in ("usvbench-code-tree-v1", "usvbench-asset-tree-v1"):
            sys.exit("unknown manifest header in %s: %r" % (m, lines[0]))
        for line in lines[1:]:
            if line.strip():
                path, kind, _mode, size, sha = json.loads(line)
                if kind != "file":
                    sys.exit("unexpected entry kind %r in %s" % (kind, m))
                expected[path] = (int(size), sha)
    return expected, digests


def tree_on_disk(root, tops):
    actual = {}
    for top in tops:
        base = root / top
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file():
                if "__pycache__" in p.parts:
                    continue
                b = p.read_bytes()
                actual[p.relative_to(root).as_posix()] = (len(b), hashlib.sha256(b).hexdigest())
    return actual


def tree_at_rev(repo, rev, tops):
    ls = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "-z", rev, "--"] + sorted(tops),
                        check=True, capture_output=True).stdout.split(b"\0")
    entries = []
    for item in ls:
        if not item:
            continue
        meta, path = item.split(b"\t", 1)
        mode, kind, sha1 = meta.split()
        if kind != b"blob":
            continue
        entries.append((path.decode("utf-8"), sha1.decode()))
    proc = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)
    actual = {}
    for path, sha1 in entries:
        proc.stdin.write((sha1 + "\n").encode())
        proc.stdin.flush()
        header = proc.stdout.readline().split()
        size = int(header[2])
        data = proc.stdout.read(size)
        proc.stdout.read(1)
        actual[path] = (size, hashlib.sha256(data).hexdigest())
    proc.stdin.close()
    proc.wait()
    return actual


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--rev")
    a = ap.parse_args()
    expected, digests = read_manifests(a.release)
    for name, d in digests.items():
        print("manifest %s sha256 %s" % (name, d))
    tops = {p.split("/")[0] for p in expected}
    actual = tree_at_rev(a.root, a.rev, tops) if a.rev else tree_on_disk(a.root, tops)
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    differ = sorted(p for p in set(expected) & set(actual) if expected[p] != actual[p])
    print("expected=%d actual=%d missing=%d extra=%d differ=%d" % (len(expected), len(actual), len(missing),
                                                                  len(extra), len(differ)))
    for tag, items in (("MISSING", missing), ("EXTRA", extra), ("DIFFER", differ)):
        for p in items[:50]:
            print(tag, p)
    ok = not (missing or extra or differ)
    print("VERIFY-OK" if ok else "VERIFY-FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
