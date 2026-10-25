"""Run D4 tests by file path, with simulator imports and caches disabled."""
import importlib.abc
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class NoSimulatorImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"isaaclab", "isaacsim", "omni"}:
            raise ImportError("D4 CPU tests forbid simulator imports: " + fullname)
        return None


sys.meta_path.insert(0, NoSimulatorImports())
HERE = Path(__file__).resolve().parent
if __name__ == "__main__":
    import pytest

    print("D4 CPU only; simulator imports blocked; bytecode/cache disabled.")
    print("Python " + sys.version.replace("\n", " "))
    raise SystemExit(pytest.main([
        "-v", "-ra", "-p", "no:cacheprovider", "--import-mode=importlib", *sys.argv[1:],
        *[str(path) for path in sorted(HERE.glob("test_*.py"))],
    ]))
