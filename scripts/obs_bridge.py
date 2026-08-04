"""CPU-only native-observation and skrl-checkpoint bridge for USVBench."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]
_OBS_PATH = _REPO_ROOT / "tasks" / "_shared" / "obs_superset.py"
_OBS_SPEC = importlib.util.spec_from_file_location("usvbench_obs_superset", _OBS_PATH)
if _OBS_SPEC is None or _OBS_SPEC.loader is None:  # pragma: no cover - import guard
    raise ImportError(f"Cannot load observation contract from {_OBS_PATH}")
_OBS = importlib.util.module_from_spec(_OBS_SPEC)
sys.modules.setdefault(_OBS_SPEC.name, _OBS)
_OBS_SPEC.loader.exec_module(_OBS)


def _layout(gym_id: str) -> list[int]:
    """Return a supported layout, preserving the contract's public errors."""

    try:
        layout = _OBS.NATIVE_LAYOUTS[gym_id]
    except KeyError:
        # Exercise the contract helper to produce its complete known-id error.
        _OBS.native_to_superset([], gym_id)
        raise AssertionError("unreachable")
    if isinstance(layout, _OBS.UnsupportedNativeLayout):
        # Exercise the same helper for one canonical unsupported-id message.
        _OBS.native_to_superset([], gym_id)
        raise AssertionError("unreachable")
    return layout


def project(vec: Any, src_gym_id: str, dst_gym_id: str) -> Any:
    """Project native source observations into the destination native order."""

    superset = _OBS.native_to_superset(vec, src_gym_id, dim=_OBS.SUPERSET_DIM_V2)
    return _OBS.extract_native(superset, dst_gym_id)


def _column_mapping(src_layout: list[int], dst_layout: list[int]) -> list[int | None]:
    src_native_by_superset = {slot: native for native, slot in enumerate(src_layout)}
    return [src_native_by_superset.get(slot) for slot in dst_layout]


def _scatter_last_dim(tensor: Any, mapping: list[int | None], fill: float = 0.0) -> Any:
    out = tensor.new_full((*tensor.shape[:-1], len(mapping)), fill)
    for dst_native, src_native in enumerate(mapping):
        if src_native is not None:
            out[..., dst_native] = tensor[..., src_native]
    return out


def _load_checkpoint(path: str | Path) -> Any:
    import torch

    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with Torch versions predating weights_only.
        return torch.load(Path(path), map_location="cpu")


def _checkpoint_structure(checkpoint: Any) -> str:
    if not isinstance(checkpoint, dict):
        return f"root={type(checkpoint).__name__}"
    return ", ".join(
        f"{key}={type(value).__name__}" for key, value in checkpoint.items()
    )


def _adapt_optimizer_state(
    value: Any,
    *,
    src_width: int,
    mapping: list[int | None],
    leading_shapes: set[tuple[int, ...]],
    path: str = "optimizer",
) -> list[str]:
    """Resize Adam moments corresponding to the resized leading weights."""

    import torch

    adapted: list[str] = []
    if isinstance(value, dict):
        for key, child in list(value.items()):
            child_path = f"{path}.{key}"
            if (
                torch.is_tensor(child)
                and child.ndim >= 2
                and child.shape[-1] == src_width
                and tuple(child.shape[:-1]) in leading_shapes
            ):
                value[key] = _scatter_last_dim(child, mapping)
                adapted.append(child_path)
            else:
                adapted.extend(
                    _adapt_optimizer_state(
                        child,
                        src_width=src_width,
                        mapping=mapping,
                        leading_shapes=leading_shapes,
                        path=child_path,
                    )
                )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            adapted.extend(
                _adapt_optimizer_state(
                    child,
                    src_width=src_width,
                    mapping=mapping,
                    leading_shapes=leading_shapes,
                    path=f"{path}[{index}]",
                )
            )
    return adapted


def _prepare_checkpoint(
    src_ckpt_path: str | Path, src_gym_id: str, dst_gym_id: str
) -> tuple[Any, dict[str, Any]]:
    """Load and resize the real skrl save structure entirely on CPU."""

    import torch

    src_layout = _layout(src_gym_id)
    dst_layout = _layout(dst_gym_id)
    src_width = len(src_layout)
    dst_width = len(dst_layout)
    mapping = _column_mapping(src_layout, dst_layout)
    checkpoint = _load_checkpoint(src_ckpt_path)

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Unsupported checkpoint structure; expected the inspected skrl dict, "
            f"got {_checkpoint_structure(checkpoint)}"
        )

    matched: list[str] = []
    leading_shapes: set[tuple[int, ...]] = set()
    for section_name in ("policy", "value"):
        section = checkpoint.get(section_name)
        if not isinstance(section, dict):
            raise TypeError(
                "Unsupported checkpoint structure; expected mapping sections "
                f"'policy' and 'value', got {_checkpoint_structure(checkpoint)}"
            )
        section_matches = 0
        for key, value in list(section.items()):
            if (
                key.endswith(".weight")
                and torch.is_tensor(value)
                and value.ndim == 2
                and value.shape[1] == src_width
            ):
                leading_shapes.add(tuple(value.shape[:-1]))
                section[key] = _scatter_last_dim(value, mapping)
                matched.append(f"{section_name}.{key}")
                section_matches += 1
        if section_matches == 0:
            available = ", ".join(
                f"{key}:{tuple(value.shape)}"
                for key, value in section.items()
                if torch.is_tensor(value)
            )
            raise ValueError(
                f"No leading Linear weight with in_features={src_width} in "
                f"checkpoint section {section_name!r}; tensors: {available}"
            )

    adapted_preprocessor: list[str] = []
    state_preprocessor = checkpoint.get("state_preprocessor")
    if isinstance(state_preprocessor, dict):
        for key, value in list(state_preprocessor.items()):
            if torch.is_tensor(value) and value.ndim == 1 and value.shape[0] == src_width:
                # Unseen destination inputs should be identity-normalised:
                # zero mean and unit variance, not zero variance.
                fill = 1.0 if "variance" in key.lower() else 0.0
                state_preprocessor[key] = _scatter_last_dim(value, mapping, fill=fill)
                adapted_preprocessor.append(f"state_preprocessor.{key}")

    adapted_optimizer: list[str] = []
    if "optimizer" in checkpoint:
        adapted_optimizer = _adapt_optimizer_state(
            checkpoint["optimizer"],
            src_width=src_width,
            mapping=mapping,
            leading_shapes=leading_shapes,
        )

    report = {
        "structure": _checkpoint_structure(checkpoint),
        "src_layout": src_layout,
        "dst_layout": dst_layout,
        "mapping": mapping,
        "matched": matched,
        "adapted_preprocessor": adapted_preprocessor,
        "adapted_optimizer": adapted_optimizer,
        "src_width": src_width,
        "dst_width": dst_width,
    }
    return checkpoint, report


def _print_report(report: dict[str, Any]) -> None:
    print(f"checkpoint structure: {report['structure']}")
    print(
        f"native widths: {report['src_width']} -> {report['dst_width']} "
        f"({sum(index is not None for index in report['mapping'])} shared, "
        f"{sum(index is None for index in report['mapping'])} zero-initialized)"
    )
    print("matched leading Linear weights:")
    for key in report["matched"]:
        print(f"  {key}")
    if report["adapted_preprocessor"]:
        print("adapted input preprocessor tensors:")
        for key in report["adapted_preprocessor"]:
            print(f"  {key}")
    if report["adapted_optimizer"]:
        print("adapted optimizer tensors:")
        for key in report["adapted_optimizer"]:
            print(f"  {key}")


def _print_mapping_table(report: dict[str, Any]) -> None:
    print("dst_native  superset  src_native")
    print("----------  --------  ----------")
    for dst_native, (slot, src_native) in enumerate(
        zip(report["dst_layout"], report["mapping"])
    ):
        source = "ZERO" if src_native is None else str(src_native)
        print(f"{dst_native:10d}  {slot:8d}  {source:>10}")


def embed_checkpoint(
    src_ckpt_path: str | Path,
    src_gym_id: str,
    dst_gym_id: str,
    out_path: str | Path,
) -> None:
    """Resize policy/value input layers by shared superset semantics and save."""

    import torch

    source = Path(src_ckpt_path).resolve()
    destination = Path(out_path).resolve()
    if source == destination:
        raise ValueError("out_path must differ from src_ckpt_path")
    checkpoint, report = _prepare_checkpoint(source, src_gym_id, dst_gym_id)
    _print_report(report)
    torch.save(checkpoint, destination)
    print(f"saved embedded checkpoint: {destination}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-ckpt", required=True, help="source skrl checkpoint")
    parser.add_argument("--src-id", required=True, help="source Gym id")
    parser.add_argument("--dst-id", required=True, help="destination Gym id")
    parser.add_argument("--out", help="output checkpoint (required unless --dry-run)")
    parser.add_argument(
        "--dry-run", action="store_true", help="inspect and print mapping without saving"
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        _, report = _prepare_checkpoint(args.src_ckpt, args.src_id, args.dst_id)
        _print_report(report)
        _print_mapping_table(report)
        print("dry-run: no checkpoint written")
        return
    if not args.out:
        parser.error("--out is required unless --dry-run is used")
    embed_checkpoint(args.src_ckpt, args.src_id, args.dst_id, args.out)


if __name__ == "__main__":
    main()
