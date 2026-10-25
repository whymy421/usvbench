"""Write assets/MANIFEST.json for the v2 deliverable (CPU, python -B)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WORK = HERE.parent


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    derived = json.loads((WORK / "evidence/vessels_v2_derived.json").read_text())
    probe = json.loads((HERE / "probe_v2.json").read_text())
    assert probe["status"] == "PASS"
    artifacts = {}
    for name in ("wamv_vrx_v2_competition.usdc", "wamv_vrx_v2_bare.usdc", "blueboat_v2.usda", "LICENSE_VRX.txt"):
        artifacts[name] = {"sha256": sha(HERE / name), "bytes": (HERE / name).stat().st_size}
    w = derived["wamv"]
    manifest = {
        "deliverable": "D5 vessel assets v2 (1:1, FLU axes, actor origin = aggregate COM)",
        "vrx_source": "https://github.com/osrf/vrx/tree/" + derived["vrx_commit"],
        "vrx_commit": derived["vrx_commit"], "vrx_checkout_clean": derived["vrx_checkout_clean"],
        "license": "VRX files are Apache-2.0 (LICENSE_VRX.txt byte-identical to the pinned LICENSE). "
                   "Modification notice is stored in each WAM-V USD root customData 'modificationNotice'. "
                   "Cite Bingham et al. 2019 (VRX README.md:33-46).",
        "artifacts": artifacts,
        "assets": {
            "wamv_vrx_v2_competition.usdc": {"mass_kg": w["competition"]["mass_kg"],
                                             "com_in_vrx_base_link_m": w["competition"]["com_base_link_m"],
                                             "inertia_com_kg_m2": w["competition"]["inertia_com_kg_m2"],
                                             "links": w["competition"]["link_count"],
                                             "colliders": w["competition"]["collider_count"],
                                             "visuals": w["competition"]["visual_count"],
                                             "included": "base_link, 2 batteries, 2 engines, 2 propellers, CPU cases, 3 cameras "
                                                         "(post, arm, camera), GPS, IMU, 3-D lidar (post, arm), acoustic "
                                                         "receiver, ball shooter (base, launcher)",
                                             "excluded": "separate release platform/dummy links, projectiles, textures, "
                                                         "Gazebo plugins, joint articulation (frozen at 0)"},
            "wamv_vrx_v2_bare.usdc": {"mass_kg": w["bare"]["mass_kg"], "com_in_vrx_base_link_m": w["bare"]["com_base_link_m"],
                                      "inertia_com_kg_m2": w["bare"]["inertia_com_kg_m2"], "links": w["bare"]["link_count"],
                                      "colliders": w["bare"]["collider_count"], "visuals": w["bare"]["visual_count"],
                                      "included": "base_link, 2 batteries, 2 engines, 2 propellers",
                                      "excluded": "all optional sensors, CPU cases, shooter, platform, textures, plugins"},
            "blueboat_v2.usda": {"mass_kg": derived["blueboat"]["mass_kg"],
                                 "source": "C:/Users/BRADY/usvbench/assets/blueboat_physics.usd",
                                 "source_sha256": "9acfd39ed31cad557ac701d57b7d56f9a420b773ffe2af01b4081dcba32a2865",
                                 "transform": "rotate 180 deg about z (source bow -X) and translate COM to origin; no scaling",
                                 "colliders": derived["blueboat"]["collision_mesh_count"]}},
        "input_sha256": derived["input_sha256"],
        "scripts_sha256": {p: sha(WORK / p) for p in ("assets/build_vessels_v2.py", "assets/probe_vessels_v2.py",
                                                     "evidence/gen_vehicles_v2_data.py", "evidence/derive_blueboat.py",
                                                     "repo/vessel_physics_ref_v2.py")},
        "probe": {"status": probe["status"],
                  "c3_wamv_windage_regression_max_relative_difference":
                      probe["c3_wamv_windage_regression"]["max_relative_difference"]},
        "superseded_not_part_of_v2": {
            "wamv_vrx_v2.usdc": "earlier partial attempt: 258 kg, base_link origin (not COM), single configuration",
            "build_wamv_v2.py / probe_wamv_v2.py / wamv_windage_v2.py / refresh_wamv_manifest.py / wamv_probe.json":
                "earlier partial attempt tooling for that asset"},
        "reproduce": ["python -B assets/build_vessels_v2.py", "python -B evidence/gen_vehicles_v2_data.py",
                      "python -B assets/probe_vessels_v2.py", "python -B assets/write_manifest_v2.py"],
    }
    (HERE / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="ascii")
    print(json.dumps(artifacts, indent=1))


if __name__ == "__main__":
    main()
