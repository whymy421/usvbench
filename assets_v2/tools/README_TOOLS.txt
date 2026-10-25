assets_v2/tools - provenance scripts for the v2 vessel assets (NOT run by any test)

These three scripts are byte-identical copies of the D5 deliverables
(D:/claude code/codex_tasks/20260930_d5_vessels_v2/work/assets/); their
SHA-256 values for build_vessels_v2.py and probe_vessels_v2.py are the ones
recorded in ../MANIFEST.json "scripts_sha256" (write_manifest_v2.py wrote
that manifest and is not listed in it).

REQUIREMENTS - they cannot run inside this repository as they are:
  * the pinned VRX checkout osrf/vrx@7609d1bd90ce7edb29d040a082f949e8b089c864
    (clean working tree; build_vessels_v2.py:50 expects it at
    D:/claude code/scratch/vrx and asserts it is clean, :689-694);
  * OpenUSD Python bindings `pxr` (build/probe), plus numpy, shapely, scipy
    and trimesh (BlueBoat CAD hydrostatics);
  * D5's work-folder layout: ../repo/vessel_physics_ref_v2.py,
    ../evidence/derive_blueboat.py, ../evidence/vessels_v2_derived.json,
    the legacy source asset C:/Users/BRADY/usvbench/assets/blueboat_physics.usd,
    the BlueBoat hull STL and C3's WKT silhouettes (paths are hard-coded at
    build_vessels_v2.py:45-63 and :555-557).

Reproduce order (from D5's work folder, per MANIFEST.json "reproduce"):
  python -B assets/build_vessels_v2.py
  python -B evidence/gen_vehicles_v2_data.py
  python -B assets/probe_vessels_v2.py
  python -B assets/write_manifest_v2.py

The generated data module is tasks/_shared/vehicles_v2_data.py; its asset
SHA-256 values are checked against the files in this folder by
tasks/_shared/tests_vessels_v2/test_vehicles_v2.py (test_asset_hashes) and by
tasks/_shared/journal_vehicle.py verify_asset().
