"""G3 appearing-obstacle suite (owner name: "sudden obstacle"): a static cylinder that was not there
appears on the boat's route when the boat comes close (a sudden unmapped obstacle).

Suite key ``appearing`` with its own dose axis D0..D4 (+ supplementary D5 under ladder
options M and E, diagnostic runs only), stackable with every other journal suite including the D7 moving intruder
(``dynamic``). Phase 1 wired scatter (HazardNav carrier) and path_following; the
G20 rollout, completed by G20b, gives every journal task a placement rule
(config.PLACEMENT_TASKS, placement.SITE_RULES; inputs from families.py) and the
env side on all twelve tasks (config.WIRED_TASKS). Stacked dynamic + appearing
cells (and moving basin with dynamic D1..D4) carry no joint feasibility
certificate: each suite is admitted on its own (documented limitation).

The package is a NAMESPACE package (no __init__.py: the C2 test test_damping_v2 pins the set of
package initializers in tasks/). This module only documents it. Submodules (import them
explicitly; the pure-stdlib ``stamp`` module stays loadable by path from analysis tools):
  ladder     G20b: THE dose-ladder switch (ACTIVE_LADDER; M = per-vessel metres (default, owner decision
             2026-10-03), k ladders A, B, E for reference; stdlib only)
  config     suite configuration, dose ladder d_k (metres per vessel under M; k * t_avoid * u_ref under the
             k ladders), slot rule
  stamp      digest-sealed episode stamps {payload, sha256} (stdlib only)
  placement  offline placement on the nominal route + per-dose witness requests
  pool       admitted / not-admitted assignments (D16, D24), pool file codec, loader
  runtime    AppearingBridge: control-rate trigger, physical+visible at once, swept contact
  families   G20: journal task -> family folder and admission inputs (lazy imports)
  occlusion  G20d: visible-at-trigger ray test (pool-time nominal flag per dose, run-time stamp; owner decision 7A)

G20d (owner placement decisions of 2026-10-04): station keeping on the approach route (BlueBoat D1 not applicable by
design), docking mouth distance + decision 5B (main tiers without the D5 witness, monotone stitching), fortress
radius rule, 16 site candidates, a fallback witness beam, the visible_at_trigger stamp (suite version v6m).

The legacy timed appearance of HazardNav (appear_count, appear_time_min_s/max_s,
appear_distance_m) is untouched; journal cfgs keep appear_count = 0.
"""
