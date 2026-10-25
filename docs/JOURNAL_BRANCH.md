# Journal branch `journal/v2.6.6`

This branch holds the exact code used by the USVBench journal campaign, so
that every campaign certificate can be traced to a commit. It starts from
`main` and replaces the code tree with the frozen releases; it is not meant
to be merged back into `main` as is.

## What is on the branch

| Path | Content | Part of the hash-pinned release? |
|---|---|---|
| `assets_v2/`, `scripts/`, `tasks/` | The release tree (code, configs, D7 pool pins and the station-keeping pool files), byte-identical to the release manifests | yes |
| `releases/<version>/` | For each release: the code manifest, the assets manifest and the release descriptor (`release.json`); `releases/verify_release_tree.py` checks a tree against them | the manifests define the release |
| `campaign/` | Controller-side campaign tooling (orchestrator fix32, config, launcher, executor, dispatch, watchers); see `campaign/README.md` | no |
| `real_data/` | Scripts that place the dose ladders in one year of UK south-coast data and pick the R4 real-condition periods; see `real_data/README.md` | no |
| `assets/`, `docs/`, `results/`, `README.md`, `TASKS.md` | Kept from `main` unchanged (`assets/` still holds the legacy USD files that the calm-water families load) | no |

The release only ships `.py .json .yaml .yml .toml .cfg` files under
`scripts/` and `tasks/`. The Markdown notes, PowerShell helpers and reference
checkpoints that `main` keeps under those folders are therefore not on this
branch.

## Tags and the four versions

| Tag | Release id | Date | Change against the previous version | Used for |
|---|---|---|---|---|
| `v2.6.6` | `2c04042a7a99-64e5932b737f` | 2026-10-14 | v2.6.5 + frozen seed registry (`scripts/seed_registry.json` and the tests that pin it) | **The campaign release.** All evaluation certificates: classical and MPPI plan `c266`, cross-machine duplicates `xmR-266`, and every PPO `final-01` test plan (also for the `PH-` and `DK-BB-` runs). PPO training for all runs except `PH-*` and `DK-BB-*`. |
| `v2.6.7` | `100c9ed2a8ed-dd62692601f1` | 2026-10-15 | D7 pools/pins of the competence-gate seed 11001 (manifests, station-keeping approach pools, tests); no behaviour change | PPO competence gate (`gate_release`) for all runs except `PH-*` and `DK-BB-*`. |
| `v2.6.8` | `304f1d3c2c5a-e2aa14a19723` | 2026-10-19 | `tasks/path_hazard/training_pool.py`: `geometry_hashes` becomes a property, fixing a RecursionError that stopped every path_hazard PPO training (evaluation does not use this class) | Training and competence gate of the PPO runs `PH-*`. |
| `v2.6.9` | `230c7a5419d1-f495de421958` | 2026-10-20 | `tasks/_shared/conditions/training_mixture.py`: BlueBoat docking appearing trains at D2 (D1 is not applicable by design); its test follows. Training-only code path | Training and competence gate of the PPO runs `DK-BB-*`. |

The "Used for" column is read from `campaign/campaign_config_v266_final_fix32.json`
(`release_id`, `gate_release`, `run_release_overrides`) and from
`campaign/orchestrator.py`: the overrides change the release of a run's
training and gate (`dispatch_train`), while every PPO test plan is generated
with the campaign `release_id`, v2.6.6 (`make_campaign_plan.py --release-id`
in the finished-training step). Each certificate also records the release id
it was produced with, and the campaign validator checks it, so the release id
inside a certificate maps directly to a tag in the table above. (This mapping
was read from the config and code; it has not been re-checked against every
certificate file.)

Release ids are `<git commit prefix>-<code tree prefix>`. The git commit
half refers to the build repository used on the controller PC, not to a
commit on this branch; the code tree half is the SHA-256 of the code
manifest and is what identifies the content. Check a tag with:

```
python releases/verify_release_tree.py --release releases/v2.6.6 --rev v2.6.6
```

It prints `VERIFY-OK` when the 607 code files and 9 asset files (621 from
v2.6.7 on) committed at the tag match the manifest byte for byte.
`.gitattributes` turns off line-ending conversion for these folders so a
Windows clone with `core.autocrlf=true` also verifies.

## Not in git

- **Pool bundles** (the `.d7pool.xz` files the pins point to):
  `g49pool-f545aa24144a` for v2.6.6 (504 files, 1.50 GB) and
  `g53pool-07650cdebac4` for v2.6.7 to v2.6.9 (544 files, 1.69 GB).
- **Release artefacts**: `usvbench-<release id>.tar.gz` (100-113 MB each) and
  the release records (they list machine addresses).
- **Campaign outputs**: certificates, pulled queue folders, ledgers,
  checkpoints and logs.

If these are to be published, attach them to a GitHub release (or another
archive with a DOI) rather than committing them; Git LFS would also work, but
its bandwidth quota is small for multi-GB bundles.

## Internal notes in comments

Comments, docstrings and file names in this tree use internal working codes:
task codes such as G23e, G51 or G58, fix codes such as FIX22, journal-note
references such as "NC 09", reviewer rounds, and owner dates. They are
internal notes kept so the release stays byte-identical to what ran; they
will be cleaned in a later version and should not be read as public
documentation.
