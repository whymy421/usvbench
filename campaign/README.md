# Campaign tooling (journal campaign, release v2.6.6)

This folder holds the controller-side tools that ran the USVBench journal
campaign: the classical + MPPI evaluation plan `c266` (2,586 cells), the
cross-machine duplicates `xmR-266`, and the 72 PPO trainings with their
54-cell `final-01` test plans (6,465 planned cells in total, see
`campaign_config_v266_final_fix32.json`).

The files are copies of what ran on the controller PC. They are kept as they
were so the campaign can be audited; they are **not** a packaged tool. Paths
such as `D:/claude code/codex_tasks/...`, `E:/Storage/...` and
`C:/Users/BRADY/.ssh/...` are that PC's layout and must be edited before the
tools can run anywhere else.

| Path | Original location (controller PC) | Role |
|---|---|---|
| `orchestrator.py` | `20261014_g52_v266_start/work/orchestrator/g52_orchestrator_fix32.py` | Unattended orchestrator, **version fix32** (the version running on 2026-10-25). Only an 8-line header comment was added; its SHA-256 before the header is in the header. |
| `campaign_config_v266_final_fix32.json` | same folder | The fix32 config: release ids, plans, PPO runs, per-run release overrides (`PH-` -> v2.6.8, `DK-BB-` -> v2.6.9), gate release v2.6.7. |
| `tools_g54/` | `.../orchestrator/tools_g54/` | Keeper, health / ETA watchers, interim report, `dispatch_g54.py` (dispatch with `--rehost`), auto shutdown, the one-off `pull_lost.py`. |
| `linux/g51_train_slot.sh` | `.../orchestrator/linux_g59/` | PPO training launcher run on a rental (one slot, one run per launch). |
| `g51_tools/` | `20261014_g51_campaign/work/tools/` | `policy_g51.py` (dispatch policy, imported by the orchestrator), `campaign_cells.py`, `make_campaign_plan.py`, `validate_campaign_certs.py`, and the G51 builders/tests. |
| `launcher/` | `20261014_g51_campaign/work/r3copy_g51/launcher/` | PowerShell launcher the orchestrator calls: `launch_rental.ps1`, `pull_rental.ps1`, `status_rental.ps1`, `abort_rental.ps1`, `lib/r3_local_common.ps1`. |
| `executor/` | `20261014_g51_campaign/work/r3copy_g51/executor/` | `usvb_dose_executor.py` (SHA-256 `0519180095...`, = `executor_sha256` in the config) and `r3_remote.py`, installed on a rental by the launcher. |
| `g12_dispatch/` | `20261002_g12_dose_smoke/work_v26/` | `tools/dispatch.py` (the config's `dispatch_py`) and `dose_plan.py` (imported by `campaign_cells.py`). |
| `hosts.example.json` | new | Shape of the hosts file the orchestrator reads (`hosts_file` in the config), with placeholder values. |

## What is deliberately not here

- **Host lists and connection tables** (`hosts_g52*.json`, the rental
  connection table): they hold rental addresses and ports. Use
  `hosts.example.json` as the template.
- **Rental addresses and ports inside copied files**: the only occurrence,
  in `tools_g54/pull_lost.py` (a one-off recovery script), was replaced by
  `REDACTED_RENTAL_SSH_HOST` and port `0`. No other copied file contained
  one. ssh key *paths* are kept; no key material is in the repository.
- **Earlier orchestrator versions** (`g52_orchestrator.py`, `..._fix1` to
  `..._fix31`) and their configs: the change history is in the campaign notes
  (controller PC, `20261014_g52_v266_start/work/orchestrator/` and the NC
  journal notes). `tools_g54/pull_lost.py` still refers to the fix16 files by
  name; it is kept as the record of that one-off recovery.
- **Run state and data**: `run/`, `pulls/`, `runs/`, `dispatch/` ledgers,
  plan files (`campaign_c266.json`, 2.6 MB, and its batch folders),
  certificates, checkpoints, console logs and the release record JSON (it
  lists host addresses).
- **The rental kit** (`usvb_kit.py`, e1-kit-4) and the release check tools
  named in the release record (`import_check.py`, `pool_lookup_check.py`,
  `appearing_pool_check.py`, `v26_pool_check.py`): not part of this copy.

Comments in these files use internal codes (G12, G51, G52, G54, FIXnn, T1.33,
"owner 10-14", ...). They are working notes and will be cleaned in a later
version.
