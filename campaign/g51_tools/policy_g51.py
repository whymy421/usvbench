"""G51 dispatch policy, shared by the simulation (sim_campaign.py) and the live orchestrator (g51_orchestrator.py).

When a slot of host H (class c) is free, choose(state, H, c) returns ONE job or None:
  1. PPO training first (critical path: training -> gate -> its 216 test cells): RENTAL slots only, at most ONE running
     training per host (the other slot keeps evaluating), at most ceil(runs / rentals online) runs per host in total (so the
     PPO test cells, which stay on the training host, are spread evenly); WAM-V runs before BlueBoat runs (longer), then
     plan order;
  2. (folded into 3) PPO test batches of a run become dispatchable when that run's training + gate finished; they may run on
     ANY rental (the orchestrator first copies the delivered checkpoint and the gate ledger rental-to-rental to the chosen
     host): binding them to the training host alone gives a 60-day tail (sim, 2026-10-14: one WAM-V run = up to 430
     slot-hours of test cells). PPO digests are equal across hosts of one type is UNVERIFIED -> REPRO-DUP PPO rows;
  3. classical / MPPI / released PPO batches: the table block (task x vessel x controller, all seed blocks) with the most remaining
     expected work first (critical path / longest chain), then that table's longest batch (LPT). A table block is locked to
     the machine CLASS of its first batch (one machine class per table block); the 5070 takes only classical and
     MPPI-BlueBoat table blocks (its WAM-V MPPI and PPO digests differ from the rentals', R2/G13), the 4080 any;
  R. REPRO-DUP batches (kind 'repro', small) right after step 1 when eligible: target class only (job['target_cls']) and never
     on the host that produced the original cells (job['avoid_hosts'], filled by the orchestrator from the main ledger once the
     originals are dispatched; a duplicate is held back until then); they take no class lock;
  4. nothing eligible -> None (the orchestrator reports 'slot idle with nothing to dispatch').
Hosts that come online later simply start receiving choose() calls; they take untaken training runs (step 1) and the
critical-path table blocks (step 3). Nothing is ever pre-empted or moved."""
import math


class State(object):
    def __init__(self, jobs, rentals_online=None):
        self.jobs = {j["id"]: j for j in jobs}
        self.taken = set()
        self.table_class = {}
        self.train_running = {}
        self.train_count = {}
        self.bound = {}                      # run -> host (released PPO test batches)
        self.rentals_online = rentals_online
        self.remaining_by_table = {}
        for j in jobs:
            if j["kind"] in ("eval", "ppo_eval"):
                self.remaining_by_table[j["table"]] = self.remaining_by_table.get(j["table"], 0.0) + j["h"]["RENTAL"]
        self.n_train = sum(1 for j in jobs if j["kind"] == "train")

    def remaining(self):
        return len(self.jobs) - len(self.taken)

    def release_run(self, run, host):
        self.bound[run] = host
        self.train_running[host] = max(0, self.train_running.get(host, 0) - 1)

    def take(self, job, host, cls):
        self.taken.add(job["id"])
        if job["kind"] in ("eval", "ppo_eval"):
            self.table_class.setdefault(job["table"], cls)
            self.remaining_by_table[job["table"]] -= job["h"]["RENTAL"]
        elif job["kind"] == "train":
            self.train_running[host] = self.train_running.get(host, 0) + 1
            self.train_count[host] = self.train_count.get(host, 0) + 1


def eligible_class(job, cls):
    if cls == "RENTAL" or cls == "4080":
        return True
    if cls == "5070":
        return job["kind"] == "eval" and (job["ctrl"] == "cl" or (job["ctrl"] == "mp" and job["vkey"] == "bb"))
    return False


def choose(state, host, cls, rentals_online=None):
    free = [j for jid, j in state.jobs.items() if jid not in state.taken]
    if cls == "RENTAL":
        n_online = rentals_online or state.rentals_online or 12
        cap = int(math.ceil(state.n_train / float(n_online)))
        if state.train_running.get(host, 0) < 1 and state.train_count.get(host, 0) < cap:
            trains = [j for j in free if j["kind"] == "train"]
            if trains:
                trains.sort(key=lambda j: (j["vkey"] != "wamv", j["id"]))
                job = trains[0]
                state.take(job, host, cls)
                return job
    reps = [j for j in free if j["kind"] == "repro" and j.get("target_cls") == cls and j.get("avoid_hosts") is not None
            and host not in j["avoid_hosts"]]
    if reps:
        job = max(reps, key=lambda j: (j["h"][cls], j["id"]))
        state.take(job, host, cls)
        return job
    evals = [j for j in free if (j["kind"] == "eval" or (j["kind"] == "ppo_eval" and j["run"] in state.bound))
             and eligible_class(j, cls) and state.table_class.get(j["table"], cls) == cls]
    if not evals:
        return None
    tables = {}
    for j in evals:
        tables.setdefault(j["table"], []).append(j)
    table = max(tables, key=lambda t: (state.remaining_by_table.get(t, 0.0), t))
    job = max(tables[table], key=lambda j: (j["h"][cls], j["id"]))
    state.take(job, host, cls)
    return job
