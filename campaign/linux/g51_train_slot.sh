#!/bin/bash
# G51 launcher for the FORMAL PPO training runs (campaign purpose 'training') = G47 g47_train_slot.sh (v2.6.4 smoke launcher,
# sha256 f917160380cd9374...) with these changes only:
#  * runs are <TC>-<BB|WAMV>-<seed> (TC = RE PF SK DK PH SC RG FO FC IB MB HM; seed 42 / 1001 / 2003 = candidates 1-3 of the
#    registry candidate_order, selection_mode fixed_first_n: train 3, deliver 3), --purpose training, arm mixture (the owner's
#    T1.33 option 2: training mixes the light doses; fixed doses 2026-10-06), budget BB 96000 / WAM-V 192000 (unchanged);
#  * validation = selection-02 (seed 11002) ONLY, on the untrained agent and EVERY 2 % checkpoint (trainer default every
#    budget/50), calm, 64 x 1, eval_v6_frozen.py; best = ppo_checkpoint_eval.py select (campaign rule: every checkpoint
#    evaluated, tie -> later); the final checkpoint is kept and reported (owner 2026-10-13 / 10-14);
#  * the pool folder is the record's bundle (v2.6.5: g49pool-f545aa24144a, which holds the seed-11002 pools);
#  * after the best checkpoint: the competence-gate evaluation of the DELIVERED (best) checkpoint through run_cell.py
#    (purpose selection, block selection-gate-01 = seed 11001, 64 x 2, nominal) and seed_registry.py gate-record (ledger v2
#    record, built here; the controller PC appends it to the study ledger); a run_<id>_DONE.txt summary;
#  * every python this launcher starts carries the dose executor's slot tag of THIS slot (-X usvbslot=ds-<release>-k<K>s<slot>),
#    and the launcher holds <usvb>/locks/slot<slot>.lock (flock) for its whole life, so a G40/G51 dose executor running in a
#    sibling slot of the same K classifies these pythons as 'sibling' (never 'foreign') and no executor can start in this slot;
#  * evaluation concurrency on the same GPU (unchanged rule, more parallel copies): --eval-par-train (default 3) evaluations
#    beside the running training, --eval-par-after (default 4) after it; each needs >= MIN_FREE_MIB free GPU memory;
#    EVAL_TIMEOUT_S 5400 for WAM-V (64 x 1 WAM-V evaluations measured up to about 20 min alone, UNVERIFIED with 3-4 copies);
#  * the negative dry-runs: the campaign seed 42 is now EXPECTED to pass (frozen registry); a selection seed, a legacy seed,
#    an --agent-set, the removed LR switch and an empty assets variable must still be refused.
# Needs a release whose seed_registry.json and competence gate are FROZEN (v2.6.5 is a proposal: the trainer refuses purpose
# 'training' there - resolve_training_block). Never deletes anything; refuses an existing queue folder; kills only the
# processes it started (heartbeat-stale trainer). NOT DEPLOYED (G51).
# usage: g51_train_slot.sh --host-key rental02 --slot-k 2 --slot 0 --runs SC-WAMV-42,PF-BB-1001 --release-id <id> \
#          --record-sha256 <sha> --predictions <file> --predictions-sha256 <sha> [--attempt N] [--eval-par-train 3] [--eval-par-after 4]
#          [--test-budget N --eval-pcts 50,100]
set -u
HOST= K= SLOT= RUNS= RID= RECSHA= PRED= PREDSHA= DR=/root/autodl-tmp TEST_BUDGET= EVAL_PCTS=all ATTEMPT=1 REUSE= GREL_ID= GATEONLY= TAGRID=
PAR_TRAIN=3 PAR_AFTER=4 MIN_FREE_MIB=5000 EVAL_SEED=11002
while [ $# -gt 0 ]; do
  case "$1" in
    --host-key) HOST=$2;; --slot-k) K=$2;; --slot) SLOT=$2;; --runs) RUNS=$2;; --release-id) RID=$2;;
    --record-sha256) RECSHA=$2;; --predictions) PRED=$2;; --predictions-sha256) PREDSHA=$2;;
    --data-root) DR=$2;; --eval-par-train) PAR_TRAIN=$2;; --eval-par-after) PAR_AFTER=$2;;
    --test-budget) TEST_BUDGET=$2;; --eval-pcts) EVAL_PCTS=$2;; --attempt) ATTEMPT=$2;; --reuse-run-dir) REUSE=$2;; --gate-release) GREL_ID=$2;; --gate-only-from) GATEONLY=$2;; --slot-tag-release) TAGRID=$2;; *) echo "unknown argument $1"; exit 2;;
  esac; shift 2
done
for v in HOST K SLOT RUNS RID RECSHA PRED PREDSHA; do [ -n "${!v}" ] || { echo "missing --$v"; exit 2; }; done
case "$K" in 1|2|3|4) ;; *) echo "--slot-k 1..4"; exit 2;; esac
[ "$SLOT" -ge 0 ] 2>/dev/null && [ "$SLOT" -lt "$K" ] || { echo "--slot must be 0..K-1"; exit 2; }
PY=$DR/usvb/env/bin/python
REL=$DR/usvb/releases/$RID
REC=$DR/usvb/records/release_record_v2_$RID.json
DESC=$DR/usvb/releases/_incoming/$RID/usvbench-$RID.release.json
MAN=$DR/usvb/releases/_incoming/$RID/usvbench-$RID.manifest
KIT=$DR/usvb/releases/_incoming/$RID/usvb_kit.py
SLOTTAG=usvbslot=ds-${TAGRID:-$RID}-k${K}s$SLOT   # G56: a run on another release keeps the campaign slot tag (executors only accept their own release's tags)
LOCK=$DR/usvb/locks/slot$SLOT.lock
OUT=$DR/usvb/ppo/$RID/formal
[ -n "$TEST_BUDGET" ] && OUT=$DR/usvb/ppo/$RID/launcher_test_g51          # launcher test: never mixed with formal runs
STAMP=$(date -u +%Y%m%dT%H%M%SZ)_$(head -c3 /dev/urandom | od -An -tx1 | tr -d ' \n')
Q=$OUT/_train_${HOST}_k${K}s${SLOT}_$STAMP
NUM_ENVS=64; SAMPLE_S=60; STALE_S=${STALE_S_OVERRIDE:-7200}   # G52 fix3: 16-CPU rentals need >30 min to 1 % on HM/FO WAM-V   # G59 (owner 10-16 option 3): harbour WAM-V gets a looser heartbeat via STALE_S_OVERRIDE
BUDGET_BB=96000; BUDGET_WAMV=192000
[ -n "$TEST_BUDGET" ] && BUDGET_BB=$TEST_BUDGET && BUDGET_WAMV=$TEST_BUDGET
task_of() { case "$1" in RE) echo reach;; PF) echo path_following;; SK) echo station_keeping;; DK) echo docking;;
  PH) echo path_hazard;; SC) echo scatter;; RG) echo ring;; FO) echo fortress;; FC) echo forced_crossing;; IB) echo iceberg;;
  MB) echo moving_basin;; HM) echo harbour_mission;; *) return 1;; esac; }
# expected training wall seconds on a 4090 rental at K=2 (G51 timing model; BlueBoat from the G45/G46 rental runs scaled by
# task, WAM-V = 2 x budget; see work/timing/timing_model.md - UNVERIFIED for WAM-V and for most tasks)
exp_of() { case "$2" in blueboat) echo "${EXP_BB:-9000}";; wamv) echo "${EXP_WAMV:-21600}";; esac; }
spec() {   # <TC>-<BB|WAMV>-<seed> -> task vessel seed
  local tc=${1%%-*} rest=${1#*-}; local v=${rest%%-*} s=${rest#*-}
  local t; t=$(task_of "$tc") || return 1
  case "$v" in BB) v=blueboat;; WAMV) v=wamv;; *) return 1;; esac
  case "$s" in 42|1001|2003) ;; *) return 1;; esac
  echo "$t $v $s"
}
timeout_s() { echo $(( ( (2 * $1 + 600) + 59 ) / 60 * 60 )); }        # G33 rule: ceil_minute(2 x expected + 600 s)
EVAL_NS=(--seed-namespace selection --seed-block selection-02 --run-purpose selection)
[ -e "$Q" ] && { echo "queue folder exists: $Q"; exit 2; }
mkdir -p "$OUT/_claims" "$DR/usvb/locks" && mkdir "$Q" "$Q/certs" "$Q/neg" "$Q/gate" || exit 2
P=$Q/progress_train.txt
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') $*" | tr -cd '\11\12\15\40-\176' >> "$P"; }
abort() { log "ABORTED $*"; echo "$*" > "$Q/train_ABORTED.txt"; exit 1; }
exec 9>>"$LOCK" || abort "cannot open $LOCK"
flock -n 9 || abort "slot lock busy: $LOCK (a dose executor or another launcher holds slot $SLOT of this host)"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1
unset USVBENCH_HARBOUR_D7_POOL USVBENCH_ASSETS_V2 USVBENCH_RELEASE_DESCRIPTOR USVB_G46_LR USVB_G45_VARIANT
[ -z "${USVBENCH_ASSETS_V2+x}" ] || { echo "USVBENCH_ASSETS_V2 still set"; exit 2; }
BID=$("$PY" -X usvbexec=g51train -c "import json,sys;print(json.load(open(sys.argv[1]))['bundle']['bundle_id'])" "$REC") || abort "record unreadable"
POOL=$DR/usvb/pools/$BID
export USVBENCH_D7_POOL_DIR=$POOL
# G53: the competence gate runs on the gate release (v2.6.7 = v2.6.6 + seed-11001 pools/pins, same behaviour); training and
# checkpoint evaluation stay on $RID
GREL=$REL; GDESC=$DESC; GPOOL=$POOL; GREGTOOL=$REL/scripts/seed_registry.py
if [ -n "$GREL_ID" ]; then
  GREL=$DR/usvb/releases/$GREL_ID; GDESC=$DR/usvb/releases/_incoming/$GREL_ID/usvbench-$GREL_ID.release.json
  GBID=$("$PY" -X usvbexec=g51train -c "import json,sys;print(json.load(open(sys.argv[1]))['bundle']['bundle_id'])" "$DR/usvb/records/release_record_v2_$GREL_ID.json") || { echo "gate record unreadable"; exit 2; }
  GPOOL=$DR/usvb/pools/$GBID; GREGTOOL=$GREL/scripts/seed_registry.py
  for f in "$GREL/scripts/run_cell.py" "$GDESC" "$GPOOL"; do [ -e "$f" ] || { echo "gate release file missing: $f"; exit 2; }; done
fi
{
  echo "host_key=$HOST release_id=$RID attempt=$ATTEMPT slot_k=$K slot=$SLOT slot_tag=$SLOTTAG launcher=g51_train_slot.sh stamp=$STAMP"
  echo "record=$REC record_sha256=$RECSHA predictions=$PRED predictions_sha256=$PREDSHA eval_seed=$EVAL_SEED pool=$POOL"
  echo "eval_par_train=$PAR_TRAIN eval_par_after=$PAR_AFTER min_free_mib=$MIN_FREE_MIB"
  echo "cell|order|id|run_dir|claim|journal|vessel|arm|budget|num_envs|timeout_ms|expected_s|purpose|training_seed"
  order=0
  for r in ${RUNS//,/ }; do
    read -r task vessel seed <<< "$(spec "$r")" || { echo "bad run $r"; exit 2; }
    [ -n "${seed:-}" ] || { echo "bad run $r"; exit 2; }
    b=$BUDGET_BB; [ "$vessel" = wamv ] && b=$BUDGET_WAMV
    exp=$(exp_of "$task" "$vessel"); order=$((order + 1))
    echo "cell|$order|$r|$OUT/$r|$OUT/_claims/$r|journal:$task|$vessel|mixture|$b|$NUM_ENVS|$(( $(timeout_s "$exp") * 1000 ))|$exp|training|$seed"
  done
} > "$Q/manifest.txt" || abort "manifest"
grep -q "^bad run" "$Q/manifest.txt" && abort "bad run id in --runs $RUNS"
echo "utc,queue_elapsed_s,cell,source,timestep,percent,iteration,wall_s,heartbeat_stamp" > "$Q/timeline.csv"
echo "utc,queue_elapsed_s,used_mib,total_mib,free_mib,util_pct,active_cell,own_pids,own_gpu_dedicated_mib,own_smi_mib,siblings,python_owners" > "$Q/gpumem.csv"
echo "run,tag,step,attempt,start_utc,end_utc,wall_s,rc,cert_nonempty,phase,concurrent_at_start" > "$Q/evals.csv"
T0=$(date +%s)
log "QUEUE-START host=$HOST K=$K slot=$SLOT runs=$RUNS release=$RID pid=$$ slot_tag=$SLOTTAG lock=$LOCK"
[ "$(sha256sum "$REC" | cut -d' ' -f1)" = "$RECSHA" ] || abort "record sha256 mismatch $REC"
[ "$(sha256sum "$PRED" | cut -d' ' -f1)" = "$PREDSHA" ] || abort "predictions sha256 mismatch $PRED"
grep -q "__FILL__" "$PRED" && abort "predictions still hold a placeholder"
[ -d "$POOL" ] || abort "pool folder missing: $POOL"
TREE=$("$PY" -X usvbexec=g51train -c "import json,sys;print(json.load(open(sys.argv[1]))['code_tree_sha256'])" "$REC") || abort "record unreadable"
verify_tree() {
  local out; out=$("$PY" -X "$SLOTTAG" -B "$KIT" verify-tree --root "$REL" --manifest "$MAN" --expect-code-tree "$TREE" 2>&1)
  local n; n=$(sed -n 's/.*TREE-OK files=\([0-9]*\).*/\1/p' <<< "$out" | head -1)
  if [ -n "$n" ]; then log "VERIFY-TREE-OK $1 files=$n"; else log "RELEASE-ROOT-DIRTY $1 $(tail -1 <<< "$out")"; abort "verify-tree $1"; fi
}
verify_tree start
[ -f "$REL/assets_v2/blueboat_v2.usda" ] || abort "release assets missing: $REL/assets_v2/blueboat_v2.usda"
TRAINER=$REL/scripts/train_journal_ppo.py
RUNCELL=$REL/scripts/run_cell.py
REGTOOL=$REL/scripts/seed_registry.py
"$PY" -X "$SLOTTAG" -B "$REGTOOL" resolve-training --seed 42 --purpose training > "$Q/neg/registry_frozen_check.json" 2>&1 \
  || abort "the release refuses campaign training seed 42 (registry or competence gate not frozen): $Q/neg/registry_frozen_check.json"
log "REGISTRY-FROZEN-OK campaign seed 42 resolves for purpose training"
for neg in "NEG_SELECTION 11001 -" "NEG_LEGACY 7 -" "NEG_AGENTSET 1001 -" "NEG_REMOVEDSWITCH 1001 lr" "NEG_EMPTYASSETS 1001 assets"; do
  set -- $neg; extra=(); [ "$1" = NEG_AGENTSET ] && extra=(--agent-set agent.learning_rate=1e-3)
  genv=(); [ "$3" = lr ] && genv=(USVB_G46_LR=kl); [ "$3" = assets ] && genv=(USVBENCH_ASSETS_V2=)
  env "${genv[@]}" "$PY" -X "$SLOTTAG" -X usvbtrain=g51-neg-$1 -B "$TRAINER" --task journal:scatter --vessel blueboat --training-seed "$2" \
     --arm mixture --budget-timesteps 96000 --num-envs $NUM_ENVS --out "$Q/neg/$1" --release-record "$REC" --dry-run "${extra[@]}" > "$Q/neg/$1.log" 2>&1
  rc=$?
  if [ $rc -eq 2 ] && grep -q REFUSED "$Q/neg/$1.log" && [ ! -e "$Q/neg/$1" ]; then log "NEGDRY-OK $1 rc=2 refused"
  else log "NEGDRY-UNEXPECTED $1 rc=$rc out_exists=$([ -e "$Q/neg/$1" ] && echo yes || echo no) log=$Q/neg/$1.log"; fi
done
declare -A EVALUATED=() EV_TAG=() EV_STEP=() EV_CERT=() EV_ATT=()
own_pids() { local c; echo -n "$1 "; for c in $(pgrep -P "$1"); do own_pids "$c"; done; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }
sample() {   # $1 run id, $2 run dir, $3 trainer pid
  local now el g own smi sib pid mem last seen=
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ); el=$(( $(date +%s) - T0 ))
  g=$(nvidia-smi --query-gpu=memory.used,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
  own=$(own_pids "$3")
  smi=0; sib=0
  while IFS=, read -r pid mem; do
    pid=$(tr -d ' ' <<< "$pid"); mem=$(tr -d ' MiB' <<< "$mem"); [ -n "$pid" ] || continue
    grep -qw "$pid" <<< "$seen" && continue; seen="$seen $pid"
    if grep -qw "$pid" <<< "$own"; then smi=$((smi + mem)); else sib=$((sib + 1)); fi
  done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
  echo "$now,$el,$g,$1,$(tr ' ' ';' <<< "$own"),,$smi,$sib," >> "$Q/gpumem.csv"
  last=$(tail -1 "$2/progress.log" 2>/dev/null)
  if [ -n "$last" ]; then
    echo "$now,$el,$1,progress.log,$(awk '{print $4}' <<< "$last" | cut -d/ -f1),$(awk '{print $2}' <<< "$last" | tr -d %),$(awk '{print $6}' <<< "$last"),$(awk '{print $8}' <<< "$last"),$(stat -c %Y "$2/heartbeat.json" 2>/dev/null)" >> "$Q/timeline.csv"
  else echo "$now,$el,$1,no-progress-yet,,,,," >> "$Q/timeline.csv"; fi
}
eval_reap() {   # $1 run id
  local pid rc s e wall ok
  for pid in "${!EV_TAG[@]}"; do
    kill -0 "$pid" 2>/dev/null && continue
    wait "$pid"; rc=$?
    read -r s e _ < "${EV_CERT[$pid]%.json}.time" 2>/dev/null || { s=; e=; }
    wall=$( [ -n "$s" ] && [ -n "$e" ] && awk -v a="$s" -v b="$e" 'BEGIN{printf "%.1f", b - a}' )
    ok=no; [ -s "${EV_CERT[$pid]}" ] && ok=yes
    echo "$1,${EV_TAG[$pid]},${EV_STEP[$pid]},${EV_ATT[$pid]},$(date -u -d "@${s%.*}" +%FT%TZ 2>/dev/null),$(date -u -d "@${e%.*}" +%FT%TZ 2>/dev/null),$wall,$rc,$ok,$PHASE,${#EV_TAG[@]}" >> "$Q/evals.csv"
    if [ $rc -eq 0 ] && [ $ok = yes ]; then log "EVAL-OK $1 tag=${EV_TAG[$pid]} step=${EV_STEP[$pid]} wall_s=$wall cert=${EV_CERT[$pid]}"
    else
      log "EVAL-FAIL $1 tag=${EV_TAG[$pid]} step=${EV_STEP[$pid]} attempt=${EV_ATT[$pid]} rc=$rc wall_s=$wall"
      [ "${EV_ATT[$pid]}" = 1 ] && EVALUATED[$1_${EV_TAG[$pid]}]=retry
    fi
    unset "EV_TAG[$pid]" "EV_STEP[$pid]" "EV_CERT[$pid]" "EV_ATT[$pid]"
  done
}
eval_launch() {   # $1 run $2 tag $3 step $4 checkpoint $5 journal id $6 vessel $7 attempt
  local cert=$Q/certs/${1}_$2.json; [ "$7" = 2 ] && cert=$Q/certs/${1}_${2}_r2.json
  local to=$EVAL_TIMEOUT_S
  ( s=$(date +%s.%N)
    timeout -k 60 "$to" "$PY" -X "$SLOTTAG" -X usvbtrain=g51-eval-$1-$2 -B "$REL/scripts/eval_v6_frozen.py" --task "$5" \
      --num_envs 64 --episodes 64 --level 0 --eval-seed "$EVAL_SEED" "${EVAL_NS[@]}" --vessel "$6" --device cuda:0 --headless --out "$cert" \
      --release-descriptor "$DESC" --checkpoint "$4" $( [[ "$5" == *-Dock-* ]] && echo --set d7_pool_dir=$POOL ) > "${cert%.json}.log" 2>&1
    rc=$?; echo "$s $(date +%s.%N) $rc" > "${cert%.json}.time"; exit $rc ) &
  local pid=$!
  EV_TAG[$pid]=$2; EV_STEP[$pid]=$3; EV_CERT[$pid]=$cert; EV_ATT[$pid]=$7
  log "EVAL-START $1 tag=$2 step=$3 attempt=$7 pid=$pid running=${#EV_TAG[@]} free_mib=$FREE cert=$cert"
}
eval_poll() {   # $1 run id, $2 run dir, $3 journal id, $4 vessel, $5 budget, $6 max parallel; starts at most ONE evaluation
  eval_reap "$1"
  [ ${#EV_TAG[@]} -ge "$6" ] && return 0
  FREE=$(gpu_free); [ -n "$FREE" ] && [ "$FREE" -ge $MIN_FREE_MIB ] || return 0
  local c t tag att
  c=$2/g46_init/agent_0.pt
  if [ "${EVALUATED[$1_init]:-}" != done ] && [ "${EVALUATED[$1_init]:-}" != done1 ] && [ -f "$c" ] && [ $(( $(date +%s) - $(stat -c %Y "$c") )) -ge 20 ]; then
    att=1; [ "${EVALUATED[$1_init]:-}" = retry ] && att=2
    EVALUATED[$1_init]=done$([ $att = 1 ] && echo 1); [ $att = 2 ] && EVALUATED[$1_init]=done
    eval_launch "$1" init 0 "$c" "$3" "$4" $att; return 0
  fi
  for c in $(ls -tr "$2"/skrl/*/checkpoints/agent_*.pt 2>/dev/null); do
    t=$(basename "$c" .pt); t=${t#agent_}
    [[ "$t" =~ ^[0-9]+$ ]] || continue
    tag=p$(( 100 * t / $5 ))
    case "${EVALUATED[$1_$tag]:-}" in done|done1) continue;; esac
    [ $(( $(date +%s) - $(stat -c %Y "$c") )) -lt 20 ] && continue
    [ "$EVAL_PCTS" = all ] || grep -qw "${tag#p}" <<< "${EVAL_PCTS//,/ }" || { EVALUATED[$1_$tag]=done; continue; }
    att=1; [ "${EVALUATED[$1_$tag]:-}" = retry ] && att=2
    EVALUATED[$1_$tag]=done$([ $att = 1 ] && echo 1); [ $att = 2 ] && EVALUATED[$1_$tag]=done
    eval_launch "$1" "$tag" "$t" "$c" "$3" "$4" $att; return 0
  done
}
pending_retry() { local k; for k in "${!EVALUATED[@]}"; do [ "${EVALUATED[$k]}" = retry ] && [[ "$k" == "$1"_* ]] && return 0; done; return 1; }
pending_ckpt() {
  local c t tag
  [ -f "$2/g46_init/agent_0.pt" ] && case "${EVALUATED[$1_init]:-}" in done|done1) ;; *) return 0;; esac
  for c in $(ls "$2"/skrl/*/checkpoints/agent_*.pt 2>/dev/null); do
    t=$(basename "$c" .pt); t=${t#agent_}; [[ "$t" =~ ^[0-9]+$ ]] || continue
    tag=p$(( 100 * t / $3 )); case "${EVALUATED[$1_$tag]:-}" in done|done1) ;; *) return 0;; esac
  done; return 1
}
for r in ${RUNS//,/ }; do
  read -r task vessel SEED <<< "$(spec "$r")"
  b=$BUDGET_BB; [ "$vessel" = wamv ] && b=$BUDGET_WAMV
  EVAL_TIMEOUT_S=${EVAL_TO_BB:-3600}; [ "$vessel" = wamv ] && EVAL_TIMEOUT_S=${EVAL_TO_WAMV:-5400}   # G58: overridable (moving-basin WAM-V evals exceeded 90 min)
  every=$(( b / 50 ))
  claim=$OUT/_claims/$r; [ "$ATTEMPT" -gt 1 ] && claim=$OUT/_claims/${r}_a$ATTEMPT; [ -n "$REUSE" ] && claim=$OUT/_claims/${r}_reeval$ATTEMPT
  if ! mkdir "$claim" 2>/dev/null; then log "SKIPPED-CLAIMED $r claim=$claim"; continue; fi
  echo "$Q" > "$claim/queue.txt"
  if [ -n "$REUSE" ]; then   # G52 d7fix: re-evaluate an already trained run (evals failed: docking pool dir); no training
    dir=$REUSE; [ -f "$dir/training_record.json" ] || { log "NOT-STARTED-REUSE-NO-RECORD $r dir=$dir"; continue; }
    args=(--task "journal:$task" --vessel "$vessel" --training-seed "$SEED" --purpose training --arm mixture
          --budget-timesteps "$b" --num-envs $NUM_ENVS --out "$Q/reuse_dry_$r" --host-class "$HOST" --concurrency "$K" --release-record "$REC")
    "$PY" -X "$SLOTTAG" -X usvbtrain=g51-dry-$r -B "$TRAINER" "${args[@]}" --dry-run --preflight-out "$Q/dryrun_$r.json" > "$Q/dryrun_$r.log" 2>&1       || { log "NOT-STARTED-DRYRUN-FAIL $r log=$Q/dryrun_$r.log"; continue; }
    jid=$("$PY" -X usvbexec=g51train -c "import json,sys;print(json.load(open(sys.argv[1]))['journal_id'])" "$Q/dryrun_$r.json")
    log "REUSE-RUN $r dir=$dir jid=$jid (re-evaluation only)"
    if [ -n "$GATEONLY" ]; then
      [ -f "$GATEONLY/best_$r/best_eval.pt" ] || { log "NOT-STARTED-GATEONLY-NO-BEST $r from=$GATEONLY"; continue; }
      cp -r "$GATEONLY/best_$r" "$Q/best_$r" && cp "$GATEONLY/best_$r.log" "$Q/best_$r.log" 2>/dev/null
      log "GATE-ONLY $r best copied from $GATEONLY"
    fi
    state=ENDED; PHASE=train; TS=$(date +%s); rc=0
  else
  dir=$OUT/$r; n=2; while [ -e "$dir" ]; do dir=$OUT/${r}_a$n; n=$((n + 1)); done
  args=(--task "journal:$task" --vessel "$vessel" --training-seed "$SEED" --purpose training --arm mixture
        --budget-timesteps "$b" --num-envs $NUM_ENVS --out "$dir" --host-class "$HOST" --concurrency "$K" --release-record "$REC")
  "$PY" -X "$SLOTTAG" -X usvbtrain=g51-dry-$r -B "$TRAINER" "${args[@]}" --dry-run --preflight-out "$Q/dryrun_$r.json" > "$Q/dryrun_$r.log" 2>&1
  rc=$?; if [ $rc -ne 0 ]; then log "NOT-STARTED-DRYRUN-FAIL $r rc=$rc log=$Q/dryrun_$r.log"; continue; fi
  log "DRYRUN-OK $r preflight=$Q/dryrun_$r.json"
  jid=$("$PY" -X usvbexec=g51train -c "import json,sys;print(json.load(open(sys.argv[1]))['journal_id'])" "$Q/dryrun_$r.json")
  g46=$("$PY" -X usvbexec=g51train -c "import json,sys;p=json.load(open(sys.argv[1]));g=p['g46'];print(g['version'], g['lr_mode']['mode'], g['gamma'], g['top_speed_mps'], p['checkpoint_every'])" "$Q/dryrun_$r.json")
  read -r g_ver g_mode g_gamma g_speed g_every <<< "$g46"
  if [ "$g_mode" != kl ] || [ "$g_every" != "$every" ]; then log "NOT-STARTED-PLAN-MISMATCH $r plan=$g46 want=kl every=$every"; continue; fi
  log "FORMAL-PLAN $r version=$g_ver lr=$g_mode gamma=$g_gamma top_speed=$g_speed checkpoint_every=$g_every seed=$SEED eval_seed=$EVAL_SEED"
  exp=$(exp_of "$task" "$vessel"); T=$(timeout_s "$exp")
  ( cd "$Q" && exec timeout -k 60 "$T" "$PY" -X "$SLOTTAG" -X usvbtrain=g51-$r -B "$TRAINER" "${args[@]}" --headless ) > "$Q/$r.log" 2>&1 &
  pid=$!
  log "START-CELL $r TIMEOUT_$r=$((T * 1000)) expected_s=$exp cmd_pid=$pid utc=$(date -u +%FT%TZ) log=$Q/$r.log run_dir=$dir"
  state=ENDED; PHASE=train; TS=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    sample "$r" "$dir" "$pid"
    eval_poll "$r" "$dir" "$jid" "$vessel" "$b" "$PAR_TRAIN"
    if [ -f "$dir/heartbeat.json" ] && [ $(( $(date +%s) - $(stat -c %Y "$dir/heartbeat.json") )) -gt $STALE_S ]; then
      state=HEARTBEAT-STALE; log "HEARTBEAT-STALE-KILL $r pid=$pid (own trainer)"; pkill -TERM -P "$pid"; kill -TERM "$pid"; sleep 60
      pkill -KILL -P "$pid"; kill -KILL "$pid" 2>/dev/null; break
    fi
    sleep $SAMPLE_S
  done
  wait "$pid"; rc=$?
  [ $rc -eq 124 ] || [ $rc -eq 137 ] && [ "$state" = ENDED ] && state=TIMEOUT
  fi
  valid=invalid
  if [ -f "$dir/training_record.json" ] && "$PY" -X "$SLOTTAG" -B "$REL/scripts/ppo_training_record.py" "$dir/training_record.json" --run-dir "$dir" > "$Q/record_check_$r.txt" 2>&1; then valid=valid; fi
  log "$state $r rc=$rc record_valid=$valid run_dir=$dir wall_s=$(( $(date +%s) - T0 )) train_wall_s=$(( $(date +%s) - TS ))"
  PHASE=after
  while [ -z "$GATEONLY" ]; do
    eval_poll "$r" "$dir" "$jid" "$vessel" "$b" "$PAR_AFTER"
    [ ${#EV_TAG[@]} -eq 0 ] && ! pending_retry "$r" && ! pending_ckpt "$r" "$dir" "$b" && break
    sleep 20
  done
  log "EVALS-DONE $r certs=$(ls "$Q"/certs/${r}_p*.json 2>/dev/null | wc -l) after_wall_s=$(( $(date +%s) - TS ))"
  best=-; gate=-; grec=-
  [ -n "$GATEONLY" ] && best=$Q/best_$r/best_eval.pt
  if [ "$valid" = valid ] && [ -z "$GATEONLY" ]; then
    certs=(); for c in "$Q"/certs/${r}_p*.json; do
      [ -s "$c" ] || continue; p=$(basename "$c" .json); p=${p#${r}_p}
      case "$p" in *_r2) p=${p%_r2}; [ -s "$Q/certs/${r}_p$p.json" ] && continue;; esac
      certs+=(--cert "$(( p * b / 100 ))=$c"); done
    if "$PY" -X "$SLOTTAG" -B "$REL/scripts/ppo_checkpoint_eval.py" select --run-dir "$dir" "${certs[@]}" --out-dir "$Q/best_$r" > "$Q/best_$r.log" 2>&1
    then best=$Q/best_$r/best_eval.pt; log "BEST-OK $r $(tail -1 "$Q/best_$r.log")"; else log "BEST-FAIL $r log=$Q/best_$r.log"; fi
  fi
  if [ "$best" != - ] && [ -f "$best" ]; then
    gate=$Q/gate/${r}_gate.json
    USVBENCH_D7_POOL_DIR=$GPOOL "$PY" -X "$SLOTTAG" -X usvbtrain=g51-gate-$r -B "$GREL/scripts/run_cell.py" --task "$jid" --vessel "$vessel" --controller ppo --checkpoint "$best" \
      --training-seed "$SEED" --run-purpose selection --seed-namespace selection --eval-seed-block selection-gate-01 --level 0 \
      --episodes 128 --num_envs 64 --device cuda:0 --python "$PY" --release-descriptor "$GDESC" --out "$gate" $( [[ "$jid" == *-Dock-* ]] && echo --set d7_pool_dir=$GPOOL ) > "${gate%.json}.log" 2>&1
    grc=$?
    if [ $grc -eq 0 ] && [ -s "$gate" ]; then
      log "GATE-EVAL-OK $r cert=$gate gate_release=${GREL_ID:-$RID}"
      completed=$("$PY" -X usvbexec=g51train -c "import json,sys;r=json.load(open(sys.argv[1]));print(r.get('completed_timesteps', r.get('timesteps_completed', '')))" "$dir/training_record.json" 2>/dev/null)
      grec=$Q/gate/${r}_gate_record.json
      PYTHONPATH=$DR/usvb/gatelibs "$PY" -X "$SLOTTAG" -B "$GREGTOOL" gate-record --certificate "$gate" --task "$jid" --vessel "$vessel" --training-seed "$SEED" \
        --training-status completed --budget-timesteps "$b" --completed-timesteps "${completed:-$b}" --checkpoint "$best" \
        --training-record "$dir/training_record.json" > "$grec" 2> "${grec%.json}.err" \
        && log "GATE-RECORD-OK $r $grec" || { log "GATE-RECORD-FAIL $r ${grec%.json}.err"; mv "$grec" "${grec%.json}.failed_empty"; grec=-; }
    else log "GATE-EVAL-FAIL $r rc=$grc log=${gate%.json}.log"; gate=-; fi
  fi
  {
    echo "run=$r task=$task vessel=$vessel training_seed=$SEED state=$state rc=$rc record_valid=$valid run_dir=$dir"
    echo "training_record=$dir/training_record.json sha256=$(sha256sum "$dir/training_record.json" 2>/dev/null | cut -d' ' -f1)"
    echo "best_eval=$best sha256=$( [ -f "$best" ] && sha256sum "$best" | cut -d' ' -f1)"
    echo "best_eval_json=$Q/best_$r/best_eval.json"
    echo "gate_certificate=$gate sha256=$( [ -f "$gate" ] && sha256sum "$gate" | cut -d' ' -f1)"
    echo "gate_record=$grec"
    echo "evals_csv=$Q/evals.csv certs=$(ls "$Q"/certs/${r}_p*.json 2>/dev/null | wc -l)"
  } > "$Q/run_${r}_DONE.txt"
  log "RUN-DONE $r summary=$Q/run_${r}_DONE.txt"
done
verify_tree end
log "ALL-CELLS-ENDED queue=$Q"
date -u +%FT%TZ > "$Q/train_FINISHED.txt"
