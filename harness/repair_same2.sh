#!/usr/bin/env bash
# Repair pass: the two same-socket configs failed (numactl mempolicy blocked in
# container). Waits out the in-flight BF16 ppl run, re-runs same2 with taskset
# (allowed) instead of numactl, then takes over the tar + grace + self-stop that
# the (killed) original runner would have done.
set -uo pipefail
BIN=/root/lbuild/bin; M=/workspace/models; R=/workspace/bench_results; B=/workspace/bench
PORT=8090; CTX=262144
XL=$M/Qwen3.6-27B-UD-Q6_K_XL.gguf
INSTANCE_ID="${CONTAINER_ID:-46531992}"
EVEN_CPUS=$(lscpu -p=CPU,NODE | awk -F, '/^[0-9]/ && $2==0 {printf "%s%s", sep, $1; sep=","}')
log(){ echo "[$(date -u +%F' '%T)] REPAIR: $*" | tee -a "$R/RUN.log"; }

log "waiting for in-flight llama-perplexity (bf16) to finish"
while pgrep -x llama-perplexity >/dev/null; do sleep 30; done
grep -E "Final estimate" "$R/ppl_bf16_q8kv.log" | tail -1 | \
  sed "s/^/[ppl_bf16_q8kv] /" | tee -a "$R/RUN.log"

run_same2(){ # spec tag meta
  local spec=$1 tag=$2 meta=$3 extra=()
  [ "$spec" = on ] && extra=(--spec-type draft-mtp --spec-draft-n-max 4)
  log "CONFIG $tag starting (taskset -c $EVEN_CPUS)"
  local t0=$SECONDS pid ok=0
  CUDA_VISIBLE_DEVICES=0,1 nohup taskset -c "$EVEN_CPUS" "$BIN/llama-server" \
    -m "$XL" --host 127.0.0.1 --port $PORT -ngl 999 -c $CTX \
    -fa on -ctk q8_0 -ctv q8_0 -sm tensor --parallel 1 --jinja \
    "${extra[@]}" > "$R/server_$tag.log" 2>&1 &
  pid=$!
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $pid 2>/dev/null || break
    sleep 5
  done
  if [ $ok = 1 ]; then
    log "CONFIG $tag loaded in $((SECONDS-t0))s"
    python3 "$B/bench_driver.py" --port $PORT --meta "$meta" \
      --depths 32768,131072,261120 >> "$R/RUN.log" 2>&1 || log "CONFIG $tag driver FAILED"
  else
    log "CONFIG $tag FAILED to become healthy; see server_$tag.log"
  fi
  kill $pid 2>/dev/null
  for _ in $(seq 1 24); do kill -0 $pid 2>/dev/null || break; sleep 5; done
  kill -9 $pid 2>/dev/null; sleep 3
  log "CONFIG $tag done ($((SECONDS-t0))s)"
}

run_same2 on  xl_same2_on  '{"weights":"XL","topo":"same2","mtp":"on","kv":"q8_0"}'
run_same2 off xl_same2_off '{"weights":"XL","topo":"same2","mtp":"off","kv":"q8_0"}'

pkill -f "nvidia-smi dmon" 2>/dev/null
cp /dev/shm/vast_build_VERSION.txt "$R/" 2>/dev/null
tar czf /workspace/bench_results.tgz -C /workspace bench_results
log "REPAIR_DONE tarball refreshed; stopping instance $INSTANCE_ID in 15 min"
sleep 900
/opt/instance-tools/bin/vastai stop instance "$INSTANCE_ID" \
  --api-key "$(cat /root/.vast_api_key)" >> "$R/RUN.log" 2>&1
log "stop issued"
