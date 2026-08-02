#!/usr/bin/env bash
# P40 box overnight bench: speed matrix -> vision smoke -> perplexity stages.
# Auto-stops (not destroys) the Vast instance 30 min after the results tarball
# is written. Everything logs under /workspace/bench_results/.
set -uo pipefail

BIN=/root/lbuild/bin
M=/workspace/models
R=/workspace/bench_results
B=/workspace/bench
PORT=8090
CTX=262144
DEPTH_ALL="32768,131072,261120"   # 256k cell leaves room for 200-token gen
DEPTH_TOP="261120"
XL=$M/Qwen3.6-27B-UD-Q6_K_XL.gguf
Q8=$M/Qwen3.6-27B-Q8_0.gguf
BF16=$M/Qwen3.6-27B-BF16-00001-of-00002.gguf
MMPROJ=$M/mmproj-F16.gguf
INSTANCE_ID="${CONTAINER_ID:-46531992}"

mkdir -p "$R" "$B"
log(){ echo "[$(date -u +%F' '%T)] $*" | tee -a "$R/RUN.log"; }

# background GPU telemetry for the whole run (30s cadence, like June's dmon logs)
nvidia-smi dmon -s pucm -d 30 -o T > "$R/dmon.log" 2>&1 &
DMON_PID=$!

SERVER_PID=""
start_server(){ # gpus numanode(0|-) model kv spec(on|off) mmproj(path|-) tag
  local gpus=$1 numa=$2 model=$3 kv=$4 spec=$5 mmproj=$6 tag=$7
  local pre=() extra=()
  [ "$numa" != "-" ] && pre=(numactl --cpunodebind="$numa" --membind="$numa")
  [ "$spec" = "on" ] && extra+=(--spec-type draft-mtp --spec-draft-n-max 4)
  [ "$mmproj" != "-" ] && extra+=(--mmproj "$mmproj")
  CUDA_VISIBLE_DEVICES=$gpus nohup "${pre[@]}" "$BIN/llama-server" \
    -m "$model" --host 127.0.0.1 --port $PORT -ngl 999 -c $CTX \
    -fa on -ctk "$kv" -ctv "$kv" -sm tensor --parallel 1 --jinja \
    "${extra[@]}" > "$R/server_$tag.log" 2>&1 &
  SERVER_PID=$!
}

wait_health(){ # returns 0 when /health is up, 1 on timeout/death
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 "$SERVER_PID" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

stop_server(){
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
  for _ in $(seq 1 24); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 5; done
  kill -9 "$SERVER_PID" 2>/dev/null; SERVER_PID=""; sleep 3
}

run_config(){ # gpus numa model kv spec depths meta tag
  local gpus=$1 numa=$2 model=$3 kv=$4 spec=$5 depths=$6 meta=$7 tag=$8
  log "CONFIG $tag starting (gpus=$gpus kv=$kv spec=$spec depths=$depths)"
  local t0=$SECONDS
  start_server "$gpus" "$numa" "$model" "$kv" "$spec" "-" "$tag"
  if ! wait_health; then
    log "CONFIG $tag FAILED to become healthy; see server_$tag.log"
    stop_server; return 1
  fi
  log "CONFIG $tag loaded in $((SECONDS-t0))s"
  python3 "$B/bench_driver.py" --port $PORT --meta "$meta" --depths "$depths" \
    >> "$R/RUN.log" 2>&1 || log "CONFIG $tag driver FAILED"
  stop_server
  log "CONFIG $tag done ($((SECONDS-t0))s total)"
}

log "===== P40 bench run starting; engine b9496 sm_61 ====="

########## Stage A: speed matrix (tensor split throughout) ##########
mj(){ printf '{"weights":"%s","topo":"%s","mtp":"%s","kv":"%s"}' "$1" "$2" "$3" "$4"; }

# XL, quad
run_config 0,1,2,3 - "$XL" q8_0 on  "$DEPTH_ALL" "$(mj XL quad on q8_0)"  xl_quad_on
run_config 0,1,2,3 - "$XL" q8_0 off "$DEPTH_ALL" "$(mj XL quad off q8_0)" xl_quad_off
# XL, 2-card same socket (GPU0,1 on NUMA0)
run_config 0,1 0 "$XL" q8_0 on  "$DEPTH_ALL" "$(mj XL same2 on q8_0)"  xl_same2_on
run_config 0,1 0 "$XL" q8_0 off "$DEPTH_ALL" "$(mj XL same2 off q8_0)" xl_same2_off
# XL, 2-card cross socket (GPU0,2) - 256k only
run_config 0,2 - "$XL" q8_0 on  "$DEPTH_TOP" "$(mj XL cross2 on q8_0)"  xl_cross2_on
run_config 0,2 - "$XL" q8_0 off "$DEPTH_TOP" "$(mj XL cross2 off q8_0)" xl_cross2_off
# Q8_0 weights, quad
run_config 0,1,2,3 - "$Q8" q8_0 on  "$DEPTH_ALL" "$(mj Q8_0 quad on q8_0)"  q8_quad_on
run_config 0,1,2,3 - "$Q8" q8_0 off "$DEPTH_ALL" "$(mj Q8_0 quad off q8_0)" q8_quad_off
# BF16 weights, quad
run_config 0,1,2,3 - "$BF16" q8_0 on  "$DEPTH_ALL" "$(mj BF16 quad on q8_0)"  bf16_quad_on
run_config 0,1,2,3 - "$BF16" q8_0 off "$DEPTH_ALL" "$(mj BF16 quad off q8_0)" bf16_quad_off
# Stage 3: KV f16 on Pascal, XL quad, 256k only
run_config 0,1,2,3 - "$XL" f16 on  "$DEPTH_TOP" "$(mj XL quad on f16)"  xl_quad_on_f16kv
run_config 0,1,2,3 - "$XL" f16 off "$DEPTH_TOP" "$(mj XL quad off f16)" xl_quad_off_f16kv

########## Stage B: vision smoke (quad, ~128k ctx, MTP on) ##########
log "VISION smoke starting"
t0=$SECONDS
CTX=131072
start_server 0,1,2,3 - "$XL" q8_0 on "$MMPROJ" vision
if wait_health; then
  python3 "$B/vision_smoke.py" --port $PORT --image /workspace/vision_test.png \
    --out "$R/vision.jsonl" >> "$R/RUN.log" 2>&1 || log "VISION driver FAILED"
else
  log "VISION server FAILED to load"
fi
stop_server
log "VISION done ($((SECONDS-t0))s)"

########## Stage C: perplexity at 128K (stages 5+6; ~130GB host buffer) ##########
ppl(){ # model kv tag
  local model=$1 kv=$2 tag=$3
  log "PPL $tag starting"
  local t0=$SECONDS
  "$BIN/llama-perplexity" -m "$model" -f /workspace/wiki.test.raw \
    -c 131072 --chunks 2 -ngl 999 -sm tensor -fa on -ctk "$kv" -ctv "$kv" \
    > "$R/ppl_$tag.log" 2>&1 || log "PPL $tag FAILED (rc=$?)"
  grep -E "Final estimate|PPL" "$R/ppl_$tag.log" | tail -2 | \
    sed "s/^/[ppl_$tag] /" | tee -a "$R/RUN.log"
  log "PPL $tag done ($((SECONDS-t0))s)"
}
ppl "$XL"   q8_0 xl_q8kv
ppl "$XL"   f16  xl_f16kv
ppl "$Q8"   q8_0 q8w_q8kv
ppl "$BF16" q8_0 bf16_q8kv

########## Wrap up: tar, grace window, self-stop ##########
kill $DMON_PID 2>/dev/null
cp /dev/shm/vast_build_VERSION.txt "$R/" 2>/dev/null
tar czf /workspace/bench_results.tgz -C /workspace bench_results
log "ALL_STAGES_DONE tarball=/workspace/bench_results.tgz"
log "Grace window: stopping instance $INSTANCE_ID in 30 min (results persist on disk)"
sleep 1800
/opt/instance-tools/bin/vastai stop instance "$INSTANCE_ID" \
  --api-key "$(cat /root/.vast_api_key)" >> "$R/RUN.log" 2>&1
log "stop issued"
