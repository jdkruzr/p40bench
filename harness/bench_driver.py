#!/usr/bin/env python3
"""Speed-matrix cell driver for the P40 box. Talks to a llama-server already
started by run_all.sh, measures decode-at-depth cells with shared-prefix
incremental prefill (depths ascending), appends JSONL rows.

Methodology matches ceres June runs: code-completion workload (passB.py's
CODE_CTX), sampling temp=1.0/top-p=0.95/top-k=20, n_predict=200, seed=42.
"""
import argparse, ast, json, os, subprocess, sys, time, urllib.request

def http(port, path, payload=None, timeout=7200):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def n_tokens(port, text):
    return len(http(port, "/tokenize", {"content": text})["tokens"])

def code_ctx(passb_path):
    """Extract CODE_CTX string literal from passB.py without importing it."""
    tree = ast.parse(open(passb_path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if getattr(t, "id", None) == "CODE_CTX":
                    return ast.literal_eval(node.value)
    raise RuntimeError("CODE_CTX not found in passB.py")

def calibrate(port, filler, depths, cache_path):
    """char offsets such that tokens(filler[:off]) is within 64 below target."""
    cache = {}
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path))
    out = {}
    for d in depths:
        k = str(d)
        if k in cache:
            out[d] = cache[k]; continue
        lo, hi = 0, len(filler)
        # initial guess by ratio to cut iterations
        while lo < hi:
            mid = (lo + hi + 1) // 2
            t = n_tokens(port, filler[:mid])
            if t <= d - 8:
                lo = mid
            else:
                hi = mid - 1
            if d - 64 <= t <= d - 8:
                lo = mid if t <= d - 8 else lo
                break
        out[d] = lo
        cache[k] = lo
        json.dump(cache, open(cache_path, "w"))
        print(f"[calib] depth {d}: {lo} chars -> {n_tokens(port, filler[:lo])} tokens", flush=True)
    return out

def vram_snapshot():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30).stdout
        return [int(x) for x in o.split()]
    except Exception:
        return []

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--meta", required=True, help="JSON: weights/topo/mtp/kv")
    p.add_argument("--depths", required=True, help="comma-separated token depths")
    p.add_argument("--filler", default="/workspace/needle_filler.txt")
    p.add_argument("--passb", default="/workspace/passB.py")
    p.add_argument("--out", default="/workspace/bench_results/matrix.jsonl")
    p.add_argument("--offsets-cache", default="/workspace/bench/offsets.json")
    args = p.parse_args()

    meta = json.loads(args.meta)
    depths = [int(x) for x in args.depths.split(",")]
    filler = open(args.filler, errors="replace").read()
    code = code_ctx(args.passb)
    code_toks = n_tokens(args.port, "\n\n" + code)
    offsets = calibrate(args.port, filler, [d - code_toks for d in depths],
                        args.offsets_cache)

    for d in depths:
        prompt = filler[:offsets[d - code_toks]] + "\n\n" + code
        t0 = time.time()
        resp = http(args.port, "/completion", {
            "prompt": prompt, "n_predict": 200, "cache_prompt": True,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "seed": 42,
        })
        wall = time.time() - t0
        row = dict(meta)
        row.update({
            "target_depth": d,
            "wall_s": round(wall, 1),
            "timings": resp.get("timings", {}),
            "tokens_predicted": resp.get("tokens_predicted"),
            "tokens_evaluated": resp.get("tokens_evaluated"),
            "content_head": resp.get("content", "")[:80],
            "vram_used_mib": vram_snapshot(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        tps = row["timings"].get("predicted_per_second")
        print(f"[cell] {meta.get('weights')}/{meta.get('topo')}/mtp-{meta.get('mtp')}"
              f"/kv-{meta.get('kv')} depth={d} decode={tps and round(tps,2)} t/s"
              f" wall={row['wall_s']}s", flush=True)

if __name__ == "__main__":
    main()
