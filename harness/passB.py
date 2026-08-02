#!/usr/bin/env python3
"""Pass B: sustained real-code completion on ceres prod MTP server (P2P-on).
Captures decode t/s AND draft acceptance (draft_n / draft_n_accepted). Cross-engine
comparable to Vast (same engine, real code workload). Usage: passB.py"""
import urllib.request, json, os, statistics, sys

BASE = "http://localhost:8080"; MODEL = "qwen3.6-27b-mtp"
KEY = os.environ.get("LLAMA_API_KEY", "")
HEAD = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}

# Substantial, realistic repo-pack-style code context: a partial module to extend.
CODE_CTX = '''# file: cache.py  -- an in-memory LRU+TTL cache for a web service
import time
from collections import OrderedDict
from threading import RLock

class LRUTTLCache:
    """Thread-safe LRU cache with per-entry TTL and metrics.
    Evicts least-recently-used entries when capacity is exceeded, and lazily
    expires entries past their TTL on access."""
    def __init__(self, capacity=1024, default_ttl=300.0):
        self._cap = capacity
        self._ttl = default_ttl
        self._data = OrderedDict()   # key -> (value, expires_at)
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key, default=None):
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self._misses += 1
                return default
            value, expires_at = item
            if expires_at is not None and time.monotonic() > expires_at:
                del self._data[key]
                self._misses += 1
                return default
            self._data.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key, value, ttl=None):
        with self._lock:
            ttl = self._ttl if ttl is None else ttl
            expires_at = None if ttl is None else time.monotonic() + ttl
            self._data[key] = (value, expires_at)
            self._data.move_to_end(key)
            while len(self._data) > self._cap:
                self._data.popitem(last=False)
                self._evictions += 1

# TODO: implement the rest of the public API, fully commented:
#   - delete(key) -> bool
#   - clear()
#   - __len__, __contains__ (respecting TTL)
#   - stats() -> dict with hits, misses, hit_rate, evictions, size
#   - purge_expired() -> int  (sweep & drop all expired, return count)
#   - get_or_set(key, factory, ttl=None)  (compute-and-cache on miss)
# Implement all of the above now, matching the existing style and locking.
'''

def filler_code(depth_tokens):
    if depth_tokens <= 0:
        return ""
    # repeat code context (stays in-domain) to reach ~depth tokens (~3.8 chars/tok)
    chunk = CODE_CTX + "\n\n"
    need_chars = int(depth_tokens * 3.8)
    reps = need_chars // len(chunk) + 1
    return ("# ---- prior module ----\n" + chunk) * reps

def run(prefix, max_tokens, temp=0.2):
    body = {"model": MODEL, "prompt": prefix + CODE_CTX, "max_tokens": max_tokens,
            "temperature": temp, "cache_prompt": False, "stream": False}
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(), headers=HEAD)
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    t = d.get("timings", {})
    dn, da = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
    return {"decode_tps": round(t.get("predicted_per_second", 0), 2),
            "prompt_n": t.get("prompt_n", 0) + t.get("cache_n", 0),
            "gen_n": t.get("predicted_n", 0),
            "draft_n": dn, "draft_accepted": da,
            "accept_pct": round(100.0 * da / dn, 1) if dn else 0.0}

DEPTHS = [0, 65536]
REPS = 3
print(f"{'depth':>7} {'rep':>3} {'decode_tps':>10} {'accept%':>8} {'draft_n':>8} {'gen_n':>6}")
agg = {}
for depth in DEPTHS:
    pre = filler_code(depth)
    cells = []
    for rep in range(REPS):
        r = run(pre, 1536)
        cells.append(r)
        print(f"{depth:>7} {rep:>3} {r['decode_tps']:>10} {r['accept_pct']:>8} {r['draft_n']:>8} {r['gen_n']:>6}")
        sys.stdout.flush()
    agg[depth] = cells
print("\n=== Pass B summary (mean of reps) ===")
print(f"{'depth':>7} {'decode_tps':>10} {'accept%':>8}")
for depth, cells in agg.items():
    mt = statistics.mean(c["decode_tps"] for c in cells)
    ac = statistics.mean(c["accept_pct"] for c in cells)
    print(f"{depth:>7} {mt:>10.2f} {ac:>8.1f}")
    agg[depth] = {"decode_tps": round(mt, 2), "accept_pct": round(ac, 1)}
json.dump(agg, open("/home/sysop/bench/perf/passB_results.json", "w"), indent=2)
print("\nwrote perf/passB_results.json")
