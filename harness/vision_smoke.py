#!/usr/bin/env python3
"""Vision smoke test: one chart PNG through mmproj on the P40s, record timings."""
import argparse, base64, json, time, urllib.request

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    b64 = base64.b64encode(open(args.image, "rb").read()).decode()
    payload = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text",
             "text": "Describe this chart: what is plotted, the axis labels, "
                     "and the approximate values of the key data points."},
        ]}],
        "max_tokens": 300, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
        "seed": 42, "timings_per_token": False,
    }
    t0 = time.time()
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    wall = time.time() - t0
    row = {
        "test": "vision_smoke", "wall_s": round(wall, 1),
        "usage": resp.get("usage", {}),
        "timings": resp.get("timings", {}),
        "content": resp["choices"][0]["message"]["content"],
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[vision] wall={wall:.1f}s usage={row['usage']}", flush=True)

if __name__ == "__main__":
    main()
