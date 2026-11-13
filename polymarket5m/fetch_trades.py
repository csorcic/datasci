"""Download every trade for the markets in data/markets.parquet (one asset per run)."""
import sys, time, requests, pandas as pd

asset = sys.argv[1]
S = requests.Session()
mk = pd.read_parquet("data/markets.parquet")
mk = mk[mk.asset == asset].sort_values("start")
start_of = dict(zip(mk.condition, mk.start))

def get(params, tries=8):
    for i in range(tries):
        try:
            r = S.get("https://data-api.polymarket.com/v2/trades", params=params, timeout=40)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(5)
        except Exception:
            pass
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"failed {params}")

# optional sharding: python fetch_trades.py btc <shard> <nshards>
shard, nsh = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
conds = list(mk.condition)[shard::nsh]
suffix = f"_{shard}" if nsh > 1 else ""
out = []
B = 10
for i in range(0, len(conds), B):
    batch = conds[i:i + B]
    cursor = None
    while True:
        p = {"condition": ",".join(batch), "limit": 1000}
        if cursor:
            p["cursor"] = cursor
        j = get(p)
        for x in j["data"]:
            out.append((start_of[x["condition_id"]], x["timestamp"], x["outcome_index"],
                        1 if x["side"] == "BUY" else 0, x["price"], x["size"]))
        cursor = (j.get("pagination") or {}).get("next_cursor")
        if not cursor:
            break
    if (i // B) % 20 == 0:
        print(asset, i, len(conds), len(out), flush=True)
df = pd.DataFrame(out, columns=["start", "ts", "outcome", "buy", "price", "size"])
df = df.sort_values(["start", "ts"]).reset_index(drop=True)
df.to_parquet(f"data/trades_{asset}{suffix}.parquet")
print("done", asset, len(df))
