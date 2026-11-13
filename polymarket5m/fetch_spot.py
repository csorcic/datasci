"""Download Binance 1-second klines (spot reference price) covering the market sample."""
import sys, time, requests, pandas as pd

S = requests.Session()
mk = pd.read_parquet("data/markets.parquet")
t0, t1 = int(mk.start.min()) - 3600, int(mk.start.max()) + 600
for asset in ["btc", "eth", "xrp"]:
    sym = asset.upper() + "USDT"
    rows, t = [], t0 * 1000
    while t < t1 * 1000:
        for i in range(8):
            try:
                r = S.get("https://data-api.binance.vision/api/v3/klines",
                          params={"symbol": sym, "interval": "1s", "startTime": t, "limit": 1000}, timeout=30)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(2 * (i + 1))
        k = r.json()
        if not k:
            break
        rows += [(x[0] // 1000, float(x[4]), float(x[5])) for x in k]
        t = k[-1][0] + 1000
        if len(rows) % 100000 < 1000:
            print(asset, len(rows), flush=True)
    df = pd.DataFrame(rows, columns=["ts", "close", "vol"]).drop_duplicates("ts")
    df.to_parquet(f"data/spot_{asset}.parquet")
    print("done", asset, len(df))
