"""Download metadata + resolutions for Polymarket 5-minute crypto Up/Down markets."""
import sys, json, time, requests, pandas as pd

DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 14
ASSETS = ["btc", "eth", "xrp"]
S = requests.Session()

def get(url, params, tries=6):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            pass
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"failed {url} {params}")

def fetch_asset(asset, t_min):
    series = get("https://gamma-api.polymarket.com/series", {"slug": f"{asset}-up-or-down-5m"})[0]["id"]
    rows, off, cursor = [], 0, None
    while True:
        p = {"series_id": series, "closed": "true", "limit": 100, "order": "startTime", "ascending": "false"}
        if cursor:
            p["after_cursor"] = cursor
        j = get("https://gamma-api.polymarket.com/events/keyset", p)
        ev, cursor = j["events"], j.get("next_cursor")
        if not ev:
            break
        stop = False
        for e in ev:
            m = e["markets"][0]
            start = int(e["slug"].rsplit("-", 1)[1])
            if start < t_min:
                stop = True
                continue
            prices = json.loads(m.get("outcomePrices") or "[]")
            toks = json.loads(m.get("clobTokenIds") or "[]")
            md = e.get("eventMetadata") or {}
            cfg = m.get("cryptoMarketConfig") or {}
            fs = m.get("feeSchedule") or {}
            rows.append(dict(asset=asset, start=start, slug=e["slug"], condition=m["conditionId"],
                             up_token=toks[0] if toks else None, down_token=toks[1] if len(toks) > 1 else None,
                             up_won=(prices[0] == "1") if len(prices) == 2 and set(prices) == {"0", "1"} else None,
                             price_to_beat=md.get("priceToBeat"), final_price=md.get("finalPrice"),
                             twap=cfg.get("twapEnabled"), twap_lookback=cfg.get("twapLookbackSeconds"),
                             fee_rate=fs.get("rate"), open_interest=e.get("openInterest")))
        off += len(ev)
        print(asset, off, len(rows), flush=True)
        if stop or not cursor:
            break
    return rows

if __name__ == "__main__":
    t_min = int(time.time()) - DAYS * 86400
    allrows = []
    for a in ASSETS:
        allrows += fetch_asset(a, t_min)
    df = pd.DataFrame(allrows).drop_duplicates("slug").sort_values(["asset", "start"])
    df.to_parquet("data/markets.parquet")
    print(df.groupby("asset").agg(n=("slug", "size"), up=("up_won", "mean"), first=("start", "min"), last=("start", "max")))
    print(df[["twap", "twap_lookback", "fee_rate"]].value_counts(dropna=False))
