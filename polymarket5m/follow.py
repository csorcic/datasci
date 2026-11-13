"""Momentum / 'follow the informed taker' test.
Trigger: a real taker trade at second tau prints through our 5s-old model price by >= theta
         (buys Up above phat+theta, or sells Up below phat-theta).
Action : we take the same side; fill = next real same-side taker print in [tau+lat, tau+lat+win), if its
         price is within `slip` of the trigger price.  Taker fee charged."""
import numpy as np, pandas as pd
from maker import prep
FEE = 0.07

def run(asset, theta=0.05, lat=1, win=5, slip=0.02, t0=10, t1=300, pmin=0.05, pmax=0.95, first_only=True):
    tr = prep(asset)
    out = []
    trig = tr[((tr.side == 1) & (tr.up_price > tr.phat + theta) | (tr.side == -1) & (tr.up_price < tr.phat - theta)) & (tr.t >= t0) & (tr.t < t1)]
    if first_only:
        trig = trig.drop_duplicates("start")
    # index trades per (start, side)
    grp = {k: (g.t.values, g.up_price.values) for k, g in tr.groupby(["start", "side"])}
    for r in trig.itertuples():
        tt, pp = grp[(r.start, r.side)]
        lo = np.searchsorted(tt, r.t + lat, "left"); hi = np.searchsorted(tt, r.t + lat + win, "left")
        if hi <= lo:
            continue
        p = pp[lo]
        if r.side == 1:
            if p > r.up_price + slip: continue
            px, won = p, r.up
        else:
            if p < r.up_price - slip: continue
            px, won = 1 - p, (not r.up)
        if not (pmin <= px <= pmax): continue
        out.append((r.start, r.t, r.side, px, won, won / px - 1 - FEE * (1 - px), r.train))
    return pd.DataFrame(out, columns=["start", "t", "side", "px", "won", "roi", "train"])

def st(f):
    if len(f) < 3: return "n<3"
    return f"n={len(f):5d} px={f.px.mean():.2f} win={f.won.mean():.3f} roi={f.roi.mean():+.4f} t={f.roi.mean()/(f.roi.std()/np.sqrt(len(f))):+.1f}"

if __name__ == "__main__":
    for a in ("btc", "eth", "xrp"):
        print("####", a)
        for (t0, t1) in ((10, 300), (10, 120), (120, 240), (240, 300)):
            for th in (0.03, 0.06, 0.10, 0.15):
                for lat in (1, 2):
                    f = run(a, theta=th, lat=lat, t0=t0, t1=t1)
                    print(f"t[{t0},{t1}) th={th} lat={lat}: TRAIN {st(f[f.train])} || TEST {st(f[~f.train])}")
