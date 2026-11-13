"""Fair-value maker test.  For every real trade, compute the Binance-implied fair value using spot up to
(ts - L) seconds, L = assumed delay between match and on-chain timestamp (+ our own latency).
We are the maker on a trade only if the taker paid at least `theta` MORE than that fair value
(taker buys Up above fair+theta  -> we sold Up / hold Down;  taker sells Up below fair-theta -> we hold Up).
Fill at the actual trade price, size capped per market.  No maker fee; rebates ignored."""
import numpy as np, pandas as pd
from scipy.stats import norm
from lib import *

def build(asset):
    mk, tr, sp = load(asset)
    P = build_panel(mk, tr, sp)
    idx = {s: i for i, s in enumerate(mk.start.values)}
    tr = tr[(tr.t >= 12) & (tr.t < 240)].copy()          # before the averaging window (clean closed form)
    tr["i"] = tr.start.map(idx).values
    tr["up"] = P["up"][tr.i.values]
    tr["train"] = tr.start < mk.start.min() + 9 * 86400
    S = P["S"] * P["basis"][:, None]; K = mk.price_to_beat.values
    for L in (2, 4, 6, 9):
        te = tr.t.values - L                               # effective info time
        now = S[tr.i.values, 60 + te]
        sig = P["sigma"][tr.i.values] * now
        z = (now - K[tr.i.values]) / (sig * np.sqrt((240 - te) + 20))
        tr[f"fv{L}"] = norm.cdf(z)
    return tr

def run(tr, L, theta, cap=100.0, pmin=0.08, pmax=0.92, t0=12, t1=240):
    fv = tr[f"fv{L}"]
    sel_a = (tr.side == 1) & (tr.up_price >= fv + theta)      # taker overpays for Up -> we take Down at 1-p
    sel_b = (tr.side == -1) & (tr.up_price <= fv - theta)     # taker dumps Up cheap  -> we take Up at p
    m = (sel_a | sel_b) & (tr.up_price >= pmin) & (tr.up_price <= pmax) & (tr.t >= t0) & (tr.t < t1)
    f = tr[m].copy()
    f["long_up"] = f.side == -1
    f["px"] = np.where(f.long_up, f.up_price, 1 - f.up_price)
    f["want"] = f["size"] * f.px
    cum = f.groupby(["start", "long_up"]).want.cumsum()
    f["stake"] = np.clip(cap - (cum - f.want), 0, f.want)
    f = f[f.stake > 0.5]
    f["won"] = np.where(f.long_up, f.up, ~f.up)
    f["pnl"] = f.stake / f.px * f.won - f.stake
    return f

def stats(f):
    if len(f) < 5: return "n<5"
    pm = f.groupby("start").agg(pnl=("pnl", "sum"), stake=("stake", "sum"))
    return f"mkts={len(pm):4d} staked=${pm.stake.sum():8.0f} pnl=${pm.pnl.sum():7.0f} roi={pm.pnl.sum()/pm.stake.sum():+.4f} t={pm.pnl.mean()/(pm.pnl.std()/np.sqrt(len(pm))):+.1f}"

if __name__ == "__main__":
    for a in ("btc", "eth", "xrp"):
        tr = build(a); tr.to_parquet(f"data/fv_{a}.parquet")
        print("####", a)
        for L in (2, 4, 6, 9):
            for th in (0.0, 0.02, 0.04, 0.07, 0.10):
                f = run(tr, L, th)
                print(f"L={L} th={th}: TRAIN {stats(f[f.train])} || TEST {stats(f[~f.train])}")
