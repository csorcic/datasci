"""Passive (maker) backtest.

Every 5 seconds we re-quote around our model probability phat (information up to second t):
    bid for Up   at qb = floor_tick(phat - theta)        (we get long Up if a taker sells down to us)
    bid for Down at 1 - qa, qa = ceil_tick(phat + theta) (we get long Down if a taker buys Up from us)
Quotes are live for seconds [t+1, t+6).  Fill rule (conservative, price priority): our bid at q is filled only if a
REAL taker trade prints on that side at a price strictly through our quote (p < qb, or p > qa).  We are then
filled at OUR quote, for at most the size that actually traded, up to a per-market dollar cap.
Makers pay no fee on Polymarket (the 20% maker rebate is ignored = extra margin of safety).
"""
import numpy as np, pandas as pd
from lib import load

TICK = 0.01
_cache = {}

def prep(asset):
    if asset in _cache:
        return _cache[asset]
    mk, tr, sp = load(asset)
    sig = pd.read_parquet("data/signals_all.parquet", columns=["asset", "start", "t", "phat", "train", "up", "z", "last"])
    sig = sig[sig.asset == asset]
    tr = tr[(tr.t >= 11) & (tr.t < 300)].copy()
    tr["tg"] = 10 + 5 * ((tr.t - 11) // 5)
    tr = tr.merge(sig[["start", "t", "phat", "train", "up", "z", "last"]].rename(columns={"t": "tg"}), on=["start", "tg"], how="inner")
    tr = tr.sort_values(["start", "ts"]).reset_index(drop=True)
    _cache[asset] = tr
    return tr

def run(asset, theta=0.03, t0=10, t1=300, strict=True, cap=50.0, pmin=0.05, pmax=0.95, zmin=None, theta_fn=None):
    tr = prep(asset)
    th = theta if theta_fn is None else theta_fn(tr)
    qb = np.floor((tr.phat - th) / TICK + 1e-9) * TICK
    qa = np.ceil((tr.phat + th) / TICK - 1e-9) * TICK
    eps = 1e-9
    inwin = (tr.tg >= t0) & (tr.tg < t1)
    if strict:
        hit_b = (tr.side == -1) & (tr.up_price < qb - eps); hit_a = (tr.side == 1) & (tr.up_price > qa + eps)
    else:
        hit_b = (tr.side == -1) & (tr.up_price <= qb + eps); hit_a = (tr.side == 1) & (tr.up_price >= qa - eps)
    hit_b &= inwin & (qb >= pmin) & (qb <= pmax)
    hit_a &= inwin & (qa >= pmin) & (qa <= pmax)
    if zmin is not None:           # only quote the side the spot signal agrees with
        hit_b &= tr.z >= zmin; hit_a &= tr.z <= -zmin
    b = tr[hit_b].assign(long_up=True, px=qb[hit_b]); a = tr[hit_a].assign(long_up=False, px=1 - qa[hit_a])
    f = pd.concat([b, a]).sort_values(["start", "ts"])
    f["want"] = f["size"] * f.px                                  # dollars available at our quote from this taker
    # per-market, per-side dollar cap
    cum = f.groupby(["start", "long_up"]).want.cumsum()
    f["stake"] = np.clip(cap - (cum - f.want), 0, f.want)
    f = f[f.stake > 0.5]
    f["won"] = np.where(f.long_up, f.up, ~f.up)
    f["pnl"] = f.stake / f.px * f.won - f.stake
    return f[["start", "ts", "t", "tg", "long_up", "px", "stake", "won", "pnl", "train", "phat", "z"]]

def stats(f, label=""):
    if len(f) == 0:
        return f"{label} n=0"
    pm = f.groupby("start").agg(pnl=("pnl", "sum"), stake=("stake", "sum"))
    t = pm.pnl.mean() / (pm.pnl.std() / np.sqrt(len(pm))) if len(pm) > 2 else np.nan
    return f"{label} fills={len(f):6d} mkts={len(pm):4d} staked=${pm.stake.sum():9.0f} pnl=${pm.pnl.sum():8.0f} roi={pm.pnl.sum()/pm.stake.sum():+.4f} t={t:+.1f}"

if __name__ == "__main__":
    for a in ("btc", "eth", "xrp"):
        print("####", a)
        for (t0, t1) in ((10, 300), (10, 120), (120, 240), (240, 300)):
            for th in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08):
                f = run(a, theta=th, t0=t0, t1=t1)
                print(f"t[{t0},{t1}) th={th}: " + stats(f[f.train], "TRAIN") + " || " + stats(f[~f.train], "TEST"))
