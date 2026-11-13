"""Build the per-(market, decision-second) research table with features and realistic fills.

Fill model ("copy the next real taker"): a decision made at the end of second t is filled by the
next real taker trades on the same side of the book during seconds [t+LAT, t+LAT+WIN), walking them
in time order until `stake` dollars are bought.  We therefore never assume a price that nobody
actually paid, and size is capped by what really traded (partial fills allowed).
"""
import sys, numpy as np, pandas as pd
from lib import *

LAT, WIN = 2, 10
STAKE = {"btc": 50.0, "eth": 20.0, "xrp": 10.0}
TIMES = list(range(10, 296, 5))


def build(asset):
    mk, tr, sp = load(asset)
    P = build_panel(mk, tr, sp)
    n = len(mk)
    stake = STAKE[asset]
    tw = tr[(tr.t >= 0) & (tr.t < W)]
    bounds = np.searchsorted(tw.start.values, mk.start.values, side="left"), np.searchsorted(tw.start.values, mk.start.values, side="right")
    tt, side, upx, usd = tw.t.values, tw.side.values, tw.up_price.values, tw.usd.values
    S = P["S"]
    rows = []
    for ti, t in enumerate(TIMES):
        pm, z = model_prob(P, t)
        f = dict(i=np.arange(n), t=t, start=P["start"], up=P["up"], last=P["last"][:, t], ask=P["ask"][:, t], bid=P["bid"][:, t],
                 pm=pm, z=z, sigma=P["sigma"],
                 mom30=S[:, 60 + t] / S[:, 60 + t - 30] - 1, mom10=S[:, 60 + t] / S[:, 60 + t - 10] - 1,
                 ret=S[:, 60 + t] / P["k_spot"] - 1,
                 dlast30=P["last"][:, t] - P["last"][:, max(t - 30, 0)])
        px = {1: np.full(n, np.nan), -1: np.full(n, np.nan)}; got = {1: np.zeros(n), -1: np.zeros(n)}
        for i in range(n):
            a, b = bounds[0][i], bounds[1][i]
            lo = a + np.searchsorted(tt[a:b], t + LAT, "left"); hi = a + np.searchsorted(tt[a:b], t + LAT + WIN, "left")
            if hi <= lo:
                continue
            for d in (1, -1):
                m = side[lo:hi] == d
                if not m.any():
                    continue
                c = upx[lo:hi][m] if d == 1 else 1 - upx[lo:hi][m]          # cost of OUR token
                # usd traded expressed in our token's notional
                sh = usd[lo:hi][m] / np.where(tw.price.values[lo:hi][m] > 0, tw.price.values[lo:hi][m], np.nan)  # shares
                u = sh * c
                cum = np.cumsum(u); k = np.searchsorted(cum, stake) + 1
                u2 = u[:k].copy()
                if cum[min(k, len(cum)) - 1] > stake:
                    u2[-1] -= cum[k - 1] - stake
                tot = u2.sum()
                if tot > 0:
                    px[d][i] = tot / (u2 / c[:k]).sum(); got[d][i] = tot
        f.update(px_up=px[1], usd_up=got[1], px_dn=px[-1], usd_dn=got[-1])
        rows.append(pd.DataFrame(f))
        print(asset, t, flush=True)
    df = pd.concat(rows, ignore_index=True)
    df["prev_up"] = np.r_[np.nan, P["up"][:-1].astype(float)][df.i.values]
    # per-dollar net return of each side incl. taker fee
    for d, won in (("up", df.up), ("dn", ~df.up)):
        p = df[f"px_{d}"]
        df[f"roi_{d}"] = won / p - 1 - FEE_RATE * (1 - p)
    df.to_parquet(f"data/signals_{asset}.parquet")
    return df


if __name__ == "__main__":
    for a in sys.argv[1:]:
        build(a)
