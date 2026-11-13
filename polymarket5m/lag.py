"""1) Estimate the delay between Binance moves and Polymarket trade timestamps (on-chain settlement lag).
   2) Look at pre-open / at-open flow: the strike is a 60s TWAP, so spot vs its own trailing TWAP at the open
      predicts the outcome before the window even starts."""
import numpy as np, pandas as pd
from lib import *
from scipy.stats import norm
for a in ("btc", "eth", "xrp"):
    mk, tr, sp = load(a)
    P = build_panel(mk, tr, sp)
    # --- lag estimate: 5s change in last-trade Up price vs 5s Binance log-return shifted by L seconds
    last = P["last"]; S = P["S"]           # S[:, 60+t]
    dP = last[:, 65:240:1] - last[:, 60:235:1]            # t from 60..234 -> change over (t, t+5]
    mid = (last[:, 60:235] > 0.25) & (last[:, 60:235] < 0.75)
    res = []
    for L in range(-2, 13):
        # spot return over (t-L, t+5-L]
        r = np.log(S[:, 60 + 65 - L:60 + 240 - L] / S[:, 60 + 60 - L:60 + 235 - L])
        res.append((L, np.corrcoef(dP[mid], r[mid])[0, 1]))
    print(a, "corr(dPrice_5s, spot ret shifted by L):", " ".join(f"{L}:{c:.3f}" for L, c in res))
    # --- open signal
    s0 = sp.index[0]; st = mk.start.values
    def spot(off):  # spot at start+off
        return sp.values[st + off - s0]
    sig = P["sigma"]
    for tau in (-30, -15, -5, 0, 5):
        n = len(st)
        seg = sp.values[(st - 60 - s0)[:, None] + np.arange(0, 60 + min(tau, 0))[None, :]]   # seconds [-60, tau)
        now = spot(tau)
        if tau < 0:
            k_est = (seg.sum(axis=1) + (-tau) * now) / 60
        else:
            k_est = P["k_spot"]
        z = (now - k_est) / (sig * now * np.sqrt(260 - tau))
        pm = norm.cdf(z)
        up = P["up"]
        q = pd.qcut(z, 5, labels=False)
        print(f"  tau={tau}: |z| median {np.median(np.abs(z)):.2f}; actual up-rate by model quintile:",
              " ".join(f"{pm[q==k].mean():.2f}->{up[q==k].mean():.2f}" for k in range(5)))
        # real taker prints in [tau+3, tau+13): average price paid for Up (ask side) and Down, by quintile
        w = tr[(tr.t >= tau + 3) & (tr.t < tau + 13)]
        idx = {s: i for i, s in enumerate(st)}
        qi = w.start.map(idx).map(lambda i: q[i]).values
        wu = w[w.side == 1]; wd = w[w.side == -1]
        gu = wu.groupby(qi[w.side.values == 1]).apply(lambda g: np.average(g.up_price, weights=g["size"]))
        gd = wd.groupby(qi[w.side.values == -1]).apply(lambda g: np.average(g.up_price, weights=g["size"]))
        print("      Up ask (vw) by quintile:", " ".join(f"{v:.3f}" for v in gu.values), "| Up bid (vw):", " ".join(f"{v:.3f}" for v in gd.values),
              "| $/mkt in window: %.0f" % (w.usd.sum() / n))
