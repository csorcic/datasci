"""Sanity checks + descriptive research for one asset."""
import sys, numpy as np, pandas as pd
from lib import *
a = sys.argv[1]
mk, tr, sp = load(a)
P = build_panel(mk, tr, sp)
n = len(mk)
print(a, "markets", n, "trades", len(tr), "usd vol/market median", tr.groupby("start").usd.sum().median())
print("trades in-window per market (median):", tr[(tr.t >= 0) & (tr.t < 300)].groupby("start").size().median())
print("median trade usd", tr.usd.median(), "mean", tr.usd.mean())
# is `side` the taker side?  ask-side trades should print above bid-side trades in the same second
g = tr[(tr.t >= 0) & (tr.t < 300)].groupby(["start", "t", "side"]).up_price.mean().unstack()
d = (g[1] - g[-1]).dropna()
print("ask-side minus bid-side price, same second: mean %.4f median %.4f n=%d" % (d.mean(), d.median(), len(d)))
# Binance TWAP proxy vs real resolution
proxy_up = P["f_spot"] >= P["k_spot"]
print("Binance-TWAP proxy agrees with resolution: %.4f" % (proxy_up == P["up"]).mean())
ret = P["f_spot"] / P["k_spot"] - 1
big = np.abs(ret) > np.nanmedian(np.abs(ret))
print("  agreement when |move| > median: %.4f" % (proxy_up == P["up"])[big].mean())
cl_ret = mk.final_price.values / mk.price_to_beat.values - 1
print("  corr(binance ret, chainlink ret) = %.4f ; basis mean %.6f sd %.6f" % (np.corrcoef(ret, cl_ret)[0, 1], P["basis"].mean(), P["basis"].std()))
print("up rate %.4f ; sigma/sec median %.2e (5-min sd of ret ~ %.4f%%)" % (P["up"].mean(), np.median(P["sigma"]), 100 * np.median(P["sigma"]) * np.sqrt(300)))
# calibration of market price and of the model at several times
for t in (30, 60, 120, 180, 210, 240, 255, 270, 285):
    mid = P["last"][:, t]; pm, z = model_prob(P, t)
    rows = []
    for lo, hi in ((0, .05), (.05, .15), (.15, .3), (.3, .45), (.45, .55), (.55, .7), (.7, .85), (.85, .95), (.95, 1.01)):
        m = (mid >= lo) & (mid < hi)
        if m.sum() > 20:
            rows.append("%.2f-%.2f n=%4d mkt=%.3f mdl=%.3f act=%.3f" % (lo, hi, m.sum(), mid[m].mean(), pm[m].mean(), P["up"][m].mean()))
    brier_m = np.mean((mid - P["up"]) ** 2); brier_p = np.mean((pm - P["up"]) ** 2)
    print(f"t={t}: brier market {brier_m:.4f} model {brier_p:.4f}"); print("   " + "\n   ".join(rows))
