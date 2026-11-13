import numpy as np, pandas as pd
pd.set_option("display.width", 250)
def tstat(pm): return pm.mean() / (pm.std() / np.sqrt(len(pm)))
for a in ("btc", "eth", "xrp"):
    f = pd.read_parquet(f"data/flow_{a}.parquet")
    f["mk_pnl"] = -f.tk_pnl
    f["usd"] = f.tk_cost
    f["tb"] = pd.cut(f.t, [-1, 30, 60, 120, 180, 240, 270, 300]).astype(str)
    f["pb"] = pd.cut(f.tk_px, [0, .05, .2, .4, .6, .8, .95, 1.0]).astype(str)
    f["sz"] = pd.cut(f.usd, [0, 2, 10, 50, 250, 1e9]).astype(str)
    print("#####", a, "total taker $", round(f.usd.sum()), " maker pnl (pre rebate) $", round(f.mk_pnl.sum()), " = %.3f%% of taker volume; taker fees ~$%d" % (100 * f.mk_pnl.sum() / f.usd.sum(), (0.07 * f.shares * f.tk_px * (1 - f.tk_px)).sum()))
    for col in ("tb", "pb", "sz"):
        rows = []
        for k, g in f.groupby(col):
            r = [k]
            for m in (g.train, ~g.train):
                gg = g[m]; pm = gg.groupby("start").mk_pnl.sum()
                r += [round(gg.mk_cost.sum()), round(gg.mk_pnl.sum()), round(gg.mk_pnl.sum() / gg.mk_cost.sum(), 4), round(tstat(pm), 1)]
            rows.append(r)
        print(pd.DataFrame(rows, columns=[col, "tr_mk$", "tr_pnl", "tr_roi", "tr_t", "te_mk$", "te_pnl", "te_roi", "te_t"]).to_string(index=False))
