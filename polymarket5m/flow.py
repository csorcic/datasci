"""Who loses money?  P&L of ALL real taker flow, bucketed by time and by the price of the token the
taker ends up long.  The maker on the other side earns the mirror image (and pays no fee)."""
import numpy as np, pandas as pd
from lib import *
pd.set_option("display.width", 250)
for a in ("btc", "eth", "xrp"):
    mk, tr, sp = load(a)
    tr = tr[(tr.t >= 0) & (tr.t < 300)].copy()
    up = tr.start.map(dict(zip(mk.start, mk.up_won))).values
    tr["tk_px"] = np.where(tr.side == 1, tr.up_price, 1 - tr.up_price)        # price of taker's long token
    tr["tk_won"] = np.where(tr.side == 1, up, ~up)
    tr["shares"] = tr["size"]
    tr["tk_cost"] = tr.shares * tr.tk_px
    tr["tk_pnl"] = tr.shares * tr.tk_won - tr.tk_cost                          # before fee
    tr["mk_cost"] = tr.shares * (1 - tr.tk_px)                                 # maker's stake on the opposite token
    tr["train"] = tr.start < mk.start.min() + 9 * 86400
    tr["tb"] = pd.cut(tr.t, [-1, 60, 120, 180, 240, 270, 300])
    tr["pb"] = pd.cut(tr.tk_px, [0, .02, .05, .10, .20, .35, .5, .65, .8, .9, .95, .98, 1.0])
    print("#####", a, " maker ROI on maker capital (= -taker pnl / maker stake); and taker-$ volume")
    for nm, m in (("TRAIN", tr.train), ("TEST", ~tr.train)):
        g = tr[m].groupby(["pb", "tb"], observed=True).agg(pnl=("tk_pnl", "sum"), mk=("mk_cost", "sum"), tk=("tk_cost", "sum"), n=("t", "size"))
        print(nm, "maker ROI"); print((-g.pnl / g.mk).unstack().round(4))
        if nm == "TRAIN":
            print("maker $ staked"); print(g.mk.unstack().round(0))
    tr[["start", "t", "tk_px", "tk_won", "shares", "tk_cost", "mk_cost", "tk_pnl", "train", "side"]].to_parquet(f"data/flow_{a}.parquet")
