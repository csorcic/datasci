"""Shared loaders / feature builders / backtest engine for Polymarket 5-minute crypto markets.

Conventions
-----------
* Every market is a 300-second window [start, start+300).  Outcome "Up" wins if the Chainlink
  60s-TWAP at the end >= the 60s-TWAP at the start ("price to beat").
* The Up and Down books are mirror images, so every trade is mapped to an *Up price*:
    taker BUY Up @p   or taker SELL Down @(1-p)  -> trade at the Up ASK = p   (side = +1)
    taker SELL Up @p  or taker BUY Down @(1-p)   -> trade at the Up BID = p   (side = -1)
* Taker fee (crypto_fees_v2): fee = shares * 0.07 * p * (1-p), paid by takers only.
"""
import glob
import numpy as np
import pandas as pd
from scipy.stats import norm

FEE_RATE = 0.07
W = 300  # seconds per market


def load(asset):
    mk = pd.read_parquet("data/markets.parquet")
    mk = mk[(mk.asset == asset) & mk.up_won.notna() & mk.price_to_beat.notna()].sort_values("start").reset_index(drop=True)
    mk["up_won"] = mk.up_won.astype(bool)
    tr = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"data/trades_{asset}*.parquet"))])
    tr = tr[tr.start.isin(mk.start)].copy()
    tr["t"] = tr.ts - tr.start                       # seconds since window open
    is_up = tr.outcome == 0
    tr["up_price"] = np.where(is_up, tr.price, 1 - tr.price)
    tr["side"] = np.where(is_up == (tr.buy == 1), 1, -1)   # +1 traded at Up ask, -1 at Up bid
    tr["usd"] = tr.price * tr["size"]
    tr = tr.sort_values(["start", "ts"]).reset_index(drop=True)
    sp = pd.read_parquet(f"data/spot_{asset}.parquet").set_index("ts").close
    full = pd.RangeIndex(sp.index.min(), sp.index.max() + 1)
    sp = sp.reindex(full).ffill()
    return mk, tr, sp


def build_panel(mk, tr, sp, vol_lookback=1800):
    """Per-market, per-second arrays (shape n_markets x 300), using only information available
    at the END of each second t (so a decision at second t may use column t, and must fill later)."""
    n = len(mk)
    idx = {s: i for i, s in enumerate(mk.start.values)}
    row = tr.start.map(idx).values
    ask = np.full((n, W), np.nan); bid = np.full((n, W), np.nan); last = np.full((n, W), np.nan)
    inwin = (tr.t.values >= 0) & (tr.t.values < W)
    t = tr.t.values
    for arr, mask in ((ask, tr.side.values == 1), (bid, tr.side.values == -1), (last, np.ones(len(tr), bool))):
        m = mask & inwin
        arr[row[m], t[m]] = tr.up_price.values[m]     # last write wins = last trade in that second (sorted by ts)
    # pre-open trades seed the opening quote
    pre = tr[tr.t < 0].groupby("start").up_price.last()
    seed = mk.start.map(pre).fillna(0.5).values
    def ffill(a, seed):
        a = a.copy(); a[:, 0] = np.where(np.isnan(a[:, 0]), seed, a[:, 0])
        return pd.DataFrame(a).ffill(axis=1).values
    P = dict(ask_raw=ask, bid_raw=bid, last=ffill(last, seed), ask=ffill(ask, seed), bid=ffill(bid, seed))
    # spot path: S[i, j] = Binance close at second start-60+j, j = 0..359  (j=60 is first second of window)
    s0 = sp.index[0]
    base = mk.start.values - 60 - s0
    S = sp.values[base[:, None] + np.arange(360)[None, :]]
    P["S"] = S
    P["k_spot"] = S[:, :60].mean(axis=1)                       # Binance analogue of the price to beat
    P["basis"] = mk.price_to_beat.values / P["k_spot"]          # Chainlink/Binance basis, known at open
    P["f_spot"] = S[:, 300:360].mean(axis=1)                    # Binance analogue of the final TWAP
    # trailing realised vol (per-second sigma of log returns from 10s bars), known at window open
    lr = np.log(sp.values[10::10] / sp.values[:-10:10])
    rv = pd.Series(lr ** 2).rolling(vol_lookback // 10).mean().values
    pos = np.clip((mk.start.values - s0) // 10 - 1, 0, len(rv) - 1)
    P["sigma"] = np.sqrt(rv[pos] / 10)
    P["up"] = mk.up_won.values
    P["start"] = mk.start.values
    return P


def model_prob(P, t, sigma_mult=1.0):
    """Model P(Up) at the end of second t (0..299) from the Binance path, the published price to beat
    and trailing vol, accounting for the 60s-TWAP settlement (prices in [240,300) are averaged)."""
    S = P["S"] * P["basis"][:, None]              # Binance path expressed in Chainlink terms
    K = P["k_spot"] * P["basis"]                  # == price_to_beat
    now = S[:, 60 + t]
    sig = P["sigma"] * sigma_mult * now           # $ per sqrt(second)
    if t < 240:
        mean = now
        var = sig ** 2 * ((240 - t) + 60 / 3)
    else:
        done = t - 240 + 1                        # seconds of the averaging window already observed
        rem = 60 - done
        locked = S[:, 300:300 + done].sum(axis=1)
        mean = (locked + rem * now) / 60
        var = sig ** 2 * (rem ** 3 / 3) / 3600 + 1e-12
    z = (mean - K) / np.sqrt(var)
    return norm.cdf(z), z


def fee_per_share(p):
    return FEE_RATE * p * (1 - p)


def next_fill(P, t, want_up, latency=2, window=10):
    """Fill model: after deciding at end of second t we take the price of the next REAL taker trade
    on our side of the book in seconds (t+latency .. t+latency+window].  Returns price paid for our
    token (Up price if want_up, else Down price) or NaN if nobody traded that side in the window."""
    lo, hi = t + latency, min(t + latency + window, W)
    if lo >= W:
        return np.full(len(P["up"]), np.nan)
    a = P["ask_raw"][:, lo:hi]; b = P["bid_raw"][:, lo:hi]
    src = np.where(want_up[:, None], a, 1 - b)
    first = np.argmax(~np.isnan(src), axis=1)
    px = src[np.arange(len(src)), first]
    return px


def settle(price, won, stake):
    """PnL of spending `stake` dollars (incl. nothing for fee; fee charged on top) at `price`."""
    shares = stake / price
    fee = shares * fee_per_share(price)
    return shares * won - stake - fee, fee


def summarize(tr_df, bankroll=1000.0):
    if len(tr_df) == 0:
        return dict(n=0)
    pnl = tr_df.pnl.values
    eq = bankroll + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.r_[bankroll, eq])[1:]
    return dict(n=len(tr_df), win=float((pnl > 0).mean()), avg_px=float(tr_df.px.mean()),
                staked=float(tr_df.stake.sum()), pnl=float(pnl.sum()),
                roi_turnover=float(pnl.sum() / tr_df.stake.sum()),
                ret_bankroll=float(pnl.sum() / bankroll),
                max_dd=float(((peak - eq) / peak).max()), min_eq=float(eq.min()))
