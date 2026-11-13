# Polymarket 5-minute crypto Up/Down markets — research & backtests (BTC, ETH, XRP)

**Bottom line: none of the strategies tested produced a real, repeatable edge after fees and realistic
fills. No strategy here legitimately meets a "50%+ return" bar, and I have not tuned one until it did.**
Sample: 2026-10-30 → 2026-11-13 (14 days), 4,005 resolved markets per asset, 7.05M real trades.
Design period = first 9 days, hold-out test = last 5 days.

## How the markets work (as of Nov 2026)
* A new market every 5 minutes per asset (`{btc,eth,xrp}-updown-5m-<unix start>`).
* Resolves **Up** if the Chainlink **60-second TWAP** at the end of the window is >= the 60s TWAP at the
  start ("price to beat"; it equals the previous market's final price). Ties go to Up.
* Taker fee `shares * 0.07 * p * (1-p)` (1.75% of notional at 50c); makers pay nothing and share 20% of fees.
* Liquidity (median taker $ per market): BTC ~$21k, ETH ~$1.4k, XRP ~$210. Median trade ~$4.

## Data caveats that matter
* Trade timestamps are on-chain settlement times: Polymarket prices lag Binance by ~4s (range 0-9s) in this
  data (`lag.py`). Anything that depends on second-level timing cannot be validated here; several apparent
  edges disappeared once this was accounted for.
* Binance is only a proxy for the Chainlink feed (agrees with the actual resolution 97.5-98.5% of the time).

## What was tested (all with real traded prices; nothing is filled at a price nobody paid)
| # | Idea | Result |
|---|------|--------|
| 1 | Taker: buy when a Binance-based fair-value model (TWAP-aware) disagrees with the market | Loses 3-30% of stake, train and test, all assets (`scan.py`) |
| 2 | Taker: gradient-boosted model on market price + spot signal (beats market log-loss out of sample) | Edge does not survive fills + fees: -2% to -15% ROI (`fit.py`, `taker_scan*.py`) |
| 3 | Taker: buy late favourites (90-99.5c), with/without spot confirmation | Indistinguishable from zero (+-0.5%); a few buckets positive in one period, negative in the other (`scan2.py`) |
| 4 | Taker: buy longshots (<10c) | Loses 17-87% of stake |
| 5 | Taker: follow fast/informed flow (momentum) | Loses 5-25% (`follow.py`) |
| 6 | Maker: quote around the model, refreshed every 5s | Picked off: -3% to -20% (`maker.py`) |
| 7 | Maker: only fill takers who overpay vs spot fair value | Profit exists only if quotes react within ~0-2s of Binance (+2-5%, t~2); zero for BTC with 2s+ latency; weak, insignificant +1-2% for XRP (`fvmaker.py`) |
| 8 | Pre-open / at-open TWAP-strike momentum | Signal is real (33% vs 64% Up by quintile) but already priced to the cent (`lag.py`) |
| 9 | Post-close "known outcome" buying | Essentially no volume; nothing to capture |

Aggregate picture (`flow.py`, `flow2.py`): across all $85M of BTC taker flow, takers lost ~0.0% before fees and
paid $1.63M in fees (1.9%). Makers as a group earned ~0 before rebates (BTC -0.04%, ETH +0.36%, XRP +0.62% of
volume). The game is zero-sum before fees and the fee is the house edge; the only structural winners are
fast makers collecting rebates.

## Reproduce
```
pip install pandas pyarrow requests scipy scikit-learn
python fetch_markets.py 14
for a in xrp eth; do python fetch_trades.py $a; done
for i in 0 1 2 3 4 5; do python fetch_trades.py btc $i 6 & done; wait
python fetch_spot.py
python explore.py btc            # calibration / sanity
python signals.py btc eth xrp    # features + realistic fills
python fit.py && python taker_scan.py && python taker_scan2.py
python scan.py btc && python scan2.py
python flow.py && python flow2.py
python maker.py && python follow.py && python fvmaker.py && python lag.py
```
Needs network access to gamma-api/data-api.polymarket.com and data-api.binance.vision.
