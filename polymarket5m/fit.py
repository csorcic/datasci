"""Fit P(Up) = f(market price, Binance-implied z-score, momentum, time) on the first 9 days,
pooled across assets, and store out-of-fold (train) / out-of-sample (test) predictions."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss

def feats(df):
    lg = lambda p: np.log(np.clip(p, .002, .998) / (1 - np.clip(p, .002, .998)))
    X = pd.DataFrame(dict(l_last=lg(df["last"]), l_ask=lg(df.ask), l_bid=lg(df.bid), z=np.clip(df.z, -8, 8),
                          l_pm=lg(df.pm), t=df.t, m10=np.clip(df.mom10 / df.sigma / np.sqrt(10), -8, 8),
                          m30=np.clip(df.mom30 / df.sigma / np.sqrt(30), -8, 8), d30=df.dlast30,
                          zr=np.clip(df.ret / df.sigma / np.sqrt(300), -8, 8)))
    return X

dfs = []
for a in ("btc", "eth", "xrp"):
    d = pd.read_parquet(f"data/signals_{a}.parquet"); d["asset"] = a; dfs.append(d)
df = pd.concat(dfs, ignore_index=True)
split = df.start.min() + 9 * 86400
df["train"] = df.start < split
X = feats(df); y = df.up.astype(int).values
tr = df.train.values
df["phat"] = np.nan
day = ((df.start - df.start.min()) // 86400).values
mk = lambda: HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05, min_samples_leaf=400, l2_regularization=1.0)
# out-of-fold predictions on the training period (folds = whole days) so thresholds can be tuned honestly
for f, (i0, i1) in enumerate(GroupKFold(n_splits=9).split(X[tr], y[tr], day[tr])):
    m = mk().fit(X[tr].iloc[i0], y[tr][i0])
    df.loc[np.where(tr)[0][i1], "phat"] = m.predict_proba(X[tr].iloc[i1])[:, 1]
m = mk().fit(X[tr], y[tr])
df.loc[~tr, "phat"] = m.predict_proba(X[~tr])[:, 1]
for nm, msk in (("train(oof)", tr), ("test", ~tr)):
    for a in ("btc", "eth", "xrp"):
        s = msk & (df.asset == a).values
        mid = np.clip(df["last"][s], .005, .995)
        print(f"{nm:10s} {a}  logloss market(last) {log_loss(y[s], mid):.4f}  model(binance only) {log_loss(y[s], np.clip(df.pm[s], .005, .995)):.4f}  combined {log_loss(y[s], np.clip(df.phat[s], .001, .999)):.4f}")
df.to_parquet("data/signals_all.parquet")
