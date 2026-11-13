import sys, numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
a = sys.argv[1]
df = pd.read_parquet(f"data/signals_{a}.parquet")
split = df.start.min() + 9 * 86400
df["train"] = df.start < split
# favourite side view
df["fav_up"] = df["last"] >= 0.5
df["fav_px"] = np.where(df.fav_up, df.px_up, df.px_dn)
df["fav_roi"] = np.where(df.fav_up, df.roi_up, df.roi_dn)
df["dog_px"] = np.where(df.fav_up, df.px_dn, df.px_up)
df["dog_roi"] = np.where(df.fav_up, df.roi_dn, df.roi_up)
df["favlast"] = np.where(df.fav_up, df["last"], 1 - df["last"])
df["zfav"] = np.where(df.fav_up, df.z, -df.z)
df["tb"] = pd.cut(df.t, [0, 60, 120, 180, 210, 240, 255, 270, 285, 300])
df["pb"] = pd.cut(df.fav_px, [.5, .6, .7, .8, .9, .95, .97, .98, .985, .99, .995, 1.0])
def tab(g, col):
    return g.groupby(["tb", "pb"], observed=True)[col].agg(["size", "mean"]).unstack(0).round(4)
print("=== favourite ROI by time x fill price (TRAIN) [n, mean roi]")
x = df[df.train & df.fav_px.notna()].groupby(["pb", "tb"], observed=True).fav_roi.agg(["size", "mean"])
print(x["mean"].unstack().round(4)); print(x["size"].unstack())
print("=== same on TEST")
x = df[~df.train & df.fav_px.notna()].groupby(["pb", "tb"], observed=True).fav_roi.agg(["size", "mean"])
print(x["mean"].unstack().round(4))
print("=== model-edge: buy side where model prob - fill px is large (TRAIN / TEST), by time")
df["e_up"] = df.pm - df.px_up; df["e_dn"] = (1 - df.pm) - df.px_dn
for thr in (0.03, 0.06, 0.10, 0.15):
    for nm, m in (("train", df.train), ("test", ~df.train)):
        u = df[m & (df.e_up > thr)]; d = df[m & (df.e_dn > thr)]
        r = pd.concat([u.assign(roi=u.roi_up, px=u.px_up), d.assign(roi=d.roi_dn, px=d.px_dn)])
        r = r[(r.px > 0.05) & (r.px < 0.95)]
        g = r.groupby("tb", observed=True).roi.agg(["size", "mean"])
        print(thr, nm, " ".join(f"{k.right}:{int(v['size'])}/{v['mean']:+.3f}" for k, v in g.iterrows()))
