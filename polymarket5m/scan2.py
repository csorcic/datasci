import sys, numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
for a in ("btc", "eth", "xrp"):
    df = pd.read_parquet(f"data/signals_{a}.parquet")
    split = df.start.min() + 9 * 86400
    df["train"] = df.start < split
    fu = df["last"] >= 0.5
    df["fav_px"] = np.where(fu, df.px_up, df.px_dn); df["fav_roi"] = np.where(fu, df.roi_up, df.roi_dn)
    df["fav_won"] = np.where(fu, df.up, ~df.up); df["zfav"] = np.where(fu, df.z, -df.z)
    df["dog_px"] = np.where(fu, df.px_dn, df.px_up); df["dog_roi"] = np.where(fu, df.roi_dn, df.roi_up)
    print("#####", a)
    for lo, hi in ((0.90, 0.95), (0.95, 0.97), (0.97, 0.985), (0.985, 0.996)):
        for t0, t1 in ((120, 180), (180, 240), (240, 270), (270, 296)):
            out = []
            for zmin in (-9, 1.5, 2.5):
                for nm, m in (("tr", df.train), ("te", ~df.train)):
                    g = df[m & (df.fav_px >= lo) & (df.fav_px < hi) & (df.t >= t0) & (df.t < t1) & (df.zfav >= zmin)]
                    g = g.drop_duplicates("i")           # first signal per market only
                    out.append(f"z>{zmin} {nm}: n={len(g):4d} L={int((~g.fav_won.astype(bool)).sum()):3d} roi={g.fav_roi.mean():+.4f}")
            print(f"px[{lo},{hi}) t[{t0},{t1})  " + " | ".join(out))
    # longshot check: buying the dog
    for lo, hi in ((0.0, 0.03), (0.03, 0.1), (0.1, 0.25)):
        for t0, t1 in ((120, 240), (240, 296)):
            g = df[(df.dog_px >= lo) & (df.dog_px < hi) & (df.t >= t0) & (df.t < t1)].drop_duplicates("i")
            print(f"dog px[{lo},{hi}) t[{t0},{t1}) n={len(g)} roi={g.dog_roi.mean():+.3f}")
