import numpy as np, pandas as pd
df = pd.read_parquet("data/signals_all.parquet")
df["e_up"] = df.phat - df.px_up - 0.07 * df.px_up * (1 - df.px_up)
df["e_dn"] = (1 - df.phat) - df.px_dn - 0.07 * df.px_dn * (1 - df.px_dn)
for a in ("btc", "eth", "xrp"):
    d = df[df.asset == a]
    print("####", a)
    for t0, t1 in ((10, 300), (10, 120), (120, 240), (240, 300)):
        for th in (0.02, 0.04, 0.06, 0.08, 0.12):
            out = []
            for nm, m in (("tr", d.train), ("te", ~d.train)):
                u = d[m & (d.e_up > th) & (d.t >= t0) & (d.t < t1)]; w = d[m & (d.e_dn > th) & (d.t >= t0) & (d.t < t1)]
                r = pd.concat([u.assign(roi=u.roi_up, px=u.px_up, usd=u.usd_up), w.assign(roi=w.roi_dn, px=w.px_dn, usd=w.usd_dn)]).sort_values(["start", "t"])
                r1 = r.drop_duplicates("start")
                se = r1.roi.std() / np.sqrt(max(len(r1), 1))
                out.append(f"{nm}: all n={len(r):5d} roi={r.roi.mean():+.3f} | first/mkt n={len(r1):4d} roi={r1.roi.mean():+.3f} (t={r1.roi.mean()/se:+.1f}) px={r1.px.mean():.2f}")
            print(f"t[{t0},{t1}) th={th}: " + " || ".join(out))
