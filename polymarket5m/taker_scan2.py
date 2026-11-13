import numpy as np, pandas as pd
df = pd.read_parquet("data/signals_all.parquet")
SL = 0.01
qa, qb = df.ask, 1 - df.bid          # last observed taker price for Up / Down (known at decision time)
df["e_up"] = df.phat - qa - 0.07 * qa * (1 - qa)
df["e_dn"] = (1 - df.phat) - qb - 0.07 * qb * (1 - qb)
df["ok_up"] = df.px_up <= qa + SL; df["ok_dn"] = df.px_dn <= qb + SL
for a in ("btc", "eth", "xrp"):
    d = df[df.asset == a]
    print("####", a)
    for t0, t1 in ((10, 300), (10, 120), (120, 240), (240, 300)):
        for th in (0.01, 0.02, 0.03, 0.05, 0.08):
            out = []
            for nm, m in (("tr", d.train), ("te", ~d.train)):
                u = d[m & (d.e_up > th) & d.ok_up & (d.t >= t0) & (d.t < t1)]; w = d[m & (d.e_dn > th) & d.ok_dn & (d.t >= t0) & (d.t < t1)]
                r = pd.concat([u.assign(roi=u.roi_up, px=u.px_up), w.assign(roi=w.roi_dn, px=w.px_dn)]).sort_values(["start", "t"])
                r1 = r.drop_duplicates("start")
                se = r1.roi.std() / np.sqrt(max(len(r1), 1)); se2 = r.roi.std() / np.sqrt(max(r.start.nunique(), 1))
                out.append(f"{nm}: all n={len(r):5d} roi={r.roi.mean():+.3f} (t~{r.roi.mean()/se2:+.1f}) | first n={len(r1):4d} roi={r1.roi.mean():+.3f} (t={r1.roi.mean()/se:+.1f}) px={r1.px.mean():.2f}")
            print(f"t[{t0},{t1}) th={th}: " + " || ".join(out))
