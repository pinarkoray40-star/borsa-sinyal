import glob

import numpy as np
import pandas as pd
import yfinance as yf

from tavan_aday import sembol_oku

HORIZON = (20, 60, 120)
LAG = {3: 50, 6: 65, 9: 50, 12: 75}
KALEMLER = ["3C", "3D", "3DF", "3L", "2N"]
BAZ = ["d_opm", "d_nm", "d_gm", "roe", "d_roe", "nm"]
TUM = BAZ + ["ivme3", "ivme_kalite"]


def bilanco_yukle():
    yol = glob.glob("**/bilanco.csv.gz", recursive=True)[0]
    b = pd.read_csv(yol, dtype={"kalem": str})
    b = b[b["kalem"].isin(KALEMLER)].copy()
    b["deger"] = pd.to_numeric(b["deger"], errors="coerce")
    w = b.pivot_table(index=["kod", "yil", "ay"], columns="kalem",
                      values="deger", aggfunc="last").reset_index()
    w = w.dropna(subset=KALEMLER)
    w = w[(w["3C"] > 0) & (w["2N"] > 0)].copy()
    w["opm"] = (w["3DF"] / w["3C"]).clip(-1, 1)
    w["nm"] = (w["3L"] / w["3C"]).clip(-1, 1)
    w["gm"] = (w["3D"] / w["3C"]).clip(-1, 1)
    w["roe"] = (w["3L"] / w["2N"] * (12 / w["ay"])).clip(-2, 2)
    onceki = w[["kod", "yil", "ay", "opm", "nm", "gm", "roe"]].copy()
    onceki["yil"] += 1
    w = w.merge(onceki, on=["kod", "yil", "ay"], suffixes=("", "_p"))
    w["d_opm"] = w["opm"] - w["opm_p"]
    w["d_nm"] = w["nm"] - w["nm_p"]
    w["d_gm"] = w["gm"] - w["gm_p"]
    w["d_roe"] = w["roe"] - w["roe_p"]
    return w


def fiyatlar(semboller):
    kap, tl = {}, {}
    for i in range(0, len(semboller), 40):
        grup = semboller[i:i + 40]
        try:
            df = yf.download(grup, start="2019-06-01", interval="1d",
                             group_by="ticker", auto_adjust=True,
                             progress=False, threads=True)
        except Exception as e:
            print("hata:", e, flush=True)
            continue
        for s in grup:
            try:
                d = df[s] if isinstance(df.columns, pd.MultiIndex) else df
                d = d.dropna(subset=["Close"])
                if len(d) < 300:
                    continue
                idx = pd.to_datetime(d.index).tz_localize(None).normalize()
                ok = ~idx.duplicated()
                k = s.replace(".IS", "")
                kap[k] = pd.Series(d["Close"].values[ok], index=idx[ok])
                tl[k] = pd.Series((d["Close"] * d["Volume"].fillna(0)).values[ok],
                                  index=idx[ok])
            except KeyError:
                pass
        print(f"fiyat indirildi {min(i + 40, len(semboller))}/{len(semboller)}",
              flush=True)
    return pd.DataFrame(kap).sort_index(), pd.DataFrame(tl).sort_index()


def yaz(ad, h, lst):
    if len(lst) < 6:
        print(f"{ad:<11}{h:>4}g  (az olay: {len(lst)})")
        return
    arr = np.array([m for _, m in lst]) * 100
    a = arr.mean(axis=0)
    sp = arr[:, 4] - arr[:, 0]
    n = len(sp)
    sd = sp.std(ddof=1)
    t = sp.mean() / (sd / np.sqrt(n)) if sd > 0 else 0
    y = n // 2
    print(f"{ad:<11}{h:>4}g Q1 {a[0]:+5.1f} Q2 {a[1]:+5.1f} Q3 {a[2]:+5.1f}"
          f" Q4 {a[3]:+5.1f} Q5 {a[4]:+5.1f} | Q5-Q1 {sp.mean():+5.1f}"
          f" t={t:4.1f} hit %{(sp > 0).mean() * 100:3.0f}"
          f" Y1 {sp[:y].mean():+5.1f} Y2 {sp[y:].mean():+5.1f}")


def main():
    w = bilanco_yukle()
    print(f"bilanco olay-hisse: {len(w)}  hisse: {w['kod'].nunique()}", flush=True)
    closes, tl = fiyatlar(sembol_oku())
    print(f"fiyat: {closes.shape[1]} hisse, {closes.index[0].date()} -> "
          f"{closes.index[-1].date()}", flush=True)
    med60 = tl.rolling(60, min_periods=30).median()
    lik_rank = med60.rank(axis=1, pct=True)
    fwd = {h: closes.shift(-h) / closes - 1 for h in HORIZON}

    sonuc = {h: {f: [] for f in TUM} for h in HORIZON}
    olay_sayisi = 0
    for (yil, ay), g in w.groupby(["yil", "ay"]):
        avail = (pd.Timestamp(int(yil), int(ay), 1) + pd.offsets.MonthEnd(0)
                 + pd.Timedelta(days=LAG[int(ay)]))
        pos = closes.index.searchsorted(avail)
        if pos >= len(closes):
            continue
        g = g.set_index("kod")
        lik = lik_rank.iloc[pos]
        likit_kodlar = set(lik[lik >= 0.3].index)
        for h in HORIZON:
            if pos + h >= len(closes):
                continue
            r = fwd[h].iloc[pos].dropna()
            ortak = [k for k in g.index if k in r.index and k in likit_kodlar]
            if len(ortak) < 100:
                continue
            rr = r[ortak]
            rr = rr.clip(rr.quantile(0.01), rr.quantile(0.99))
            sub = g.loc[ortak]
            rk = sub[BAZ].rank(pct=True)
            seri = {f: sub[f] for f in BAZ}
            seri["ivme3"] = rk[["d_opm", "d_nm", "d_roe"]].mean(axis=1)
            seri["ivme_kalite"] = rk[["d_opm", "d_nm", "d_roe", "roe"]].mean(axis=1)
            ex = rr - rr.mean()
            for f in TUM:
                b = np.ceil(seri[f].rank(pct=True) * 5).clip(1, 5)
                m = [ex[b == k].mean() for k in range(1, 6)]
                if not np.any(np.isnan(m)):
                    sonuc[h][f].append((avail, m))
            if h == HORIZON[0]:
                olay_sayisi += 1
    print(f"olay (donem) sayisi (20g): {olay_sayisi}")
    print("Q1 = en kotu ivme, Q5 = en iyi ivme; degerler evrene gore fazla getiri"
          " (puan). hit = Q5>Q1 olan donem orani.")
    print("(60g/120g pencereler birbirine biner, t degerleri sisik olabilir)")

    for f in TUM:
        print("-" * 100)
        for h in HORIZON:
            yaz(f, h, sonuc[h][f])

    print("=" * 100)
    son = None
    for (y, a), g in w.groupby(["yil", "ay"]):
        if len(g) >= 200:
            son = (int(y), int(a), g)
    y, a, g = son
    g = g.set_index("kod")
    lik = lik_rank.iloc[-1]
    ok = [k for k in g.index if k in lik.index and lik[k] >= 0.3]
    s = g.loc[ok]
    rk = s[["d_opm", "d_nm", "d_roe", "roe"]].rank(pct=True)
    s = s.assign(skor=rk.mean(axis=1)).sort_values("skor", ascending=False).head(15)
    print(f"GUNCEL ADAYLAR (donem {y}/{a}, likit hisseler, ivme+kalite skoru):")
    for kod, r in s.iterrows():
        print(f"  {kod:<7} skor {r['skor']:.2f}  faaliyet marji degisimi {r['d_opm'] * 100:+6.1f}"
              f"  net marj degisimi {r['d_nm'] * 100:+6.1f}  ROE(yil) %{r['roe'] * 100:5.1f}")


if __name__ == "__main__":
    main()
