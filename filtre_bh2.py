"""
filtre_bh2.py - BH2 işlemlerine aday filtreleri uygular ve etkisini ölçer. Siteye dokunmaz.
buyuk.py, buyuk2.py ve ayrim_bh2.py'yi içe aktarır (sinyal/takip/veri çekme aynen aynı).
Filtreler sinyal gününde bilinen bilgilerle çalışır. Kalan işlemler tabanın alt kümesidir.
Not: filtre sonradan uygulanıyor; canlıda cooldown etkisiyle küçük farklar olabilir.
"""
import warnings

import numpy as np
import pandas as pd

import ayrim_bh2 as AY
import buyuk as BY
import buyuk2 as B2

warnings.filterwarnings("ignore")
YEARS = 5


def welch(a, b):
    if len(a) < 15 or len(b) < 15:
        return np.nan
    return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))


def med_edge(d, m):
    a, b = d.loc[m, "pnl"], d.loc[~m, "pnl"]
    if len(a) < 15 or len(b) < 15:
        return np.nan
    return a.median() - b.median()


def main():
    syms = AY.read_symbols()
    print("sembol sayısı:", len(syms))
    close, volume = AY.load(syms)
    ist = pd.Timestamp.now(tz="UTC").tz_convert("Europe/Istanbul")
    if close.index[-1].date() == ist.date() and (ist.hour, ist.minute) < (18, 40):
        close, volume = close.iloc[:-1], volume.iloc[:-1]
    print("hisse:", close.shape[1], "gün:", close.shape[0], "son gün:", close.index[-1].date())

    idx = B2.fetch_index(close)
    ix = idx.reindex(close.index).ffill(limit=3)
    ri = ix.pct_change(fill_method=None)
    r = close.pct_change(fill_method=None)
    dn, dr = B2.dayaniklilik(close, idx, B2.PARAM["window"])

    P = B2.PARAM
    bh = BY.ozellikler(close, volume)
    DR = dr.to_numpy()
    mask = bh & (np.nan_to_num(DR, nan=0.0) >= P["dn_min"])
    ii, jj = BY.sinyaller(mask, P["cooldown"])
    L = BY.likit_maske(close, volume)
    C = close.to_numpy()

    # --- sinyal günü özellikleri ---
    cov = r.rolling(120, min_periods=60).cov(ri)
    var = ri.rolling(120, min_periods=60).var()
    F = {
        "ciro_sira": (close * volume).rolling(20).mean().rank(axis=1, pct=True) * 100,
        "fiyat_sira": close.rank(axis=1, pct=True) * 100,
        "ret5": (close / close.shift(5) - 1) * 100,
        "oyn_sira": r.rolling(60).std().rank(axis=1, pct=True) * 100,
        "kirilim": (close / close.shift(1).rolling(20).max() - 1) * 100,
        "beta120": cov.div(var, axis=0),
    }
    FA = {k: v.to_numpy() for k, v in F.items()}

    start = close.index[-1] - pd.DateOffset(years=YEARS)
    rows = []
    for p, j in zip(ii, jj):
        if close.index[p] < start:
            continue
        entry = float(C[p, j])
        if not np.isfinite(entry) or entry <= 0:
            continue
        path = (C[p + 1:p + P["hold"] + 1, j] / entry - 1) * 100
        st, k, xr, peak = BY.takip(path, P["sl"], P["hold"])
        if st == "açık":
            continue
        b = BY.kiyas(C, L, p, int(k))
        row = {"sym": close.columns[j].replace(".IS", ""), "date": close.index[p], "year": close.index[p].year,
               "pnl": float(xr), "st": st, "diff": (float(xr) - b) if b is not None else np.nan}
        for f in FA:
            row[f] = FA[f][p, j]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df.to_csv("filtre_bh2.csv", index=False)
    n = len(df)
    mid = df["date"].iloc[n // 2]
    df["yari"] = np.where(df["date"] < mid, 1, 2)
    years = sorted(df.year.unique())
    print(f"\nBH2 sonuçlanmış işlem: {n} | bölme {mid.date()} | beta boş değil: {df.beta120.notna().sum()}")

    top20 = set(df.nlargest(20, "pnl").index)

    # --- filtreler ---
    d = df
    FILT = [
        ("ciro>=50", d.ciro_sira >= 50),
        ("ciro>=60", d.ciro_sira >= 60),
        ("ciro>=70", d.ciro_sira >= 70),
        ("fiyat_sira<=50", d.fiyat_sira <= 50),
        ("fiyat_sira<=70", d.fiyat_sira <= 70),
        ("ret5<=10", d.ret5 <= 10),
        ("ret5<=15", d.ret5 <= 15),
        ("kirilim<=2", d.kirilim <= 2.0),
        ("oyn_sira<90", d.oyn_sira < 90),
        ("oyn_sira<80", d.oyn_sira < 80),
        ("c60+fiyat70", (d.ciro_sira >= 60) & (d.fiyat_sira <= 70)),
        ("c60+ret5<=10", (d.ciro_sira >= 60) & (d.ret5 <= 10)),
        ("c60+oyn<90", (d.ciro_sira >= 60) & (d.oyn_sira < 90)),
        ("c60+f70+r5<=10", (d.ciro_sira >= 60) & (d.fiyat_sira <= 70) & (d.ret5 <= 10)),
    ]

    def summ(x):
        if len(x) == 0:
            return [0, np.nan, np.nan, np.nan, np.nan, np.nan, 0]
        return [len(x), x.pnl.mean(), x.pnl.median(), (x.pnl > 0).mean() * 100,
                (x.st == "stop").mean() * 100, x["diff"].mean(), sum(i in top20 for i in x.index)]

    cols = ["filtre", "n", "ort", "med", "kaz%", "stop%", "fark", "top20"]
    t1 = [["TABAN"] + summ(df)]
    base_ex = df.drop(index=list(top20))
    print("\n=== 1) FİLTRELER (kalan işlemler) ===")
    t2 = []
    for name, m in FILT:
        m = m.fillna(False)
        kept = df[m]
        t1.append([name] + summ(kept))
        kept_ex = kept.drop(index=[i for i in kept.index if i in top20])
        e_all = kept.pnl.mean() - df.loc[~m, "pnl"].mean() if m.sum() and (~m).sum() else np.nan
        t = welch(kept.pnl, df.loc[~m, "pnl"])
        mf = med_edge(df, m)
        y = [med_edge(df[df.yari == h], m[df.yari == h]) for h in (1, 2)]
        ye = [med_edge(df[df.year == yr], m[df.year == yr]) for yr in years]
        ye = [x for x in ye if np.isfinite(x)]
        pos = sum(x > 0 for x in ye)
        ok = (np.isfinite(mf) and mf > 0 and all(np.isfinite(x) and x > 0 for x in y)
              and len(ye) >= 3 and pos >= int(np.ceil(0.7 * len(ye)))
              and kept.pnl.mean() >= df.pnl.mean() and summ(kept)[6] >= 14 and len(kept) >= 0.4 * n)
        t2.append([name, kept_ex.pnl.mean() if len(kept_ex) else np.nan,
                   kept_ex.pnl.median() if len(kept_ex) else np.nan, e_all, t, mf, y[0], y[1],
                   f"{pos}/{len(ye)}", "ADAY" if ok else ""])
    print(pd.DataFrame(t1, columns=cols).to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    print(f"\n=== 2) DAYANIKLILIK (ilk 20 çıkınca tabanın ort {base_ex.pnl.mean():.1f}, medyan {base_ex.pnl.median():.1f}) ===")
    print("ex20 = ilk 20 işlem çıkarılınca kalanın ort/med | ort_f = kalan-elenen ortalama farkı | t = Welch")
    print("med_f, y1, y2 = kalan-elenen MEDYAN farkı (tüm dönem, yarı1, yarı2) | yıl = medyan farkı + olan yıllar")
    c2 = ["filtre", "ex20_ort", "ex20_med", "ort_f", "t", "med_f", "y1", "y2", "yıl", "karar"]
    print(pd.DataFrame(t2, columns=c2).to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print("(ADAY = medyan farkı tüm dönemde ve iki yarıda pozitif, yılların >=%70'inde pozitif, ortalama düşmüyor,")
    print(" ilk 20'nin en az 14'ü kalıyor, işlemlerin en az %40'ı kalıyor)")

    # --- beta (düzeltilmiş) ---
    print("\n=== 3) BETA (düzeltilmiş) beşli dilim, hücre = ort işlem % ===")
    bd = df[["pnl", "yari", "beta120"]].dropna()
    if len(bd) >= 100:
        q = pd.qcut(bd.beta120.rank(method="first"), 5, labels=False)
        qm = bd.groupby(q)["pnl"].mean()
        print("Q1(düşük beta) ... Q5(yüksek):", " | ".join(f"{qm.get(i):.1f}" for i in range(5)))
        print("rho:", f"{AY.rho(bd.beta120, bd.pnl):.2f}",
              "| yarı1:", f"{AY.rho(bd.loc[bd.yari == 1, 'beta120'], bd.loc[bd.yari == 1, 'pnl']):.2f}",
              "| yarı2:", f"{AY.rho(bd.loc[bd.yari == 2, 'beta120'], bd.loc[bd.yari == 2, 'pnl']):.2f}",
              "| beta medyan:", f"{bd.beta120.median():.2f}")
    else:
        print("beta hâlâ boş, düzeltme çalışmadı.")
    print("\nCSV kaydedildi: filtre_bh2.csv")


if __name__ == "__main__":
    main()
