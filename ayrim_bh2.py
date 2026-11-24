"""
ayrim_bh2.py - BH2 işlemlerinde kazananları kaybedenlerden ayıran özellik analizi.
Siteye dokunmaz. buyuk.py ve buyuk2.py'yi içe aktarır (sinyal/takip aynen siteyle aynı).
Kural: giriş sinyal günü kapanışı, izleyen stop %25, 84 gün süre. Sadece sonuçlanmış işlemler.
"""
import warnings

import numpy as np
import pandas as pd

import buyuk as BY
import buyuk2 as B2

warnings.filterwarnings("ignore")

YEARS = 5      # analiz penceresi
DL_YEARS = 8   # ısınma dahil indirilecek yıl


def read_symbols():
    syms = []
    with open("bist_symbols.txt", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            for tok in ln.replace(",", " ").split():
                tok = tok.strip().upper()
                if tok:
                    syms.append(tok if "." in tok else tok + ".IS")
    return list(dict.fromkeys(syms))


def load(symbols):
    import yfinance as yf

    cs, vs = [], []
    for i in range(0, len(symbols), 80):
        chunk = symbols[i:i + 80]
        try:
            raw = yf.download(chunk, period=f"{DL_YEARS}y", interval="1d", auto_adjust=True,
                              progress=False, threads=True)
        except Exception as e:
            print("indirme hatası", i, e)
            continue
        if raw is None or raw.empty:
            continue
        c, v = raw["Close"], raw["Volume"]
        if isinstance(c, pd.Series):
            c, v = c.to_frame(chunk[0]), v.to_frame(chunk[0])
        cs.append(c)
        vs.append(v)
    close = pd.concat(cs, axis=1)
    volume = pd.concat(vs, axis=1)
    close = close.loc[:, ~close.columns.duplicated()]
    volume = volume.loc[:, ~volume.columns.duplicated()]
    for d in (close, volume):
        d.index = pd.to_datetime(d.index).tz_localize(None)
    close = close.sort_index().dropna(how="all").dropna(how="all", axis=1)
    volume = volume.reindex(index=close.index, columns=close.columns)
    return close, volume


def rho(a, b):
    d = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(d) < 15:
        return np.nan
    return d["a"].rank().corr(d["b"].rank())


def main():
    syms = read_symbols()
    print("sembol sayısı:", len(syms))
    close, volume = load(syms)
    # yarım günlük son mumu at (buyuk2 ile aynı mantık)
    ist = pd.Timestamp.now(tz="UTC").tz_convert("Europe/Istanbul")
    if close.index[-1].date() == ist.date() and (ist.hour, ist.minute) < (18, 40):
        close, volume = close.iloc[:-1], volume.iloc[:-1]
    print("hisse:", close.shape[1], "gün:", close.shape[0], "son gün:", close.index[-1].date())

    idx = B2.fetch_index(close)
    ix = idx.reindex(close.index).ffill(limit=3)
    ri = ix.pct_change(fill_method=None)
    r = close.pct_change(fill_method=None)
    dn, dr = B2.dayaniklilik(close, idx, B2.PARAM["window"])

    # --- sinyaller (siteyle aynı) ---
    P = B2.PARAM
    bh = BY.ozellikler(close, volume)
    DR = dr.to_numpy()
    mask = bh & (np.nan_to_num(DR, nan=0.0) >= P["dn_min"])
    ii, jj = BY.sinyaller(mask, P["cooldown"])
    L = BY.likit_maske(close, volume)
    C = close.to_numpy()
    T = close.shape[0]

    # --- sinyal günü özellikleri ---
    F = {}
    F["ret5"] = (close / close.shift(5) - 1) * 100
    F["ret20"] = (close / close.shift(20) - 1) * 100
    F["ret60"] = (close / close.shift(60) - 1) * 100
    F["ret120"] = (close / close.shift(120) - 1) * 100
    F["zirve60_uzak"] = (close / close.rolling(60).max() - 1) * 100
    F["zirve250_uzak"] = (close / close.rolling(250, min_periods=120).max() - 1) * 100
    F["kirilim_gucu"] = (close / close.shift(1).rolling(20).max() - 1) * 100
    F["hacim_oran"] = volume.rolling(5).mean() / volume.rolling(60).mean()
    F["oynaklik60"] = r.rolling(60).std() * 100
    F["bant20"] = (close.rolling(20).max() - close.rolling(20).min()) / close * 100
    F["ciro_sira"] = (close * volume).rolling(20).mean().rank(axis=1, pct=True) * 100
    F["dayaniklilik_sira"] = dr * 100
    F["beta120"] = r.rolling(120, min_periods=60).cov(ri) / ri.rolling(120, min_periods=60).var()
    F["goreceli60"] = F["ret60"].sub((ix / ix.shift(60) - 1) * 100, axis=0)
    F["fiyat_tl"] = close
    # piyasa özellikleri (aynı gün tüm hisseler için aynı)
    sma50 = close.rolling(50).mean()
    gen = ((close > sma50).where(sma50.notna()).mean(axis=1) * 100)
    M = {"piyasa_genislik": gen,
         "endeks_sma50": (ix / ix.rolling(50).mean() - 1) * 100,
         "endeks_ret60": (ix / ix.shift(60) - 1) * 100}
    FA = {k: v.to_numpy() for k, v in F.items()}
    MA = {k: v.to_numpy() for k, v in M.items()}
    FEATS = list(F) + list(M)

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
        row = {"sym": close.columns[j].replace(".IS", ""), "date": close.index[p],
               "year": close.index[p].year, "pnl": float(xr), "st": st, "days": int(k),
               "bench": b, "diff": (float(xr) - b) if b is not None else np.nan, "peak": peak}
        for f in F:
            row[f] = FA[f][p, j]
        for f in M:
            row[f] = MA[f][p]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df.to_csv("ayrim_bh2.csv", index=False)
    n = len(df)
    print(f"\nBH2 sonuçlanmış işlem: {n}")
    mid = df["date"].iloc[n // 2]
    df["yari"] = np.where(df["date"] < mid, 1, 2)
    ff = lambda x: f"{x:.1f}"

    # 1) genel
    print("\n=== 1) GENEL ===")
    print(f"ort {df.pnl.mean():.1f} | medyan {df.pnl.median():.1f} | kazanan %{(df.pnl > 0).mean()*100:.0f} "
          f"| stop %{(df.st == 'stop').mean()*100:.0f} | evrene fark ort {df['diff'].mean():.1f}")
    print(f"yarı1 ort {df[df.yari==1].pnl.mean():.1f} (n{(df.yari==1).sum()}) | "
          f"yarı2 ort {df[df.yari==2].pnl.mean():.1f} (n{(df.yari==2).sum()}) | bölme tarihi {mid.date()}")

    # 2) en iyi 20 işlem tek tek
    print("\n=== 2) EN İYİ 20 İŞLEM (tek tek) ===")
    show = ["ret20", "ret60", "zirve60_uzak", "hacim_oran", "oynaklik60", "bant20", "beta120",
            "dayaniklilik_sira", "piyasa_genislik"]
    top = df.nlargest(20, "pnl")
    t = top[["sym", "date", "pnl", "days"] + show].copy()
    t["date"] = t["date"].dt.strftime("%Y-%m-%d")
    print(t.to_string(index=False, float_format=ff))

    # 3) grup medyanları
    print("\n=== 3) GRUP MEDYANLARI ===")
    q90 = df.pnl.quantile(0.90)
    groups = {"ilk20": top, f"ilk%10(>={q90:.0f})": df[df.pnl >= q90],
              "kazanan(>0)": df[df.pnl > 0], "kaybeden(<=0)": df[df.pnl <= 0], "tümü": df}
    med = pd.DataFrame({g: d[FEATS].median() for g, d in groups.items()})
    print(med.to_string(float_format=ff))

    # 4) beşli dilim + tutarlılık
    print("\n=== 4) BEŞLİ DİLİM (Q1=özellik düşük ... Q5=yüksek; hücre = ort işlem %) ===")
    years = sorted(df.year.unique())
    out = []
    for f in FEATS:
        d = df[["pnl", "diff", "year", "yari", f]].dropna(subset=[f])
        if len(d) < 100 or d[f].nunique() < 5:
            continue
        q = pd.qcut(d[f].rank(method="first"), 5, labels=False)
        qm = d.groupby(q)["pnl"].mean()
        r_all = rho(d[f], d["pnl"])
        r1 = rho(d.loc[d.yari == 1, f], d.loc[d.yari == 1, "pnl"])
        r2 = rho(d.loc[d.yari == 2, f], d.loc[d.yari == 2, "pnl"])
        yr = [rho(d.loc[d.year == y, f], d.loc[d.year == y, "pnl"]) for y in years if (d.year == y).sum() >= 25]
        same = sum(np.sign(x) == np.sign(r_all) for x in yr if np.isfinite(x))
        ok = (abs(r_all) >= 0.08 and np.sign(r1) == np.sign(r2) == np.sign(r_all)
              and len(yr) > 0 and same >= int(np.ceil(0.7 * len(yr))))
        out.append({"özellik": f, "Q1": qm.get(0), "Q2": qm.get(1), "Q3": qm.get(2), "Q4": qm.get(3),
                    "Q5": qm.get(4), "rho": r_all, "rho_y1": r1, "rho_y2": r2,
                    "yıl_uyum": f"{same}/{len(yr)}", "tutarlı": "EVET" if ok else ""})
    res = pd.DataFrame(out).sort_values("rho", key=lambda s: s.abs(), ascending=False)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("(tutarlı = |rho|>=0.08, iki yarıda aynı yön, yılların >=%70'inde aynı yön)")

    # 5) tuzak kontrolü
    print("\n=== 5) TUZAK KONTROLÜ ===")
    print("ilk20 içinde farklı hisse:", top.sym.nunique(), "| yıl dağılımı:", dict(top.year.value_counts().sort_index()))
    print("ilk20 hisseler:", ", ".join(top.sym.tolist()))
    pos = df.loc[df.pnl > 0, "pnl"].sum()
    print(f"ilk10 işlem, tüm pozitif toplamın %{df.nlargest(10,'pnl').pnl.sum()/pos*100:.0f}'i")
    rest = df.drop(top.index)
    print(f"ilk20'yi çıkarınca: ort {rest.pnl.mean():.1f} | medyan {rest.pnl.median():.1f} | evrene fark {rest['diff'].mean():.1f}")
    print("\nCSV kaydedildi: ayrim_bh2.csv")


if __name__ == "__main__":
    main()
