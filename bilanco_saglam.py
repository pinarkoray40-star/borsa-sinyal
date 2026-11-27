import numpy as np
import pandas as pd
import yfinance as yf

from bilanco_test import bilanco_yukle, fiyatlar, LAG
from tavan_aday import sembol_oku
from bilanco_portfoy import simule, ozet, yillik, donem_getiri

NLER = (20, 30, 40)
EVRENLER = (("LIKIT (ilk %70)", 0.3), ("BUYUK (ilk %40)", 0.6))


def roe_sec(sub, n, temiz):
    if temiz:
        sub = sub[(sub["roe"] <= 0.8) & (sub["d_opm"] <= 0.5)
                  & (sub["d_nm"] <= 0.5)]
    return list(sub["roe"].sort_values(ascending=False).index[:n])


def cagr_haric(eq, yil_haric=2022):
    r = eq.dropna().pct_change().dropna()
    r = r[r.index.year != yil_haric]
    if len(r) < 100:
        return float("nan")
    return ((1 + r).prod() ** (252 / len(r)) - 1) * 100


def calistir(ad, esik, w, closes, lik_rank, xc):
    olaylar = []
    for (yil, ay), g in sorted(w.groupby(["yil", "ay"]), key=lambda x: x[0]):
        avail = (pd.Timestamp(int(yil), int(ay), 1) + pd.offsets.MonthEnd(0)
                 + pd.Timedelta(days=LAG[int(ay)]))
        pos = closes.index.searchsorted(avail)
        if pos >= len(closes) - 1:
            continue
        olaylar.append((pos, g.set_index("kod")))

    pozlar, port = [], {}
    for pos, g in olaylar:
        lik = lik_rank.iloc[pos]
        px0 = closes.iloc[pos]
        ok = [k for k in g.index
              if k in lik.index and lik[k] >= esik and pd.notna(px0.get(k))]
        if len(ok) < 80:
            continue
        sub = g.loc[ok]
        pozlar.append(pos)
        port.setdefault("evren", []).append(list(sub.index))
        for n in NLER:
            port.setdefault(f"roe{n}", []).append(roe_sec(sub, n, False))
            port.setdefault(f"roe_temiz{n}", []).append(roe_sec(sub, n, True))

    print("=" * 100)
    print(f"EVREN: {ad}  rebalance: {len(pozlar)}  "
          f"ilk giris: {closes.index[pozlar[0]].date()}", flush=True)

    sonuc = {k: simule(v, pozlar, closes) for k, v in port.items()}
    idx = closes.index[pozlar[0]:]
    x = xc.reindex(idx).ffill()
    sonuc["XU100"] = x / x.iloc[0]

    evren_r = donem_getiri(sonuc["evren"], pozlar, closes)
    c_ev, _, _ = ozet(sonuc["evren"])
    h_ev = cagr_haric(sonuc["evren"])
    print(f"{'STRATEJI':<14}{'YILLIK%':>9}{'2022 HARIC%':>13}{'MAKS DUSUS%':>13}"
          f"{'SON(x)':>8}{'EVREN FARK':>12}{'FARK 22 HARIC':>15}{'EVRENI GECEN%':>15}")
    for k, eq in sonuc.items():
        c, d, s = ozet(eq)
        h = cagr_haric(eq)
        if k in ("evren", "XU100"):
            hit, f1, f2 = "", "", ""
        else:
            r = donem_getiri(eq, pozlar, closes)
            hit = f"{(r > evren_r).mean() * 100:.0f}"
            f1, f2 = f"{c - c_ev:+.1f}", f"{h - h_ev:+.1f}"
        print(f"{k:<14}{c:>9.1f}{h:>13.1f}{d:>13.1f}{s:>8.2f}"
              f"{f1:>12}{f2:>15}{hit:>15}")

    print("-" * 100)
    print("YIL YIL GETIRI (%):")
    sec_ad = ["roe20", "roe30", "roe40", "roe_temiz20", "evren", "XU100"]
    tablo = pd.DataFrame({k: yillik(sonuc[k]) for k in sec_ad})
    print("yil   " + " ".join(f"{c[:11]:>11}" for c in tablo.columns))
    for y, satir in tablo.iterrows():
        print(f"{y}  " + " ".join(f"{v:11.1f}" for v in satir.values))

    # yarim yarim: ilk ve ikinci yari (zaman kararliligi)
    ort = pozlar[len(pozlar) // 2]
    print("-" * 100)
    print(f"IKI YARI (bolme: {closes.index[ort].date()}), yillik %:")
    for k in ("roe20", "roe30", "roe40", "evren"):
        eq = sonuc[k].dropna()
        for ad2, parca in (("Y1", eq[eq.index < closes.index[ort]]),
                           ("Y2", eq[eq.index >= closes.index[ort]])):
            p = parca / parca.iloc[0]
            yil = (p.index[-1] - p.index[0]).days / 365.25
            print(f"  {k:<12}{ad2}: {(p.iloc[-1] ** (1 / yil) - 1) * 100:7.1f}",
                  end="")
        print()


def main():
    w = bilanco_yukle()
    closes, tl = fiyatlar(sembol_oku())
    med60 = tl.rolling(60, min_periods=30).median()
    lik_rank = med60.rank(axis=1, pct=True)

    xu = yf.download("XU100.IS", start="2019-06-01", interval="1d",
                     auto_adjust=True, progress=False)
    xc = xu["Close"].iloc[:, 0] if isinstance(xu.columns, pd.MultiIndex) else xu["Close"]
    xc.index = pd.to_datetime(xc.index).tz_localize(None).normalize()
    xc = xc[~xc.index.duplicated()]

    for ad, esik in EVRENLER:
        calistir(ad, esik, w, closes, lik_rank, xc)

    print("=" * 100)
    print("Okuma: 'EVREN FARK' = stratejinin yillik getirisi - ayni evrenin esit agirlikli getirisi (puan).")
    print("2022 haric sutunu 2022 gunlerini cikarip yilliklastirir; fark orada da pozitifse 2022'ye bagimli degil.")


if __name__ == "__main__":
    main()
