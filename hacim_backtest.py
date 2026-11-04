"""
hacim_backtest.py
hacim_birikim.py'deki sinyalin GEÇMİŞTE işe yarayıp yaramadığını test eder.

Her hisse ve her gün için sinyal o günün kapanışına kadar olan veriyle
hesaplanır. Alış, sinyalden SONRAKİ günün kapanışından yapılır (gerçekçi).
Sonra 5 ve 10 işlem günü sonraki getiri ölçülür. Aynı hissede sinyalden sonra
10 gün yeni sinyal sayılmaz (üst üste binmeyi önler). Her işlemden %0.4
maliyet düşülür.

Sonuçlar "tüm günler" (rastgele alış) ile karşılaştırılır.

Not: Sadece bugün borsada olan hisseler test edilir (hayatta kalma yanlılığı
olabilir), tavan/taban kilitlerinde gerçekte alamama durumu hesaba katılmaz.

Kullanım: python hacim_backtest.py [HISSE ...]
Gereken: pip install borsapy pandas numpy
"""
import sys
import numpy as np
import pandas as pd
import borsapy as bp

UFUKLAR = [5, 10]
MALIYET = 0.004       # gidiş-dönüş toplam
COOLDOWN = 10         # aynı hissede sinyal sonrası beklenen gün


def kolon_bul(df, ad):
    for c in df.columns:
        if str(c).lower() == ad.lower():
            return c
    raise KeyError(ad)


def hisseleri_al():
    if len(sys.argv) > 1:
        return [t.upper() for t in sys.argv[1:]]
    df = bp.companies()
    for c in df.columns:
        if str(c).lower() in ("ticker", "symbol", "kod", "code"):
            return [str(x).strip().upper() for x in df[c].dropna()]
    return [str(x).strip().upper() for x in df.iloc[:, 0].dropna()]


def veri_al(kod):
    for periyot in ("2y", "1y", "6mo"):
        try:
            df = bp.Ticker(kod).history(period=periyot)
            if df is not None and len(df) >= 100:
                return df
        except Exception:
            continue
    return None


def hazirla(df):
    k = df[kolon_bul(df, "Close")].astype(float).reset_index(drop=True)
    h = df[kolon_bul(df, "Volume")].astype(float).reset_index(drop=True)
    d = pd.DataFrame({"k": k, "h": h})
    onceki20 = h.shift(5).rolling(20).mean()
    d["oran"] = h.rolling(5).mean() / onceki20.replace(0, np.nan)
    d["getiri5"] = k / k.shift(5) - 1
    mx15 = k.rolling(15).max()
    mn15 = k.rolling(15).min()
    d["bant"] = (mx15 - mn15) / mn15
    d["zirve"] = k / k.rolling(60).max()
    for u in UFUKLAR:
        d[f"f{u}"] = k.shift(-1 - u) / k.shift(-1) - 1 - MALIYET
    return d.dropna(subset=["oran", "getiri5", "bant", "zirve"])


VARYANTLAR = {
    "A hacim>=1.5": lambda d: (d.getiri5 > 0) & (d.oran >= 1.5),
    "A hacim>=2": lambda d: (d.getiri5 > 0) & (d.oran >= 2),
    "A hacim>=3": lambda d: (d.getiri5 > 0) & (d.oran >= 3),
    "A hacim>=4": lambda d: (d.getiri5 > 0) & (d.oran >= 4),
    "A>=3, tepede degil": lambda d: (d.getiri5 > 0) & (d.oran >= 3) & (d.zirve < 0.95),
    "A>=3, 5g getiri<=%15": lambda d: (d.getiri5 > 0) & (d.getiri5 <= 0.15) & (d.oran >= 3),
    "B yatay birikim>=3": lambda d: (d.bant <= 0.08) & (d.oran >= 3),
}


def cooldown_sec(maske):
    """Sinyal sonrası COOLDOWN gün yeni sinyalleri atla."""
    secilen = []
    son = -10**9
    for i in np.flatnonzero(maske.values):
        if i - son > COOLDOWN:
            secilen.append(i)
            son = i
    return secilen


def main():
    kodlar = hisseleri_al()
    sonuc = {v: {u: [] for u in UFUKLAR} for v in VARYANTLAR}
    taban = {u: [] for u in UFUKLAR}
    test_edilen = 0

    for kod in kodlar:
        try:
            df = veri_al(kod)
            if df is None:
                continue
            d = hazirla(df)
            if len(d) < 50:
                continue
        except Exception as e:
            print(f"{kod}: atlandı ({e})")
            continue
        test_edilen += 1
        for u in UFUKLAR:
            taban[u].extend(d[f"f{u}"].dropna().tolist())
        for ad, fn in VARYANTLAR.items():
            idx = cooldown_sec(fn(d))
            for u in UFUKLAR:
                sonuc[ad][u].extend(d.iloc[idx][f"f{u}"].dropna().tolist())

    print(f"Test edilen hisse: {test_edilen}\n")
    for u in UFUKLAR:
        print(f"=== {u} işlem günü sonra (maliyet düşülmüş) ===")
        satirlar = []
        t = np.array(taban[u])
        satirlar.append(["TUM GUNLER (kıyas)", len(t), t.mean() * 100,
                         np.median(t) * 100, (t > 0).mean() * 100])
        for ad in VARYANTLAR:
            a = np.array(sonuc[ad][u])
            if len(a) == 0:
                satirlar.append([ad, 0, np.nan, np.nan, np.nan])
            else:
                satirlar.append([ad, len(a), a.mean() * 100,
                                 np.median(a) * 100, (a > 0).mean() * 100])
        tablo = pd.DataFrame(satirlar, columns=["varyant", "adet", "ort_%", "medyan_%", "kazanma_%"])
        pd.set_option("display.width", 200)
        print(tablo.round(2).to_string(index=False))
        print()
    tablo.to_csv("hacim_backtest_sonuc.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
