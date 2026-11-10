"""
buyuk_yukselis_analiz.py
Tüm BIST'te geçmişte "aşırı yükselen" hisseleri bulur ve yükselişten ÖNCE
hangi özelliklerin öne çıktığını ölçer.

Tanım: Bir gün t için, ertesi günün kapanışından (alış) sonraki UFUK işlem günü
içinde fiyat en az ESIK kat olduysa o gün "yükseliş öncesi" sayılır.
(varsayılan: 60 işlem günü içinde 2 kat = +%100)

Her gün için sadece o güne kadar bilinen özellikler hesaplanır:
  bant15/bant30  : son 15/30 günde (max-min)/min, %
  hacim_oran     : son 5 gün ort. hacim / önceki 20 gün ort. hacim
  mom20/mom60    : 20/60 günlük getiri, %
  rs20           : 20 günlük getiri - BIST100 20 günlük getiri, puan
  zirve120       : kapanış / 120 günlük zirve, %
  dip120         : kapanış / 120 günlük dip, %
  vol20          : son 20 günün günlük getiri std'si, %
  tavan10        : son 10 günde >= %9.5 yükselen gün sayısı
  hacim_tl       : son 20 gün ort. işlem hacmi (milyon TL)
  fiyat          : kapanış, TL

Çıktı:
  1) Kaç yükseliş olayı, kaç hissede, taban oran
  2) Her özellik için 5 dilim: o dilimdeki günlerin yüzde kaçı yükseliş öncesiydi
  3) Yükseliş öncesi günlerin ve diğer günlerin medyan özellikleri
  4) En büyük yükselişler ve ismini verdiğin hisselerin yükselişleri

Kullanım: python buyuk_yukselis_analiz.py [HISSE ...]
Gereken: pip install borsapy pandas numpy
"""
import sys
import numpy as np
import pandas as pd
import borsapy as bp
from concurrent.futures import ThreadPoolExecutor

UFUK = 60
ESIK = 2.0
KUME_ARASI = 5          # aynı yükselişi tek olay saymak için gün farkı
ISIMLI = ["ANELE", "KTLEV", "GUNDG", "ASTOR", "SELEC"]

OZELLIKLER = ["bant15", "bant30", "hacim_oran", "mom20", "mom60", "rs20",
              "zirve120", "dip120", "vol20", "tavan10", "hacim_tl", "fiyat"]


def kolon_bul(df, ad):
    for c in df.columns:
        if str(c).lower() == ad.lower():
            return c
    raise KeyError(ad)


def tarihler(df):
    idx = None
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    else:
        for c in df.columns:
            if str(c).lower() in ("date", "datetime", "tarih", "time"):
                idx = pd.DatetimeIndex(pd.to_datetime(df[c]))
                break
    if idx is None:
        return None
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def hisseleri_al():
    if len(sys.argv) > 1:
        return [t.upper() for t in sys.argv[1:]]
    df = bp.companies()
    for c in df.columns:
        if str(c).lower() in ("ticker", "symbol", "kod", "code"):
            return [str(x).strip().upper() for x in df[c].dropna()]
    return [str(x).strip().upper() for x in df.iloc[:, 0].dropna()]


def veri_al(kod):
    for periyot in ("2y", "1y"):
        try:
            df = bp.Ticker(kod).history(period=periyot)
            if df is not None and len(df) >= 200:
                return df
        except Exception:
            continue
    return None


def endeks_al():
    denemeler = [lambda: bp.Index("XU100").history(period="2y"),
                 lambda: bp.Ticker("XU100").history(period="2y")]
    for dene in denemeler:
        try:
            df = dene()
            if df is None or len(df) < 100:
                continue
            idx = tarihler(df)
            if idx is None:
                continue
            s = pd.Series(df[kolon_bul(df, "Close")].astype(float).values, index=idx)
            return s[~s.index.duplicated(keep="last")].sort_index()
        except Exception:
            continue
    return None


def hazirla(kod, df, endeks):
    k = df[kolon_bul(df, "Close")].astype(float).reset_index(drop=True)
    h = df[kolon_bul(df, "Volume")].astype(float).reset_index(drop=True)
    dt = tarihler(df)
    getiri = k.pct_change()

    d = pd.DataFrame({"hisse": kod})
    d["tarih"] = dt.strftime("%Y-%m-%d") if dt is not None else range(len(k))
    d["kapanis"] = k
    d["bant15"] = (k.rolling(15).max() - k.rolling(15).min()) / k.rolling(15).min() * 100
    d["bant30"] = (k.rolling(30).max() - k.rolling(30).min()) / k.rolling(30).min() * 100
    onceki20 = h.shift(5).rolling(20).mean()
    d["hacim_oran"] = h.rolling(5).mean() / onceki20.replace(0, np.nan)
    d["mom20"] = (k / k.shift(20) - 1) * 100
    d["mom60"] = (k / k.shift(60) - 1) * 100
    d["rs20"] = np.nan
    if endeks is not None and dt is not None:
        e = pd.Series(endeks.reindex(dt, method="ffill").values)
        d["rs20"] = (k / k.shift(20) - 1) * 100 - (e / e.shift(20) - 1) * 100
    d["zirve120"] = k / k.rolling(120).max() * 100
    d["dip120"] = k / k.rolling(120).min() * 100
    d["vol20"] = getiri.rolling(20).std() * 100
    d["tavan10"] = (getiri >= 0.095).rolling(10).sum()
    d["hacim_tl"] = (k * h).rolling(20).mean() / 1e6
    d["fiyat"] = k

    # Sonraki UFUK günde (alıştan sonra) ulaşılan en yüksek kapanış / alış
    ileri_max = k[::-1].rolling(UFUK).max()[::-1].shift(-2)
    alis = k.shift(-1)
    d["oran"] = ileri_max / alis

    cekirdek = [c for c in OZELLIKLER if c != "rs20"]
    d = d.dropna(subset=cekirdek).reset_index(drop=True)

    # Olaylar: yükseliş öncesi günleri kümele, her kümeden en iyi günü al
    olaylar = []
    q = np.flatnonzero((d["oran"] >= ESIK).values)
    if len(q):
        kume = [q[0]]
        for i in q[1:]:
            if i - kume[-1] <= KUME_ARASI:
                kume.append(i)
            else:
                olaylar.append(kume)
                kume = [i]
        olaylar.append(kume)
    ol = []
    for kume in olaylar:
        j = max(kume, key=lambda x: d["oran"].iloc[x])
        ol.append({"hisse": kod, "tarih": d["tarih"].iloc[j],
                   "kapanis": round(float(d["kapanis"].iloc[j]), 2),
                   "yukselis_%": round((float(d["oran"].iloc[j]) - 1) * 100)})
    return d, ol


def guvenli(args):
    kod, endeks = args
    try:
        df = veri_al(kod)
        if df is None:
            return None
        return hazirla(kod, df, endeks)
    except Exception:
        return None


def main():
    endeks = endeks_al()
    print("BIST100 verisi:", "alındı" if endeks is not None else "ALINAMADI (rs20 boş)")
    kodlar = hisseleri_al()
    with ThreadPoolExecutor(max_workers=4) as ex:
        sonuc = [r for r in ex.map(guvenli, [(k, endeks) for k in kodlar]) if r]
    if not sonuc:
        print("Veri yok.")
        return

    havuz = pd.concat([r[0] for r in sonuc], ignore_index=True)
    olaylar = pd.DataFrame([o for r in sonuc for o in r[1]])
    etiketli = havuz[havuz["oran"].notna()].copy()
    etiketli["y"] = (etiketli["oran"] >= ESIK).astype(int)
    taban = etiketli["y"].mean() * 100

    print(f"\nTaranan hisse: {len(sonuc)} / {len(kodlar)}")
    print(f"Tanım: alıştan sonra {UFUK} işlem günü içinde en az {ESIK:g} kat")
    print(f"Yükseliş olayı: {len(olaylar)} (kaç hissede: {olaylar['hisse'].nunique() if len(olaylar) else 0})")
    print(f"Taban oran (rastgele bir günün yükseliş öncesi olma ihtimali): %{taban:.1f}\n")

    print("=== Özellik dilimleri: o dilimdeki günlerin yüzde kaçı yükseliş öncesiydi ===")
    print(f"(taban oran %{taban:.1f}; sütunlar düşükten yükseğe 5 dilim)")
    for f in OZELLIKLER:
        s = etiketli[[f, "y"]].dropna()
        if len(s) < 500:
            continue
        kenar = s[f].quantile([0.2, 0.4, 0.6, 0.8]).round(1).tolist()
        dilim = pd.cut(s[f], [-np.inf] + kenar + [np.inf], duplicates="drop")
        p = s.groupby(dilim, observed=True)["y"].mean() * 100
        print(f"{f:11s} kenar: {kenar}")
        print(f"{'':11s} %: {[round(x, 1) for x in p.tolist()]}")

    print("\n=== Medyan özellikler: yükseliş öncesi günler vs diğer günler ===")
    med = etiketli.groupby("y")[OZELLIKLER].median().round(1).T
    med.columns = ["diger", "yukselis_oncesi"]
    print(med.to_string())

    if len(olaylar):
        print("\n=== En büyük 20 yükseliş ===")
        print(olaylar.sort_values("yukselis_%", ascending=False).head(20).to_string(index=False))
        print("\n=== İsmini verdiğin hisseler ===")
        sec = olaylar[olaylar["hisse"].isin(ISIMLI)]
        print(sec.to_string(index=False) if len(sec) else "Bu hisselerde olay bulunamadı.")
        olaylar.to_csv("buyuk_yukselis_olaylar.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
