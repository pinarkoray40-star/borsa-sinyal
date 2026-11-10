"""
yakin_aday.py
Dar aralık sinyalinin (hacim_backtest_c.py ile aynı mantık) bugünkü durumunu kontrol eder.

Sinyal: son 15 günde fiyat bandı <= %4 VE son 5 günün ortalama hacmi,
önceki 20 günün ortalamasının en az 1.5 katı.

Çıktı:
  1) Son 5 işlem gününün her biri için: o gün kaç hisse sinyal verirdi
     (sistemin çalışıp çalışmadığını anlamak için)
  2) Bugün sinyal verenler
  3) Şartı kıl payı kaçıranlar (en yakın adaylar) ve neyi kaçırdıkları

Kullanım: python yakin_aday.py [HISSE ...]
Gereken: pip install borsapy pandas numpy
"""
import sys
import numpy as np
import pandas as pd
import borsapy as bp
from concurrent.futures import ThreadPoolExecutor

BANT_MAX = 0.04
HACIM_MIN = 1.5
GERIYE_GUN = 5          # kaç günün sinyal sayısı gösterilsin
ADAY_ESIK = 0.5         # yakınlık skoru bu değerin altındaysa aday sayılır
ADAY_SAYISI = 25


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


def analiz(kod):
    df = bp.Ticker(kod).history(period="6mo")
    if df is None or len(df) < 40:
        return None
    k = df[kolon_bul(df, "Close")].astype(float).reset_index(drop=True)
    h = df[kolon_bul(df, "Volume")].astype(float).reset_index(drop=True)

    onceki20 = h.shift(5).rolling(20).mean()
    oran = h.rolling(5).mean() / onceki20.replace(0, np.nan)
    bant = (k.rolling(15).max() - k.rolling(15).min()) / k.rolling(15).min()
    sinyal = (bant <= BANT_MAX) & (oran >= HACIM_MIN)

    son = len(k) - 1
    if pd.isna(oran.iloc[son]) or pd.isna(bant.iloc[son]):
        return None

    # Son N günün sinyal durumu (en yeni sonda)
    gecmis = [bool(sinyal.iloc[son - i]) for i in range(GERIYE_GUN - 1, -1, -1)]

    b = float(bant.iloc[son])
    o = float(oran.iloc[son])
    bant_asim = max(b / BANT_MAX - 1, 0)
    hacim_eksik = max(1 - o / HACIM_MIN, 0)
    return {
        "hisse": kod,
        "bant_%": round(b * 100, 2),
        "hacim_orani": round(o, 2),
        "sinyal_bugun": bool(sinyal.iloc[son]),
        "uzaklik": round(bant_asim + hacim_eksik, 3),
        "kacirdigi": ("bant geniş " if bant_asim > 0 else "") + ("hacim düşük" if hacim_eksik > 0 else ""),
        "gecmis": gecmis,
    }


def guvenli(kod):
    try:
        return analiz(kod)
    except Exception:
        return None


def main():
    kodlar = hisseleri_al()
    with ThreadPoolExecutor(max_workers=4) as ex:
        sonuc = [r for r in ex.map(guvenli, kodlar) if r]

    if not sonuc:
        print("Sonuç yok (veri alınamadı).")
        return

    print(f"Taranan: {len(sonuc)} / {len(kodlar)} hisse\n")

    print(f"Son {GERIYE_GUN} işlem günü, o gün sinyal verecek hisse sayısı (eskiden yeniye):")
    sayilar = [sum(r["gecmis"][i] for r in sonuc) for i in range(GERIYE_GUN)]
    print("  " + "  ".join(str(s) for s in sayilar) + "\n")

    t = pd.DataFrame(sonuc).drop(columns=["gecmis"])
    sinyalli = t[t.sinyal_bugun].sort_values("hacim_orani", ascending=False)
    print(f"Bugün sinyal veren: {len(sinyalli)}")
    if len(sinyalli):
        print(sinyalli[["hisse", "bant_%", "hacim_orani"]].to_string(index=False))
    print()

    aday = t[(~t.sinyal_bugun) & (t.uzaklik <= ADAY_ESIK)].sort_values("uzaklik").head(ADAY_SAYISI)
    print(f"En yakın adaylar (şartı kıl payı kaçıranlar, eşik: bant <= %{BANT_MAX*100:g}, hacim >= {HACIM_MIN}x):")
    if len(aday):
        pd.set_option("display.width", 200)
        print(aday[["hisse", "bant_%", "hacim_orani", "kacirdigi"]].to_string(index=False))
    else:
        print("  Yakın aday bile yok.")

    print("\nİpucu: bant sütunu 4'ün biraz üstü, hacim 1.5'a yakınsa sistem sağlıklı, eşik sıkı demektir.")
    t.drop(columns=["sinyal_bugun"]).to_csv("yakin_aday_sonuc.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
