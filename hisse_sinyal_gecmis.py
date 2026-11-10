"""
hisse_sinyal_gecmis.py
Tek bir hissenin geçmişinde sinyallerimizin ne zaman çıktığını gösterir.

Sinyaller (günlük, o günün verisiyle):
  s4 : dar aralık (canlı sistem) -> 15 günlük bant <= %4 VE hacim oranı >= 1.5
  s8 : gevşek dar aralık         -> 15 günlük bant <= %8 VE hacim oranı >= 1.5
  A  : yükselirken hacim         -> 5 günlük getiri > 0 VE hacim oranı >= 3
Hacim oranı = son 5 günün ort. hacmi / ondan önceki 20 günün ort. hacmi.

Çıktı:
  1) Son GOSTER işlem günü: tarih, kapanış, günlük %, bant %, hacim oranı, sinyaller
  2) Veri boyunca her sinyalin çıktığı günler ve sonraki 5/10/20 gün getirisi
     (alış: sinyalden sonraki günün kapanışı, maliyet %0.4 düşülmüş)

Kullanım: python hisse_sinyal_gecmis.py [HISSE]   (varsayılan DOGUB)
Gereken: pip install borsapy pandas numpy
"""
import sys
import numpy as np
import pandas as pd
import borsapy as bp

BANT_DAR = 0.04
BANT_GEVSEK = 0.08
HACIM_MIN = 1.5
HACIM_A = 3.0
UFUKLAR = [5, 10, 20]
MALIYET = 0.004
GOSTER = 30


def kolon_bul(df, ad):
    for c in df.columns:
        if str(c).lower() == ad.lower():
            return c
    raise KeyError(ad)


def tarih_indeksi(df):
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


def veri_al(kod):
    for periyot in ("2y", "1y", "6mo"):
        try:
            df = bp.Ticker(kod).history(period=periyot)
            if df is not None and len(df) >= 60:
                return df
        except Exception:
            continue
    return None


def main():
    kod = (sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].strip() else "DOGUB").upper()
    df = veri_al(kod)
    if df is None:
        print(f"{kod}: veri alınamadı.")
        return

    tarih = tarih_indeksi(df)
    k = df[kolon_bul(df, "Close")].astype(float).reset_index(drop=True)
    h = df[kolon_bul(df, "Volume")].astype(float).reset_index(drop=True)

    onceki20 = h.shift(5).rolling(20).mean()
    oran = h.rolling(5).mean() / onceki20.replace(0, np.nan)
    bant = (k.rolling(15).max() - k.rolling(15).min()) / k.rolling(15).min()
    getiri5 = k / k.shift(5) - 1

    d = pd.DataFrame({
        "tarih": tarih.strftime("%Y-%m-%d") if tarih is not None else range(len(k)),
        "kapanis": k.round(2),
        "gun_%": (k.pct_change() * 100).round(1),
        "bant_%": (bant * 100).round(1),
        "hacim_orani": oran.round(2),
    })
    d["s4"] = ((bant <= BANT_DAR) & (oran >= HACIM_MIN)).map({True: "EVET", False: "-"})
    d["s8"] = ((bant <= BANT_GEVSEK) & (oran >= HACIM_MIN)).map({True: "EVET", False: "-"})
    d["A"] = ((getiri5 > 0) & (oran >= HACIM_A)).map({True: "EVET", False: "-"})
    for u in UFUKLAR:
        d[f"f{u}_%"] = ((k.shift(-1 - u) / k.shift(-1) - 1 - MALIYET) * 100).round(1)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(f"{kod}: {len(d)} işlem günü verisi, {d['tarih'].iloc[0]} -> {d['tarih'].iloc[-1]}\n")

    print(f"=== Son {GOSTER} işlem günü ===")
    print(d[["tarih", "kapanis", "gun_%", "bant_%", "hacim_orani", "s4", "s8", "A"]]
          .tail(GOSTER).to_string(index=False))

    for ad in ("s4", "s8", "A"):
        s = d[d[ad] == "EVET"]
        print(f"\n=== Sinyal {ad}: veri boyunca {len(s)} gün ===")
        if len(s):
            print(s[["tarih", "kapanis", "bant_%", "hacim_orani",
                     "f5_%", "f10_%", "f20_%"]].tail(25).to_string(index=False))
        else:
            print("  Hiç çıkmadı.")

    d.to_csv(f"{kod}_sinyal_gecmis.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
