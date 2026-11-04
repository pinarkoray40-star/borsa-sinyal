"""
takip.py - Canlı takip defteri (dar aralık sinyali)

Her gün borsa kapandıktan sonra çalışır:
  1. Tüm BIST'i tarar. Kriter: son 15 günde fiyat bandı <= %4 VE son 5 günün
     hacmi, önceki 20 günün ortalamasının en az 1.5 katı.
  2. Sinyal veren hisse deftere "giris_bekliyor" olarak girer; sinyal günü
     kapanışı kaydedilir. Giriş fiyatı, sinyalden SONRAKİ günün kapanışıdır
     (geçmiş testle aynı kural).
  3. Açık pozisyonlar için her gün kapanışta günlük % ve girişe göre toplam %
     yazılır. Stop yoktur. Her pozisyon 6 ay (183 gün) izlenir.
  4. 10. ve 20. işlem gününün getirisi ayrıca kaydedilir (test vadeleri).

Dosyalar (repoya geri yazılır):
  takip_pozisyon.csv  -> her sinyalin özeti
  takip_gunluk.csv    -> günlük kayıt (tarih, hisse, kapanış, günlük %, toplam %)

Rapor GitHub'daki çalıştırma sayfasının en üstünde (Summary) görünür.
Borsa açıkken (hafta içi 10:00-18:25) çalıştırılırsa gün sonu verisi olmadığı
için durur. FORCE=1 ile zorlanabilir.

Gereken: pip install borsapy pandas numpy
"""
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import borsapy as bp

BANT_MAX = 0.04
HACIM_MIN = 1.5
COOLDOWN_GUN = 14      # aynı hissede bu kadar gün yeni sinyal yazılmaz
TAKIP_GUN = 183        # yaklaşık 6 ay
POZ = "takip_pozisyon.csv"
GUN = "takip_gunluk.csv"

POZ_KOLON = ["hisse", "sinyal_tarihi", "sinyal_fiyati", "bant_%", "hacim_orani",
             "zirveye_oran_%", "durum", "giris_tarihi", "giris_fiyati",
             "son_tarih", "son_fiyat", "toplam_%", "gun_sayisi", "g10_%", "g20_%"]
GUN_KOLON = ["tarih", "hisse", "kapanis", "gunluk_%", "toplam_%"]


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


def tum_hisseler():
    df = bp.companies()
    for c in df.columns:
        if str(c).lower() in ("ticker", "symbol", "kod", "code"):
            return [str(x).strip().upper() for x in df[c].dropna()]
    return [str(x).strip().upper() for x in df.iloc[:, 0].dropna()]


def getir(kod):
    try:
        df = bp.Ticker(kod).history(period="6mo")
        if df is None or len(df) < 65:
            return None
        dt = tarihler(df)
        if dt is None:
            return None
        k = pd.Series(df[kolon_bul(df, "Close")].astype(float).values, index=dt)
        h = pd.Series(df[kolon_bul(df, "Volume")].astype(float).values, index=dt)
        k = k[~k.index.duplicated(keep="last")]
        h = h[~h.index.duplicated(keep="last")]
        return kod, k, h
    except Exception:
        return None


def sinyal_hesapla(k, h):
    son5 = h.iloc[-5:].mean()
    onceki20 = h.iloc[-25:-5].mean()
    if not np.isfinite(son5) or not np.isfinite(onceki20) or onceki20 <= 0:
        return None
    son15 = k.iloc[-15:]
    bant = (son15.max() - son15.min()) / son15.min()
    zirve = k.iloc[-1] / k.iloc[-60:].max()
    return {"oran": son5 / onceki20, "bant": bant, "zirve": zirve}


def tarih_yaz(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def bos(x):
    return x is None or (isinstance(x, float) and np.isnan(x))


def pozisyonu_guncelle(p, k, gunluk_set, yeni_gunluk):
    """Tek pozisyonun girişini, günlük kayıtlarını ve özetini günceller."""
    if p["durum"] == "giris_bekliyor":
        sonra = k[k.index > pd.Timestamp(p["sinyal_tarihi"])]
        if len(sonra) == 0:
            return
        p["giris_tarihi"] = tarih_yaz(sonra.index[0])
        p["giris_fiyati"] = float(sonra.iloc[0])
        p["durum"] = "acik"

    gt = pd.Timestamp(p["giris_tarihi"])
    gf = float(p["giris_fiyati"])
    seri = k[k.index >= gt]
    onceki = k.shift(1)
    for d, c in seri.items():
        anahtar = (tarih_yaz(d), p["hisse"])
        if anahtar in gunluk_set:
            continue
        pr = onceki.loc[d]
        gp = (c / pr - 1) * 100 if np.isfinite(pr) and pr > 0 else np.nan
        yeni_gunluk.append({"tarih": tarih_yaz(d), "hisse": p["hisse"],
                            "kapanis": round(float(c), 4),
                            "gunluk_%": round(gp, 2) if np.isfinite(gp) else np.nan,
                            "toplam_%": round((c / gf - 1) * 100, 2)})
        gunluk_set.add(anahtar)

    p["son_tarih"] = tarih_yaz(seri.index[-1])
    p["son_fiyat"] = round(float(seri.iloc[-1]), 4)
    p["toplam_%"] = round((seri.iloc[-1] / gf - 1) * 100, 2)
    p["gun_sayisi"] = len(seri) - 1
    if len(seri) > 10:
        p["g10_%"] = round((seri.iloc[10] / gf - 1) * 100, 2)
    if len(seri) > 20:
        p["g20_%"] = round((seri.iloc[20] / gf - 1) * 100, 2)
    if (seri.index[-1] - gt).days >= TAKIP_GUN:
        p["durum"] = "kapandi"


def rapor_olustur(kayitlar, yeni_sinyal, veriler, son_gun):
    L = []
    L.append(f"# Takip defteri - {tarih_yaz(son_gun)}\n")
    L.append("Kural: 15 günlük bant <= %4 ve hacim >= 1.5 kat. Stop yok.\n")

    L.append(f"## Yeni sinyaller ({len(yeni_sinyal)})")
    if yeni_sinyal:
        L.append("| Hisse | Sinyal fiyatı | Bant % | Hacim oranı | Zirveye oran % |")
        L.append("|---|---|---|---|---|")
        for s in yeni_sinyal:
            L.append(f"| {s['hisse']} | {s['sinyal_fiyati']:.2f} | {s['bant_%']:.1f} "
                     f"| {s['hacim_orani']:.2f} | {s['zirveye_oran_%']:.0f} |")
        L.append("\nGiriş fiyatı yarının kapanışı olacak.\n")
    else:
        L.append("Bugün yeni sinyal yok.\n")

    bekleyen = [p for p in kayitlar if p["durum"] == "giris_bekliyor"]
    acik = [p for p in kayitlar if p["durum"] == "acik"]
    L.append(f"## Açık pozisyonlar ({len(acik)})")
    if acik:
        L.append("| Hisse | Giriş | Giriş fiyatı | Son | Bugün % | Toplam % | Gün | Durum |")
        L.append("|---|---|---|---|---|---|---|---|")
        for p in sorted(acik, key=lambda x: -float(x["toplam_%"])):
            k = veriler.get(p["hisse"], (None,))[0]
            bugun = ""
            if k is not None and len(k) > 1:
                bugun = f"{(k.iloc[-1] / k.iloc[-2] - 1) * 100:+.1f}"
            t = float(p["toplam_%"])
            durum = "🟢 KÂR" if t > 0 else ("🔴 ZARAR" if t < 0 else "⚪ başabaş")
            L.append(f"| {p['hisse']} | {p['giris_tarihi']} | {float(p['giris_fiyati']):.2f} "
                     f"| {float(p['son_fiyat']):.2f} | {bugun} | {t:+.1f} "
                     f"| {int(p['gun_sayisi'])} | {durum} |")
        top = [float(p["toplam_%"]) for p in acik]
        L.append(f"\nOrtalama toplam: {np.mean(top):+.2f}% | "
                 f"Kârda: {sum(t > 0 for t in top)}/{len(top)}\n")
    else:
        L.append("Henüz açık pozisyon yok.\n")

    if bekleyen:
        L.append(f"## Giriş bekleyen ({len(bekleyen)})")
        L.append(", ".join(f"{p['hisse']} ({p['sinyal_tarihi']})" for p in bekleyen) + "\n")

    hepsi = [p for p in kayitlar if p["durum"] in ("acik", "kapandi")]
    g10 = [float(p["g10_%"]) for p in hepsi if not bos(p.get("g10_%"))]
    g20 = [float(p["g20_%"]) for p in hepsi if not bos(p.get("g20_%"))]
    L.append("## Vade özeti (testle karşılaştırmak için)")
    if g10:
        L.append(f"- 10. gün: {len(g10)} işlem, ort. {np.mean(g10):+.2f}%, "
                 f"kazanma %{np.mean([x > 0 for x in g10]) * 100:.0f}")
    else:
        L.append("- 10. gün: henüz dolan yok")
    if g20:
        L.append(f"- 20. gün: {len(g20)} işlem, ort. {np.mean(g20):+.2f}%, "
                 f"kazanma %{np.mean([x > 0 for x in g20]) * 100:.0f}")
    else:
        L.append("- 20. gün: henüz dolan yok")
    L.append("\nTest beklentisi (geçmiş): 10 günde ort. +1.95%, kazanma %60; "
             "20 günde ort. +2.95%, kazanma %59.")
    return "\n".join(L)


def main():
    simdi = datetime.now(ZoneInfo("Europe/Istanbul"))
    dk = simdi.hour * 60 + simdi.minute
    if os.environ.get("FORCE") != "1" and simdi.weekday() < 5 and 600 <= dk < 18 * 60 + 25:
        print("Borsa açık, gün sonu verisi henüz yok. 18:30'dan sonra çalıştır "
              "(zorlamak için FORCE=1).")
        return

    if os.path.exists(POZ):
        poz = pd.read_csv(POZ, dtype=object)
        for c in POZ_KOLON:
            if c not in poz.columns:
                poz[c] = np.nan
        kayitlar = poz[POZ_KOLON].astype(object).where(poz[POZ_KOLON].notna(), None).to_dict("records")
    else:
        kayitlar = []
    gunluk = pd.read_csv(GUN) if os.path.exists(GUN) else pd.DataFrame(columns=GUN_KOLON)
    gunluk_set = set(zip(gunluk["tarih"].astype(str), gunluk["hisse"].astype(str)))

    kodlar = tum_hisseler()
    print(f"{len(kodlar)} hisse taranıyor...")
    with ThreadPoolExecutor(max_workers=4) as ex:
        veriler = {r[0]: (r[1], r[2]) for r in ex.map(getir, kodlar) if r}
    if not veriler:
        print("Veri alınamadı.")
        return
    son_gun = max(v[0].index[-1] for v in veriler.values())
    print(f"Veri alınan hisse: {len(veriler)} | Son işlem günü: {tarih_yaz(son_gun)}")

    # 1) Mevcut pozisyonları güncelle
    yeni_gunluk = []
    for p in kayitlar:
        if p["durum"] == "kapandi":
            continue
        v = veriler.get(p["hisse"])
        if v is None:
            continue
        pozisyonu_guncelle(p, v[0], gunluk_set, yeni_gunluk)

    # 2) Yeni sinyalleri bul
    yeni_sinyal = []
    for kod, (k, h) in veriler.items():
        if k.index[-1] != son_gun:
            continue
        s = sinyal_hesapla(k, h)
        if s is None or not (s["bant"] <= BANT_MAX and s["oran"] >= HACIM_MIN):
            continue
        if any((son_gun - pd.Timestamp(p["sinyal_tarihi"])).days < COOLDOWN_GUN
               for p in kayitlar if p["hisse"] == kod):
            continue
        kayit = {c: None for c in POZ_KOLON}
        kayit.update({"hisse": kod, "sinyal_tarihi": tarih_yaz(son_gun),
                      "sinyal_fiyati": round(float(k.iloc[-1]), 4),
                      "bant_%": round(s["bant"] * 100, 2),
                      "hacim_orani": round(s["oran"], 2),
                      "zirveye_oran_%": round(s["zirve"] * 100, 1),
                      "durum": "giris_bekliyor"})
        kayitlar.append(kayit)
        yeni_sinyal.append(kayit)

    # 3) Kaydet
    pd.DataFrame(kayitlar, columns=POZ_KOLON).to_csv(POZ, index=False, encoding="utf-8-sig")
    if yeni_gunluk:
        gunluk = pd.concat([gunluk, pd.DataFrame(yeni_gunluk, columns=GUN_KOLON)],
                           ignore_index=True)
    gunluk.to_csv(GUN, index=False, encoding="utf-8-sig")

    # 4) Rapor
    rapor = rapor_olustur(kayitlar, yeni_sinyal, veriler, son_gun)
    print("\n" + rapor)
    ozet = os.environ.get("GITHUB_STEP_SUMMARY")
    if ozet:
        with open(ozet, "a", encoding="utf-8") as f:
            f.write(rapor + "\n")


if __name__ == "__main__":
    main()
