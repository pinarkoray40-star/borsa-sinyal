"""
takip_dip.py - Canlı takip defteri: Dip OBV (ayar seti 2)

Mevcut dar aralık defterine (takip.py) DOKUNMAZ, onun kurallarını aynen kullanır:
  - Sinyal günü kapanışı kaydedilir, giriş fiyatı ertesi günün kapanışıdır.
  - Stop yok, her pozisyon 6 ay (183 gün) izlenir, günlük % ve toplam % yazılır.
  - 10. ve 20. gün getirisi ayrıca kaydedilir.

Sinyal mantığı dip_obv_2.py ile aynıdır (RelVol 14, OBV 98, NetFlow 41, RSI 3, Vol 1.1).

Dosyalar (repoya geri yazılır, dar aralık defteriyle karışmaz):
  takip_dip_pozisyon.csv  -> her sinyalin özeti
  takip_dip_gunluk.csv    -> günlük kayıt

Borsa açıkken (hafta içi 10:00-18:25) çalıştırılırsa durur. FORCE=1 ile zorlanır.
Gereken: pip install borsapy pandas numpy
"""
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import borsapy as bp

import takip as t          # mevcut defterin yardımcı fonksiyonları (değiştirilmez)
import dip_obv_2 as dip    # sinyal mantığı

POZ = "takip_dip_pozisyon.csv"
GUN = "takip_dip_gunluk.csv"

POZ_KOLON = ["hisse", "sinyal_tarihi", "sinyal_fiyati", "skor", "hacim_orani",
             "zirveden_dusus_%", "durum", "giris_tarihi", "giris_fiyati",
             "son_tarih", "son_fiyat", "toplam_%", "gun_sayisi", "g10_%", "g20_%"]
GUN_KOLON = ["tarih", "hisse", "kapanis", "gunluk_%", "toplam_%"]


def getir(kod, xclose):
    """Hissenin verisini çeker; (kod, kapanış serisi, sinyal bilgisi veya None) döner."""
    try:
        df = bp.Ticker(kod).history(period="2y")
        if df is None or len(df) < 150:
            return None
        d = dip.ohlcv(df)
        if len(d) < 150:
            return None
        sinyal = sinyal_bul(d, xclose)
        return kod, d["close"], sinyal
    except Exception:
        return None


def sinyal_bul(d, xclose):
    """Son barda Dip OBV (ayar seti 2) AL sinyali varsa bilgi sözlüğü döner."""
    r = dip.hesapla(d, xclose)
    s = dip.cooldown_uygula(r["core"])
    if not bool(s.iloc[-1]):
        return None
    return {"skor": int(r["score"].iloc[-1]),
            "hacim": float(r["rel_vol"].iloc[-1]),
            "dusus": float(r["dd"].iloc[-1])}


def rapor_olustur(kayitlar, yeni_sinyal, veriler, son_gun):
    L = [f"# Dip OBV takip defteri (ayar seti 2) - {t.tarih_yaz(son_gun)}\n",
         "Kural: Dip OBV AL sinyali. Giriş ertesi gün kapanışı, stop yok.\n",
         f"## Yeni sinyaller ({len(yeni_sinyal)})"]
    if yeni_sinyal:
        L += ["| Hisse | Sinyal fiyatı | Skor | Hacim oranı | Zirveden düşüş % |",
              "|---|---|---|---|---|"]
        for s in yeni_sinyal:
            L.append(f"| {s['hisse']} | {s['sinyal_fiyati']:.2f} | {s['skor']} "
                     f"| {s['hacim_orani']:.2f} | {s['zirveden_dusus_%']:.1f} |")
        L.append("\nGiriş fiyatı yarının kapanışı olacak.\n")
    else:
        L.append("Bugün yeni sinyal yok.\n")

    bekleyen = [p for p in kayitlar if p["durum"] == "giris_bekliyor"]
    acik = [p for p in kayitlar if p["durum"] == "acik"]
    L.append(f"## Açık pozisyonlar ({len(acik)})")
    if acik:
        L += ["| Hisse | Giriş | Giriş fiyatı | Son | Bugün % | Toplam % | Gün | Durum |",
              "|---|---|---|---|---|---|---|---|"]
        for p in sorted(acik, key=lambda x: -float(x["toplam_%"])):
            k = veriler.get(p["hisse"], (None,))[0]
            bugun = ""
            if k is not None and len(k) > 1:
                bugun = f"{(k.iloc[-1] / k.iloc[-2] - 1) * 100:+.1f}"
            tp = float(p["toplam_%"])
            durum = "🟢 KÂR" if tp > 0 else ("🔴 ZARAR" if tp < 0 else "⚪ başabaş")
            L.append(f"| {p['hisse']} | {p['giris_tarihi']} | {float(p['giris_fiyati']):.2f} "
                     f"| {float(p['son_fiyat']):.2f} | {bugun} | {tp:+.1f} "
                     f"| {int(p['gun_sayisi'])} | {durum} |")
        top = [float(p["toplam_%"]) for p in acik]
        L.append(f"\nOrtalama toplam: {np.mean(top):+.2f}% | Kârda: {sum(x > 0 for x in top)}/{len(top)}\n")
    else:
        L.append("Henüz açık pozisyon yok.\n")

    if bekleyen:
        L.append(f"## Giriş bekleyen ({len(bekleyen)})")
        L.append(", ".join(f"{p['hisse']} ({p['sinyal_tarihi']})" for p in bekleyen) + "\n")

    hepsi = [p for p in kayitlar if p["durum"] in ("acik", "kapandi")]
    g10 = [float(p["g10_%"]) for p in hepsi if not t.bos(p.get("g10_%"))]
    g20 = [float(p["g20_%"]) for p in hepsi if not t.bos(p.get("g20_%"))]
    L.append("## Vade özeti")
    for ad, g in (("10. gün", g10), ("20. gün", g20)):
        if g:
            L.append(f"- {ad}: {len(g)} işlem, ort. {np.mean(g):+.2f}%, "
                     f"kazanma %{np.mean([x > 0 for x in g]) * 100:.0f}")
        else:
            L.append(f"- {ad}: henüz dolan yok")
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

    xclose = dip.xu100_al()
    if xclose is None:
        print("UYARI: XU100 verisi alınamadı, piyasa ve göreceli güç filtreleri devre dışı!")

    kodlar = t.tum_hisseler()
    print(f"{len(kodlar)} hisse taranıyor...")
    with ThreadPoolExecutor(max_workers=4) as ex:
        sonuc = [r for r in ex.map(lambda k: getir(k, xclose), kodlar) if r]
    veriler = {kod: (k, s) for kod, k, s in sonuc}
    if not veriler:
        print("Veri alınamadı.")
        return
    son_gun = max(v[0].index[-1] for v in veriler.values())
    print(f"Veri alınan hisse: {len(veriler)} | Son işlem günü: {t.tarih_yaz(son_gun)}")

    # 1) Mevcut pozisyonları güncelle (mevcut defterin fonksiyonuyla)
    yeni_gunluk = []
    for p in kayitlar:
        if p["durum"] == "kapandi":
            continue
        v = veriler.get(p["hisse"])
        if v is None:
            continue
        t.pozisyonu_guncelle(p, v[0], gunluk_set, yeni_gunluk)

    # 2) Yeni sinyaller
    yeni_sinyal = []
    for kod, (k, s) in veriler.items():
        if s is None or k.index[-1] != son_gun:
            continue
        if any((son_gun - pd.Timestamp(p["sinyal_tarihi"])).days < t.COOLDOWN_GUN
               for p in kayitlar if p["hisse"] == kod):
            continue
        kayit = {c: None for c in POZ_KOLON}
        kayit.update({"hisse": kod, "sinyal_tarihi": t.tarih_yaz(son_gun),
                      "sinyal_fiyati": round(float(k.iloc[-1]), 4),
                      "skor": s["skor"], "hacim_orani": round(s["hacim"], 2),
                      "zirveden_dusus_%": round(s["dusus"], 1),
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
