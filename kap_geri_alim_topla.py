"""KAP'tan pay geri alim kayitlarini toplar.

Gunluk pencerelerle gecmise dogru tarar ve su konulardaki kayitlari
kap_geri_alim_kayit.csv dosyasina yazar:
  - "Paylarin Geri Alinmasina Iliskin Bildirim" (program baslangici, gunluk geri alim
    islemleri, program sonlandirilmasi)
  - "Geri Alinan Paylarin Elden Cikarilmasi" (geri alinan payin satisi)
Asama (program / gunluk islem / sonlandirma / elden cikarma) baslik (summary) alanindan
sonraki adimda ayrilir; burada sadece ham kayit toplanir ve ozet sayilar yazilir.

HTTP 429 (hiz siniri) gelirse bekleyip ayni gunu tekrar dener; alinamayan gunleri
kap_geri_alim_alinamayan.txt dosyasina yazar.

Kullanim: python kap_geri_alim_topla.py [--bas 2023-11-01] [--sure-dk 130]
Yatirim tavsiyesi degildir.
"""
import argparse
import csv
import functools
import re
import time
from collections import Counter
from datetime import date, timedelta

import requests

import kap_probe2 as kp

print = functools.partial(print, flush=True)


def konu_uygun(r):
    k = kp.norm(r.get("subject", ""))
    return ("geri alinmasina" in k) or ("geri alinan paylarin" in k)


def tur(r):
    """Ham siniflama (sadece ozet sayilar icin): sonraki adimda netlestirilecek."""
    k = kp.norm(r.get("subject", ""))
    s = kp.norm(r.get("summary", "") + " " + r.get("kapTitle", ""))
    if "elden cikar" in k:
        return "elden cikarma"
    if "sonlandir" in s:
        return "program sonlandirma"
    if re.search(r"\d{2}\.\d{2}\.\d{4}\s+tarihli", s) or "geri alim islemleri" in s:
        return "gunluk islem"
    if "program" in s:
        return "program"
    return "diger"


def gun_al(s, iso, durum):
    """Bir gunun kayitlarini al. HTTP 429'da bekleyip tekrar dener, bekleme suresini kalici artirir."""
    for deneme in range(1, 7):
        junk = []
        data = kp.fetch_window(s, iso, iso, junk)
        if data is not None:
            return data, ""
        son = junk[-1] if junk else ""
        if "429" in son:
            durum["bekle"] = min(durum["bekle"] + 0.5, 4.0)
            durum["n429"] += 1
            time.sleep(30 * deneme)
        else:
            time.sleep(3.0 * deneme)
    return None, son


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bas", default="2023-11-01")
    ap.add_argument("--out", default="kap_geri_alim_kayit.csv")
    ap.add_argument("--sure-dk", type=int, default=130)
    args = ap.parse_args()

    s = requests.Session()
    s.headers.update({"User-Agent": kp.UA, "Accept-Language": "tr-TR,tr;q=0.9"})
    try:
        s.get(kp.PAGE_URL, timeout=30)
    except Exception as exc:  # noqa: BLE001
        print(f"Sayfa istegi hata verdi (devam): {type(exc).__name__}: {exc}")

    bas = date.fromisoformat(args.bas)
    gunler, d = [], date.today()
    while d >= bas:
        if d.weekday() < 5:
            gunler.append(d)
        d -= timedelta(days=1)

    t0 = time.time()
    toplam, hata, yazilan = 0, 0, 0
    turler, ozetler = Counter(), Counter()
    en_eski = None
    durum = {"bekle": 1.5, "n429": 0}
    alinamayan = []
    with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["yayin", "kodlar", "ad", "baslik", "konu", "no", "tur"])
        for n, g in enumerate(gunler, 1):
            if time.time() - t0 > args.sure_dk * 60:
                print(f"SURE LIMITI: {n - 1}/{len(gunler)} gun tarandi, en eski tarih {gunler[n - 2]}")
                break
            time.sleep(durum["bekle"])
            iso = g.isoformat()
            data, son = gun_al(s, iso, durum)
            if data is None:
                hata += 1
                alinamayan.append(iso)
                print(f"  {iso}: alinamadi {son}")
                continue
            toplam += len(data)
            en_eski = iso
            for r in data:
                if konu_uygun(r):
                    t = tur(r)
                    turler[t] += 1
                    ozetler[(r.get("summary") or "").replace("\n", " ")[:70]] += 1
                    w.writerow([r.get("publishDate", ""), r.get("stockCodes", ""), r.get("kapTitle", ""),
                                (r.get("summary") or "").replace("\n", " "), r.get("subject", ""),
                                r.get("disclosureIndex", ""), t])
                    yazilan += 1
            if n % 50 == 0:
                print(f"  {n}/{len(gunler)} gun ({iso}), toplam {toplam} kayit, konuya uyan {yazilan}, "
                      f"hata {hata}, 429 sayisi {durum['n429']}, bekleme {durum['bekle']}s, "
                      f"gecen {int((time.time() - t0) / 60)} dk")
    print(f"\nBitti: {toplam} kayit tarandi, konuya uyan {yazilan}, alinamayan gun {hata}, "
          f"en eski taranan gun {en_eski} -> {args.out}")
    with open("kap_geri_alim_alinamayan.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(alinamayan))
    print(f"429 sayisi {durum['n429']}, son bekleme {durum['bekle']}s, alinamayan gun {len(alinamayan)}")
    if alinamayan:
        print("Alinamayan gunler (ilk 15):", ", ".join(alinamayan[:15]))
    print("\nTur sayilari:")
    for k, v in turler.most_common():
        print(f"  {v} x {k}")
    print("\nEn sik ozetler (ilk 25):")
    for k, v in ozetler.most_common(25):
        print(f"  {v} x {k}")


if __name__ == "__main__":
    main()
