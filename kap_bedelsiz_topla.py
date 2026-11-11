"""KAP'tan 'Sermaye Artirimi - Azaltimi Islemlerine Iliskin Bildirim' kayitlarini toplar.

Gunluk pencerelerle (saniyede 1 istek) gecmise dogru tarar ve sadece bu konudaki kayitlari
kap_bedelsiz_kayit.csv dosyasina yazar. Bedelsiz yonetim kurulu karari / SPK basvurusu /
SPK onayi / tescil bildirimleri bu konunun altindadir; asama, baslik (summary) alanindan ayrilir.

Kullanim: python kap_bedelsiz_topla.py [--bas 2023-11-01] [--sure-dk 45]
Yatirim tavsiyesi degildir.
"""
import argparse
import csv
import functools
import time
from datetime import date, timedelta

import requests

import kap_probe2 as kp

print = functools.partial(print, flush=True)


def konu_uygun(r):
    k = kp.norm(r.get("subject", ""))
    return "sermaye artirimi" in k and "azaltimi" in k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bas", default="2023-11-01")
    ap.add_argument("--out", default="kap_bedelsiz_kayit.csv")
    ap.add_argument("--sure-dk", type=int, default=45)
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
    with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["yayin", "kodlar", "ad", "baslik", "konu", "no"])
        for n, g in enumerate(gunler, 1):
            if time.time() - t0 > args.sure_dk * 60:
                print(f"SURE LIMITI: {n - 1}/{len(gunler)} gun tarandi, en eski tarih {gunler[n - 2]}")
                break
            time.sleep(1.0)
            iso = g.isoformat()
            junk = []
            data = kp.fetch_window(s, iso, iso, junk)
            if data is None:
                time.sleep(3.0)
                data = kp.fetch_window(s, iso, iso, junk)
            if data is None:
                hata += 1
                print(f"  {iso}: alinamadi {junk[-1:] if junk else ''}")
                continue
            toplam += len(data)
            for r in data:
                if konu_uygun(r):
                    w.writerow([r.get("publishDate", ""), r.get("stockCodes", ""), r.get("kapTitle", ""),
                                (r.get("summary") or "").replace("\n", " "), r.get("subject", ""),
                                r.get("disclosureIndex", "")])
                    yazilan += 1
            if n % 50 == 0:
                print(f"  {n}/{len(gunler)} gun ({iso}), toplam {toplam} kayit, konuya uyan {yazilan}, "
                      f"hata {hata}, gecen {int((time.time() - t0) / 60)} dk")
    print(f"Bitti: {toplam} kayit tarandi, konuya uyan {yazilan}, alinamayan gun {hata} -> {args.out}")


if __name__ == "__main__":
    main()
