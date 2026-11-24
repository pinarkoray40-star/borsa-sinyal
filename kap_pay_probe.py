"""KAP pay alim / geri alim kesfi.

Amac: Yonetici / ortak pay alimi ve sirketin kendi payini geri alimi KAP'ta hangi konu
(subject) ve baslik adlariyla geciyor, kayitta kim / kac lot / hangi fiyat bilgisi var?
Bunu ogrenirsek "alim haberi cikan gun al" testini yazabiliriz.

Yapar:
1) Son 25 is gunu + gecmisten 10 is gunu icin GUNLUK pencereler cekilir (saniyede 1 istek).
2) Tum konularin (subject) sayilari yazilir; pay alim / geri alim ile ilgili olanlar ayrica isaretlenir.
3) Her ilgili konudan ornek kayitlar (tarih, hisse, baslik, ozet) listelenir.
4) Farkli konulardan birkac kaydin METNI okunur (alici, adet, fiyat var mi anlasilsin).

kap_probe2.py ve kap_bedelsiz_probe.py'deki ayni gayriresmi uc nokta kullanilir.
Yatirim tavsiyesi degildir.
"""
import argparse
import os
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import requests

import kap_probe2 as kp
from kap_bedelsiz_probe import is_gunleri
from kap_bedelsiz_probe2 import metin_al

ANAHTAR = ("pay alim", "pay satim", "geri alim", "geri alin", "yonetim kontrol",
           "isleme iliskin", "alim satim")


def ilgili(r):
    k = kp.norm(r.get("subject", "")) + " " + kp.norm(r.get("kapTitle", ""))
    return any(a in k for a in ANAHTAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kap_pay_probe.md")
    ap.add_argument("--son", type=int, default=25)
    ap.add_argument("--oku", type=int, default=6, help="metni okunacak kayit sayisi")
    args = ap.parse_args()

    out = ["# KAP pay alim / geri alim kesfi"]
    s = requests.Session()
    s.headers.update({"User-Agent": kp.UA, "Accept-Language": "tr-TR,tr;q=0.9"})
    try:
        s.get(kp.PAGE_URL, timeout=30)
    except Exception as exc:
        out.append(f"- Sayfa istegi hata verdi (devam): {type(exc).__name__}: {exc}")

    gunler = is_gunleri(date.today(), args.son) + is_gunleri(date(2025, 6, 13), 10)

    toplam = 0
    konu_say = Counter()
    konu_sinif = defaultdict(Counter)
    konu_ornek = defaultdict(list)
    for d in gunler:
        time.sleep(1.0)
        g = d.isoformat()
        data = kp.fetch_window(s, g, g, out)
        if data is None:
            continue
        toplam += len(data)
        for r in data:
            konu = str(r.get("subject"))
            konu_say[konu] += 1
            if ilgili(r):
                konu_sinif[konu][str(r.get("disclosureClass"))] += 1
                konu_ornek[konu].append(r)

    out.append(f"\n## Ozet: {len(gunler)} gun, toplam {toplam} kayit")
    out.append("\n### Tum konular (ilk 40)")
    for k, n in konu_say.most_common(40):
        out.append(f"- {n} × {k}")

    out.append("\n### Pay alim / geri alim ile ilgili konular")
    if not konu_ornek:
        out.append("- yok")
    for k in sorted(konu_ornek, key=lambda x: -len(konu_ornek[x])):
        out.append(f"\n#### {k} · {len(konu_ornek[k])} kayit · sinif: {dict(konu_sinif[k])}")
        for r in konu_ornek[k][:8]:
            ozet = (r.get("summary") or "").replace("\n", " ")[:140]
            out.append(f"- {r.get('publishDate')} · {r.get('stockCodes')} · {r.get('kapTitle')} · {ozet}")

    # Metni okunacaklar: her konudan sirayla birer kayit
    secim = []
    konular = sorted(konu_ornek, key=lambda x: -len(konu_ornek[x]))
    i = 0
    while len(secim) < args.oku and konular and i < 10:
        for k in konular:
            if i < len(konu_ornek[k]) and len(secim) < args.oku:
                secim.append(konu_ornek[k][i])
        i += 1

    out.append(f"\n## Metni okunan {len(secim)} kayit")
    for r in secim:
        idx = r.get("disclosureIndex")
        if not idx:
            continue
        time.sleep(1.0)
        t = metin_al(s, idx)
        out.append(f"\n### {r.get('publishDate')} · {r.get('stockCodes')} · no {idx} · konu: {r.get('subject')}")
        out.append(f"- Liste ozeti: {(r.get('summary') or '')[:140]}")
        out.append(f"- Metin ({len(t)} karakter, ilk 1500): {t[:1500]}")

    out.append("\n> Not: gayriresmi uc nokta, yaklasik 45 istek. Yatirim tavsiyesi degildir.")
    text = "\n".join(out)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
