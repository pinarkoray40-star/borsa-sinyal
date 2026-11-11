"""KAP bedelsiz kesif - 2. adim.

1. adimda 'Sermaye Artirimi - Azaltimi Islemlerine Iliskin Bildirim' konusu (40 gunde 81 kayit)
one cikti. Bu adimda:
1) Son 25 is gununde sadece bu konudaki kayitlari toplar, hisse kodu + tarih + ozetle listeler.
2) 'bedelsiz' gecen ve gecmeyen birkac kaydin METNINI okur (HTML etiketleri temizlenir).
3) Metinde basvuru / onay / tescil / karar tarihi / bedelsiz / bedelli ifadelerinin hangisinin
   gectigini isaretler. Boylece hangi kayit 'SPK basvurusu', hangisi 'onay', hangisi 'tescil'
   anlasilir ve 'haber cikan gun al' testi icin dogru tarihi secebiliriz.

Yaklasik 35 istek, saniyede 1. Yatirim tavsiyesi degildir.
"""
import argparse
import html
import json
import os
import re
import time
from collections import Counter
from datetime import date
from pathlib import Path

import requests

import kap_probe2 as kp
from kap_bedelsiz_probe import is_gunleri

ANAHTARLAR = ("bedelsiz", "bedelli", "spk", "basvuru", "onay", "tescil", "karar tarihi",
              "pay bolun", "iç kaynak", "ic kaynak")


def konu_uygun(r):
    k = kp.norm(r.get("subject", ""))
    return "sermaye artirimi" in k and "azaltimi" in k


def temiz_metin(d):
    parcalar = kp.texts(d)
    t = " ".join(parcalar)
    t = re.sub(r"<[^>]+>", " ", t)
    t = html.unescape(t)
    return " ".join(t.split())


def metin_al(s, idx):
    try:
        r = s.get(f"{kp.BASE}/tr/api/notification/attachment-detail/{idx}", timeout=60,
                  headers={"Referer": f"{kp.BASE}/tr/Bildirim/{idx}", "Accept": "application/json"})
    except Exception as exc:
        return f"HATA {type(exc).__name__}: {exc}"
    if r.status_code != 200:
        return f"HTTP {r.status_code}"
    try:
        return temiz_metin(r.json())
    except Exception:
        return "JSON degil"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kap_bedelsiz_probe2.md")
    ap.add_argument("--son", type=int, default=25)
    ap.add_argument("--oku", type=int, default=10, help="metni okunacak kayit sayisi")
    args = ap.parse_args()

    out = ["# KAP bedelsiz kesif - 2. adim"]
    s = requests.Session()
    s.headers.update({"User-Agent": kp.UA, "Accept-Language": "tr-TR,tr;q=0.9"})
    try:
        s.get(kp.PAGE_URL, timeout=30)
    except Exception as exc:
        out.append(f"- Sayfa istegi hata verdi (devam): {type(exc).__name__}: {exc}")

    kayit = []
    for d in is_gunleri(date.today(), args.son):
        time.sleep(1.0)
        g = d.isoformat()
        data = kp.fetch_window(s, g, g, out)
        if data is None:
            continue
        kayit += [r for r in data if konu_uygun(r)]

    out.append(f"\n## 'Sermaye Artirimi - Azaltimi' kayitlari: {len(kayit)} (son {args.son} is gunu)")
    kodlar = Counter(str(r.get("stockCodes")) for r in kayit)
    out.append("- Ayni hissede birden cok kayit: " + ", ".join(f"{k}×{n}" for k, n in kodlar.most_common(12) if n > 1))
    out.append("\n### Tum kayitlar (tarih · hisse · ozet)")
    for r in kayit[:60]:
        ozet = (r.get("summary") or "")[:120]
        out.append(f"- {r.get('publishDate')} · {r.get('stockCodes')} · {r.get('kapTitle')} · {ozet}")

    # Metni okunacaklar: once 'bedelsiz' gecenler, sonra digerleri
    def bedelsiz_mi(r):
        return "bedelsiz" in kp.norm(json.dumps(r, ensure_ascii=False))

    secim = [r for r in kayit if bedelsiz_mi(r)][: args.oku // 2]
    secim += [r for r in kayit if not bedelsiz_mi(r)][: args.oku - len(secim)]

    out.append(f"\n## Metni okunan {len(secim)} kayit")
    for r in secim:
        idx = r.get("disclosureIndex")
        if not idx:
            continue
        time.sleep(1.0)
        t = metin_al(s, idx)
        n = kp.norm(t)
        bulunan = [a for a in ANAHTARLAR if a in n]
        out.append(f"\n### {r.get('publishDate')} · {r.get('stockCodes')} · no {idx} · liste ozeti: "
                   f"{(r.get('summary') or '')[:100]}")
        out.append(f"- Metinde gecenler: {', '.join(bulunan) if bulunan else 'hicbiri'}")
        out.append(f"- Metin ({len(t)} karakter, ilk 1800): {t[:1800]}")

    out.append("\n> Not: gayriresmi uc nokta, yaklasik 35 istek. Yatirim tavsiyesi degildir.")
    text = "\n".join(out)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
