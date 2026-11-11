"""KAP bedelsiz duyuru kesfi.

Amac: Bedelsiz sermaye artirimi icin SPK basvurusu / onayi / yonetim kurulu karari
bildirimleri KAP'ta hangi baslik ve konu adlariyla geciyor, hangi gun yayinlaniyor?
Bunu ogrenirsek "haber cikan gun al" testini yazabiliriz.

Yapar:
1) Son ~30 is gunu + gecmisten bir donem icin GUNLUK pencereler cekilir (saniyede 1 istek).
2) Kayitlardan "bedelsiz" / "sermaye art" / "spk" gecenler ayiklanir.
3) Konu (subject) ve baslik (kapTitle) sayilari + ornek kayitlar yazilir.
4) Ilk eslesen birkac bildirimin metni okunur (SPK onayi mi, basvuru mu anlasilsin).

kap_probe2.py'deki ayni gayriresmi uc nokta kullanilir. Yatirim tavsiyesi degildir.
"""
import argparse
import json
import os
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import requests

import kap_probe2 as kp

ANAHTAR = ("bedelsiz", "sermaye art", "sermayenin art", "spk", "sermaye piyasasi kurulu")
GUCLU = ("bedelsiz",)


def is_gunleri(bitis, adet):
    d, out = bitis, []
    while len(out) < adet:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kap_bedelsiz_probe.md")
    ap.add_argument("--son", type=int, default=30, help="son kac is gunu")
    args = ap.parse_args()

    out = ["# KAP bedelsiz duyuru kesfi"]
    s = requests.Session()
    s.headers.update({"User-Agent": kp.UA, "Accept-Language": "tr-TR,tr;q=0.9"})
    try:
        s.get(kp.PAGE_URL, timeout=30)
    except Exception as exc:
        out.append(f"- Sayfa istegi hata verdi (devam): {type(exc).__name__}: {exc}")

    gunler = is_gunleri(date.today(), args.son)
    gunler += is_gunleri(date(2025, 6, 13), 10)   # gecmisten bir donem (kiyas icin)

    eslesen, konu_say, baslik_say, guclu = [], Counter(), Counter(), []
    toplam = 0
    for d in gunler:
        time.sleep(1.0)
        g = d.isoformat()
        data = kp.fetch_window(s, g, g, out)
        if data is None:
            continue
        toplam += len(data)
        bul = [r for r in data
               if any(k in kp.norm(json.dumps(r, ensure_ascii=False)) for k in ANAHTAR)]
        gl = [r for r in bul if any(k in kp.norm(json.dumps(r, ensure_ascii=False)) for k in GUCLU)]
        out.append(f"- {g}: {len(data)} kayit · anahtar kelime {len(bul)} · 'bedelsiz' gecen {len(gl)}")
        for r in bul:
            konu_say[str(r.get("subject"))] += 1
            baslik_say[str(r.get("kapTitle"))] += 1
        eslesen += bul
        guclu += gl

    out.append(f"\n## Ozet: {len(gunler)} gun, toplam {toplam} kayit, anahtar kelimeli {len(eslesen)}, 'bedelsiz' gecen {len(guclu)}")
    out.append("\n### Konu (subject) sayilari")
    for k, n in konu_say.most_common(15):
        out.append(f"- {n} × {k}")
    out.append("\n### Baslik (kapTitle) sayilari")
    for k, n in baslik_say.most_common(15):
        out.append(f"- {n} × {k}")

    out.append("\n### 'bedelsiz' gecen ornek kayitlar")
    for r in guclu[:25]:
        out.append(f"- {kp.compact(r, 700)}")
    if not guclu:
        out.append("- yok")

    # Ilk 2 'bedelsiz' kaydinin metnini oku
    for r in guclu[:2]:
        if r.get("disclosureIndex"):
            time.sleep(1.0)
            kp.detail(s, r["disclosureIndex"], out)

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
