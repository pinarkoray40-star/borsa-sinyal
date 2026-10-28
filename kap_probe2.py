"""KAP ikinci deneme: Pabrai bildirimini bul, kayıt alanlarını ve bildirim metnini göster.

kap_probe.py'nin devamı (erişim çalıştı). Bu sefer:
1) 28 Mayıs - 6 Haziran 2025 arası GÜNLÜK pencereler: Dalal Street / Pabrai / RYGYO kayıtlarını arar.
2) 26-28 Ekim 2026: RYGYO'nun son bildirimlerini özetiyle ve metniyle gösterir.
Toplam yaklaşık 15 istek, saniyede en fazla 1 istek. Gayriresmi uç nokta kullanır.
Yatırım tavsiyesi değildir.
"""
import argparse
import json
import os
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import requests

BASE = "https://www.kap.org.tr"
PAGE_URL = BASE + "/tr/bildirim-sorgu"
LIST_URL = BASE + "/tr/api/disclosure/members/byCriteria"
UA = "borsa-sinyal-probe/0.2 (research; low rate)"
KEYWORDS = ("rygyo", "reysas gayrimenkul", "dalal", "pabrai")


def norm(s) -> str:
    return (str(s).replace("İ", "i").replace("I", "i").replace("ı", "i")
            .replace("ş", "s").replace("Ş", "s").replace("ğ", "g").replace("Ğ", "g")
            .replace("ü", "u").replace("Ü", "u").replace("ö", "o").replace("Ö", "o")
            .replace("ç", "c").replace("Ç", "c").lower())


def fetch_window(s, d1, d2, out):
    body = {"fromDate": d1, "toDate": d2, "mkkMemberOidList": [], "subjectList": []}
    try:
        r = s.post(LIST_URL, json=body, timeout=60,
                   headers={"Referer": PAGE_URL, "Origin": BASE,
                            "Content-Type": "application/json", "Accept": "application/json"})
    except Exception as exc:
        out.append(f"- {d1}→{d2}: **hata** {type(exc).__name__}: {exc}")
        return None
    if r.status_code != 200:
        out.append(f"- {d1}→{d2}: HTTP {r.status_code} `{r.text[:120]!r}`")
        return None
    try:
        data = r.json()
    except Exception:
        out.append(f"- {d1}→{d2}: JSON değil")
        return None
    if not isinstance(data, list):
        out.append(f"- {d1}→{d2}: beklenmeyen biçim {type(data).__name__}")
        return None
    return data


def compact(r: dict, limit: int = 600) -> str:
    keep = {k: r.get(k) for k in ("publishDate", "stockCodes", "relatedStocks", "kapTitle", "subject",
                                  "disclosureClass", "disclosureType", "fundCode", "disclosureIndex",
                                  "modifyStatus", "summary")}
    txt = json.dumps(keep, ensure_ascii=False)
    return txt if len(txt) <= limit else txt[:limit] + "…"


def texts(obj, acc=None):
    """Bir JSON ağacındaki uzun metin değerlerini toplar (bildirim gövdesini okumak için)."""
    if acc is None:
        acc = []
    if isinstance(obj, str):
        t = " ".join(obj.split())
        if len(t) > 25 and t not in acc:
            acc.append(t)
    elif isinstance(obj, dict):
        for v in obj.values():
            texts(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            texts(v, acc)
    return acc


def detail(s, idx, out):
    out.append(f"\n### Bildirim metni (no {idx})")
    try:
        r = s.get(f"{BASE}/tr/api/notification/attachment-detail/{idx}", timeout=60,
                  headers={"Referer": f"{BASE}/tr/Bildirim/{idx}", "Accept": "application/json"})
    except Exception as exc:
        out.append(f"- Hata: {type(exc).__name__}: {exc}")
        return
    out.append(f"- HTTP {r.status_code} · {len(r.content)} bayt")
    if r.status_code != 200:
        return
    try:
        d = r.json()
    except Exception:
        out.append(f"- JSON değil: `{r.text[:200]!r}`")
        return
    first = d[0] if isinstance(d, list) and d else d
    if isinstance(first, dict):
        out.append(f"- İlk öğe alanları: `{', '.join(sorted(first.keys()))[:300]}`")
    body = " | ".join(texts(d))
    out.append(f"- Metin (ilk 1500 karakter): {body[:1500]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="5y", help="Analiz calistir akışı için; kullanılmaz")
    ap.add_argument("--out", default="kap_probe2.md")
    args = ap.parse_args()

    out = ["# KAP ikinci deneme"]
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Language": "tr-TR,tr;q=0.9"})
    try:
        s.get(PAGE_URL, timeout=30)
    except Exception as exc:
        out.append(f"- Sayfa isteği hata verdi (devam): {type(exc).__name__}: {exc}")

    windows = []
    d = date(2025, 5, 28)
    while d <= date(2025, 6, 6):
        windows.append((d.isoformat(), d.isoformat()))
        d += timedelta(days=1)
    windows.append(("2026-10-26", "2026-10-28"))

    hits_all = []
    for d1, d2 in windows:
        time.sleep(1.0)
        data = fetch_window(s, d1, d2, out)
        if data is None:
            continue
        pay = [r for r in data if "pay alim" in norm(r.get("subject", ""))]
        cls = dict(Counter(str(r.get("disclosureClass")) for r in pay))
        hits = [r for r in data if any(k in norm(json.dumps(r, ensure_ascii=False)) for k in KEYWORDS)]
        out.append(f"\n## {d1} → {d2}: {len(data)} kayıt · pay alım satım {len(pay)} {cls} · aranan kelime: **{len(hits)}**")
        for r in hits[:8]:
            out.append(f"- {compact(r)}")
        hits_all += hits

    # Örnek: ODA sınıfı pay alım satım kaydı nasıl görünüyor (alıcı bilgisi var mı?)
    out.append("\n## Not")
    out.append("- Aranan kelimeler: " + ", ".join(KEYWORDS))

    pick = None
    for r in hits_all:
        if "pay alim" in norm(r.get("subject", "")):
            pick = r
            break
    if pick is None and hits_all:
        pick = hits_all[-1]
    if pick is not None and pick.get("disclosureIndex"):
        time.sleep(1.0)
        detail(s, pick["disclosureIndex"], out)
    # RYGYO'nun 26 Ekim bildirimini de oku (özel durum açıklaması)
    last = [r for r in hits_all if str(r.get("publishDate", "")).startswith("26.10.2026")]
    if last and (pick is None or last[0].get("disclosureIndex") != pick.get("disclosureIndex")):
        time.sleep(1.0)
        detail(s, last[0]["disclosureIndex"], out)

    if not hits_all:
        out.append("- Aranan kelimelerle eşleşen kayıt bulunamadı.")
    out.append("\n> Not: Gayriresmi uç nokta, toplam birkaç istek yapıldı. Yatırım tavsiyesi değildir.")

    text = "\n".join(out)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
