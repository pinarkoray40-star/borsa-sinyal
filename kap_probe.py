"""KAP erişim denemesi: Actions'tan KAP'a ulaşılabiliyor mu, hangi alanlar geliyor?

Küçük bir deneme, toplam birkaç istek (saniyede en fazla 1 istek):
1) 27-29 Mayıs 2025: Pabrai fonlarının RYGYO alım bildiriminin (28.05.2025) geldiği pencere.
2) Son 3 gün: güncel veri geliyor mu.
Gayriresmi uç nokta kullanır; KAP sitesi bu isteklere izin vermeyebilir. Sonuca göre devam edilir.
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
UA = "borsa-sinyal-probe/0.1 (research; low rate)"


def norm(s) -> str:
    """Türkçe büyük/küçük harf farklarını yok sayarak karşılaştırma için sadeleştirir."""
    return str(s).replace("İ", "i").replace("I", "i").replace("ı", "i").lower()


def pick(r: dict, *keys):
    for k in keys:
        v = r.get(k)
        if v not in (None, ""):
            return v
    return "-"


def fetch_window(s, d1: str, d2: str, out: list):
    out.append(f"\n## Pencere {d1} → {d2}")
    body = {"fromDate": d1, "toDate": d2, "mkkMemberOidList": [], "subjectList": []}
    try:
        r = s.post(LIST_URL, json=body, timeout=40,
                   headers={"Referer": PAGE_URL, "Origin": BASE,
                            "Content-Type": "application/json", "Accept": "application/json"})
    except Exception as exc:
        out.append(f"- **Erişim hatası:** {type(exc).__name__}: {exc}")
        return None
    out.append(f"- HTTP {r.status_code} · {len(r.content)} bayt")
    if r.status_code != 200:
        out.append(f"- Yanıt başı: `{r.text[:300]!r}`")
        return None
    try:
        data = r.json()
    except Exception:
        out.append(f"- JSON değil. Yanıt başı: `{r.text[:300]!r}`")
        return None
    if not isinstance(data, list):
        keys = list(data)[:10] if isinstance(data, dict) else ""
        out.append(f"- Beklenmeyen biçim: {type(data).__name__} {keys}")
        return None
    return data


def row_text(r: dict) -> str:
    return (f"{pick(r, 'publishDate', 'disclosureDate', 'time')} · {pick(r, 'stockCode', 'stockCodes')} · "
            f"{pick(r, 'kapTitle', 'companyTitle', 'memberTitle')} · {pick(r, 'subject')} · "
            f"sınıf {pick(r, 'disclosureClass')} · no {pick(r, 'disclosureIndex')}")


def summarize(data: list, out: list):
    out.append(f"- Kayıt sayısı: **{len(data)}**"
               + (" (2.000 sınırına dayandı, pencereyi daralt)" if len(data) >= 2000 else ""))
    if not data:
        return
    out.append(f"- İlk kaydın alanları: `{', '.join(sorted(data[0].keys()))}`")
    cnt = Counter((str(r.get("subject")), str(r.get("disclosureClass"))) for r in data)
    out.append("- En sık konu / sınıf çiftleri:")
    for (subj, cls), n in cnt.most_common(8):
        out.append(f"  - {subj} · {cls} · {n}")
    pay = [r for r in data if "pay alim" in norm(r.get("subject", ""))]
    cls_cnt = dict(Counter(str(r.get("disclosureClass")) for r in pay))
    out.append(f"- 'Pay Alım Satım Bildirimi' kayıtları: **{len(pay)}** (sınıflar: {cls_cnt})")
    for r in pay[:8]:
        out.append(f"  - {row_text(r)}")


def rygyo_rows(data: list) -> list:
    return [r for r in data if "rygyo" in norm(json.dumps(r, ensure_ascii=False))
            or "reysas gayrimenkul" in norm(json.dumps(r, ensure_ascii=False))]


def detail(s, idx, out: list):
    url = f"{BASE}/tr/api/notification/attachment-detail/{idx}"
    out.append(f"\n### Detay denemesi (no {idx})")
    try:
        r = s.get(url, timeout=40, headers={"Referer": f"{BASE}/tr/Bildirim/{idx}", "Accept": "application/json"})
    except Exception as exc:
        out.append(f"- Hata: {type(exc).__name__}: {exc}")
        return
    out.append(f"- HTTP {r.status_code} · {len(r.content)} bayt")
    if r.status_code != 200:
        out.append(f"- Yanıt başı: `{r.text[:200]!r}`")
        return
    try:
        d = r.json()
    except Exception:
        out.append(f"- JSON değil: `{r.text[:200]!r}`")
        return
    if isinstance(d, dict):
        out.append(f"- Alanlar: `{', '.join(sorted(d.keys()))}`")
        att = d.get("attachments")
        if att:
            out.append(f"- Ek sayısı: {len(att)} · ilk ek alanları: "
                       f"`{', '.join(sorted(att[0].keys())) if isinstance(att[0], dict) else type(att[0]).__name__}`")
    else:
        out.append(f"- Biçim: {type(d).__name__}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="5y", help="Analiz calistir akışı için; kullanılmaz")
    ap.add_argument("--out", default="kap_probe.md")
    args = ap.parse_args()

    today = date.today()
    windows = [("2025-05-27", "2025-05-29"),
               ((today - timedelta(days=3)).isoformat(), today.isoformat())]

    out = ["# KAP erişim denemesi"]
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Language": "tr-TR,tr;q=0.9"})
    try:
        r = s.get(PAGE_URL, timeout=30)
        out.append(f"- Sayfa isteği (çerez için): HTTP {r.status_code}")
    except Exception as exc:
        out.append(f"- Sayfa isteği hata verdi (devam ediliyor): {type(exc).__name__}: {exc}")

    got_any = False
    first_rygyo = None
    for d1, d2 in windows:
        time.sleep(1.0)
        data = fetch_window(s, d1, d2, out)
        if data is None:
            continue
        got_any = True
        summarize(data, out)
        rows = rygyo_rows(data)
        out.append(f"- RYGYO ile ilgili kayıt: **{len(rows)}**")
        for r in rows[:10]:
            out.append(f"  - {row_text(r)}")
        if rows and first_rygyo is None:
            first_rygyo = pick(rows[0], "disclosureIndex")

    if first_rygyo not in (None, "-"):
        time.sleep(1.0)
        detail(s, first_rygyo, out)

    out.insert(1, "**Sonuç: ERİŞİM VAR** (liste döndü)" if got_any
               else "**Sonuç: ERİŞİM YOK** (aşağıdaki hata satırlarına bak)")
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
