import requests

KODLAR = ["THYAO", "ASELS", "EREGL"]
URL = ("https://www.isyatirim.com.tr/_layouts/15/IsYatirim.Website/"
       "Common/Data.aspx/MaliTablo")
H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def cek(kod, grup, donemler):
    p = {"companyCode": kod, "exchange": "TRY", "financialGroup": grup}
    for i, (y, m) in enumerate(donemler, 1):
        p[f"year{i}"] = y
        p[f"period{i}"] = m
    return requests.get(URL, params=p, headers=H, timeout=30)


def dene(kod, grup, donemler):
    print(f"--- {kod} {grup} {donemler}")
    try:
        r = cek(kod, grup, donemler)
        print("durum:", r.status_code, "uzunluk:", len(r.text))
        j = r.json()
        satirlar = j.get("value") or []
        print("ok:", j.get("ok"), "satir sayisi:", len(satirlar))
        if not satirlar:
            print("govde:", r.text[:200])
            return
        print("ornek satir:", {k: satirlar[0][k] for k in list(satirlar[0])[:8]})
        for s in satirlar:
            ad = str(s.get("itemDescTr", ""))
            if "Net Dönem" in ad or "Satış Gelirleri" in ad or ad.strip() == "Hasılat":
                print(f"  {s.get('itemCode')} {ad[:40]}: "
                      f"{s.get('value1')} | {s.get('value2')} | "
                      f"{s.get('value3')} | {s.get('value4')}")
    except Exception as e:
        print("hata:", e)


def main():
    yeni = [(2026, 6), (2026, 3), (2025, 12), (2025, 9)]
    eski = [(2022, 12), (2022, 9), (2022, 6), (2022, 3)]
    cok_eski = [(2019, 12), (2019, 9), (2019, 6), (2019, 3)]
    for kod in KODLAR:
        for grup in ("XI_29", "UFRS"):
            dene(kod, grup, yeni)
    dene("THYAO", "XI_29", eski)
    dene("THYAO", "UFRS", eski)
    dene("THYAO", "UFRS", cok_eski)

    print("=" * 70)
    print("borsapy:")
    try:
        import borsapy as bp
        t = bp.Ticker("THYAO")
        print([a for a in dir(t) if not a.startswith("_")])
    except Exception as e:
        print("hata:", e)

    print("=" * 70)
    print("yfinance:")
    try:
        import yfinance as yf
        q = yf.Ticker("THYAO.IS").quarterly_income_stmt
        print("ceyrek tablo sekli:", q.shape)
        print("donemler:", [str(c)[:10] for c in q.columns])
    except Exception as e:
        print("hata:", e)


if __name__ == "__main__":
    main()
