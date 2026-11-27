import json
import time

import requests

ORNEK = ["THYAO", "ASELS", "EREGL", "AKBNK", "TUPRS", "BIMAS", "KCHOL",
         "SISE", "ENKAI", "PGSUS", "ALARK", "DOGUB"]


def kaynak_borsapy():
    print("=" * 80)
    print("KAYNAK 1: borsapy")
    try:
        import borsapy as bp
    except Exception as e:
        print("borsapy yuklenemedi:", e)
        return
    print("borsapy ust seviye adlar:",
          [a for a in dir(bp) if not a.startswith("_")][:60])
    for ad in ("companies", "search_companies", "sectors", "get_sectors"):
        f = getattr(bp, ad, None)
        if f is None:
            continue
        try:
            r = f() if ad in ("companies", "sectors", "get_sectors") else f("THYAO")
            print(f"bp.{ad}() -> tip {type(r).__name__}")
            try:
                print(r.head(8).to_string())
                print("kolonlar:", list(r.columns))
                print("satir:", len(r))
            except Exception:
                print(str(r)[:500])
        except Exception as e:
            print(f"bp.{ad} hata:", repr(e)[:200])
    for s in ORNEK[:6]:
        try:
            t = bp.Ticker(s)
            info = getattr(t, "info", None)
            if info is None:
                print(s, "info yok; ozellikler:",
                      [a for a in dir(t) if not a.startswith("_")][:40])
                continue
            try:
                d = dict(info)
            except Exception:
                d = {}
            anahtar = {k: v for k, v in d.items()
                       if any(x in str(k).lower() for x in
                              ("sector", "sektor", "industry", "sanayi"))}
            print(s, "->", anahtar if anahtar else ("anahtarlar:", list(d)[:25]))
        except Exception as e:
            print(s, "hata:", repr(e)[:200])
        time.sleep(0.5)


def kaynak_yfinance():
    print("=" * 80)
    print("KAYNAK 2: yfinance (sector / industry)")
    try:
        import yfinance as yf
    except Exception as e:
        print("yfinance yok:", e)
        return
    dolu = 0
    for s in ORNEK:
        try:
            info = yf.Ticker(s + ".IS").info
            sec, ind = info.get("sector"), info.get("industry")
            print(f"{s:<7} sector={sec} | industry={ind}")
            dolu += int(bool(sec))
        except Exception as e:
            print(s, "hata:", repr(e)[:150])
        time.sleep(0.7)
    print(f"yfinance sektor dolu: {dolu}/{len(ORNEK)}")


def kaynak_isyatirim():
    print("=" * 80)
    print("KAYNAK 3: Is Yatirim hisse bilgisi")
    url = ("https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/"
           "Common/Data.aspx/HisseBilgi")
    h = {"User-Agent": "Mozilla/5.0"}
    for s in ORNEK[:3]:
        try:
            r = requests.get(url, params={"hisse": s}, headers=h, timeout=30)
            print(s, "HTTP", r.status_code, "|", r.text[:300].replace("\n", " "))
        except Exception as e:
            print(s, "hata:", repr(e)[:150])
    url2 = ("https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/"
            "Common/Data.aspx/HisseSenetleri")
    try:
        r = requests.get(url2, headers=h, timeout=30)
        print("HisseSenetleri HTTP", r.status_code, "|", r.text[:300].replace("\n", " "))
        try:
            veri = r.json()
            print("json tip:", type(veri).__name__)
            print(json.dumps(veri, ensure_ascii=False)[:600])
        except Exception:
            pass
    except Exception as e:
        print("HisseSenetleri hata:", repr(e)[:150])


def kaynak_maliTablo():
    print("=" * 80)
    print("KAYNAK 4: Is Yatirim MaliTablo cevabinda sektor bilgisi var mi")
    url = ("https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/"
           "Common/Data.aspx/MaliTablo")
    h = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, params={
            "companyCode": "THYAO", "exchange": "TRY", "financialGroup": "XI_29",
            "year1": 2025, "period1": 9, "year2": 2025, "period2": 6,
            "year3": 2025, "period3": 3, "year4": 2024, "period4": 12},
            headers=h, timeout=30)
        print("HTTP", r.status_code, "| ilk 300:", r.text[:300].replace("\n", " "))
    except Exception as e:
        print("hata:", repr(e)[:150])


def main():
    kaynak_borsapy()
    kaynak_yfinance()
    kaynak_isyatirim()
    kaynak_maliTablo()
    print("=" * 80)
    print("bitti")


if __name__ == "__main__":
    main()
