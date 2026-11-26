import csv
import gzip
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from tavan_aday import sembol_oku

URL = ("https://www.isyatirim.com.tr/_layouts/15/IsYatirim.Website/"
       "Common/Data.aspx/MaliTablo")
H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
YILLAR = list(range(2019, 2027))
AYLAR = [12, 9, 6, 3]
ANAHTAR = ("SALES", "PROFIT", "INCOME", "TOTAL ASSETS", "EQUITY", "CASH",
           "DEPRECIATION", "DEBT", "BORROW", "FINANC")


def cek_yil(kod, yil):
    p = {"companyCode": kod, "exchange": "TRY", "financialGroup": "XI_29"}
    for i, ay in enumerate(AYLAR, 1):
        p[f"year{i}"] = yil
        p[f"period{i}"] = ay
    for deneme in range(4):
        try:
            r = requests.get(URL, params=p, headers=H, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(3 * (deneme + 1))
                continue
            j = r.json()
            time.sleep(0.15)
            return kod, yil, j.get("value") or []
        except Exception:
            time.sleep(2 * (deneme + 1))
    return kod, yil, None


def main():
    kodlar = [s.replace(".IS", "") for s in sembol_oku()]
    gorevler = [(k, y) for k in kodlar for y in YILLAR]
    print(f"hisse: {len(kodlar)}  istek: {len(gorevler)}", flush=True)

    satirlar = []
    kalemler = {}
    veri_olan = set()
    basarisiz = 0
    bitti = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        isler = [ex.submit(cek_yil, k, y) for k, y in gorevler]
        for f in as_completed(isler):
            kod, yil, rows = f.result()
            bitti += 1
            if bitti % 300 == 0:
                print(f"  {bitti}/{len(gorevler)}  satir: {len(satirlar)}", flush=True)
            if rows is None:
                basarisiz += 1
                continue
            for s in rows:
                eng = str(s.get("itemDescEng", "")).upper()
                if not any(a in eng for a in ANAHTAR):
                    continue
                ic = s.get("itemCode")
                kalemler.setdefault(ic, (s.get("itemDescTr"), s.get("itemDescEng")))
                for i, ay in enumerate(AYLAR, 1):
                    v = s.get(f"value{i}")
                    if v in (None, "", "null"):
                        continue
                    satirlar.append((kod, yil, ay, ic, v))
                    veri_olan.add(kod)

    with gzip.open("bilanco.csv.gz", "wt", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kod", "yil", "ay", "kalem", "deger"])
        w.writerows(satirlar)
    with open("kalemler.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kalem", "tr", "en"])
        for ic, (tr, en) in sorted(kalemler.items()):
            w.writerow([ic, tr, en])

    print("=" * 70)
    print(f"veri olan hisse: {len(veri_olan)}/{len(kodlar)}  satir: {len(satirlar)}"
          f"  basarisiz istek: {basarisiz}  kalem sayisi: {len(kalemler)}")
    print("KALEMLER (kod: Turkce ad):")
    for ic, (tr, en) in sorted(kalemler.items()):
        ad = str(tr)
        if any(x in str(en).upper() for x in
               ("SALES", "PROFIT", "NET INCOME", "TOTAL ASSETS", "EQUITY")):
            print(f"  {ic}: {ad[:55]}")


if __name__ == "__main__":
    main()
