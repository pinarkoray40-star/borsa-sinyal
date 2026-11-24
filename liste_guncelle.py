"""
liste_guncelle.py - bist_symbols.txt'yi günceller (siteye ve diğer dosyalara dokunmaz).
1) borsapy'den güncel BIST hisse listesini toplar (birkaç endeksin birleşimi).
2) Yahoo'da son 3 ayda verisi olanları doğrular.
3) Yeni olanları ekler; Yahoo'da hiç verisi olmayan eskileri çıkarır.
APPLY=1 ise dosyayı yazar, değilse sadece rapor basar.
"""
import os
import re

import pandas as pd

FILE = "bist_symbols.txt"
APPLY = os.environ.get("APPLY", "0") == "1"
INDEXES = ["XUTUM", "XTUMY", "XKTUM", "XU100", "XUSIN", "XUMAL", "XUHIZ", "XUTEK", "XHOLD", "XYLDZ"]
PATTERN = re.compile(r"^[A-Z0-9]{3,6}$")


def components(idx_symbol):
    import borsapy as bp

    idx = bp.Index(idx_symbol)
    for attr in ("component_symbols", "components"):
        v = getattr(idx, attr, None)
        if v is None:
            continue
        if callable(v):
            v = v()
        out = []
        for x in v:
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, dict):
                out.append(x.get("symbol") or x.get("ticker") or "")
        out = [s.strip().upper().replace(".IS", "") for s in out if s]
        if out:
            return out
    return []


def collect():
    import borsapy as bp

    names = list(INDEXES)
    try:
        allx = bp.all_indices()
        print("borsapy endeks sayısı:", len(allx))
        for d in allx:
            if isinstance(d, dict) and d.get("count", 0) >= 300 and d.get("symbol") not in names:
                names.append(d["symbol"])
    except Exception as e:
        print("all_indices hatası:", e)
    found = set()
    for n in names:
        try:
            c = components(n)
            print(f"  {n}: {len(c)} hisse")
            found.update(c)
        except Exception as e:
            print(f"  {n}: hata {type(e).__name__}: {str(e)[:80]}")
    return sorted(s for s in found if PATTERN.match(s))


def check(symbols):
    """Yahoo'da son 3 ayda >=5 gün verisi var mı. True/False; indirme hatasında None (bilinmiyor)."""
    import yfinance as yf

    res = {}
    for i in range(0, len(symbols), 80):
        part = [s + ".IS" for s in symbols[i:i + 80]]
        try:
            raw = yf.download(part, period="3mo", interval="1d", auto_adjust=True,
                              progress=False, threads=True)
        except Exception as e:
            print("indirme hatası:", e)
            raw = None
        if raw is None or raw.empty:
            for s in symbols[i:i + 80]:
                res[s] = None
            continue
        c = raw["Close"]
        if isinstance(c, pd.Series):
            c = c.to_frame(part[0])
        for s in symbols[i:i + 80]:
            col = s + ".IS"
            res[s] = bool(col in c.columns and c[col].notna().sum() >= 5)
    return res


def main():
    lines = open(FILE, encoding="utf-8").read().splitlines()
    existing = [ln.strip().upper() for ln in lines if ln.strip() and not ln.startswith("#")]
    ex_set = set(existing)
    print("dosyadaki hisse:", len(existing))

    cand = collect()
    print("borsapy toplam aday:", len(cand))
    new = [s for s in cand if s not in ex_set]
    print("dosyada olmayan:", len(new))

    ok_new = check(new) if new else {}
    to_add = sorted(s for s, v in ok_new.items() if v)
    skipped = sorted(s for s, v in ok_new.items() if not v)

    ok_old = check(existing)
    dead = sorted(s for s, v in ok_old.items() if v is False)
    unknown = sorted(s for s, v in ok_old.items() if v is None)

    print("\n=== RAPOR ===")
    print(f"EKLENECEK ({len(to_add)}):", ", ".join(to_add) or "-")
    print(f"Yahoo'da verisi yok, eklenmedi ({len(skipped)}):", ", ".join(skipped) or "-")
    print(f"ÇIKARILACAK - 3 ayda verisi yok ({len(dead)}):", ", ".join(dead) or "-")
    if unknown:
        print(f"Kontrol edilemedi, dokunulmadı ({len(unknown)}):", ", ".join(unknown))

    # güvenlik: kaynak az hisse döndürdüyse hiçbir şeyi çıkarma / ekleme
    if len(cand) < 350:
        print("\nUYARI: borsapy kaynağı beklenenden az hisse verdi (<350). Hiçbir değişiklik yapılmayacak.")
        return
    if len(dead) > 40:
        print("\nUYARI: çıkarılacak sayısı çok fazla (>40). Çıkarma yapılmayacak, sadece eklenecek.")
        dead = []

    if not APPLY:
        print("\nRAPOR MODU: dosya değiştirilmedi.")
        return
    keep = [ln for ln in lines if (not ln.strip()) or ln.startswith("#") or ln.strip().upper() not in set(dead)]
    while keep and not keep[-1].strip():
        keep.pop()
    keep += to_add
    open(FILE, "w", encoding="utf-8").write("\n".join(keep) + "\n")
    print(f"\nbist_symbols.txt yazıldı: +{len(to_add)} / -{len(dead)} -> {len(existing) + len(to_add) - len(dead)} hisse")


if __name__ == "__main__":
    main()
