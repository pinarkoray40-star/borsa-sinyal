"""tani.py - bedelsiz olaylari neden bulunamiyor? Hizli tani (1-2 dk surmeli).

Ne yapar:
1) borsapy surumunu ve Ticker'in bedelsiz/sermaye/bolunme/temettu ile ilgili
   ozelliklerini listeler
2) Birkac hissede bu ozellikleri gercekten cagirip ham ciktiyi yazdirir
3) Fiyat gecmisinde bolunme/bedelsiz izi (Splits sutunu) var mi bakar

Calistir: bedelsiz.yml icinde `python bedelsiz_test.py` yerine `python tani.py`
Sonra Actions logunun ekran goruntusunu at.
"""
import sys
import time
import traceback

try:
    import borsapy as bp
except Exception:
    print("borsapy import hatasi")
    traceback.print_exc()
    sys.exit(1)

print("python:", sys.version.split()[0])
print("borsapy surum:", getattr(bp, "__version__", "?"))
print("borsapy ust duzey adlar:", [n for n in dir(bp) if not n.startswith("_")][:60])
print()

HISSELER = ["THYAO", "RYGYO", "DOGUB", "ASELS", "EREGL"]
ANAHTAR = ("split", "bolun", "bedelsiz", "sermaye", "capital", "dividend",
           "temettu", "action", "event")


def kisalt(x, n=1500):
    s = str(x)
    return s if len(s) <= n else s[:n] + " ...[kesildi]"


for kod in HISSELER:
    print("=" * 60)
    print("HISSE:", kod)
    t0 = time.time()
    try:
        t = bp.Ticker(kod)
    except Exception:
        print("Ticker olusturulamadi")
        traceback.print_exc()
        continue

    adaylar = [n for n in dir(t) if not n.startswith("_")
               and any(k in n.lower() for k in ANAHTAR)]
    print("ilgili ozellikler:", adaylar)

    for ad in adaylar:
        try:
            v = getattr(t, ad)
            if callable(v):
                v = v()
            print(f"--- {ad} (tip: {type(v).__name__}) ---")
            print(kisalt(v))
        except Exception as e:
            print(f"--- {ad} HATA: {type(e).__name__}: {e}")

    # fiyat gecmisi: bolunme/bedelsiz sutunu var mi?
    try:
        h = t.history(period="max")
        print("history satir:", len(h), "| sutunlar:", list(h.columns))
        for c in h.columns:
            if any(k in str(c).lower() for k in ("split", "bolun", "bedelsiz")):
                nz = h[h[c] != 0] if h[c].dtype != object else h[h[c].notna()]
                print(f"history[{c}] sifir olmayan satir:", len(nz))
                print(kisalt(nz[[c]].tail(10)))
    except Exception as e:
        print(f"history HATA: {type(e).__name__}: {e}")

    print(f"sure: {time.time() - t0:.1f} sn")

print()
print("TANI BITTI")
