"""
Bedelsiz oncesi davranis testi.

Soru: Bedelsiz sermaye artirimi yapan hisseler, hak kullanim tarihinden once
endekse (XU100) gore fazla getiri sagliyor mu?

Kaynak: borsapy Ticker.splits (BonusFromCapital + BonusFromDividend, yuzde).
Olay tarihi = bedelsiz hakkinin islem gordugu tarih.

Cikti: bedelsiz_olaylar.csv ve bedelsiz_rapor.txt
"""
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import borsapy as bp
except ImportError:
    sys.exit("borsapy kurulu degil: pip install borsapy")

# (baslangic, bitis) -> olaydan kac islem gunu once
PENCERELER = [(60, 40), (40, 20), (20, 1)]
MIN_BEDELSIZ = 20.0  # en az %20 bedelsiz
ISCI = 6
ERKEN_DUR = 60  # bu kadar hissede hic olay yoksa dur


def hisse_listesi():
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as f:
            return [s.strip().upper() for s in f if s.strip()]
    for endeks in ("XUTUM", "XU100"):
        try:
            idx = bp.Index(endeks)
            for ad in ("component_symbols", "symbols", "components"):
                v = getattr(idx, ad, None)
                if v is not None:
                    v = list(v)
                    if v and isinstance(v[0], dict):
                        v = [d.get("symbol") for d in v]
                    v = [str(x).replace(".IS", "").upper() for x in v if x]
                    if len(v) > 20:
                        print(f"{endeks}: {len(v)} hisse bulundu")
                        return v
        except Exception as e:  # noqa: BLE001
            print(f"{endeks} bilesenleri alinamadi: {e}")
    sys.exit("Hisse listesi alinamadi.")


def kapanis(df):
    df = df.copy()
    for c in ("Close", "close", "Kapanis"):
        if c in df.columns:
            s = df[c].astype(float)
            break
    else:
        s = df.iloc[:, 3].astype(float)
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    return s[~s.index.duplicated()].sort_index().dropna()


def pencere_getiri(seri, i_son, bas, bit):
    a, b = i_son - bas, i_son - bit
    if a < 0 or b <= a:
        return np.nan
    return seri.iloc[b] / seri.iloc[a] - 1


def bedelsiz_serisi(t):
    """Tarih -> bedelsiz yuzdesi (sadece >= MIN_BEDELSIZ). Yoksa None."""
    sp = t.splits
    if sp is None or len(sp) == 0:
        return None
    sp = sp.copy()
    for c in ("BonusFromCapital", "BonusFromDividend"):
        if c not in sp.columns:
            sp[c] = 0.0
    b = (pd.to_numeric(sp["BonusFromCapital"], errors="coerce").fillna(0)
         + pd.to_numeric(sp["BonusFromDividend"], errors="coerce").fillna(0))
    idx = pd.to_datetime(sp.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    b.index = idx
    b = b[b >= MIN_BEDELSIZ]
    return b if len(b) else None


def isle(kod, xu):
    hata = None
    for deneme in range(3):
        try:
            t = bp.Ticker(kod)
            ev = bedelsiz_serisi(t)
            if ev is None:
                return kod, [], [], None
            fiyat = kapanis(t.history(period="max"))
            break
        except Exception as e:  # noqa: BLE001
            hata = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (deneme + 1))
    else:
        return kod, [], [], hata

    ort = fiyat.index.intersection(xu.index)
    f, x = fiyat.loc[ort], xu.loc[ort]
    if len(f) < 150:
        return kod, [], [], None

    baz = []
    for i in range(60, len(f), 20):
        baz.append((f.iloc[i] / f.iloc[i - 20] - 1) - (x.iloc[i] / x.iloc[i - 20] - 1))

    olaylar = []
    for tarih, yuzde in ev.items():
        pos = f.index.searchsorted(pd.Timestamp(tarih).normalize())
        if pos >= len(f) or pos < 61:
            continue
        satir = {"hisse": kod, "tarih": f.index[pos].date(), "bedelsiz_yuzde": float(yuzde)}
        for bas, bit in PENCERELER:
            satir[f"fazla_{bas}_{bit}"] = (pencere_getiri(f, pos, bas, bit)
                                           - pencere_getiri(x, pos, bas, bit))
        olaylar.append(satir)
    return kod, olaylar, baz, None


def main():
    hisseler = hisse_listesi()
    print("Endeks fiyatlari aliniyor...")
    xu = kapanis(bp.Index("XU100").history(period="max"))

    olaylar, baz, hatalar = [], [], []
    islenen = 0
    ex = ThreadPoolExecutor(max_workers=ISCI)
    isler = [ex.submit(isle, k, xu) for k in hisseler]
    for fut in as_completed(isler):
        kod, ol, bz, hata = fut.result()
        islenen += 1
        olaylar += ol
        baz += bz
        if hata:
            hatalar.append((kod, hata))
        if islenen % 50 == 0:
            print(f"  {islenen}/{len(hisseler)} hisse, {len(olaylar)} olay, {len(hatalar)} hata")
        if islenen >= ERKEN_DUR and not olaylar:
            ex.shutdown(wait=False, cancel_futures=True)
            print("Ilk hisselerde hic olay yok, duruyorum. Ornek hatalar:", hatalar[:5])
            sys.exit(1)
    ex.shutdown()

    if hatalar:
        print(f"{len(hatalar)} hissede hata, ornek:", hatalar[:5])
    if not olaylar:
        sys.exit("Hic olay bulunamadi.")

    df = pd.DataFrame(olaylar).sort_values(["hisse", "tarih"])
    df.to_csv("bedelsiz_olaylar.csv", index=False)
    baz = np.array(baz)

    satirlar = [f"Olay sayisi: {len(df)}  (hisse: {df.hisse.nunique()})",
                f"Tarih araligi: {df.tarih.min()} -> {df.tarih.max()}", ""]
    satirlar.append("Olaydan ONCEKI donemlerde endekse gore fazla getiri (ortanca / ortalama / pozitif orani)")
    for bas, bit in PENCERELER:
        s = df[f"fazla_{bas}_{bit}"].dropna()
        satirlar.append(f"  -{bas}..-{bit} gun: ortanca %{s.median()*100:.1f} | ort. %{s.mean()*100:.1f} | "
                        f"pozitif %{(s > 0).mean()*100:.0f} | n={len(s)}")
    if len(baz):
        satirlar.append("")
        satirlar.append("Karsilastirma (ayni hisselerin rastgele 20 gunluk dilimleri):")
        satirlar.append(f"  ortanca %{np.median(baz)*100:.1f} | ort. %{baz.mean()*100:.1f} | "
                        f"pozitif %{(baz > 0).mean()*100:.0f} | n={len(baz)}")

    df["yil"] = pd.to_datetime(df.tarih).dt.year
    satirlar.append("")
    satirlar.append("Yila gore -20..-1 gun ortanca fazla getiri:")
    for y, g in df.groupby("yil"):
        s = g["fazla_20_1"].dropna()
        if len(s) >= 5:
            satirlar.append(f"  {y}: %{s.median()*100:.1f} (n={len(s)})")

    rapor = "\n".join(satirlar)
    print("\n" + rapor)
    with open("bedelsiz_rapor.txt", "w", encoding="utf-8") as f:
        f.write(rapor + "\n")


if __name__ == "__main__":
    main()
