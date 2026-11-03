"""
Bedelsiz + momentum + goreceli guc birlesim testi.

Giris: olaydan 20 islem gunu once. Sonuc: giristen olaydan 1 gun oncesine
endekse gore fazla getiri. Filtre degerleri giris gununde, sadece gecmis veriyle
olculur. Karsilastirma: ayni filtre bedelsiz olmayan donemlerde (ayni hisseler).

Cikti: birlesim_satirlar.csv ve birlesim_rapor.txt
"""
import functools
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as ZamanAsimi

import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)
warnings.filterwarnings("ignore")

try:
    import borsapy as bp
except ImportError:
    sys.exit("borsapy kurulu degil: pip install borsapy")

MIN_BEDELSIZ = 20.0   # en az %20 bedelsiz
GIRIS = 20            # olaydan kac islem gunu once girilir
BASLANGIC = 120       # faktorler icin gereken gecmis gun
ADIM = 10             # bedelsiz olmayan donemlerde ornekleme adimi
ISCI = 6
ERKEN_DUR = 60
SURE_LIMITI = 150 * 60  # saniye


def bitir(kod):
    sys.stdout.flush()
    os._exit(kod)


def hisse_listesi():
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


def kolon(df, adlar, poz):
    s = None
    for c in adlar:
        if c in df.columns:
            s = df[c].astype(float)
            break
    if s is None:
        s = df.iloc[:, poz].astype(float)
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    return s[~s.index.duplicated()].sort_index().dropna()


def bedelsiz_serisi(t):
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


def satir(kod, f, x, v, i, cikis, olay):
    mom = f.iloc[i] / f.iloc[i - 20] - 1
    rs = (f.iloc[i] / f.iloc[i - 60] - 1) - (x.iloc[i] / x.iloc[i - 60] - 1)
    zirve = f.iloc[i] / f.iloc[i - 120:i + 1].max()
    v60 = v.iloc[i - 59:i + 1].mean()
    hacim = v.iloc[i - 19:i + 1].mean() / v60 if v60 > 0 else np.nan
    sonuc = (f.iloc[cikis] / f.iloc[i] - 1) - (x.iloc[cikis] / x.iloc[i] - 1)
    return {"hisse": kod, "giris": f.index[i].date(), "olay": olay, "mom": mom,
            "rs": rs, "zirve": zirve, "hacim": hacim, "sonuc": sonuc}


def isle_ic(kod, xu):
    hata = None
    for deneme in range(3):
        try:
            t = bp.Ticker(kod)
            ev = bedelsiz_serisi(t)
            if ev is None:
                return kod, [], None
            df = t.history(period="max")
            break
        except Exception as e:  # noqa: BLE001
            hata = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (deneme + 1))
    else:
        return kod, [], hata

    fiyat = kolon(df, ("Close", "close"), 3)
    hacim = kolon(df, ("Volume", "volume"), 4)
    ort = fiyat.index.intersection(xu.index)
    f, x = fiyat.loc[ort], xu.loc[ort]
    if len(f) < BASLANGIC + 60:
        return kod, [], None
    v = hacim.reindex(ort).fillna(0)

    evpos = []
    for tarih in ev.index:
        pos = f.index.searchsorted(pd.Timestamp(tarih).normalize())
        if pos < len(f):
            evpos.append(pos)

    satirlar = []
    for pos in evpos:
        i = pos - GIRIS
        if i >= BASLANGIC:
            satirlar.append(satir(kod, f, x, v, i, pos - 1, True))
    for i in range(BASLANGIC, len(f) - GIRIS, ADIM):
        if any(i < e <= i + 2 * GIRIS for e in evpos):
            continue
        satirlar.append(satir(kod, f, x, v, i, i + GIRIS - 1, False))
    return kod, satirlar, None


def isle(kod, xu):
    t0 = time.time()
    try:
        sonuc = isle_ic(kod, xu)
    except Exception as e:  # noqa: BLE001
        sonuc = (kod, [], f"{type(e).__name__}: {e}")
    sure = time.time() - t0
    if sure > 20:
        print(f"  yavas: {kod} {sure:.0f} sn")
    return sonuc


FILTRELER = {
    "Filtre yok": lambda d: np.ones(len(d), bool),
    "A  momentum > 0": lambda d: d.mom > 0,
    "B  goreceli guc > 0": lambda d: d.rs > 0,
    "C  momentum > 0 ve guc > 0": lambda d: (d.mom > 0) & (d.rs > 0),
    "D  C + zirveden en az %5 geride": lambda d: (d.mom > 0) & (d.rs > 0) & (d.zirve < 0.95),
    "E  D + hacim artiyor": lambda d: ((d.mom > 0) & (d.rs > 0) & (d.zirve < 0.95)
                                       & (d.hacim > 1)),
}


def ozet(s):
    s = s.dropna()
    if len(s) == 0:
        return "n=0"
    return (f"n={len(s)} | ortanca %{s.median()*100:.1f} | ort. %{s.mean()*100:.1f} | "
            f"pozitif %{(s > 0).mean()*100:.0f}")


def main():
    hisseler = hisse_listesi()
    print("Endeks fiyatlari aliniyor...")
    xu = kolon(bp.Index("XU100").history(period="max"), ("Close", "close"), 3)
    print(f"Endeks hazir ({len(xu)} gun). {len(hisseler)} hisse taranacak.")

    rows, hatalar = [], []
    islenen = 0
    ex = ThreadPoolExecutor(max_workers=ISCI)
    isler = [ex.submit(isle, k, xu) for k in hisseler]
    sure_doldu = False
    try:
        for fut in as_completed(isler, timeout=SURE_LIMITI):
            kod, sat, hata = fut.result()
            islenen += 1
            rows += sat
            if hata:
                hatalar.append((kod, hata))
            if islenen % 10 == 0:
                n_olay = sum(1 for r in rows if r["olay"])
                print(f"  {islenen}/{len(hisseler)} hisse, {n_olay} olay, {len(hatalar)} hata")
            if islenen >= ERKEN_DUR and not rows:
                print("Ilk hisselerde hic olay yok, duruyorum. Hatalar:", hatalar[:5])
                bitir(1)
    except ZamanAsimi:
        sure_doldu = True
        print(f"SURE LIMITI DOLDU: {islenen}/{len(hisseler)} hisse tamamlandi")
    ex.shutdown(wait=False, cancel_futures=True)

    if hatalar:
        print(f"{len(hatalar)} hissede hata, ornek:", hatalar[:5])
    if not rows:
        print("Hic satir yok.")
        bitir(1)

    df = pd.DataFrame(rows)
    df.to_csv("birlesim_satirlar.csv", index=False)
    ev = df[df.olay]
    bz = df[~df.olay]

    out = [f"Taranan hisse: {islenen}/{len(hisseler)}" + (" (sure doldu)" if sure_doldu else ""),
           f"Bedelsiz olayi (giris yapilabilen): {len(ev)} | bedelsiz olmayan donem: {len(bz)}",
           f"Giris: olaydan {GIRIS} gun once, sonuc: olaydan 1 gun oncesine endekse gore fazla getiri",
           "Fark > 0 ise bedelsiz, ayni filtrenin ustune ek katki saglar", ""]
    for ad, fn in FILTRELER.items():
        e = ev.loc[np.asarray(fn(ev)), "sonuc"]
        b = bz.loc[np.asarray(fn(bz)), "sonuc"]
        out.append(ad)
        out.append("  bedelsiz      : " + ozet(e))
        out.append("  bedelsiz degil: " + ozet(b))
        if e.notna().any() and b.notna().any():
            out.append(f"  fark (ortanca): %{(e.median() - b.median())*100:+.1f}")
        out.append("")

    rapor = "\n".join(out)
    print("\n" + rapor)
    with open("birlesim_rapor.txt", "w", encoding="utf-8") as fh:
        fh.write(rapor + "\n")
    bitir(0)


if __name__ == "__main__":
    main()
