"""
Bedelsiz oncesi davranis testi (kaba hali).

Soru: Bedelsiz sermaye artirimi yapan hisseler, hak kullanim (bolunme) tarihinden
once endekse (XU100) gore fazla getiri sagliyor mu?

Kaynak: borsapy (hisse.splits + fiyat gecmisi). SPK basvuru/onay tarihleri YOK;
bu yuzden olay tarihi = bedelsiz hakkinin islem gordugu tarih.

Kullanim:
    pip install borsapy pandas numpy
    python bedelsiz_test.py                # hisse listesi endeksten alinir
    python bedelsiz_test.py hisseler.txt   # istersen kendi listen (satir basi bir kod)

Cikti: bedelsiz_olaylar.csv ve bedelsiz_rapor.txt
"""
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import borsapy as bp
except ImportError:
    sys.exit("borsapy kurulu degil: pip install borsapy")

# (baslangic, bitis) -> olaydan kac islem gunu once; 1 = olaydan bir gun once
PENCERELER = [(60, 40), (40, 20), (20, 1)]
MIN_ORAN = 1.2  # split orani >= 1.2 (yani en az %20 bedelsiz) olanlar


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
    sys.exit("Hisse listesi alinamadi. hisseler.txt verip tekrar dene.")


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
    """i_son: olay gununun pozisyonu. bas/bit: olaydan kac gun once."""
    a, b = i_son - bas, i_son - bit
    if a < 0 or b <= a:
        return np.nan
    return seri.iloc[b] / seri.iloc[a] - 1


def main():
    hisseler = hisse_listesi()
    print("Endeks fiyatlari aliniyor...")
    xu = kapanis(bp.Index("XU100").history(period="max"))

    olaylar, baz = [], []
    for n, kod in enumerate(hisseler, 1):
        try:
            t = bp.Ticker(kod)
            sp = t.splits
            if sp is None or len(sp) == 0:
                continue
            fiyat = kapanis(t.history(period="max"))
        except Exception as e:  # noqa: BLE001
            print(f"  atlandi {kod}: {e}")
            continue
        if len(fiyat) < 150:
            continue

        # fiyat ve endeksi ortak gunlere hizala
        ort = fiyat.index.intersection(xu.index)
        f, x = fiyat.loc[ort], xu.loc[ort]
        if len(f) < 150:
            continue

        # baz: bu hissenin her 20 gunluk dilimdeki fazla getirisi (olaydan bagimsiz)
        for i in range(60, len(f), 20):
            baz.append((f.iloc[i] / f.iloc[i - 20] - 1) - (x.iloc[i] / x.iloc[i - 20] - 1))

        sp = sp.copy()
        sp.index = pd.to_datetime(sp.index)
        if getattr(sp.index, "tz", None) is not None:
            sp.index = sp.index.tz_localize(None)
        for tarih, oran in sp.items():
            try:
                oran = float(oran)
            except (TypeError, ValueError):
                continue
            if oran < MIN_ORAN:
                continue
            pos = f.index.searchsorted(pd.Timestamp(tarih).normalize())
            if pos >= len(f) or pos < 61:
                continue
            satir = {"hisse": kod, "tarih": f.index[pos].date(), "oran": oran}
            for bas, bit in PENCERELER:
                h = pencere_getiri(f, pos, bas, bit)
                e = pencere_getiri(x, pos, bas, bit)
                satir[f"fazla_{bas}_{bit}"] = h - e
            olaylar.append(satir)
        if n % 50 == 0:
            print(f"  {n}/{len(hisseler)} hisse tarandi, {len(olaylar)} olay")

    if not olaylar:
        sys.exit("Hic olay bulunamadi (borsapy veri/sutun yapisi degismis olabilir).")

    df = pd.DataFrame(olaylar)
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
