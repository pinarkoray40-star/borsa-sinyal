"""Tavan sinyalini alsaydık ne olurdu? (tavan.py'nin devamı)

Sinyal günü kapanışında işaretler görülür. Alım ertesi gün kapanıştan yapılır (sinyali gördükten sonra
alabileceğin en erken fiyat). Ertesi gün zaten tavan olduysa alım yapılamaz kabul edilir ("kaçan tren").
Satış: alımdan 1, 3 veya 5 işlem günü sonra kapanıştan. Her işlemden gidiş-dönüş maliyet düşülür.
Karşılaştırma: aynı koşulda (son 10 günde tavan yok) rastgele alınan hisse.
Yatırım tavsiyesi değildir.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from signals import fetch_all, read_symbols

TAVAN = 0.095
QUIET = 10
HOLDS = (1, 3, 5)


def pr(df):
    return df.rank(axis=1, pct=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="5y")
    ap.add_argument("--symbols-file", default="bist_symbols.txt")
    ap.add_argument("--cost", type=float, default=0.004, help="gidiş-dönüş maliyet (0.004 = %0,4)")
    ap.add_argument("--out", default="tavan_getiri.md")
    args = ap.parse_args()

    symbols = read_symbols(args.symbols_file)
    print(f"{len(symbols)} sembol indiriliyor ({args.period})...")
    close, volume = fetch_all(symbols, args.period)
    if close.shape[1] < 20:
        raise SystemExit(f"Yeterli veri yok ({close.shape[1]} hisse).")

    chg = close.pct_change(fill_method=None)
    is_tavan = chg >= TAVAN
    prev_tavan = is_tavan.shift(1, fill_value=False).rolling(QUIET).max().fillna(1).astype(bool)
    quiet = ~prev_tavan

    r20 = close / close.shift(20) - 1
    r126 = close / close.shift(126) - 1
    vrat = volume / volume.rolling(20).mean().replace(0, np.nan)
    vol20 = chg.rolling(20).std()
    vol120 = chg.rolling(120).std()
    turn = (close * volume).rolling(20).median()

    valid = quiet & vrat.notna() & turn.notna() & vol120.notna() & r126.notna() & (close > 0)

    # alım: ertesi gün kapanış; ertesi gün tavan olduysa alınamaz
    next_tavan = is_tavan.shift(-1)
    entry = close.shift(-1)
    fill = valid & next_tavan.eq(False) & entry.notna()
    cost = args.cost

    ret = {}
    for h in HOLDS:
        ret[h] = (close.shift(-1 - h) / entry - 1 - cost)

    S = {
        "Hepsi (aynı koşulda rastgele hisse)": valid,
        "A: hacim 3x+ ve 20 günde -%15'ten fazla düşüş": (vrat >= 3.0) & (r20 < -0.15),
        "B: 20 günde +%30'dan fazla ve hacim 2x+": (r20 > 0.30) & (vrat >= 2.0),
        "C: oynaklık en yüksek %20 ve hacim 3x+": (pr(vol20) >= 0.80) & (vrat >= 3.0),
        "D: 20 günde -%15'ten fazla düşüş ve hacim 2x+": (r20 < -0.15) & (vrat >= 2.0),
        "E: sadece hacim 3x+": vrat >= 3.0,
    }
    S = {k: (v.fillna(False) & valid) for k, v in S.items()}

    idx = close.index[valid.any(axis=1)]
    mid = idx[len(idx) // 2]
    first = (close.index <= mid)[:, None]

    def stats(mask, h, sub=None):
        m = (mask & fill).values
        if sub is not None:
            m = m & sub
        v = ret[h].values[m]
        v = v[~np.isnan(v)]
        return v

    L = []
    L.append("# Tavan sinyalini alsaydık ne olurdu?")
    L.append(f"{close.shape[1]} hisse · {close.index[0].date()} → {close.index[-1].date()} · "
             f"maliyet gidiş-dönüş %{cost * 100:.1f} düşüldü.")
    L.append("**İşlem kuralı:** sinyal günü kapanışında işaret görülür → ertesi gün kapanıştan al → "
             "1 / 3 / 5 gün sonra kapanıştan sat. Ertesi gün tavan olduysa alınamaz sayılır.\n")

    for name, mask in S.items():
        sig_all = int((mask).values.sum())
        sig_fill = int((mask & fill).values.sum())
        missed = sig_all - sig_fill
        # alınamayanların oranı: sinyal sonrası ertesi gün tavan olan
        nt = int((mask & next_tavan.eq(True)).values.sum())
        L.append(f"## {name}")
        L.append(f"- Sinyal sayısı {sig_all:,} · alınabilen {sig_fill:,} · "
                 f"ertesi gün tavan olup kaçan %{(nt / sig_all * 100) if sig_all else 0:.0f}".replace(",", "."))
        for h in HOLDS:
            v = stats(mask, h)
            if len(v) < 100:
                L.append(f"- {h} gün tutuş: yeterli işlem yok")
                continue
            v1 = stats(mask, h, first)
            v2 = stats(mask, h, ~first)
            L.append(f"- **{h} gün tutuş:** ortalama {v.mean() * 100:+.1f}% · medyan {np.median(v) * 100:+.1f}% · "
                     f"kazanan %{(v > 0).mean() * 100:.0f} · en kötü %10'luk dilim {np.percentile(v, 10) * 100:+.0f}% · "
                     f"ilk yarı {v1.mean() * 100:+.1f}% / ikinci yarı {v2.mean() * 100:+.1f}%")
        L.append("")

    L.append("> Uyarı: Alım ve satış kapanış fiyatından yapılmış sayılır; gerçekte fiyat kayar (özellikle "
             "düşük hacimli hisselerde). Taban olan gün satış yapılamaz, bu hesaba katılmadı. Veri bugünkü "
             "hisse listesidir; işlemden kalkan şirketler yoktur. Gözlemler üst üste bindiği için sayılar "
             "göründüğünden daha az bağımsızdır. Yatırım tavsiyesi değildir.")

    text = "\n".join(L)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
