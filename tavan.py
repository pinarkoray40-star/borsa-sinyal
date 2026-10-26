"""Tavan öncesi iz var mı? Tavan olan günlerin ÖNCESİNE bakar.

Soru: Bir hisse tavan olmadan önce (bir gün, üç gün, beş gün önce) sıradan hisselerden farklı görünüyor mu?
Tavan = günlük kapanış getirisi >= %9,5 (BIST günlük sınırı %10; yüksek/düşük verisi olmadığı için
kapanıştan yaklaşık hesaplanır).
Hedef: "Önümüzdeki 5 işlem günü içinde ilk tavan" (son 10 günde tavan yoksa).
Her işaret için: o işaret varken hedefin gerçekleşme ihtimali ÷ genel ihtimal = lift.
Yatırım tavsiyesi değildir.
"""
import argparse
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd

from signals import fetch_all, read_symbols

TAVAN = 0.095     # >= %9,5 kapanış getirisi = tavan sayılır
AHEAD = 5         # kaç işlem günü ileriye bakılır
QUIET = 10        # "ilk tavan" için önceki kaç günde tavan olmamalı


def pr(df):
    return df.rank(axis=1, pct=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="5y")
    ap.add_argument("--symbols-file", default="bist_symbols.txt")
    ap.add_argument("--out", default="tavan.md")
    args = ap.parse_args()

    symbols = read_symbols(args.symbols_file)
    print(f"{len(symbols)} sembol indiriliyor ({args.period})...")
    close, volume = fetch_all(symbols, args.period)
    if close.shape[1] < 20:
        raise SystemExit(f"Yeterli veri yok ({close.shape[1]} hisse).")

    chg = close.pct_change(fill_method=None)
    is_tavan = chg >= TAVAN

    # önceki QUIET günde tavan yok
    prev_tavan = is_tavan.shift(1, fill_value=False).rolling(QUIET).max().fillna(1).astype(bool)
    quiet = ~prev_tavan

    # hedef: sonraki AHEAD günden herhangi birinde tavan
    fut = is_tavan.shift(-1).fillna(False).astype(bool)
    for k in range(2, AHEAD + 1):
        fut = fut | is_tavan.shift(-k).fillna(False).astype(bool)
    # son AHEAD günün sonucu bilinmiyor
    known = pd.Series(False, index=close.index)
    known.iloc[:-AHEAD] = True
    known_df = pd.DataFrame(np.repeat(known.values[:, None], close.shape[1], axis=1),
                            index=close.index, columns=close.columns)

    r1 = chg
    r5 = close / close.shift(5) - 1
    r20 = close / close.shift(20) - 1
    r126 = close / close.shift(126) - 1
    hi55 = close.rolling(55).max()
    hi252 = close.rolling(252, min_periods=200).max()
    lo252 = close.rolling(252, min_periods=200).min()
    pos52 = (close - lo252) / (hi252 - lo252)
    vrat = volume / volume.rolling(20).mean().replace(0, np.nan)
    vrat5 = volume.rolling(5).mean() / volume.rolling(60).mean().replace(0, np.nan)
    vol20 = chg.rolling(20).std()
    vol120 = chg.rolling(120).std()
    turn = (close * volume).rolling(20).median()
    n_tavan_90 = is_tavan.rolling(90).sum()
    drawdown = close / hi252 - 1

    valid = (known_df & quiet & vrat.notna() & turn.notna() & vol120.notna()
             & r126.notna() & (close > 0))
    W = fut & valid
    vm = valid.values
    n_all = int(vm.sum())
    base = W.values[vm].mean()

    B = {
        "Dünkü getiri +%3'ten fazla (kıpırdanma)": r1 > 0.03,
        "Son 5 gün +%10'dan fazla": r5 > 0.10,
        "Son 20 gün -%15'ten fazla düşüş (dipte)": r20 < -0.15,
        "Son 20 gün +%30'dan fazla": r20 > 0.30,
        "Bugün hacim ortalamanın 2 katı+": vrat >= 2.0,
        "Bugün hacim ortalamanın 3 katı+": vrat >= 3.0,
        "Son 5 gün hacmi 60 gün ortalamasının 1,5 katı+": vrat5 >= 1.5,
        "Sessizlik: son 5 gün hacmi 60 gün ortalamasının %60'ından az": vrat5 < 0.6,
        "Sıkışma (20g oynaklık < 120g'nin yarısı)": vol20 < 0.5 * vol120,
        "Oynaklık en yüksek %20": pr(vol20) >= 0.80,
        "İşlem hacmi (TL) en düşük %20 (çok küçük)": pr(turn) <= 0.20,
        "İşlem hacmi (TL) en düşük %40": pr(turn) <= 0.40,
        "Son 90 günde 1+ tavan yapmış (spekülatif geçmiş)": n_tavan_90 >= 1,
        "Son 90 günde 3+ tavan yapmış": n_tavan_90 >= 3,
        "52 hafta aralığının alt %20'sinde": pos52 <= 0.20,
        "52 hafta aralığının üst %20'sinde": pos52 >= 0.80,
        "Zirveden %50+ düşmüş": drawdown < -0.50,
        "55 gün zirvenin %3 yakınında": close >= hi55 * 0.97,
        "Fiyat 20 TL altı": close < 20,
        "Fiyat 100 TL üstü": close > 100,
    }
    R = {k: v.fillna(False) for k, v in B.items()}   # ham işaretler (son gün için de kullanılır)
    B = {k: (v & valid) for k, v in R.items()}

    idx = close.index[valid.any(axis=1)]
    mid = idx[len(idx) // 2]
    first = (close.index <= mid)[:, None]

    def hit(mask, sub=None):
        m = mask.values & vm
        if sub is not None:
            m = m & sub
        n = int(m.sum())
        return n, int((W.values & m).sum())

    b1 = hit(valid, first)
    b2 = hit(valid, ~first)

    L = []
    L.append("# Tavan öncesi iz var mı?")
    L.append(f"{close.shape[1]} hisse · {close.index[0].date()} → {close.index[-1].date()} · "
             f"{n_all:,} (hisse-gün) gözlemi · tavan = günlük getiri ≥ %{TAVAN * 100:.1f}.".replace(",", "."))
    L.append(f"**Genel olasılık:** son {QUIET} günde tavan yapmamış rastgele bir hissenin önümüzdeki "
             f"{AHEAD} işlem günü içinde tavan olma ihtimali **%{base * 100:.1f}**.\n")
    L.append("**Okuma:** *Lift* = işaret varken ihtimal ÷ genel ihtimal. 1 = fark yok, 3 = üç kat daha sık. "
             "*İsabet* = işaret çıkan 100 hisse-günden kaçı sonraki 5 günde tavan oldu. "
             "Tutarlılık = ilk yarı / ikinci yarı lift (ikisi de 1'in üstündeyse işaret güvenilir).\n")

    rows = []
    for name, mask in B.items():
        n, w = hit(mask)
        if n < 500 or w < 20:
            continue
        p = w / n
        n1, a1 = hit(mask, first)
        n2, a2 = hit(mask, ~first)
        l1 = (a1 / n1) / (b1[1] / b1[0]) if n1 and b1[1] else np.nan
        l2 = (a2 / n2) / (b2[1] / b2[0]) if n2 and b2[1] else np.nan
        rows.append(dict(name=name, n=n, w=w, p=p, lift=p / base, l1=l1, l2=l2))
    rows.sort(key=lambda r: -r["lift"])

    L.append("## Tek tek işaretler (en yüksek lift üstte)\n")
    for r in rows:
        cons = "✔ tutarlı" if (r["l1"] > 1 and r["l2"] > 1) else "✖ tutarsız"
        L.append(f"- **{r['name']}**\n"
                 f"  - isabet %{r['p'] * 100:.1f} (lift {r['lift']:.1f}x) · gözlem {r['n']:,} · "
                 f"tavan olan {r['w']:,}\n"
                 f"  - tutarlılık {r['l1']:.1f} / {r['l2']:.1f} {cons}".replace(",", "."))
    L.append("")

    good = [r for r in rows if r["w"] >= 30 and r["lift"] > 1.2]
    names = [r["name"] for r in good[:8]]
    combos = []
    for a, b in itertools.combinations(names, 2):
        m = B[a] & B[b]
        n, w = hit(m)
        if n < 200 or w < 15:
            continue
        n1, a1 = hit(m, first)
        n2, a2 = hit(m, ~first)
        combos.append((w / n / base, a, b, n, w, (a1 / n1 if n1 else 0), (a2 / n2 if n2 else 0)))
    combos.sort(key=lambda x: -x[0])
    L.append("## En iyi ikili kombinasyonlar\n")
    if not combos:
        L.append("- Yeterli gözlemli kombinasyon yok.")
    for lift, a, b, n, w, p1, p2 in combos[:6]:
        L.append(f"- **{a} + {b}**\n"
                 f"  - isabet %{w / n * 100:.1f} (lift {lift:.1f}x) · {w} / {n:,} gözlem\n"
                 f"  - ilk yarı %{p1 * 100:.1f} · ikinci yarı %{p2 * 100:.1f}".replace(",", "."))
    L.append("")

    # gerçek örnekler: en son ilk-tavanlar ve bir gün öncesindeki profil
    ev = []
    for j, sym in enumerate(close.columns):
        col = is_tavan.iloc[:, j].values
        for i in np.where(col)[0]:
            if i >= 1 and not prev_tavan.iat[i, j] and not np.isnan(vrat.iat[i - 1, j]):
                ev.append((i, sym))
    ev.sort(reverse=True)
    L.append("## Son ilk-tavanlar ve bir gün ÖNCEKİ görünümleri\n")
    for i, sym in ev[:15]:
        j = list(close.columns).index(sym)
        k = i - 1
        L.append(f"- **{sym.replace('.IS', '')}** · tavan günü {close.index[i]:%d.%m.%Y} ({chg.iat[i, j] * 100:+.1f}%)\n"
                 f"  - bir gün önce: hacim {vrat.iat[k, j]:.1f}x · son 5 gün {r5.iat[k, j] * 100:+.0f}% · "
                 f"son 20 gün {r20.iat[k, j] * 100:+.0f}% · hacim sırası %{pr(turn).iat[k, j] * 100:.0f} · "
                 f"fiyat {close.iat[k, j]:.1f} TL")

    # tavan öncesi medyan profil (t-1) ile tüm gözlemler
    if len(ev) >= 30:
        def med(df):
            vals = [df.iat[i - 1, list(close.columns).index(s)] for i, s in ev]
            return float(np.nanmedian(vals))
        L.append("\n## Tavandan bir gün önce medyan profil ve tüm gözlemler\n")
        allv = vm
        def mall(df):
            return float(np.nanmedian(df.values[allv]))
        L.append(f"- Hacim oranı: tavan öncesi {med(vrat):.2f}x · hepsi {mall(vrat):.2f}x")
        L.append(f"- 5 gün getiri: tavan öncesi {med(r5) * 100:+.1f}% · hepsi {mall(r5) * 100:+.1f}%")
        L.append(f"- 20 gün getiri: tavan öncesi {med(r20) * 100:+.1f}% · hepsi {mall(r20) * 100:+.1f}%")
        L.append(f"- İşlem hacmi sırası (0=en az): tavan öncesi {med(pr(turn)):.2f} · hepsi {mall(pr(turn)):.2f}")
        L.append(f"- Fiyat: tavan öncesi {med(close):.1f} TL · hepsi {mall(close):.1f} TL")

    # bugünün adayları: en iyi tutarlı kombinasyon
    L.append("\n## Bugünün adayları (son gün, en iyi tutarlı kombinasyon)\n")
    cons_combos = [c for c in combos if c[5] > base and c[6] > base]
    if cons_combos:
        _, a, b, *_ = cons_combos[0]
        last_ok = R[a].iloc[-1] & R[b].iloc[-1] & ~prev_tavan.iloc[-1]
        cands = [s for s in close.columns[last_ok.values]]
        L.append(f"- Kombinasyon: **{a} + {b}** · son gün {close.index[-1]:%d.%m.%Y}")
        if cands:
            L.append("- Adaylar: " + ", ".join(c.replace(".IS", "") for c in cands[:25])
                     + " (bunlar tavan olacak demek değil; sadece iz taşıyorlar)")
        else:
            L.append("- Bugün koşulu sağlayan hisse yok.")
    else:
        L.append("- Tutarlı bir kombinasyon bulunamadı: tavanı önceden yakalamak için güvenilir iz yok.")

    L.append("\n> Uyarı: Tavan günü yaklaşık kapanış getirisinden bulunur (yüksek/düşük verisi yok). "
             "Veri bugünkü hisse listesidir; işlemden kalkan şirketler yoktur. Gözlemler üst üste bindiği için "
             "sayılar göründüğünden daha az bağımsızdır. Tavanın nedeni çoğu zaman KAP haberidir ve haber "
             "önceden bilinemez; bu analiz sadece haber öncesi iz olup olmadığına bakar. "
             "Yatırım tavsiyesi değildir.")

    text = "\n".join(L)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
