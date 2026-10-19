"""Büyük kazananları (6 ayda 2x / 3x) başlangıçta nasıl görünüyordu diye inceler.

Her (hisse, gün) için o günkü bilgilerle ileriki 126 işlem günü getirisine bakılır.
"Kazanan" = 126 işlem günü (~6 ay) sonra en az +%100 (2x) veya +%200 (3x).
Soru: hangi işaretler kazananlarda sıradan hisselerden daha sık görülüyor? (lift)
Yatırım tavsiyesi değildir.
"""
import argparse
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd

from signals import fetch_all, read_symbols, rsi

H = 126


def pr(df):
    return df.rank(axis=1, pct=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="5y")
    ap.add_argument("--symbols-file", default="bist_symbols.txt")
    ap.add_argument("--out", default="winners.md")
    args = ap.parse_args()

    symbols = read_symbols(args.symbols_file)
    print(f"{len(symbols)} sembol indiriliyor ({args.period})...")
    close, volume = fetch_all(symbols, args.period)
    if close.shape[1] < 20:
        raise SystemExit(f"Yeterli veri yok ({close.shape[1]} hisse).")

    chg = close.pct_change(fill_method=None)
    r126 = close / close.shift(126) - 1
    r20 = close / close.shift(20) - 1
    hi55 = close.rolling(55).max()
    hi252 = close.rolling(252, min_periods=200).max()
    lo252 = close.rolling(252, min_periods=200).min()
    sma200 = close.rolling(200).mean()
    vrat = volume / volume.rolling(20).mean().replace(0, np.nan)
    vol20 = chg.rolling(20).std()
    vol120 = chg.rolling(120).std()
    turn = (close * volume).rolling(20).median()
    rs = close.apply(lambda s: rsi(s, 14))
    pos52 = (close - lo252) / (hi252 - lo252)

    fwd = close.shift(-H) / close - 1
    valid = fwd.notna() & r126.notna() & sma200.notna() & vrat.notna() & turn.notna()
    fv = fwd.where(valid)
    W2 = (fv >= 1.0)
    W3 = (fv >= 2.0)
    n_all = int(valid.values.sum())
    base2 = W2.values[valid.values].mean()
    base3 = W3.values[valid.values].mean()

    B = {
        "Momentum: 6 ay en iyi %10": pr(r126) >= 0.90,
        "Momentum: 6 ay orta (%30-%70)": (pr(r126) >= 0.30) & (pr(r126) < 0.70),
        "Momentum: 6 ay en kötü %30": pr(r126) < 0.30,
        "Son 1 ay +%20'den fazla": r20 > 0.20,
        "Son 1 ay -%10'dan fazla düşüş": r20 < -0.10,
        "55 gün zirvenin %3 yakınında": close >= hi55 * 0.97,
        "55 gün zirveden %30+ uzakta": close < hi55 * 0.70,
        "52 hafta aralığının üst %20'sinde": pos52 >= 0.80,
        "52 hafta aralığının alt %20'sinde": pos52 <= 0.20,
        "Hacim ortalamanın 1,5 katı+": vrat >= 1.5,
        "Hacim ortalamanın 3 katı+": vrat >= 3.0,
        "Oynaklık en yüksek %20": pr(vol20) >= 0.80,
        "Oynaklık en düşük %20": pr(vol20) <= 0.20,
        "Sıkışma (20g oynaklık < 120g'nin yarısı)": vol20 < 0.5 * vol120,
        "İşlem hacmi (TL) en düşük %30 (küçük/illikit)": pr(turn) <= 0.30,
        "İşlem hacmi (TL) en yüksek %30 (büyük/likit)": pr(turn) >= 0.70,
        "SMA200 üstünde": close > sma200,
        "SMA200 altında": close < sma200,
        "RSI > 70": rs > 70,
        "RSI < 30": rs < 30,
        "Fiyat 10 TL altı": close < 10,
        "Fiyat 100 TL üstü": close > 100,
    }
    B = {k: (v & valid) for k, v in B.items()}

    idx = close.index[valid.any(axis=1)]
    mid = idx[len(idx) // 2]
    first = (close.index <= mid)[:, None]
    vm = valid.values

    def hit(mask, W, sub=None):
        m = mask.values & vm
        if sub is not None:
            m = m & sub
        n = int(m.sum())
        return n, int((W.values & m).sum())

    L = []
    L.append("# Büyük kazananlar başlangıçta nasıl görünüyordu?")
    L.append(f"{close.shape[1]} hisse · {close.index[0].date()} → {close.index[-1].date()} · "
             f"{n_all:,} (hisse-gün) gözlemi · ileriye bakış: {H} işlem günü (~6 ay).".replace(",", "."))
    L.append(f"**Genel olasılık:** rastgele bir gün alınan bir hissenin 6 ay sonra 2 katına çıkma ihtimali "
             f"**%{base2 * 100:.1f}**, 3 katına çıkma ihtimali **%{base3 * 100:.1f}**.\n")
    L.append("**Okuma:** *Lift* = o işaret varken kazanma ihtimali ÷ genel ihtimal. "
             "1 = fark yok, 2 = iki kat daha sık, 0,5 = yarı sıklıkta. "
             "*Tutarlılık* = ilk yarı / ikinci yarı lift (ikisi de 1'in üstündeyse işaret güvenilir).\n")

    rows = []
    for name, mask in B.items():
        n, w2 = hit(mask, W2)
        _, w3 = hit(mask, W3)
        if n < 500:
            continue
        p2, p3 = w2 / n, w3 / n
        n1, a1 = hit(mask, W2, first)
        n2, a2 = hit(mask, W2, ~first)
        b1 = hit(valid, W2, first)
        b2 = hit(valid, W2, ~first)
        l1 = (a1 / n1) / (b1[1] / b1[0]) if n1 and b1[1] else np.nan
        l2 = (a2 / n2) / (b2[1] / b2[0]) if n2 and b2[1] else np.nan
        rows.append(dict(name=name, n=n, p2=p2, p3=p3, l2=p2 / base2, l3=p3 / base3 if base3 else np.nan,
                         w2=w2, w3=w3, l1=l1, l1b=l2))
    rows.sort(key=lambda r: -r["l2"])

    L.append("## İşaretlere göre 2x ve 3x ihtimali (en yüksek lift üstte)\n")
    for r in rows:
        cons = "✔ tutarlı" if (r["l1"] > 1 and r["l1b"] > 1) else "✖ tutarsız"
        L.append(f"- **{r['name']}**\n"
                 f"  - 2x ihtimali %{r['p2'] * 100:.1f} (lift {r['l2']:.1f}x) · 3x ihtimali %{r['p3'] * 100:.1f} "
                 f"(lift {r['l3']:.1f}x)\n"
                 f"  - gözlem {r['n']:,} · 2x yapan {r['w2']:,} · tutarlılık {r['l1']:.1f} / {r['l1b']:.1f} {cons}"
                 .replace(",", "."))
    L.append("")

    # ikili kombinasyonlar
    good = [r for r in rows if r["w2"] >= 80 and r["l2"] > 1.2]
    names = [r["name"] for r in good[:8]]
    combos = []
    for a, b in itertools.combinations(names, 2):
        m = B[a] & B[b]
        n, w2 = hit(m, W2)
        _, w3 = hit(m, W3)
        if n < 400 or w2 < 25:
            continue
        n1, a1 = hit(m, W2, first)
        n2, a2 = hit(m, W2, ~first)
        combos.append((w2 / n / base2, a, b, n, w2, w3, (a1 / n1 if n1 else 0), (a2 / n2 if n2 else 0), m))
    combos.sort(key=lambda x: -x[0])
    L.append("## En iyi ikili kombinasyonlar\n")
    if not combos:
        L.append("- Yeterli gözlemli kombinasyon yok.")
    for lift, a, b, n, w2, w3, p1, p2, m in combos[:6]:
        fm = fv.values[m.values & vm]
        L.append(f"- **{a} + {b}**\n"
                 f"  - 2x ihtimali %{w2 / n * 100:.1f} (lift {lift:.1f}x) · 3x yapan {w3} / {n:,} gözlem\n"
                 f"  - 6 ay sonra ortalama {fm.mean() * 100:+.0f}% · medyan {np.median(fm) * 100:+.0f}% · "
                 f"zararda bitenler %{(fm < 0).mean() * 100:.0f}".replace(",", "."))
    L.append("")

    # gerçek örnekler: 3x başlangıçları
    starts = W3 & ~W3.shift(1, fill_value=False)
    ev = []
    for j, sym in enumerate(close.columns):
        for i in np.where(starts.iloc[:, j].values)[0]:
            ev.append((sym, i, float(fv.iat[i, j])))
    ev.sort(key=lambda x: -x[2])
    L.append("## Gerçek örnekler: 6 ayda en çok kazananlar (başlangıç günü)\n")
    seen = set()
    cnt = 0
    for sym, i, ret in ev:
        if sym in seen:
            continue
        seen.add(sym)
        j = list(close.columns).index(sym)
        L.append(f"- **{sym.replace('.IS', '')}** · {close.index[i]:%d.%m.%Y} · 6 ay sonra **{ret * 100:+.0f}%**\n"
                 f"  - o gün: 6 ay momentum {r126.iat[i, j] * 100:+.0f}% (sıra %{pr(r126).iat[i, j] * 100:.0f}) · "
                 f"1 ay {r20.iat[i, j] * 100:+.0f}% · zirveye uzaklık {(close.iat[i, j] / hi55.iat[i, j] - 1) * 100:+.0f}% · "
                 f"hacim {vrat.iat[i, j]:.1f}x · fiyat {close.iat[i, j]:.1f} TL")
        cnt += 1
        if cnt >= 15:
            break
    L.append("")

    # kazananların medyan profili
    m3 = (W3.values & vm)
    if m3.sum() >= 30:
        def med(df, mask):
            return float(np.nanmedian(df.values[mask]))
        L.append("## 3x yapanların ortalama profili (medyan) ve tüm gözlemler\n")
        allm = vm
        L.append(f"- 6 ay momentum: kazananlar {med(r126, m3) * 100:+.0f}% · hepsi {med(r126, allm) * 100:+.0f}%")
        L.append(f"- 1 ay getiri: kazananlar {med(r20, m3) * 100:+.0f}% · hepsi {med(r20, allm) * 100:+.0f}%")
        L.append(f"- Hacim oranı: kazananlar {med(vrat, m3):.2f}x · hepsi {med(vrat, allm):.2f}x")
        L.append(f"- Zirveye uzaklık: kazananlar {med(close / hi55 - 1, m3) * 100:+.0f}% · hepsi {med(close / hi55 - 1, allm) * 100:+.0f}%")
        L.append(f"- İşlem hacmi sırası (0=en az, 1=en çok): kazananlar {med(pr(turn), m3):.2f} · hepsi {med(pr(turn), allm):.2f}")
        L.append(f"- Fiyat: kazananlar {med(close, m3):.1f} TL · hepsi {med(close, allm):.1f} TL")
        L.append("")

    L.append("> Uyarı: Veri bugünkü hisse listesidir; batan veya borsadan çıkan şirketler yoktur, "
             "bu yüzden kazananların oranı gerçekte daha düşüktür. Gözlemler üst üste bindiği için "
             "sayılar göründüğünden daha az bağımsızdır. Geçmiş gelecek için garanti değildir. "
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
