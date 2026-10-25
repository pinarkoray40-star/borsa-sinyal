"""Stres testi: 'Birleşik (3+4+5)' varyantı gerçekçi koşullarda da ayakta kalıyor mu?

Test edilen varyant: RSI>70'te satma yok + trend onaylı giriş + momentum sıralı + 10 hisse.

Senaryolar:
  A) Orijinal test (sinyal günü kapanıştan al/sat, maliyet %0,2)
  B) Tavan kapatan günde alım yok, taban kapatan günde çıkış yok (ertesi güne kalır)
  C) B + maliyet/kayma %0,7 (tek yön)
  D) C + test dönemindeki EN İYİ 5 İŞLEM çıkarıldı (bir kaç fırlayan hisseye mi bağlı?)
  E) C ama 7 hisse
  F) C ama 15 hisse
  G) C ama sinyalden 1 GÜN SONRA işlem (alarm gelir, ertesi gün kapanışta alırım)

Ayrıca en iyi 5 işlem listelenir ve yıl yıl getiri verilir.

Kullanım (GitHub'da "Analiz calistir", dosya adı: stress.py):
    python stress.py --period 5y

Yatırım tavsiyesi değildir.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import WINDOW, indicators
from diagnose import WARMUP, clean_prices, md_table
from signals import fetch_all, read_symbols
from variants import summarize

LIMIT_UP = 0.095     # günlük +%9,5 ve üstü kapanış ~ tavan, alınamaz
LIMIT_DN = -0.095    # günlük -%9,5 ve altı kapanış ~ taban, satılamaz


def strat_x(px, ind, liquid, no_buy=None, no_exit=None, slots=10, stop=0.08, max_age=WINDOW) -> pd.DataFrame:
    """Birleşik varyant (exit_rsi=False, trend_entry=True, rank='mom'); alım/çıkış engeli eklenebilir."""
    rsi_, line, sig = ind["rsi"], ind["macd"], ind["macd_sig"]
    sma20, sma50 = ind["sma20"], ind["sma50"]
    up = ((line.shift(1) <= sig.shift(1)) & (line > sig)).astype(float)
    dn = ((line.shift(1) >= sig.shift(1)) & (line < sig)).astype(float)
    gc = ((sma20.shift(1) <= sma50.shift(1)) & (sma20 > sma50)).astype(float)
    dc = ((sma20.shift(1) >= sma50.shift(1)) & (sma20 < sma50)).astype(float)
    score = ((rsi_ < 30).astype(float) + up - dn + gc - dc).where(sma50.notna())
    mom = px / px.shift(126) - 1
    trend_ok = (px > sma50) & (sma20 > sma50)

    P, S, Mom = px.values, score.values, mom.values
    Lq, Tr = liquid.values, trend_ok.values
    NB = no_buy if no_buy is not None else np.zeros(P.shape, dtype=bool)
    NE = no_exit if no_exit is not None else np.zeros(P.shape, dtype=bool)
    W = np.zeros(P.shape)
    held = {}
    with np.errstate(invalid="ignore"):
        for i in range(len(px)):
            for j in list(held):
                e, a = held[j]
                a += 1
                held[j] = (e, a)
                p = P[i, j]
                hit = (S[i, j] <= -1) or (np.isfinite(p) and p <= e * (1 - stop)) or a >= max_age
                if hit and not NE[i, j]:
                    del held[j]
            free = slots - len(held)
            if free > 0:
                ok = (S[i] >= 1) & Lq[i] & Tr[i] & np.isfinite(P[i]) & ~NB[i]
                for j in held:
                    ok[j] = False
                idx = np.where(ok)[0]
                if len(idx):
                    key = {j: (-S[i, j], -Mom[i, j] if np.isfinite(Mom[i, j]) else np.inf) for j in idx}
                    for j in sorted(idx, key=lambda j: key[j])[:free]:
                        held[j] = (P[i, j], 0)
            for j in held:
                W[i, j] = 1.0 / slots
    return pd.DataFrame(W, index=px.index, columns=px.columns)


def daily(w, rets, cost, cash_rate):
    wl = w.shift(1).fillna(0.0)
    gross = (wl * rets.fillna(0.0)).sum(axis=1)
    turnover = (wl - wl.shift(1).fillna(0.0)).abs().sum(axis=1)
    idle = (1.0 - wl.sum(axis=1)).clip(lower=0.0)
    return gross - turnover * cost + idle * ((1 + cash_rate) ** (1 / 252) - 1)


def trades_detail(w, px):
    """(sütun, giriş günü, çıkış günü, getiri) listesi."""
    W = w.values > 0
    P = px.values
    n = len(px)
    out = []
    for j in range(W.shape[1]):
        col = W[:, j]
        if not col.any():
            continue
        d = np.diff(np.concatenate(([0], col.astype(int), [0])))
        for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1):
            e = min(b + 1, n - 1)
            p0, p1 = P[a, j], P[e, j]
            if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                out.append((j, a, e, p1 / p0 - 1))
    return out


def report(close, volume, xu100, cash_rate) -> str:
    px = close.sort_index().ffill(limit=5)
    vol = volume.reindex(index=px.index, columns=px.columns)
    pxc, rc, _, _ = clean_prices(px)
    liquid = ((px * vol).rolling(20).median().rank(axis=1, pct=True) >= 0.30)
    ind = indicators(pxc)

    tavan = (rc >= LIMIT_UP).values
    taban = (rc <= LIMIT_DN).values
    cols_n = list(pxc.columns)

    rows, series = {}, {}

    def add(name, w, cost, lag=0):
        wl = w.shift(lag).fillna(0.0) if lag else w
        d = daily(wl, rc, cost, cash_rate)
        rows[name] = summarize(d, wl, pxc)
        series[name] = d
        print(f"  {name}: yıllık {rows[name]['yillik_%']}%")

    wA = strat_x(pxc, ind, liquid)
    add("A) Orijinal (maliyet %0,2)", wA, 0.002)

    wB = strat_x(pxc, ind, liquid, no_buy=tavan, no_exit=taban)
    add("B) Tavanda alım yok, tabanda çıkış yok", wB, 0.002)

    wC = strat_x(pxc, ind, liquid, no_buy=tavan, no_exit=taban)
    add("C) B + maliyet %0,7", wC, 0.007)

    # D) en iyi 5 işlemi çıkar
    tr = sorted(trades_detail(wC, pxc), key=lambda t: -t[3])
    top5 = tr[:5]
    nb = tavan.copy()
    for j, a, e, r in top5:
        nb[a, j] = True
    wD = strat_x(pxc, ind, liquid, no_buy=nb, no_exit=taban)
    add("D) C + en iyi 5 işlem çıkarıldı", wD, 0.007)

    add("E) C ama 7 hisse", strat_x(pxc, ind, liquid, no_buy=tavan, no_exit=taban, slots=7), 0.007)
    add("F) C ama 15 hisse", strat_x(pxc, ind, liquid, no_buy=tavan, no_exit=taban, slots=15), 0.007)
    add("G) C ama sinyalden 1 gün sonra işlem", wC, 0.007, lag=1)

    bx = None
    if xu100 is not None and len(xu100) > WARMUP:
        bx = xu100.reindex(px.index).ffill().pct_change(fill_method=None).fillna(0.0)
        rows["Kıyas: BIST 100 endeksi"] = summarize(bx)

    cols = ["yillik_%", "1.yari_%", "2.yari_%", "max_dusus_%", "ort_kasa_%", "islem", "kazanma_%",
            "beklenti_%/islem"]
    L = [f"# Stres testi: Birleşik varyant\n\n{px.shape[1]} hisse, {px.index[0].date()} -> {px.index[-1].date()}. "
         f"Temiz veri, sadece likit hisseler, boş kasa yıllık %{cash_rate * 100:.0f} faiz kazanır.\n",
         md_table(rows, cols), ""]

    L.append("## Sonuca en çok katkı yapan 5 işlem (senaryo C)\n")
    for j, a, e, r in top5:
        L.append(f"- {cols_n[j]}: giriş {pxc.index[a].date()}, çıkış {pxc.index[e].date()} "
                 f"({e - a} gün), getiri {r * 100:+.0f}%")
    n_all = len(tr)
    tot = sum(t[3] for t in tr)
    L.append(f"\nToplam {n_all} işlem. İlk 5'in toplamı: {sum(t[3] for t in top5) * 100:+.0f} puan, "
             f"tüm işlemlerin toplamı: {tot * 100:+.0f} puan.\n")

    L.append("## Yıl yıl getiri (%)\n")
    ycols = ["Orijinal (A)", "Gerçekçi (C)", "1 gün gecikmeli (G)"]
    ys = {}
    srcs = {"Orijinal (A)": series["A) Orijinal (maliyet %0,2)"],
            "Gerçekçi (C)": series["C) B + maliyet %0,7"],
            "1 gün gecikmeli (G)": series["G) C ama sinyalden 1 gün sonra işlem"]}
    if bx is not None:
        ycols = ["BIST 100"] + ycols
        srcs = {"BIST 100": bx, **srcs}
    for name, d in srcs.items():
        d = d.fillna(0.0).iloc[WARMUP:]
        y = ((1 + d).groupby(d.index.year).prod() - 1) * 100
        for yr, v in y.items():
            ys.setdefault(str(yr), {})[name] = round(float(v), 1)
    L.append(md_table(ys, ycols))

    L.append("\nNasıl okunur: Gerçekçi (C) ve gecikmeli (G) satırlar BIST 100 satırının belirgin üstündeyse, "
             "D satırı (en iyi 5 işlem çıkınca) hâlâ pozitifse ve getiri yıllara dağılmışsa varyant sağlam sayılır. "
             "C/G çökerse ya da D negatife düşerse A'daki sonuç tavan/taban ve birkaç fırlayan hisseden geliyordu.")
    L.append("\n> Uyarı: Bugünkü hisse listesiyle test (batan şirketler yok), sonuçlar iyimser. "
             "Tavan/taban filtresi yaklaşık bir temsildir (günlük ±%9,5). Yatırım tavsiyesi değildir.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols-file", default="bist_symbols.txt")
    ap.add_argument("--period", default="10y")
    ap.add_argument("--cash-rate", type=float, default=0.30)
    ap.add_argument("--out", default="stress.md")
    args = ap.parse_args()

    symbols = read_symbols(args.symbols_file)
    print(f"{len(symbols)} sembol indiriliyor ({args.period})...")
    close, volume = fetch_all(symbols, args.period)

    xu100 = None
    try:
        import yfinance as yf
        raw = yf.download("XU100.IS", period=args.period, auto_adjust=True, progress=False)
        c = raw["Close"]
        xu100 = (c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c).dropna()
    except Exception as exc:
        print("Endeks alınamadı:", exc)

    text = report(close, volume, xu100, args.cash_rate)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
