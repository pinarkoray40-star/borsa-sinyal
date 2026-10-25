"""'Sinyal (RSI+MACD+SMA)' stratejisinin düzeltilmiş varyantları.

Hepsi aynı koşullarda test edilir: temizlenmiş veri (günlük %12 üstü hareketler silinir),
sadece likit hisseler (hacim sıralamasında alt %30 elenir), maliyet dahil, boş kasa nakit faizi kazanır.

Varyantlar tek tek ve birlikte denenir. DİKKAT: aynı veride çok varyant denemek "şans eseri iyi çıkma"
riski taşır. Bu yüzden her varyant için 1. yarı ve 2. yarı ayrı verilir; sadece İKİ YARIDA DA
iyi olan varyanta güvenilir.

Kullanım (GitHub'da "Analiz calistir", dosya adı: variants.py):
    python variants.py --period 5y

Yatırım tavsiyesi değildir.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import WINDOW, indicators, metrics, run
from diagnose import WARMUP, clean_prices, md_table, trade_table
from signals import fetch_all, read_symbols


def strat_v(px, ind, liquid, mkt_ok, entry_min=1, exit_rsi=True, slots=5,
            trend_entry=False, rank="rsi", stop=0.08, max_age=WINDOW) -> pd.DataFrame:
    """backtest.strat_signals'ın ayarlanabilir hali. Varsayılanlar mevcut kuralla aynıdır."""
    rsi_, line, sig = ind["rsi"], ind["macd"], ind["macd_sig"]
    sma20, sma50 = ind["sma20"], ind["sma50"]
    up = ((line.shift(1) <= sig.shift(1)) & (line > sig)).astype(float)
    dn = ((line.shift(1) >= sig.shift(1)) & (line < sig)).astype(float)
    gc = ((sma20.shift(1) <= sma50.shift(1)) & (sma20 > sma50)).astype(float)
    dc = ((sma20.shift(1) >= sma50.shift(1)) & (sma20 < sma50)).astype(float)
    rsi_pt = (rsi_ < 30).astype(float)
    if exit_rsi:
        rsi_pt = rsi_pt - (rsi_ > 70).astype(float)
    score = (rsi_pt + up - dn + gc - dc).where(sma50.notna())

    mom = px / px.shift(126) - 1
    trend_ok = ((px > sma50) & (sma20 > sma50)) if trend_entry else pd.DataFrame(True, index=px.index, columns=px.columns)

    P, S, R, Mom = px.values, score.values, rsi_.values, mom.values
    Lq, Tr, mk = liquid.values, trend_ok.values, mkt_ok.values
    W = np.zeros(P.shape)
    held = {}  # sütun -> (giriş fiyatı, yaş)
    with np.errstate(invalid="ignore"):
        for i in range(len(px)):
            for j in list(held):
                e, a = held[j]
                a += 1
                held[j] = (e, a)
                p = P[i, j]
                if (S[i, j] <= -1) or (np.isfinite(p) and p <= e * (1 - stop)) or a >= max_age or not mk[i]:
                    del held[j]
            free = slots - len(held)
            if free > 0 and mk[i]:
                ok = (S[i] >= entry_min) & Lq[i] & Tr[i] & np.isfinite(P[i])
                for j in held:
                    ok[j] = False
                idx = np.where(ok)[0]
                if len(idx):
                    if rank == "mom":
                        key = {j: (-S[i, j], -Mom[i, j] if np.isfinite(Mom[i, j]) else np.inf) for j in idx}
                    else:
                        key = {j: (-S[i, j], R[i, j]) for j in idx}
                    for j in sorted(idx, key=lambda j: key[j])[:free]:
                        held[j] = (P[i, j], 0)
            for j in held:
                W[i, j] = 1.0 / slots
    return pd.DataFrame(W, index=px.index, columns=px.columns)


def daily_returns(w, rets, cash_rate):
    d = run(w, rets)
    idle = (1.0 - w.shift(1).fillna(0.0).sum(axis=1)).clip(lower=0.0)
    return d + idle * ((1 + cash_rate) ** (1 / 252) - 1)


def summarize(d: pd.Series, w=None, px=None) -> dict:
    d = d.fillna(0.0).iloc[WARMUP:]
    m = metrics(d)
    half = len(d) // 2

    def ann(s):
        return round(((1 + s).prod() ** (252 / max(len(s), 1)) - 1) * 100, 1)

    out = {"yillik_%": m["yillik_%"], "1.yari_%": ann(d.iloc[:half]), "2.yari_%": ann(d.iloc[half:]),
           "max_dusus_%": m["max_dusus_%"], "40g_hedefi_tutan_%": m["40g_hedefi_tutan_%"]}
    if w is not None:
        out["ort_kasa_%"] = round(float(w.shift(1).fillna(0.0).sum(axis=1).iloc[WARMUP:].mean()) * 100, 0)
        tr, _ = trade_table(w, px)
        out["islem"] = len(tr)
        out["kazanma_%"] = round(float((tr > 0).mean()) * 100, 0) if len(tr) else ""
        out["beklenti_%/islem"] = round(float(tr.mean()) * 100, 2) if len(tr) else ""
    else:
        out["ort_kasa_%"] = 100
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols-file", default="bist_symbols.txt")
    ap.add_argument("--period", default="10y")
    ap.add_argument("--cash-rate", type=float, default=0.30)
    ap.add_argument("--out", default="variants.md")
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


def report(close, volume, xu100, cash_rate) -> str:
    px = close.sort_index().ffill(limit=5)
    vol = volume.reindex(index=px.index, columns=px.columns)
    pxc, rc, _, _ = clean_prices(px)
    liquid = ((px * vol).rolling(20).median().rank(axis=1, pct=True) >= 0.30)
    ind = indicators(pxc)
    all_ok = pd.Series(True, index=px.index)

    mkt = None
    if xu100 is not None and len(xu100) > 250:
        bx = xu100.reindex(px.index).ffill()
        mkt = (bx > bx.rolling(200).mean()).fillna(True)

    combo = dict(exit_rsi=False, trend_entry=True, rank="mom", slots=10)
    variants = {
        "1) Mevcut kural": dict(),
        "2) Sıkı giriş (skor ≥ 2)": dict(entry_min=2),
        "3) RSI>70'te satma": dict(exit_rsi=False),
        "4) Trend onaylı giriş + momentum sıralı": dict(trend_entry=True, rank="mom"),
        "5) 10 hisseye yay": dict(slots=10),
        "6) Zarar-kes %15": dict(stop=0.15),
        "7) Birleşik: 3+4+5": dict(combo),
        "8) Birleşik + zarar-kes %15": dict(combo, stop=0.15),
    }
    if mkt is not None:
        variants["9) Piyasa filtresi (BIST100 > SMA200)"] = dict(mkt=True)
        variants["10) Birleşik + piyasa filtresi"] = dict(combo, mkt=True)
        variants["11) Birleşik + piyasa + zarar-kes %15"] = dict(combo, mkt=True, stop=0.15)

    rows = {}
    for name, kw in variants.items():
        kw = dict(kw)
        use_mkt = kw.pop("mkt", False)
        w = strat_v(pxc, ind, liquid, mkt if use_mkt else all_ok, **kw)
        rows[name] = summarize(daily_returns(w, rc, cash_rate), w, pxc)
        print(f"  {name}: yıllık {rows[name]['yillik_%']}%")

    rows["Kıyas: eşit ağırlık (temiz, likit)"] = summarize(rc.where(liquid).mean(axis=1))
    if xu100 is not None and len(xu100) > WARMUP:
        bx = xu100.reindex(px.index).ffill().pct_change(fill_method=None).fillna(0.0)
        rows["Kıyas: BIST 100 endeksi"] = summarize(bx)

    cols = ["yillik_%", "1.yari_%", "2.yari_%", "max_dusus_%", "40g_hedefi_tutan_%",
            "ort_kasa_%", "islem", "kazanma_%", "beklenti_%/islem"]
    L = [f"# Sinyal stratejisi varyantları\n\n{px.shape[1]} hisse, {px.index[0].date()} -> {px.index[-1].date()}. "
         f"Temiz veri, sadece likit hisseler, maliyet dahil, boş kasa yıllık %{cash_rate * 100:.0f} faiz kazanır.\n",
         md_table(rows, cols),
         "\nNasıl okunur: 1.yarı/2.yarı = dönemin ilk/son yarısındaki yıllık getiri. İkisi de BIST 100'ün üstündeyse "
         "güvenilir; sadece birinde iyiyse şans olabilir. beklenti = işlem başına ortalama getiri.",
         "\n> Uyarı: Bugünkü hisse listesiyle test (batan şirketler yok), sonuçlar iyimser. "
         "Çok varyant denemek şans eseri iyi sonuç riskini artırır. Yatırım tavsiyesi değildir."]
    return "\n".join(L)


if __name__ == "__main__":
    main()
