"""Backtest: stratejileri tüm BIST hisselerinde komisyon dahil karşılaştırır.

Kullanım:
    python download_data.py          # önce veriyi indir
    python backtest.py               # data/prices.csv üzerinde çalıştır

Hedef ölçüsü: 40 takvim günü ~ 28 işlem günü. Her strateji için tüm 28 günlük pencerelere bakılır
ve kaçında getirinin %20 ve üstü olduğu raporlanır.

Karar günün kapanışında verilir, ertesi günün getirisine uygulanır (ileriye bakma hatası yok).
Yatırım tavsiyesi değildir.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from signals import macd, rsi

WINDOW = 28          # 40 takvim günü ~ 28 işlem günü
TARGET = 0.20        # hedef getiri
N_SLOTS = 5          # aynı anda en fazla kaç hisse
COST = 0.002         # işlem başına tek yön maliyet (komisyon + kayma), %0.2
STOP = 0.08          # zarar kes


# ---------- göstergeler (tüm hisseler tek seferde) ----------

def indicators(px: pd.DataFrame) -> dict:
    line, sig = macd(px)
    return {
        "rsi": rsi(px),
        "macd": line,
        "macd_sig": sig,
        "sma20": px.rolling(20).mean(),
        "sma50": px.rolling(50).mean(),
    }


def score_matrix(px: pd.DataFrame, ind: dict) -> pd.DataFrame:
    """signals.py ile aynı puanlama: her gün her hisse için -3..+3."""
    s = pd.DataFrame(0, index=px.index, columns=px.columns, dtype=float)
    s += (ind["rsi"] < 30).astype(float) - (ind["rsi"] > 70).astype(float)
    up = (ind["macd"].shift(1) <= ind["macd_sig"].shift(1)) & (ind["macd"] > ind["macd_sig"])
    dn = (ind["macd"].shift(1) >= ind["macd_sig"].shift(1)) & (ind["macd"] < ind["macd_sig"])
    s += up.astype(float) - dn.astype(float)
    gc = (ind["sma20"].shift(1) <= ind["sma50"].shift(1)) & (ind["sma20"] > ind["sma50"])
    dc = (ind["sma20"].shift(1) >= ind["sma50"].shift(1)) & (ind["sma20"] < ind["sma50"])
    s += gc.astype(float) - dc.astype(float)
    return s.where(ind["sma50"].notna())


# ---------- stratejiler: hedef ağırlık matrisi döndürür ----------

def strat_signals(px, ind) -> pd.DataFrame:
    """AL sinyalinde gir; SAT sinyali, zarar kes ya da 28 gün dolunca çık."""
    score = score_matrix(px, ind)
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    held: dict[str, tuple[float, int]] = {}  # sembol -> (giriş fiyatı, tutulan gün)
    for i, day in enumerate(px.index):
        row_px, row_sc = px.iloc[i], score.iloc[i]
        for sym in list(held):
            entry, age = held[sym]
            age += 1
            held[sym] = (entry, age)
            if row_sc[sym] <= -1 or row_px[sym] <= entry * (1 - STOP) or age >= WINDOW:
                del held[sym]
        free = N_SLOTS - len(held)
        if free > 0:
            cands = row_sc[(row_sc >= 1) & ~row_sc.index.isin(list(held))].dropna()
            if len(cands):
                order = sorted(cands.index, key=lambda c: (-cands[c], ind["rsi"].iloc[i][c]))
                for sym in order[:free]:
                    held[sym] = (row_px[sym], 0)
        for sym in held:
            w.at[day, sym] = 1.0 / N_SLOTS
    return w


def momentum_rank(px, lookback=120, skip=5) -> pd.DataFrame:
    return px.shift(skip) / px.shift(lookback) - 1


def strat_momentum(px, ind) -> pd.DataFrame:
    """Her 28 günde bir son 120 günün en çok yükselen N hissesini al."""
    mom = momentum_rank(px)
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    cur = pd.Series(0.0, index=px.columns)
    for i, day in enumerate(px.index):
        if i % WINDOW == 0 and mom.iloc[i].notna().any():
            top = mom.iloc[i].dropna().nlargest(N_SLOTS).index
            cur = pd.Series(0.0, index=px.columns)
            cur[top] = 1.0 / N_SLOTS
        w.iloc[i] = cur
    return w


def strat_combined(px, ind) -> pd.DataFrame:
    """Momentum ilk 15 içinden, fiyatı SMA50 üstünde ve RSI < 75 olanlardan en iyi N."""
    mom = momentum_rank(px)
    ok = (px > ind["sma50"]) & (ind["rsi"] < 75)
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    cur = pd.Series(0.0, index=px.columns)
    for i, day in enumerate(px.index):
        if i % WINDOW == 0 and mom.iloc[i].notna().any():
            pool = mom.iloc[i].dropna().nlargest(15).index
            pool = [s for s in pool if ok.iloc[i][s]][:N_SLOTS]
            cur = pd.Series(0.0, index=px.columns)
            if pool:
                cur[pool] = 1.0 / N_SLOTS
        w.iloc[i] = cur
    return w


STRATEGIES = {
    "Sinyal (RSI+MACD+SMA)": strat_signals,
    "Momentum": strat_momentum,
    "Momentum + trend filtresi": strat_combined,
}


# ---------- simülasyon ve ölçüler ----------

def run(w: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    """Ağırlık t günü kapanışında verilir, t+1 getirisine uygulanır."""
    wl = w.shift(1).fillna(0.0)
    gross = (wl * rets.fillna(0.0)).sum(axis=1)
    turnover = (wl - wl.shift(1).fillna(0.0)).abs().sum(axis=1)
    return gross - turnover * COST


def benchmark(rets: pd.DataFrame) -> pd.Series:
    return rets.mean(axis=1)  # her gün eşit ağırlık, maliyetsiz


def metrics(daily: pd.Series) -> dict:
    eq = (1 + daily).cumprod()
    years = len(daily) / 252
    roll = eq / eq.shift(WINDOW) - 1
    roll = roll.dropna()
    dd = (eq / eq.cummax() - 1).min()
    return {
        "toplam_getiri_%": round((eq.iloc[-1] - 1) * 100, 1),
        "yillik_%": round((eq.iloc[-1] ** (1 / years) - 1) * 100, 1),
        "max_dusus_%": round(dd * 100, 1),
        "40g_medyan_%": round(roll.median() * 100, 1),
        "40g_hedefi_tutan_%": round((roll >= TARGET).mean() * 100, 1),
        "40g_-10_alti_%": round((roll <= -0.10).mean() * 100, 1),
        "40g_en_kotu_%": round(roll.min() * 100, 1),
        "40g_en_iyi_%": round(roll.max() * 100, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/prices.csv")
    ap.add_argument("--extra", default="data/extra.csv")
    ap.add_argument("--out", default="backtest.json")
    args = ap.parse_args()

    px = pd.read_csv(args.prices, index_col=0, parse_dates=True).sort_index()
    px = px.ffill(limit=5)
    rets = px.pct_change(fill_method=None)
    ind = indicators(px)
    print(f"{px.shape[1]} hisse, {px.index[0].date()} -> {px.index[-1].date()}\n")

    results = {}
    for name, fn in STRATEGIES.items():
        results[name] = metrics(run(fn(px, ind), rets))
    results["Tüm hisseler eşit ağırlık (kıyas)"] = metrics(benchmark(rets))

    extra = Path(args.extra)
    if extra.exists():
        ex = pd.read_csv(extra, index_col=0, parse_dates=True).sort_index().ffill()
        for col, label in (("XU100", "BIST 100 endeksi (kıyas)"), ("USDTRY", "Dolar/TL (kıyas)")):
            if col in ex:
                r = ex[col].reindex(px.index).ffill().pct_change(fill_method=None).fillna(0.0)
                results[label] = metrics(r)

    df = pd.DataFrame(results).T
    pd.set_option("display.width", 200)
    print(df.to_string())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    print("\nNot: güncel hisse listesiyle test edildiği için sonuçlar olduğundan iyimser çıkabilir "
          "(listeden çıkmış/batmış hisseler yok, 'survivorship bias').")


if __name__ == "__main__":
    main()
