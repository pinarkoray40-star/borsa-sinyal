"""Strateji araştırması v2: kurallar, OBV birikim, piyasa filtresi, portföy simülasyonu.

Ölçü: sinyalden 21 / 42 işlem günü sonraki getiri ve "fazla getiri"
(aynı gün tüm likit hisselerin ortalamasına göre fark).
Sonuç telefonda rahat okunsun diye satır satır yazılır.
Yatırım tavsiyesi değildir.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from signals import fetch_all, read_symbols, rsi

HORIZONS = (21, 42)
COST = 0.4      # bir tur (al+sat) toplam maliyet, %
STOP = 8.0      # zarar kes %
TOPN = 10       # portföy hisse sayısı


def pct_rank(df):
    return df.rank(axis=1, pct=True)


def build_features(close, volume):
    f = {}
    f["ret1"] = close.pct_change()
    f["r20"] = close / close.shift(20) - 1
    f["r126"] = close / close.shift(126) - 1
    f["r12_1"] = close.shift(21) / close.shift(252) - 1
    f["sma50"] = close.rolling(50).mean()
    f["sma200"] = close.rolling(200).mean()
    f["hi55"] = close.rolling(55).max()
    f["rsi"] = close.apply(lambda s: rsi(s, 14))
    f["volrat"] = volume / volume.rolling(20).mean()
    f["vol20"] = f["ret1"].rolling(20).std()
    f["turn"] = (close * volume).rolling(20).median()
    sign = np.sign(f["ret1"].fillna(0))
    obv = (sign * volume.fillna(0)).cumsum()
    base = volume.rolling(20).sum().replace(0, np.nan)
    f["obv20"] = (obv - obv.shift(20)) / base      # -1..+1: son 20 günde net alım baskısı
    f["obv_hi"] = obv >= obv.rolling(60).max()      # OBV 60 gün zirvesinde
    return f


def build_rules(close, f):
    up200 = close > f["sma200"]
    near_high = close >= f["hi55"] * 0.97
    down20 = f["r20"] < 0
    R = {}
    R["A) 6 ay momentum (en iyi %10)"] = pct_rank(f["r126"]) >= 0.90
    R["B) Momentum 6 ay + zirveye yakın + hacim"] = (pct_rank(f["r126"]) >= 0.80) & near_high & (f["volrat"] >= 1.2)
    R["C) Düşük oynaklık + orta üstü momentum"] = (pct_rank(f["vol20"]) <= 0.20) & (pct_rank(f["r126"]) >= 0.50)
    R["D) OBV birikim: fiyat 20g düşük, OBV güçlü artıyor (>0.30)"] = down20 & (f["obv20"] > 0.30)
    R["E) OBV birikim: fiyat 20g düşük, OBV en iyi %10"] = down20 & (pct_rank(f["obv20"].where(down20)) >= 0.90)
    R["F) OBV birikim + SMA200 üstü (trendde geri çekilme)"] = down20 & up200 & (f["obv20"] > 0.30)
    R["G) OBV birikim + RSI<45"] = down20 & (f["obv20"] > 0.20) & (f["rsi"] < 45)
    R["H) OBV 60g zirve ama fiyat SMA50 altı"] = f["obv_hi"] & (close < f["sma50"])
    R["I) Aşırı satım (RSI<30)"] = f["rsi"] < 30
    R["J) 55 gün zirve kırılımı + hacim"] = (close >= f["hi55"]) & (f["volrat"] >= 1.5)
    return R


def fwd_path(close, h):
    fwd = close.shift(-h) / close - 1
    # önümüzdeki h günde görülen en düşük fiyat
    fmin = close[::-1].rolling(h).min()[::-1].shift(-1) / close - 1
    return fwd, fmin


def evaluate(mask, fwd, fmin, exc, liquid, h, years):
    m = mask & liquid & fwd.notna()
    mv = m.values
    if mv.sum() < 50:
        return None
    r = fwd.values[mv] * 100
    e = exc.values[mv] * 100
    rs = np.where(fmin.values[mv] * 100 <= -STOP, -STOP, r)  # zarar kes uygulanmış getiri
    daily = exc.where(m).mean(axis=1).dropna() * 100
    t = np.nan
    if len(daily) > 30:
        t = daily.mean() / (daily.std() / np.sqrt(max(len(daily) / h, 1)) + 1e-12)
    half = len(daily) // 2
    byyear = {}
    for y, g in daily.groupby(daily.index.year):
        if len(g) >= 30:
            byyear[int(y)] = g.mean()
    return dict(n=int(mv.sum()), avg=r.mean(), med=np.median(r), win=(r > 0).mean() * 100,
                exc=e.mean(), t=t, h1=daily.iloc[:half].mean(), h2=daily.iloc[half:].mean(),
                big=(r >= 20).mean() * 100, stop=rs.mean(), net=r.mean() - COST, years=byyear)


def f1(x, d=1, plus=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:+.{d}f}" if plus else f"{x:.{d}f}"


def line(name, r):
    return (f"- **{name}**\n"
            f"  - {r['n']:,} sinyal · ortalama {f1(r['avg'])}% · medyan {f1(r['med'])}% · kazanan %{r['win']:.0f}\n"
            f"  - kıyasa göre fark **{f1(r['exc'], 2)} puan** (t={f1(r['t'], 1, False)}) · "
            f"1. yarı {f1(r['h1'], 2)} / 2. yarı {f1(r['h2'], 2)}\n"
            f"  - %20+ yapan oran %{r['big']:.1f} · %{STOP:.0f} zarar kesle ortalama {f1(r['stop'])}%\n"
            ).replace(",", ".")


def simulate(close, score, liquid, step=21, topn=TOPN, allow=None, cost=COST):
    """Her `step` günde en yüksek skorlu topn hisseye eşit para. Dönüş: aylık getiri serisi."""
    idx = close.index
    pos = list(range(260, len(idx) - step, step))
    rets, dates = [], []
    for p in pos:
        if allow is not None and not bool(allow.iloc[p]):
            rets.append(0.0); dates.append(idx[p + step]); continue
        sc = score.iloc[p].where(liquid.iloc[p]).dropna()
        if len(sc) < topn:
            continue
        pick = sc.nlargest(topn).index
        r = (close.iloc[p + step][pick] / close.iloc[p][pick] - 1).dropna()
        if r.empty:
            continue
        rets.append(r.mean() - cost / 100); dates.append(idx[p + step])
    return pd.Series(rets, index=dates)


def stats(rets, label):
    if len(rets) < 6:
        return f"- **{label}**: yetersiz veri\n"
    eq = (1 + rets).cumprod()
    yrs = len(rets) * 21 / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    return (f"- **{label}**\n"
            f"  - 1 TL → **{eq.iloc[-1]:.2f} TL** ({yrs:.1f} yılda) · yıllık ≈ %{cagr * 100:.0f}\n"
            f"  - en kötü düşüş %{dd * 100:.0f} · aylık kazanma %{(rets > 0).mean() * 100:.0f} · "
            f"en kötü ay %{rets.min() * 100:.0f} · en iyi ay %{rets.max() * 100:+.0f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="5y")
    ap.add_argument("--symbols-file", default="bist_symbols.txt")
    ap.add_argument("--out", default="research.md")
    args = ap.parse_args()

    symbols = read_symbols(args.symbols_file)
    print(f"{len(symbols)} sembol indiriliyor ({args.period})...")
    close, volume = fetch_all(symbols, args.period)
    if close.shape[1] < 20:
        raise SystemExit(f"Yeterli veri yok ({close.shape[1]} hisse).")

    # BIST100 endeksi (piyasa filtresi için)
    allow = None
    try:
        import yfinance as yf
        ix = yf.download("XU100.IS", period=args.period, interval="1d", auto_adjust=True, progress=False)["Close"]
        ix = ix.squeeze().reindex(close.index).ffill()
        if ix.notna().sum() > 250:
            allow = ix > ix.rolling(200).mean()
    except Exception as exc:
        print("Endeks alınamadı:", exc)

    f = build_features(close, volume)
    liquid = pct_rank(f["turn"]) >= 0.30
    rules = build_rules(close, f)
    L = []
    L.append("# Strateji araştırması v2")
    L.append(f"{close.shape[1]} hisse · {close.index[0].date()} → {close.index[-1].date()} · "
             f"düşük hacimli %30 hisse hariç · maliyet %{COST}/tur varsayımı.\n")
    L.append("**Okuma:** *Kıyasa göre fark* = o gün tüm hisselerin ortalamasına göre fazla getiri. "
             "t>2 anlamlı, t>3 güçlü. 1. ve 2. yarı ikisi de artıda ise tutarlı.\n")

    best = {}
    for h in HORIZONS:
        fwd, fmin = fwd_path(close, h)
        mkt = fwd.where(liquid).mean(axis=1)
        exc = fwd.sub(mkt, axis=0)
        allm = pd.DataFrame(True, index=close.index, columns=close.columns)
        base = evaluate(allm, fwd, fmin, exc, liquid, h, None)
        res = []
        for name, mask in rules.items():
            r = evaluate(mask, fwd, fmin, exc, liquid, h, None)
            if r:
                res.append((name, r))
        res.sort(key=lambda x: -x[1]["exc"])
        best[h] = res
        L.append(f"## {h} işlem günü ({'~1 ay' if h == 21 else '~2 ay'}) sonrası\n")
        L.append(line("Kıyas: tüm likit hisseler", base))
        for name, r in res:
            L.append(line(name, r))

    # yıl yıl (21 gün, en iyi 3 kural + OBV kuralları)
    L.append("## Yıl yıl tutarlılık (21 gün, kıyasa göre puan)\n")
    for name, r in best[21][:5]:
        ys = " · ".join(f"{y}: {v:+.1f}" for y, v in sorted(r["years"].items()))
        L.append(f"- **{name}**: {ys}")
    L.append("")

    # portföy simülasyonu
    L.append("## Portföy simülasyonu\n")
    L.append(f"Her ay (21 işlem günü) en iyi {TOPN} hisseye eşit para, ay sonunda değiştir. "
             f"Maliyet dahil. Geçmişte; geleceği garanti etmez.\n")
    eqw = simulate(close, pct_rank(f["r20"] * 0 + 1) + np.random.default_rng(0).random(close.shape) * 1e-6, liquid, topn=TOPN)
    mom = simulate(close, f["r126"], liquid)
    obv_score = f["obv20"].where(f["r20"] < 0)
    obv = simulate(close, obv_score, liquid)
    L.append(stats(eqw, f"Kıyas: rastgele {TOPN} likit hisse"))
    L.append(stats(mom, f"6 ay momentum, en iyi {TOPN}"))
    L.append(stats(obv, f"OBV birikim (fiyat düşük, OBV en güçlü), en iyi {TOPN}"))
    if allow is not None:
        L.append(stats(simulate(close, f["r126"], liquid, allow=allow), "6 ay momentum + piyasa filtresi (BIST100 SMA200 altında nakit)"))
        L.append(stats(simulate(close, obv_score, liquid, allow=allow), "OBV birikim + piyasa filtresi"))
    else:
        L.append("- Piyasa filtresi: BIST100 verisi alınamadı, atlandı.")

    L.append("\n> Uyarı: Veri bugünkü hisse listesidir (batan şirketler yok, sonuçlar iyimser çıkar). "
             "Spread/kayma tam yansıtılmamıştır. Yatırım tavsiyesi değildir.")

    text = "\n".join(L)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
