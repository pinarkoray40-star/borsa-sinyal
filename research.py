"""Strateji araştırması: birçok al kuralını geçmiş verinin tamamında yarıştırır.

Her kural için, sinyal çıktıktan 21 ve 28 işlem günü (~1 ay / ~40 gün) sonraki getiri ölçülür.
Asıl ölçü "fazla getiri": aynı gün tüm (likit) hisselerin ortalamasına göre fark.
Böylece genel borsa yükselişi ve enflasyon etkisi çıkarılır.

Kullanım: python research.py [--period 5y] [--out research.md]
Yatırım tavsiyesi değildir.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from signals import fetch_all, read_symbols, rsi

HORIZONS = (21, 28)
COST = 0.4  # alım+satım toplam maliyet varsayımı (%)


def rsi_df(close, n=14):
    return close.apply(lambda s: rsi(s, n))


def build_features(close, volume):
    f = {}
    f["ret1"] = close.pct_change()
    f["r126"] = close / close.shift(126) - 1
    f["r12_1"] = close.shift(21) / close.shift(252) - 1
    f["r5"] = close / close.shift(5) - 1
    f["sma50"] = close.rolling(50).mean()
    f["sma200"] = close.rolling(200).mean()
    f["hi55"] = close.rolling(55).max()
    f["rsi"] = rsi_df(close)
    f["volrat"] = volume / volume.rolling(20).mean()
    f["vol20"] = f["ret1"].rolling(20).std()
    f["turn"] = (close * volume).rolling(20).median()
    return f


def pct_rank(df):
    return df.rank(axis=1, pct=True)


def build_rules(close, f):
    top = lambda df, q: pct_rank(df) >= q
    bot = lambda df, q: pct_rank(df) <= q
    up200 = close > f["sma200"]
    near_high = close >= f["hi55"] * 0.97
    rules = {
        "1) Güçlü momentum (6 ay en iyi %10)": top(f["r126"], 0.90),
        "2) Klasik momentum 12-1 ay (en iyi %10)": top(f["r12_1"], 0.90),
        "3) 55 gün zirve kırılımı + hacim patlaması": (close >= f["hi55"]) & (f["volrat"] >= 1.5),
        "4) Yükselen trendde geri çekilme (RSI<40, SMA200 üstü)": up200 & (f["rsi"] < 40),
        "5) Aşırı satım (RSI<30)": f["rsi"] < 30,
        "6) Kısa vadeli dip (5 gün en kötü %10, SMA200 üstü)": bot(f["r5"], 0.10) & up200,
        "7) Düşük oynaklık + orta üstü momentum": bot(f["vol20"], 0.20) & top(f["r126"], 0.50),
        "8) Momentum + zirveye yakın + hacim": top(f["r126"], 0.80) & near_high & (f["volrat"] >= 1.2),
        "9) Momentum + SMA200 üstü + RSI 50-70": top(f["r126"], 0.80) & up200 & (f["rsi"] >= 50) & (f["rsi"] < 70),
    }
    return rules


def evaluate(mask, fwd, exc, liquid, h):
    m = (mask & liquid & fwd.notna()).values
    if m.sum() < 50:
        return None
    r = fwd.values[m] * 100
    e = exc.values[m] * 100
    # gün bazında ortalama fazla getiri -> t istatistiği (örtüşen pencereler için n/h ile düzeltilir)
    daily = (exc.where(mask & liquid)).mean(axis=1).dropna() * 100
    t = np.nan
    if len(daily) > 30:
        t = daily.mean() / (daily.std() / np.sqrt(max(len(daily) / h, 1)) + 1e-12)
    half = len(daily) // 2
    return {
        "n": int(m.sum()),
        "avg": r.mean(), "med": np.median(r), "win": (r > 0).mean() * 100,
        "exc": e.mean(), "exc_med": np.median(e),
        "net": r.mean() - COST,
        "t": t,
        "h1": daily.iloc[:half].mean() if half else np.nan,
        "h2": daily.iloc[half:].mean() if half else np.nan,
        "big": (r >= 20).mean() * 100,
    }


def fmt(x, d=1, plus=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:+.{d}f}" if plus else f"{x:.{d}f}"


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
    print(f"Veri: {close.shape[1]} hisse, {close.shape[0]} gün "
          f"({close.index[0].date()} → {close.index[-1].date()})")

    f = build_features(close, volume)
    liquid = pct_rank(f["turn"]) >= 0.30  # en az işlem gören %30'u dışarıda bırak
    rules = build_rules(close, f)
    rules["0) Kıyas: tüm likit hisseler, her gün"] = pd.DataFrame(True, index=close.index, columns=close.columns)

    lines = []
    lines.append("# Strateji araştırması")
    lines.append(f"{close.shape[1]} hisse, {close.index[0].date()} → {close.index[-1].date()} "
                 f"({close.shape[0]} işlem günü). Düşük hacimli %30 hisse hariç. "
                 f"Maliyet varsayımı: toplam %{COST} (alım+satım).\n")
    lines.append("**Nasıl okunur:** *Fazla getiri* = o gün tüm hisselerin ortalamasına göre fark "
                 "(genel yükselişi çıkarır). *t* değeri 2'nin üstündeyse fark tesadüf olmaktan uzak, "
                 "3'ün üstü güçlü. *1. yarı / 2. yarı* ikisi de artıda ise kural tutarlı demektir.\n")

    summary = {}
    for h in HORIZONS:
        fwd = close.shift(-h) / close - 1
        mkt = fwd.where(liquid).mean(axis=1)
        exc = fwd.sub(mkt, axis=0)
        rows = []
        for name, mask in rules.items():
            res = evaluate(mask, fwd, exc, liquid, h)
            if res:
                rows.append((name, res))
        rows.sort(key=lambda x: -x[1]["exc"])
        summary[h] = rows
        label = "1 ay (21 işlem günü)" if h == 21 else "~40 gün (28 işlem günü)"
        lines.append(f"## {label} sonrası\n")
        lines.append("| Kural | Sinyal | Ort. getiri | Medyan | Kazanan % | Fazla getiri | t | 1. yarı | 2. yarı | Net* | ≥%20 oranı |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, r in rows:
            lines.append(
                f"| {name} | {r['n']} | {fmt(r['avg'], 1, True)}% | {fmt(r['med'], 1, True)}% | "
                f"{fmt(r['win'], 0)} | {fmt(r['exc'], 2, True)} | {fmt(r['t'], 1)} | "
                f"{fmt(r['h1'], 2, True)} | {fmt(r['h2'], 2, True)} | {fmt(r['net'], 1, True)}% | {fmt(r['big'], 1)} |"
            )
        lines.append("\n*Net = ortalama getiri − maliyet.*\n")

    # kısa özet
    lines.append("## Özet\n")
    for h in HORIZONS:
        rows = [(n, r) for n, r in summary[h] if not n.startswith("0)")]
        good = [(n, r) for n, r in rows if r["t"] >= 2 and r["h1"] > 0 and r["h2"] > 0 and r["exc"] > 0.5]
        label = "1 ay" if h == 21 else "~40 gün"
        if good:
            lines.append(f"- **{label}:** tutarlı üstünlük gösteren kurallar: " +
                         ", ".join(f"{n.split(')')[0]}) (+{r['exc']:.2f} puan, t={r['t']:.1f})" for n, r in good))
        else:
            lines.append(f"- **{label}:** hiçbir kural hem istatistiksel olarak anlamlı hem de iki yarıda "
                         f"tutarlı değil. Dürüst sonuç: bu fikirlerle güvenilir üstünlük bulunamadı.")
    lines.append("\n> Uyarı: Veri bugünkü listede olan hisseleri içerir (hayatta kalan yanlılığı), "
                 "komisyon/spread tam yansıtılmamıştır. Geçmiş başarı gelecek için garanti değildir. "
                 "Yatırım tavsiyesi değildir.")

    text = "\n".join(lines)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
