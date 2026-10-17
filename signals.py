"""Borsa sinyal programı: RSI, MACD ve SMA kesişimi ile AL / SAT / BEKLE sinyali üretir.

Sonuçları signals.json dosyasına yazar; index.html bu dosyayı gösterir.

Kullanım:
    python signals.py                     # varsayılan BIST listesi
    python signals.py THYAO.IS ASELS.IS   # kendi sembollerin
    python signals.py AAPL MSFT --period 1y --out signals.json

Bu program yatırım tavsiyesi değildir; yalnızca eğitim amaçlıdır.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = [
    "THYAO.IS", "ASELS.IS", "GARAN.IS", "AKBNK.IS", "EREGL.IS",
    "BIMAS.IS", "KCHOL.IS", "SISE.IS", "TUPRS.IS", "FROTO.IS",
]


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = rsi(df["Close"])
    df["macd"], df["macd_signal"] = macd(df["Close"])
    df["sma_fast"] = df["Close"].rolling(20).mean()
    df["sma_slow"] = df["Close"].rolling(50).mean()
    return df


def score_last(df: pd.DataFrame) -> dict:
    """Son mum için puan hesaplar. Pozitif = al, negatif = sat."""
    d = add_indicators(df).dropna()
    if len(d) < 2:
        return {"signal": "VERİ YETERSİZ", "score": 0, "reasons": []}
    last, prev = d.iloc[-1], d.iloc[-2]
    score, reasons = 0, []

    if last["rsi"] < 30:
        score += 1
        reasons.append(f"RSI aşırı satım ({last['rsi']:.1f})")
    elif last["rsi"] > 70:
        score -= 1
        reasons.append(f"RSI aşırı alım ({last['rsi']:.1f})")

    if prev["macd"] <= prev["macd_signal"] and last["macd"] > last["macd_signal"]:
        score += 1
        reasons.append("MACD yukarı kesti")
    elif prev["macd"] >= prev["macd_signal"] and last["macd"] < last["macd_signal"]:
        score -= 1
        reasons.append("MACD aşağı kesti")

    if prev["sma_fast"] <= prev["sma_slow"] and last["sma_fast"] > last["sma_slow"]:
        score += 1
        reasons.append("SMA20, SMA50'yi yukarı kesti (golden cross)")
    elif prev["sma_fast"] >= prev["sma_slow"] and last["sma_fast"] < last["sma_slow"]:
        score -= 1
        reasons.append("SMA20, SMA50'yi aşağı kesti (death cross)")

    signal = "AL" if score >= 1 else "SAT" if score <= -1 else "BEKLE"
    return {
        "signal": signal,
        "score": score,
        "reasons": reasons,
        "price": float(last["Close"]),
        "change_pct": float((last["Close"] / prev["Close"] - 1) * 100),
        "rsi": float(last["rsi"]),
    }


def fetch(symbol: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(symbol, period=period, auto_adjust=True, progress=False, multi_level_index=False)


def build_item(symbol: str, df: pd.DataFrame) -> dict:
    r = score_last(df)
    r["symbol"] = symbol.replace(".IS", "")
    r["ticker"] = symbol
    r["spark"] = [round(float(x), 2) for x in df["Close"].dropna().tail(60)]
    return r


def main():
    parser = argparse.ArgumentParser(description="Borsa sinyal programı")
    parser.add_argument("symbols", nargs="*", default=DEFAULT_SYMBOLS)
    parser.add_argument("--period", default="1y", help="yfinance periyodu (örn. 6mo, 1y, 2y)")
    parser.add_argument("--out", default="signals.json", help="JSON çıktı yolu")
    args = parser.parse_args()

    items = []
    for sym in args.symbols:
        try:
            df = fetch(sym, args.period)
            if df.empty:
                print(f"{sym:10s} veri alınamadı")
                continue
            item = build_item(sym, df)
            items.append(item)
            why = "; ".join(item["reasons"]) or "belirgin sinyal yok"
            print(f"{sym:10s} {item['signal']:6s} fiyat={item.get('price', float('nan')):.2f}  {why}")
        except Exception as exc:  # ağ hatası vb.
            print(f"{sym:10s} hata: {exc}")

    if not items:
        raise SystemExit("Hiç veri alınamadı, JSON yazılmadı.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "items": items}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(items)} hisse -> {out}")


if __name__ == "__main__":
    main()
