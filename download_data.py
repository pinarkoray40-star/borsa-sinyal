"""BIST geçmiş fiyatlarını Yahoo Finance'ten indirip tek bir CSV'ye yazar.

Kullanım:
    python download_data.py                      # data/prices.csv
    python download_data.py --start 2015-01-01

Çıktı: tarih satırları x hisse sütunları (düzeltilmiş kapanış). Endeks ve dolar ayrı dosyalarda.
"""
import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


def read_symbols(path: str) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [s.strip() + ".IS" for s in lines if s.strip() and not s.startswith("#")]


def close_frame(tickers, start) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False, group_by="column")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    return close.dropna(how="all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="bist_symbols.txt")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--out", default="data/prices.csv")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tickers = read_symbols(args.symbols)
    prices = close_frame(tickers, args.start)
    prices = prices.dropna(axis=1, thresh=250)  # 1 yıldan az verisi olanları at
    prices.columns = [c.replace(".IS", "") for c in prices.columns]
    prices.to_csv(out)
    print(f"{prices.shape[1]} hisse, {len(prices)} gün -> {out}")

    extra = close_frame(["XU100.IS", "USDTRY=X"], args.start)
    extra.columns = ["XU100" if c == "XU100.IS" else "USDTRY" for c in extra.columns]
    extra.to_csv(out.with_name("extra.csv"))
    print(f"endeks/dolar -> {out.with_name('extra.csv')}")


if __name__ == "__main__":
    main()
