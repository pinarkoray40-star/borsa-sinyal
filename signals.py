"""Borsa sinyal programı: tüm BIST hisseleri için kademeli sinyal ve sinyal geçmişi.

Her hisse için trend, momentum, RSI, taze kesişim ve hacimden bir skor çıkarır:
    GÜÇLÜ AL (>=5) / AL (3-4) / BEKLE / SAT (-3,-4) / GÜÇLÜ SAT (<=-5)

Aynı kural geçmiş verinin tamamında da uygulanır. Böylece her kademedeki bir sinyalden
1 hafta, 1 ay ve 3 ay sonra ne olduğu (ve tüm günlerin ortalamasına kıyasla) gösterilir.

Kullanım:
    python signals.py                      # bist_symbols.txt'teki tüm hisseler
    python signals.py THYAO ASELS          # sadece bunlar
    python signals.py --out signals.json

Bu program yatırım tavsiyesi değildir; yalnızca eğitim amaçlıdır.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (5, 21, 63)  # ~1 hafta, ~1 ay, ~3 ay (işlem günü)
LABELS = {2: "GÜÇLÜ AL", 1: "AL", 0: "BEKLE", -1: "SAT", -2: "GÜÇLÜ SAT"}


# ---------- göstergeler (Series ve DataFrame ile çalışır) ----------

def rsi(close, period: int = 14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close, fast: int = 12, slow: int = 26, signal: int = 9):
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig


# ---------- skor ve kademe ----------

def compute(close: pd.DataFrame, volume: pd.DataFrame) -> dict:
    r = rsi(close)
    line, sig = macd(close)
    hist = line - sig
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    vavg = volume.rolling(20).mean().replace(0, np.nan)
    vol_ratio = volume / vavg
    chg = close.pct_change(fill_method=None)

    def sg(up, dn):
        return up.astype(float) - dn.astype(float)

    macd_up = ((line.shift(1) <= sig.shift(1)) & (line > sig)).astype(float)
    macd_dn = ((line.shift(1) >= sig.shift(1)) & (line < sig)).astype(float)
    gc = ((sma20.shift(1) <= sma50.shift(1)) & (sma20 > sma50)).astype(float)
    dc = ((sma20.shift(1) >= sma50.shift(1)) & (sma20 < sma50)).astype(float)
    fresh_macd = macd_up.rolling(3).max() - macd_dn.rolling(3).max()
    fresh_sma = gc.rolling(5).max() - dc.rolling(5).max()

    rsi_pts = (
        ((r >= 50) & (r < 70)).astype(float)
        - (r < 40).astype(float)
        - (r >= 75).astype(float)
        + ((r < 30) & (hist > hist.shift(1))).astype(float)
    )
    vol_pts = ((vol_ratio >= 1.5).astype(float) * np.sign(chg)).fillna(0.0)

    score = (
        sg(close > sma50, close < sma50)
        + sg(sma20 > sma50, sma20 < sma50)
        + sg(line > sig, line < sig)
        + rsi_pts
        + fresh_macd
        + fresh_sma
        + vol_pts
    )
    score = score.where(sma50.notna() & r.notna())

    tier = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    tier = (
        tier.mask(score >= 3, 1.0).mask(score >= 5, 2.0)
        .mask(score <= -3, -1.0).mask(score <= -5, -2.0)
        .where(score.notna())
    )
    return {
        "rsi": r, "line": line, "sig": sig, "sma20": sma20, "sma50": sma50, "vol_ratio": vol_ratio,
        "macd_up": macd_up, "macd_dn": macd_dn, "gc": gc, "dc": dc,
        "score": score, "tier": tier,
    }


def _stat(arr: np.ndarray, mask: np.ndarray):
    m = mask & np.isfinite(arr)
    v = arr[m]
    if len(v) == 0:
        return None
    return {
        "n": int(len(v)),
        "avg": round(float(v.mean()) * 100, 2),
        "median": round(float(np.median(v)) * 100, 2),
        "win": round(float((v > 0).mean()) * 100, 1),
    }


def days_since(flag: pd.DataFrame, n: int) -> pd.Series:
    """Son n satırda bayrağın en son kaç gün önce yandığı (bugün=0), yoksa NaN."""
    out = pd.Series(np.nan, index=flag.columns)
    for k in range(min(n, len(flag)) - 1, -1, -1):
        out[flag.iloc[-1 - k] > 0] = k
    return out


def ago(k: float) -> str:
    return "bugün" if k == 0 else f"{int(k)} gün önce"


def build(close: pd.DataFrame, volume: pd.DataFrame) -> dict:
    c = compute(close, volume)
    tier, score = c["tier"], c["score"]
    n_days = len(close)

    prev = tier.shift(1)
    up_ev = (tier >= 1) & prev.notna() & (tier > prev)
    dn_ev = (tier <= -1) & prev.notna() & (tier < prev)
    fwd = {h: (close.shift(-h) / close - 1).to_numpy() for h in HORIZONS}

    # --- kademe istatistikleri (tüm geçmiş) ---
    groups = {
        "GÜÇLÜ AL": (up_ev & (tier == 2)).to_numpy(),
        "AL": (up_ev & (tier == 1)).to_numpy(),
        "Tüm günler (kıyas)": score.notna().to_numpy(),
        "SAT": (dn_ev & (tier == -1)).to_numpy(),
        "GÜÇLÜ SAT": (dn_ev & (tier == -2)).to_numpy(),
    }
    stats = {}
    for name, mask in groups.items():
        stats[name] = {f"h{h}": _stat(fwd[h], mask) for h in HORIZONS}
    stats_period = f"{close.index[0]:%d.%m.%Y} - {close.index[-1]:%d.%m.%Y}"

    # --- hisse başına son AL sinyali ---
    ev = up_ev.to_numpy()
    any_ev = ev.any(axis=0)
    last_pos = np.where(any_ev, ev.shape[0] - 1 - np.argmax(ev[::-1], axis=0), -1)

    last_close = close.iloc[-1]
    prev_close = close.iloc[-2]
    macd_days = days_since(c["macd_up"], 3)
    macd_dn_days = days_since(c["macd_dn"], 3)
    gc_days = days_since(c["gc"], 5)
    dc_days = days_since(c["dc"], 5)
    L = {k: c[k].iloc[-1] for k in ("rsi", "line", "sig", "sma20", "sma50", "vol_ratio", "score", "tier")}
    chg_last = last_close / prev_close - 1

    items = []
    cols = list(close.columns)
    for j, sym in enumerate(cols):
        if pd.isna(L["tier"][sym]) or pd.isna(last_close[sym]):
            continue
        reasons = []
        reasons.append("Fiyat SMA50 üstünde" if last_close[sym] > L["sma50"][sym] else "Fiyat SMA50 altında")
        reasons.append("SMA20 > SMA50 (yükseliş trendi)" if L["sma20"][sym] > L["sma50"][sym]
                       else "SMA20 < SMA50 (düşüş trendi)")
        reasons.append("MACD sinyal çizgisinin üstünde" if L["line"][sym] > L["sig"][sym]
                       else "MACD sinyal çizgisinin altında")
        if not pd.isna(macd_days[sym]):
            reasons.append(f"MACD yukarı kesti ({ago(macd_days[sym])})")
        if not pd.isna(macd_dn_days[sym]):
            reasons.append(f"MACD aşağı kesti ({ago(macd_dn_days[sym])})")
        if not pd.isna(gc_days[sym]):
            reasons.append(f"SMA20, SMA50'yi yukarı kesti ({ago(gc_days[sym])})")
        if not pd.isna(dc_days[sym]):
            reasons.append(f"SMA20, SMA50'yi aşağı kesti ({ago(dc_days[sym])})")
        rv = L["rsi"][sym]
        if rv >= 75:
            reasons.append(f"RSI aşırı alım ({rv:.0f})")
        elif rv < 30:
            reasons.append(f"RSI aşırı satım ({rv:.0f})")
        vr = L["vol_ratio"][sym]
        if not pd.isna(vr) and vr >= 1.5:
            reasons.append(f"Hacim ortalamanın {vr:.1f} katı ({'yükselişle' if chg_last[sym] > 0 else 'düşüşle'})")

        last_buy = None
        p = int(last_pos[j])
        if p >= 0:
            entry = float(close.iat[p, j])
            last_buy = {
                "date": f"{close.index[p]:%Y-%m-%d}",
                "tier": LABELS[int(tier.iat[p, j])],
                "price": round(entry, 2),
                "ret_now": round((float(last_close[sym]) / entry - 1) * 100, 2),
                "days": int(n_days - 1 - p),
            }

        items.append({
            "symbol": sym.replace(".IS", ""),
            "price": round(float(last_close[sym]), 2),
            "change_pct": round(float(chg_last[sym]) * 100, 2) if not pd.isna(chg_last[sym]) else 0.0,
            "tier": LABELS[int(L["tier"][sym])],
            "score": int(L["score"][sym]),
            "rsi": round(float(rv), 1),
            "reasons": reasons,
            "spark": [round(float(x), 2) for x in close[sym].dropna().tail(40)],
            "last_buy": last_buy,
        })
    items.sort(key=lambda x: (-x["score"], x["symbol"]))

    # --- son 120 günün AL sinyalleri ve sonuçları ---
    start = close.index.searchsorted(close.index[-1] - pd.Timedelta(days=120))
    recent = []
    for off, j in np.argwhere(ev[start:]):
        p = start + int(off)
        entry = float(close.iat[p, j])
        row = {
            "symbol": cols[j].replace(".IS", ""),
            "date": f"{close.index[p]:%Y-%m-%d}",
            "tier": LABELS[int(tier.iat[p, j])],
            "price": round(entry, 2),
            "ret_now": round((float(last_close.iloc[j]) / entry - 1) * 100, 2),
            "days": int(n_days - 1 - p),
        }
        for h in HORIZONS:
            v = fwd[h][p, j]
            row[f"r{h}"] = round(float(v) * 100, 2) if np.isfinite(v) else None
        recent.append(row)
    recent.sort(key=lambda x: (x["date"], x["tier"] == "GÜÇLÜ AL"), reverse=True)

    return {
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_bar": f"{close.index[-1]:%Y-%m-%d}",
        "n_stocks": len(items),
        "items": items,
        "stats": {"period": stats_period, "groups": stats},
        "recent": recent[:600],
    }


# ---------- veri ----------

def read_symbols(path: str) -> list:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [s.strip() + ".IS" for s in lines if s.strip() and not s.startswith("#")]


def fetch_all(symbols: list, period: str = "3y", chunk: int = 60):
    import yfinance as yf

    closes, vols = [], []
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i + chunk]
        raw = None
        for attempt in range(3):
            try:
                raw = yf.download(part, period=period, interval="1d", auto_adjust=True,
                                  progress=False, threads=True, group_by="column")
                break
            except Exception as exc:  # ağ hatası, istek sınırı vb.
                print(f"  parça {i // chunk + 1}: deneme {attempt + 1} hata: {exc}")
                time.sleep(5 * (attempt + 1))
        if raw is None or raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            c, v = raw["Close"], raw["Volume"]
        else:
            c = raw[["Close"]].rename(columns={"Close": part[0]})
            v = raw[["Volume"]].rename(columns={"Volume": part[0]})
        closes.append(c)
        vols.append(v)
        print(f"  parça {i // chunk + 1}/{-(-len(symbols) // chunk)}: {c.shape[1]} hisse")
    if not closes:
        raise SystemExit("Hiç veri alınamadı.")
    close = pd.concat(closes, axis=1)
    volume = pd.concat(vols, axis=1)
    close = close.loc[:, ~close.columns.duplicated()].sort_index()
    volume = volume.reindex(index=close.index, columns=close.columns)
    keep = close.columns[close.notna().sum() >= 130]
    close, volume = close[keep], volume[keep]
    close = close.dropna(how="all").ffill(limit=3)
    volume = volume.reindex(close.index)
    return close, volume


def main():
    ap = argparse.ArgumentParser(description="Borsa sinyal programı")
    ap.add_argument("symbols", nargs="*", help="boşsa bist_symbols.txt kullanılır")
    ap.add_argument("--symbols-file", default="bist_symbols.txt")
    ap.add_argument("--period", default="3y", help="yfinance periyodu (örn. 2y, 3y, 5y)")
    ap.add_argument("--out", default="signals.json")
    args = ap.parse_args()

    if args.symbols:
        symbols = [s if "." in s else s + ".IS" for s in args.symbols]
    else:
        symbols = read_symbols(args.symbols_file)
    print(f"{len(symbols)} sembol indiriliyor...")
    close, volume = fetch_all(symbols, args.period)
    if close.shape[1] < 5:
        raise SystemExit(f"Yeterli veri yok ({close.shape[1]} hisse), JSON yazılmadı.")

    payload = build(close, volume)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    counts = {}
    for it in payload["items"]:
        counts[it["tier"]] = counts.get(it["tier"], 0) + 1
    print(f"{payload['n_stocks']} hisse -> {out} ({out.stat().st_size // 1024} KB)")
    print("Dağılım:", counts)


if __name__ == "__main__":
    main()
