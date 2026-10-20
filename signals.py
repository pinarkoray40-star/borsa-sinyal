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
import os
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


def momentum_block(close: pd.DataFrame, volume: pd.DataFrame, topn: int = 10, step: int = 21) -> dict:
    """6 aylık momentum sıralaması + aynı kuralın geçmiş (aylık yenilenen portföy) testi."""
    r126 = close / close.shift(126) - 1
    r20 = close / close.shift(20) - 1
    chg = close.pct_change(fill_method=None)
    vol20 = chg.rolling(20).std()
    hi55 = close.rolling(55).max()
    turn = (close * volume).rolling(20).median()
    liquid = turn.rank(axis=1, pct=True) >= 0.30
    sc = r126.where(liquid)

    last = sc.iloc[-1].dropna().sort_values(ascending=False)
    rank_pct = sc.rank(axis=1, pct=True).iloc[-1]
    vol_rank = vol20.where(liquid).rank(axis=1, pct=True).iloc[-1]
    vrat = (volume / volume.rolling(20).mean().replace(0, np.nan)).iloc[-1]
    top = []
    for k, (sym, v) in enumerate(last.items(), 1):
        px = float(close[sym].iloc[-1])
        near = bool(px >= float(hi55[sym].iloc[-1]) * 0.97)
        vr = float(vrat[sym]) if pd.notna(vrat[sym]) else 0.0
        zv = bool(rank_pct[sym] >= 0.80 and near and vr >= 1.2)
        if k > 40 and not zv:
            continue
        top.append({
            "rank": k,
            "symbol": sym.replace(".IS", ""),
            "price": round(px, 2),
            "change_pct": round(float(chg[sym].iloc[-1]) * 100, 2) if pd.notna(chg[sym].iloc[-1]) else 0.0,
            "r126": round(float(v) * 100, 1),
            "r20": round(float(r20[sym].iloc[-1]) * 100, 1) if pd.notna(r20[sym].iloc[-1]) else None,
            "calm": bool(vol_rank[sym] <= 0.20 and rank_pct[sym] >= 0.50),
            "near_high": near,
            "zv": zv,
            "volrat": round(vr, 1),
            "top10pct": bool(rank_pct[sym] >= 0.90),
            "spark": [round(float(x), 2) for x in close[sym].dropna().tail(40)],
        })

    # geçmiş test: her `step` günde en güçlü topn hisseyi al, bir sonraki yenilemede sat
    rows = []
    for pos in range(130, len(close) - step, step):
        sr = sc.iloc[pos].dropna()
        if len(sr) < 30:
            continue
        pick = sr.nlargest(topn).index
        fw = (close.iloc[pos + step] / close.iloc[pos] - 1)
        mom = fw[pick].dropna().mean()
        mkt = fw[sr.index].dropna().mean()
        if np.isnan(mom) or np.isnan(mkt):
            continue
        rows.append({"start": f"{close.index[pos]:%Y-%m-%d}", "end": f"{close.index[pos + step]:%Y-%m-%d}",
                     "mom": float(mom) - 0.004, "mkt": float(mkt),
                     "picks": [x.replace(".IS", "") for x in pick]})
    hist = None
    if len(rows) >= 6:
        m = np.array([r["mom"] for r in rows]); k = np.array([r["mkt"] for r in rows])
        em, ek = np.cumprod(1 + m), np.cumprod(1 + k)
        dd = lambda e: float((e / np.maximum.accumulate(e) - 1).min())
        hist = {
            "months": len(rows),
            "mom_total": round(float(em[-1]), 2), "mkt_total": round(float(ek[-1]), 2),
            "mom_dd": round(dd(em) * 100, 0), "mkt_dd": round(dd(ek) * 100, 0),
            "beat_pct": round(float((m > k).mean()) * 100, 0),
            "worst": round(float(m.min()) * 100, 1), "best": round(float(m.max()) * 100, 1),
            "avg": round(float(m.mean()) * 100, 2),
            "last": [{"start": r["start"], "end": r["end"], "mom": round(r["mom"] * 100, 1),
                      "mkt": round(r["mkt"] * 100, 1), "picks": r["picks"]} for r in rows[-12:]][::-1],
        }
    return {"top": top, "hist": hist, "topn": topn}


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
        "momentum": momentum_block(close, volume),
    }


# ---------- Dip OBV Dengeli v1.5 (TradingView göstergesi, 5 dakikalık mum) ----------
# Parametreler TradingView'daki varsayılanlarla aynıdır. DEĞİŞTİRMEYİN.
DIP = {"rel_vol": 8.9, "narrow": 1.0, "obv_lb": 98, "rsi_len": 3, "vol_mult": 1.1, "flow_lb": 41}
DIP_TRACK = 126          # ~6 ay (işlem günü) takip
DIP_DELAY_MIN = 22       # Yahoo ~15 dk gecikmeli; bu süreden yeni mumlar henüz kapanmamış sayılır


def _rma(x: pd.Series, n: int) -> pd.Series:
    """Pine ta.rma (Wilder)."""
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def dip_components(h: pd.Series, l: pd.Series, c: pd.Series, v: pd.Series, P: dict = DIP) -> pd.DataFrame:
    """Pine kodunun birebir karşılığı. Her satır bir 5 dk mumu."""
    avg_vol = v.rolling(30).mean()
    rel_vol = v / avg_vol
    lo15 = l.rolling(15).min()
    price_range = (h.rolling(15).max() - lo15) / lo15 * 100
    is_narrow = price_range < P["narrow"]

    d = c.diff()
    signed = pd.Series(np.where(d > 0, v, np.where(d < 0, -v, 0.0)), index=c.index)
    obv_rising = signed.rolling(P["obv_lb"]).sum() > 0          # obv > obv[lookback]

    up = lambda k: (v.shift(k) > v.shift(k + 1)).astype(int)
    vol_build = (up(0) + up(1) + up(2) + up(3)) >= 2

    ch = c.diff()
    gain = _rma(ch.clip(lower=0), P["rsi_len"])
    loss = _rma((-ch).clip(lower=0), P["rsi_len"])
    rs = gain / loss
    rsi_v = pd.Series(np.where(loss == 0, 100.0, np.where(gain == 0, 0.0, 100 - 100 / (1 + rs))), index=c.index)
    rsi_v = rsi_v.where(gain.notna() & loss.notna())
    rsi_bottom = (rsi_v >= 46) & (rsi_v <= 63)

    rng = h - l
    mfm = pd.Series(np.where(rng != 0, ((c - l) - (h - c)) / rng.replace(0, np.nan), 0.0), index=c.index)
    mfv = mfm * v
    cmf = mfv.rolling(21).mean() / v.rolling(21).mean()
    cmf_pos = (cmf > 0) & (cmf > cmf.shift(1))

    nmf = mfv.rolling(P["flow_lb"]).sum()
    flow_pos = (nmf > 0) & (nmf > nmf.shift(P["flow_lb"]))

    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = _rma(tr, 14)
    atr_avg = atr.rolling(20).mean()
    vol_ok = atr > atr_avg * P["vol_mult"]

    rv_ok = rel_vol >= P["rel_vol"]
    dip_acc = is_narrow & obv_rising & vol_build & rsi_bottom & cmf_pos & flow_pos & rv_ok & vol_ok
    score = (is_narrow * 24 + obv_rising * 28 + vol_build * 16 + rsi_bottom * 10 + cmf_pos * 8
             + flow_pos * 14 + rv_ok * 5 + vol_ok * 5).clip(upper=100)
    buy = dip_acc & (score >= 72)
    return pd.DataFrame({"buy": buy, "score": score, "rel_vol": rel_vol, "flow": nmf, "rsi": rsi_v})


# ---------- Dip OBV Dengeli v2.0 Sade (ikinci gösterge, ayrı kayıt) ----------
# Ayarlar kullanıcının TradingView'daki değerleridir. DEĞİŞTİRMEYİN.
DIP2 = {"rel_vol": 12.7, "narrow": 1.0, "obv_lb": 111, "obv_band": 6, "obv_ratio": 0.92, "rsi_len": 3,
        "vol_mult": 1.0, "dip_lb": 40, "dip_tol": 6.0, "pin": 1.0, "brk_vol": 12.7}


def dip2_components(h, l, c, o, v, P: dict = DIP2) -> pd.DataFrame:
    """Pine 'Dip OBV Dengeli v2.0 Sade' kodunun birebir karşılığı."""
    avg_vol = v.rolling(30).mean()
    rel_vol = v / avg_vol
    lo15 = l.rolling(15).min()
    is_narrow = ((h.rolling(15).max() - lo15) / lo15 * 100) < P["narrow"]

    d = c.diff()
    signed = pd.Series(np.where(d > 0, v, np.where(d < 0, -v, 0.0)), index=c.index)
    cs = signed.cumsum()
    lo_n, hi_n = max(1, P["obv_lb"] - P["obv_band"]), P["obv_lb"] + P["obv_band"]
    cnt = sum(((cs - cs.shift(n)) > 0).astype(int) for n in range(lo_n, hi_n + 1))  # obv > obv[n]
    total = P["obv_band"] * 2 + 1
    obv_ratio = cnt / total
    obv_robust = obv_ratio >= P["obv_ratio"]

    up = lambda k: (v.shift(k) > v.shift(k + 1)).astype(int)
    vol_build = (up(0) + up(1) + up(2) + up(3)) >= 2

    ch = c.diff()
    gain = _rma(ch.clip(lower=0), P["rsi_len"])
    loss = _rma((-ch).clip(lower=0), P["rsi_len"])
    rs = gain / loss
    rsi_v = pd.Series(np.where(loss == 0, 100.0, np.where(gain == 0, 0.0, 100 - 100 / (1 + rs))), index=c.index)
    rsi_v = rsi_v.where(gain.notna() & loss.notna())
    rsi_bottom = (rsi_v >= 46) & (rsi_v <= 63)

    rng = h - l
    mfm = pd.Series(np.where(rng != 0, ((c - l) - (h - c)) / rng.replace(0, np.nan), 0.0), index=c.index)
    mfv = mfm * v
    cmf = mfv.rolling(21).mean() / v.rolling(21).mean()
    cmf_pos = (cmf > 0) & (cmf > cmf.shift(1))

    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = _rma(tr, 14)
    vol_ok = atr > atr.rolling(20).mean() * P["vol_mult"]

    recent_low = l.rolling(P["dip_lb"]).min()
    near_dip = ((c - recent_low) / recent_low * 100) <= P["dip_tol"]

    body = (c - o).abs()
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
    upper_half = c >= (l + (h - l) * 0.5)
    reversal = (body > 0) & (lower_wick >= body * P["pin"]) & upper_half
    brk_ok = v > avg_vol * P["brk_vol"]
    rv_ok = rel_vol >= P["rel_vol"]

    acc = is_narrow & obv_robust & vol_build & rsi_bottom & cmf_pos & rv_ok & vol_ok & near_dip & reversal & brk_ok
    score = (is_narrow * 25 + obv_robust * 30 + vol_build * 16 + rsi_bottom * 11 + cmf_pos * 9
             + rv_ok * 5 + vol_ok * 4).clip(upper=100)
    buy = acc & (score >= 72)
    return pd.DataFrame({"buy": buy, "score": score, "rel_vol": rel_vol, "flow": obv_ratio * 100, "rsi": rsi_v})


def fetch_intraday(symbols: list, period: str, interval: str = "5m", chunk: int = 50):
    """5 dk OHLCV. Yahoo 5 dk geçmişini en fazla 60 gün verir."""
    import yfinance as yf

    parts = {k: [] for k in ("Open", "High", "Low", "Close", "Volume")}
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i + chunk]
        raw = None
        for attempt in range(3):
            try:
                raw = yf.download(part, period=period, interval=interval, auto_adjust=True,
                                  progress=False, threads=True, group_by="column")
                break
            except Exception as exc:
                print(f"  5dk parça {i // chunk + 1}: deneme {attempt + 1} hata: {exc}")
                time.sleep(5 * (attempt + 1))
        if raw is None or raw.empty:
            continue
        for k in parts:
            if isinstance(raw.columns, pd.MultiIndex):
                parts[k].append(raw[k])
            else:
                parts[k].append(raw[[k]].rename(columns={k: part[0]}))
        print(f"  5dk parça {i // chunk + 1}/{-(-len(symbols) // chunk)}: {raw.shape[0]} mum")
    if not parts["Close"]:
        return None
    out = {}
    for k, lst in parts.items():
        df = pd.concat(lst, axis=1)
        out[k] = df.loc[:, ~df.columns.duplicated()].sort_index()
    return out


def dip_scan(frames: dict, now: pd.Timestamp, kind: str = "v1") -> list:
    """Tüm hisselerde kapanmış 5 dk mumlarında AL sinyali arar."""
    H, L, C, V = frames["High"], frames["Low"], frames["Close"], frames["Volume"]
    O = frames.get("Open")
    idx = C.index
    tz = getattr(idx, "tz", None)
    cutoff = (now.tz_convert(tz) if tz is not None else now.tz_localize(None)) - pd.Timedelta(minutes=DIP_DELAY_MIN)
    found = []
    for sym in C.columns:
        cols = {"h": H[sym], "l": L[sym], "c": C[sym], "v": V[sym]}
        if kind == "v2":
            if O is None or sym not in O.columns:
                continue
            cols["o"] = O[sym]
        d = pd.DataFrame(cols).dropna()
        d = d[d.index <= cutoff]
        if len(d) < 130:  # OBV 98 ve net akış 41 uzun ısınma ister
            continue
        comp = (dip2_components(d["h"], d["l"], d["c"], d["o"], d["v"]) if kind == "v2"
                else dip_components(d["h"], d["l"], d["c"], d["v"]))
        hit = comp.index[comp["buy"].fillna(False).to_numpy()]
        for ts in hit:
            r = comp.loc[ts]
            found.append({
                "id": f"{sym}|{ts:%Y-%m-%d %H:%M}",
                "symbol": sym.replace(".IS", ""),
                "time": f"{ts:%Y-%m-%d %H:%M}",
                "date": f"{ts:%Y-%m-%d}",
                "entry": round(float(d.loc[ts, "c"]), 2),
                "score": int(r["score"]),
                "rel_vol": round(float(r["rel_vol"]), 1),
                "flow": round(float(r["flow"]), 1 if kind == "v2" else 0),
            })
    return found


def load_prev_log(url):
    """Önceki yayındaki günlüğü okur. 404 = ilk çalışma (None). Başka hata = dur (veri kaybı olmasın)."""
    import urllib.error
    import urllib.request

    err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(f"{url}?t={int(time.time())}",
                                         headers={"Cache-Control": "no-cache", "User-Agent": "borsa-sinyal"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err = e
        except Exception as e:
            err = e
        time.sleep(3 * (attempt + 1))
    raise SystemExit(f"Dip günlüğü okunamadı ({err}). Veri kaybı olmasın diye yayın durduruldu.")


def dip_block(close: pd.DataFrame, symbols: list, prev_log, now=None, kind: str = "v1", fetch=None):
    """Günlüğü günceller. Dönüş: (sayfa verisi, yazılacak günlük veya None)."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    P = DIP2 if kind == "v2" else DIP
    if prev_log is not None and prev_log.get("params") != P:
        prev_log = None  # ayarlar değişti: eski ayarla tutulan kayıt karışmasın, baştan başla
    first = prev_log is None
    entries = list(prev_log.get("entries", [])) if prev_log else []
    created = prev_log.get("created") if prev_log else now.isoformat(timespec="seconds")
    created_local = pd.Timestamp(created).tz_convert("Europe/Istanbul").strftime("%Y-%m-%d %H:%M")
    warning = None

    period = "60d" if first else "5d"
    frames = fetch(period) if fetch else fetch_intraday(symbols, period)
    if frames is None:
        warning = "5 dakikalık veri alınamadı; son bilinen kayıtlar gösteriliyor."
        if first:
            return None, None, warning
    else:
        have = {e["id"] for e in entries}
        for s in (dip_scan(frames, now) if kind == "v1" else dip_scan(frames, now, kind)):
            if s["id"] in have:
                continue
            s["src"] = "geçmiş" if (first or s["time"] < created_local) else "canlı"
            entries.append(s)
    entries.sort(key=lambda e: e["time"], reverse=True)
    log = {"version": 1, "params": P, "created": created, "entries": entries}

    # günlük kapanışlarla gün gün kâr/zarar
    dates = [f"{d:%Y-%m-%d}" for d in close.index]
    pos = {d: i for i, d in enumerate(dates)}
    cols = set(close.columns)
    arr = close.to_numpy()
    colpos = {c: j for j, c in enumerate(close.columns)}
    out_entries = []
    for e in entries:
        sym = e["symbol"] + ".IS"
        if sym not in cols:
            continue
        p0 = pos.get(e["date"])
        if p0 is None:
            later = [i for i, d in enumerate(dates) if d >= e["date"]]
            if not later:  # günlük veri henüz o günü içermiyor: kaydı koru, kâr/zarar sonra gelir
                out_entries.append({**e, "p0": len(dates), "pnl": [], "cur": None, "max": None, "min": None,
                                    "days": 0, "done": False})
                continue
            p0 = later[0]
        col = arr[p0:p0 + DIP_TRACK + 1, colpos[sym]]
        pnl = [round(float(x / e["entry"] - 1) * 100, 2) if np.isfinite(x) else None for x in col]
        vals = [x for x in pnl if x is not None]
        if not vals:
            continue
        out_entries.append({**e, "p0": p0, "pnl": pnl, "cur": vals[-1], "max": max(vals), "min": min(vals),
                            "days": len(pnl) - 1, "done": len(pnl) >= DIP_TRACK + 1})

    # özet: gün 1/5/21/63 sonrası, piyasa kıyasıyla
    stats = {}
    for hz in (1, 5, 21, 63):
        mk = (close.shift(-hz) / close - 1).mean(axis=1).to_numpy()
        r, b = [], []
        for e in out_entries:
            if len(e["pnl"]) > hz and e["pnl"][hz] is not None:
                r.append(e["pnl"][hz])
                m = mk[e["p0"]]
                if np.isfinite(m):
                    b.append(m * 100)
        if len(r) >= 3:
            stats[f"h{hz}"] = {"n": len(r), "avg": round(float(np.mean(r)), 2), "median": round(float(np.median(r)), 2),
                               "win": round(float((np.array(r) > 0).mean()) * 100, 1),
                               "mkt": round(float(np.mean(b)), 2) if b else None}
    cur = [e["cur"] for e in out_entries if e["cur"] is not None]
    live = [e for e in out_entries if e["src"] == "canlı"]
    page = {
        "params": P, "created": created, "n": len(out_entries), "n_live": len(live),
        "dates": dates, "entries": out_entries[:400], "stats": stats,
        "now": {"n": len(cur), "avg": round(float(np.mean(cur)), 2) if cur else None,
                "median": round(float(np.median(cur)), 2) if cur else None,
                "win": round(float((np.array(cur) > 0).mean()) * 100, 1) if cur else None},
        "warning": warning,
    }
    return page, log, warning


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
    ap.add_argument("--dip-log-url", default=None, help="önceki yayındaki dip_log.json adresi")
    ap.add_argument("--no-dip", action="store_true", help="Dip OBV bölümünü atla")
    args = ap.parse_args()

    if args.symbols:
        symbols = [s if "." in s else s + ".IS" for s in args.symbols]
    else:
        symbols = read_symbols(args.symbols_file)
    print(f"{len(symbols)} sembol indiriliyor...")
    close, volume = fetch_all(symbols, args.period)
    if close.shape[1] < 5:
        raise SystemExit(f"Yeterli veri yok ({close.shape[1]} hisse), JSON yazılmadı.")

    url = args.dip_log_url
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not url and repo:
        owner, name = repo.split("/", 1)
        url = f"https://{owner.lower()}.github.io/{name}/dip_log.json"
    url2 = url.replace("dip_log.json", "dip2_log.json") if url else None
    prev = load_prev_log(url) if (url and not args.no_dip) else None
    prev2 = load_prev_log(url2) if (url2 and not args.no_dip) else None
    need60 = any(pl is None or pl.get("params") != P for pl, P in ((prev, DIP), (prev2, DIP2)))
    cache = {}

    def fetch(_period):  # iki gösterge için 5 dk veriyi bir kez indir
        if "d" not in cache:
            cache["d"] = fetch_intraday(symbols, "60d" if need60 else "5d")
        return cache["d"]

    payload = build(close, volume)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_dip:
        for key, kind, pl, fname in (("dip", "v1", prev, "dip_log.json"), ("dip2", "v2", prev2, "dip2_log.json")):
            page, log, warn = dip_block(close, symbols, pl, kind=kind, fetch=fetch)
            payload[key] = page
            if log is not None:
                (out.parent / fname).write_text(json.dumps(log, ensure_ascii=False, separators=(",", ":")),
                                                encoding="utf-8")
            print(f"{key}:", f"{page['n']} sinyal ({page['n_live']} canlı)" if page else "veri yok", warn or "")
    out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    counts = {}
    for it in payload["items"]:
        counts[it["tier"]] = counts.get(it["tier"], 0) + 1
    print(f"{payload['n_stocks']} hisse -> {out} ({out.stat().st_size // 1024} KB)")
    print("Dağılım:", counts)


if __name__ == "__main__":
    main()
