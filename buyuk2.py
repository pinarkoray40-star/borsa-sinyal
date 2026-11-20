"""
buyuk2.py - "Büyük Hareket 2" sinyali (günlük veri). signals.py bu dosyayı içe aktarır; tek başına çalıştırılmaz.

Büyük Hareket (buyuk.py) ile AYNI sinyal + bir ek şart: DAYANIKLILIK.
  dn = son 120 işlem gününde, BIST 100'ün DÜŞTÜĞÜ günlerde hissenin endekse göre ortalama fazla getirisi.
  Şart: dn'in tüm hisseler arasındaki yüzdelik sırası >= %70 (yani endeks düşerken en dayanıklı %30'da).
İşlem kuralı buyuk.py ile aynı: giriş = sinyal günü kapanışı, izleyen stop %25, 84. işlem günü süre dolar. Hepsi kapanış fiyatıyla.
buyuk.py'ye dokunmaz; onun sinyal/takip fonksiyonlarını içe aktarıp kullanır.
Endeks verisini (XU100.IS) kendisi indirir; indiremezse hata verir ve signals.py bu sekmeyi boş bırakır (diğer sekmeler etkilenmez).
"""
import numpy as np
import pandas as pd

import buyuk as BY

PARAM = dict(BY.PARAM)
PARAM.update({"dn_min": 0.70, "window": 120, "live_from": "2026-11-20"})
INDEX = "XU100.IS"


def fetch_index(close: pd.DataFrame) -> pd.Series:
    import yfinance as yf

    start = (pd.Timestamp(close.index[0]) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    raw = yf.download(INDEX, start=start, interval="1d", auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("XU100 verisi alınamadı")
    c = raw["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    c = c.dropna()
    c.index = pd.to_datetime(c.index).tz_localize(None)
    return c


def dayaniklilik(close: pd.DataFrame, idx: pd.Series, window: int):
    """Her gün için hisse bazında dn değeri ve gün içindeki yüzdelik sırası."""
    ix = idx.copy()
    ix.index = pd.to_datetime(ix.index).tz_localize(None)
    cidx = pd.to_datetime(close.index).tz_localize(None)
    ix = ix.reindex(cidx).ffill(limit=3)
    ix.index = close.index
    r = close.pct_change(fill_method=None)
    ri = ix.pct_change(fill_method=None)
    e = r.sub(ri, axis=0)
    dn = e.where(ri < 0, np.nan).rolling(window, min_periods=max(20, window // 4)).mean()
    return dn, dn.rank(axis=1, pct=True)


def buyuk2_block(close: pd.DataFrame, volume: pd.DataFrame, now=None, P: dict = PARAM, idx: pd.Series = None) -> dict:
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    ist = now.tz_convert("Europe/Istanbul")
    dropped = False
    # Gün henüz bitmediyse (borsa 18:00'de kapanır, Yahoo gecikmeli) son mum yarım olabilir: sinyal hesabına katma
    if pd.Timestamp(close.index[-1]).date() == ist.date() and (ist.hour, ist.minute) < (18, 40):
        close, volume = close.iloc[:-1], volume.iloc[:-1]
        dropped = True
    if idx is None:
        idx = fetch_index(close)
    dn, dr = dayaniklilik(close, idx, P["window"])
    T = close.shape[0]
    C = close.to_numpy()
    DR = dr.to_numpy()
    cols = list(close.columns)
    bh = BY.ozellikler(close, volume)
    dn_ok = np.nan_to_num(DR, nan=0.0) >= P["dn_min"]
    mask = bh & dn_ok
    ii, jj = BY.sinyaller(mask, P["cooldown"])
    first = T - 1 - P["track"]
    entries = []
    for p, j in zip(ii, jj):
        if p < first:
            continue
        entry = float(C[p, j])
        if not np.isfinite(entry) or entry <= 0:
            continue
        path = (C[p + 1:p + P["hold"] + 1, j] / entry - 1) * 100
        st, k, xr, peak = BY.takip(path, P["sl"], P["hold"])
        seen = [0.0] + [round(float(v), 2) if np.isfinite(v) else None for v in path[:k]]
        vals = [v for v in seen if v is not None]
        cur = xr if xr is not None else (vals[-1] if vals else 0.0)
        d = f"{close.index[p]:%Y-%m-%d}"
        entries.append({
            "symbol": cols[j].replace(".IS", ""), "date": d, "entry": round(entry, 2),
            "peak_px": round(entry * (1 + peak / 100), 2),
            "stop_px": round(entry * (1 + peak / 100) * (1 - P["sl"] / 100), 2),
            "st": st, "days": int(k), "pnl": round(float(cur), 2), "max": max(vals), "min": min(vals),
            "xdate": f"{close.index[p + k]:%Y-%m-%d}" if st != "açık" else None,
            "live": d >= P["live_from"], "dnr": int(round(float(DR[p, j]) * 100)), "path": seen,
        })
    entries.sort(key=lambda e: (e["date"], e["symbol"]), reverse=True)

    done = [e for e in entries if e["st"] != "açık"]
    opn = [e for e in entries if e["st"] == "açık"]
    stats = {"n": len(entries), "n_open": len(opn), "n_done": len(done)}
    if done:
        r = np.array([e["pnl"] for e in done])
        stats.update({
            "stop": round(sum(e["st"] == "stop" for e in done) / len(done) * 100, 1),
            "sure": round(sum(e["st"] == "süre" for e in done) / len(done) * 100, 1),
            "avg": round(float(r.mean()), 2), "median": round(float(np.median(r)), 2),
            "best": round(float(r.max()), 1), "worst": round(float(r.min()), 1),
            "win": round(float((r > 0).mean()) * 100, 1)})
    if opn:
        c = np.array([e["pnl"] for e in opn])
        stats["open_avg"] = round(float(c.mean()), 2)
        stats["open_win"] = round(float((c > 0).mean()) * 100, 1)
    return {"params": P, "last_bar": f"{close.index[-1]:%Y-%m-%d}", "dropped_partial": dropped,
            "universe": int(close.shape[1]), "n": len(entries), "n_live": sum(e["live"] for e in entries),
            "stats": stats, "entries": entries[:300]}
