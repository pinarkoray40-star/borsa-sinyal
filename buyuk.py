"""
buyuk.py - "Büyük Hareket" sinyali (günlük veri). signals.py bu dosyayı içe aktarır; tek başına çalıştırılmaz.

Sinyal (hepsi aynı gün kapanışında sağlanmalı, işlem hacmi en düşük %30 hisse hariç):
  hareketli      : son 60 günün günlük oynaklığı (std) tüm hisseler arasında en yüksek %40'ta
  20g kırılım    : kapanış, önceki 20 günün en yüksek kapanışına eşit ya da üstünde
  hacim kuruması : son 5 günün ortalama hacmi, 60 günlük ortalamanın %70'inden az
  zirveye yakın  : kapanış 60 günlük zirvenin en az %90'ı
Aynı hissede sinyalden sonra 21 gün yeni sinyal sayılmaz.
İşlem kuralı (takip): giriş = sinyal günü kapanışı; 42 işlem günü içinde kapanış önce +%30 olursa HEDEF,
önce -%15 olursa STOP, ikisi de olmazsa 42. gün kapanışıyla SÜRE DOLDU. Hepsi kapanış fiyatıyla.
Kayıt tutmaz: sinyaller ve sonuçlar her seferinde günlük veriden yeniden hesaplanır.
"""
import numpy as np
import pandas as pd

PARAM = {"tp": 30.0, "sl": 15.0, "hold": 42, "track": 126, "cooldown": 21, "live_from": "2026-11-18"}


def ozellikler(close: pd.DataFrame, volume: pd.DataFrame):
    r = close.pct_change()
    vol60 = r.rolling(60).std()
    hareketli = vol60.rank(axis=1, pct=True) >= 0.6
    kirilim = close >= close.shift(1).rolling(20).max()
    kuruma = volume.rolling(5).mean() / volume.rolling(60).mean() < 0.7
    yakin = close >= 0.90 * close.rolling(60).max()
    ciro = (close * volume).rolling(20).mean()
    likit = ciro.rank(axis=1, pct=True) >= 0.30
    gecerli = likit & close.notna() & close.shift(60).notna()
    mask = gecerli & hareketli & kirilim & kuruma & yakin
    return mask.fillna(False).to_numpy()


def sinyaller(mask: np.ndarray, cooldown: int):
    out_i, out_j = [], []
    for j in range(mask.shape[1]):
        last = -10**9
        for i in np.flatnonzero(mask[:, j]):
            if i - last >= cooldown:
                out_i.append(int(i))
                out_j.append(j)
                last = i
    return out_i, out_j


def takip(path, tp, sl, hold):
    """path: girişten sonraki günlerin % getirisi. Dönüş: (durum, çıkış günü, çıkış getirisi)."""
    for k, v in enumerate(path, start=1):
        if not np.isfinite(v):
            continue
        if v >= tp:
            return "hedef", k, float(v)
        if v <= -sl:
            return "stop", k, float(v)
    if len(path) >= hold:
        fin = [v for v in path[:hold] if np.isfinite(v)]
        return "süre", hold, float(fin[-1]) if fin else 0.0
    return "açık", len(path), None


def buyuk_block(close: pd.DataFrame, volume: pd.DataFrame, now=None, P: dict = PARAM) -> dict:
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    ist = now.tz_convert("Europe/Istanbul")
    dropped = False
    # Gün henüz bitmediyse (borsa 18:00'de kapanır, Yahoo gecikmeli) son mum yarım olabilir: sinyal hesabına katma
    if pd.Timestamp(close.index[-1]).date() == ist.date() and (ist.hour, ist.minute) < (18, 40):
        close, volume = close.iloc[:-1], volume.iloc[:-1]
        dropped = True
    T = close.shape[0]
    C = close.to_numpy()
    cols = list(close.columns)
    mask = ozellikler(close, volume)
    ii, jj = sinyaller(mask, P["cooldown"])
    first = T - 1 - P["track"]
    entries = []
    for p, j in zip(ii, jj):
        if p < first:
            continue
        entry = float(C[p, j])
        if not np.isfinite(entry) or entry <= 0:
            continue
        path = (C[p + 1:p + P["hold"] + 1, j] / entry - 1) * 100
        st, k, xr = takip(path, P["tp"], P["sl"], P["hold"])
        seen = [0.0] + [round(float(v), 2) if np.isfinite(v) else None for v in path[:k]]
        vals = [v for v in seen if v is not None]
        cur = xr if xr is not None else (vals[-1] if vals else 0.0)
        d = f"{close.index[p]:%Y-%m-%d}"
        entries.append({
            "symbol": cols[j].replace(".IS", ""), "date": d, "entry": round(entry, 2),
            "tp_px": round(entry * (1 + P["tp"] / 100), 2), "sl_px": round(entry * (1 - P["sl"] / 100), 2),
            "st": st, "days": int(k), "pnl": round(float(cur), 2), "max": max(vals), "min": min(vals),
            "xdate": f"{close.index[p + k]:%Y-%m-%d}" if st != "açık" else None,
            "live": d >= P["live_from"], "path": seen,
        })
    entries.sort(key=lambda e: (e["date"], e["symbol"]), reverse=True)

    done = [e for e in entries if e["st"] != "açık"]
    opn = [e for e in entries if e["st"] == "açık"]
    stats = {"n": len(entries), "n_open": len(opn), "n_done": len(done)}
    if done:
        r = np.array([e["pnl"] for e in done])
        stats.update({
            "hedef": round(sum(e["st"] == "hedef" for e in done) / len(done) * 100, 1),
            "stop": round(sum(e["st"] == "stop" for e in done) / len(done) * 100, 1),
            "sure": round(sum(e["st"] == "süre" for e in done) / len(done) * 100, 1),
            "avg": round(float(r.mean()), 2), "median": round(float(np.median(r)), 2),
            "win": round(float((r > 0).mean()) * 100, 1)})
    if opn:
        c = np.array([e["pnl"] for e in opn])
        stats["open_avg"] = round(float(c.mean()), 2)
        stats["open_win"] = round(float((c > 0).mean()) * 100, 1)
    return {"params": P, "last_bar": f"{close.index[-1]:%Y-%m-%d}", "dropped_partial": dropped,
            "universe": int(close.shape[1]), "n": len(entries), "n_live": sum(e["live"] for e in entries),
            "stats": stats, "entries": entries[:300]}
