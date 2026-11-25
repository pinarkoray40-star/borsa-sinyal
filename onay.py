"""
onay.py - "Onaylı" sinyal (günlük veri). signals.py bu dosyayı içe aktarır; tek başına çalıştırılmaz.

Sinyal: Büyük Hareket 2 (buyuk2.py) ile AYNI sinyal. Fark sadece giriş zamanı: ONAYLI GIRIS.
  Sinyal gününden 'confirm' işlem günü sonra kapanış, sinyal günü kapanışının ÜSTÜNDEYSE giriş = o günün kapanışı.
  Değilse sinyal atlanır (girilmez).
İşlem kuralı buyuk.py ile aynı: izleyen stop %25, 84. işlem günü süre dolar; sayım giriş gününden başlar. Hepsi kapanış fiyatıyla.
Karşılaştırma için iki ek hesap yapılır:
  skip_*: Atlanan sinyaller, onaylıların giriş günüyle AYNI günden (onay günü kapanışı) girilseydi ne olurdu (adil kıyas).
  bh2_* : Aynı sinyallerin hepsine sinyal günü kapanışından girilseydi (Büyük Hareket 2 kuralı) ne olurdu.
buyuk.py ve buyuk2.py'ye dokunmaz; onların fonksiyonlarını içe aktarıp kullanır.
"""
import numpy as np
import pandas as pd

import buyuk as BY
import buyuk2 as B2

PARAM = dict(B2.PARAM)
PARAM.update({"confirm": 1, "live_from": "2026-11-25"})


def onay_block(close: pd.DataFrame, volume: pd.DataFrame, now=None, P: dict = PARAM, idx: pd.Series = None) -> dict:
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    ist = now.tz_convert("Europe/Istanbul")
    dropped = False
    # Gün henüz bitmediyse son mum yarım olabilir: sinyal hesabına katma
    if pd.Timestamp(close.index[-1]).date() == ist.date() and (ist.hour, ist.minute) < (18, 40):
        close, volume = close.iloc[:-1], volume.iloc[:-1]
        dropped = True
    if idx is None:
        idx = B2.fetch_index(close)
    dn, dr = B2.dayaniklilik(close, idx, P["window"])
    T = close.shape[0]
    C = close.to_numpy()
    DR = dr.to_numpy()
    cols = list(close.columns)
    L = BY.likit_maske(close, volume)
    bh = BY.ozellikler(close, volume)
    mask = bh & (np.nan_to_num(DR, nan=0.0) >= P["dn_min"])
    ii, jj = BY.sinyaller(mask, P["cooldown"])
    first = T - 1 - P["track"]
    c = int(P["confirm"])

    entries, pending, skip_pnl, bh2_pnl = [], [], [], []
    n_sig = n_skip = 0
    for p, j in zip(ii, jj):
        if p < first:
            continue
        sig = float(C[p, j])
        if not np.isfinite(sig) or sig <= 0:
            continue
        n_sig += 1
        d_sig = f"{close.index[p]:%Y-%m-%d}"
        dnr = int(round(float(DR[p, j]) * 100))
        # Kıyas: aynı sinyale sinyal günü kapanışından girilseydi (Büyük Hareket 2 kuralı)
        pb = (C[p + 1:p + P["hold"] + 1, j] / sig - 1) * 100
        stb, kb, xrb, _ = BY.takip(pb, P["sl"], P["hold"])
        if stb != "açık" and xrb is not None:
            bh2_pnl.append(float(xrb))
        q = p + c
        if q > T - 1:  # onay günü henüz gelmedi
            pending.append({"symbol": cols[j].replace(".IS", ""), "date": d_sig,
                            "signal_px": round(sig, 2), "dnr": dnr})
            continue
        entry = float(C[q, j])
        if not np.isfinite(entry) or entry <= 0 or entry <= sig:
            n_skip += 1
            # Atlanan sinyal: onaylılarla AYNI günden (onay günü kapanışı) girseydik sonuç ne olurdu?
            if np.isfinite(entry) and entry > 0:
                path0 = (C[q + 1:q + P["hold"] + 1, j] / entry - 1) * 100
                st0, k0, xr0, _ = BY.takip(path0, P["sl"], P["hold"])
                if st0 != "açık" and xr0 is not None:
                    skip_pnl.append(float(xr0))
            continue
        path = (C[q + 1:q + P["hold"] + 1, j] / entry - 1) * 100
        st, k, xr, peak = BY.takip(path, P["sl"], P["hold"])
        seen = [0.0] + [round(float(v), 2) if np.isfinite(v) else None for v in path[:k]]
        vals = [v for v in seen if v is not None]
        cur = xr if xr is not None else (vals[-1] if vals else 0.0)
        b = BY.kiyas(C, L, q, int(k))
        d = f"{close.index[q]:%Y-%m-%d}"
        entries.append({
            "symbol": cols[j].replace(".IS", ""), "date": d, "sdate": d_sig, "sig_px": round(sig, 2),
            "entry": round(entry, 2),
            "peak_px": round(entry * (1 + peak / 100), 2),
            "stop_px": round(entry * (1 + peak / 100) * (1 - P["sl"] / 100), 2),
            "st": st, "days": int(k), "pnl": round(float(cur), 2), "max": max(vals), "min": min(vals),
            "xdate": f"{close.index[q + k]:%Y-%m-%d}" if st != "açık" else None,
            "bench": round(b, 2) if b is not None else None,
            "diff": round(float(cur) - b, 2) if b is not None else None,
            "live": d_sig >= P["live_from"], "dnr": dnr, "path": seen,
        })
    entries.sort(key=lambda e: (e["date"], e["dnr"], e["symbol"]), reverse=True)

    done = [e for e in entries if e["st"] != "açık"]
    opn = [e for e in entries if e["st"] == "açık"]
    stats = {"n": len(entries), "n_open": len(opn), "n_done": len(done),
             "n_signals": n_sig, "skipped": n_skip, "pending": len(pending)}
    if done:
        r = np.array([e["pnl"] for e in done])
        stats.update({
            "stop": round(sum(e["st"] == "stop" for e in done) / len(done) * 100, 1),
            "sure": round(sum(e["st"] == "süre" for e in done) / len(done) * 100, 1),
            "avg": round(float(r.mean()), 2), "median": round(float(np.median(r)), 2),
            "best": round(float(r.max()), 1), "worst": round(float(r.min()), 1),
            "win": round(float((r > 0).mean()) * 100, 1)})
    if opn:
        o = np.array([e["pnl"] for e in opn])
        stats["open_avg"] = round(float(o.mean()), 2)
        stats["open_win"] = round(float((o > 0).mean()) * 100, 1)
    if skip_pnl:
        s = np.array(skip_pnl)
        stats.update({"skip_done": len(s), "skip_avg": round(float(s.mean()), 2),
                      "skip_median": round(float(np.median(s)), 2),
                      "skip_win": round(float((s > 0).mean()) * 100, 1)})
    if bh2_pnl:
        h = np.array(bh2_pnl)
        stats.update({"bh2_done": len(h), "bh2_avg": round(float(h.mean()), 2),
                      "bh2_median": round(float(np.median(h)), 2),
                      "bh2_win": round(float((h > 0).mean()) * 100, 1)})
    BY.kiyas_istat(done, opn, stats)
    return {"params": P, "last_bar": f"{close.index[-1]:%Y-%m-%d}", "dropped_partial": dropped,
            "universe": int(close.shape[1]), "n": len(entries), "n_live": sum(e["live"] for e in entries),
            "stats": stats, "pending": pending, "entries": entries[:300]}
