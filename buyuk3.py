"""
buyuk3.py - "Büyük Hareket 3" sinyali (günlük veri). signals.py bu dosyayı içe aktarır; tek başına çalıştırılmaz.

Büyük Hareket 2 (buyuk2.py) ile AYNI sinyal + bir ek şart: ALT SEKTÖR ZAYIF OLMAMALI.
  sg = hissenin alt sektöründeki (yfinance industry, sektor.csv) DİĞER hisselerin son 20 günlük ortalama getirisi
       - tüm hisselerin ortalama getirisi (hissenin kendisi hesaba katılmaz).
  Şart: sg > 0 olmalı. sg <= 0 (alt sektör piyasadan zayıf) ise sinyal elenir.
  Alt sektörü bilinmeyen ya da çok küçük (6 hisseden az) olan hisselerin sinyali ELENMEZ.
İşlem kuralı aynı: giriş = sinyal günü kapanışı, izleyen stop %25, 84. işlem günü süre dolar. Hepsi kapanış fiyatıyla.
buyuk.py ve buyuk2.py'ye dokunmaz; onların fonksiyonlarını içe aktarıp kullanır.
sektor.csv yoksa hata verir ve signals.py bu sekmeyi boş bırakır (diğer sekmeler etkilenmez).
"""
import os

import numpy as np
import pandas as pd

import buyuk as BY
import buyuk2 as B2

PARAM = dict(B2.PARAM)
PARAM.update({"sektor_lb": 20, "min_uye": 6, "live_from": "2026-11-27"})
SEKTOR_DOSYA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sektor.csv")


def sektor_haritasi(close: pd.DataFrame):
    """Her hisse için alt sektör kimliği (-1 = bilinmiyor), alt sektör adları ve hisse başına ad."""
    if not os.path.exists(SEKTOR_DOSYA):
        raise RuntimeError("sektor.csv bulunamadı")
    df = pd.read_csv(SEKTOR_DOSYA)
    kol = {str(c).replace(".IS", ""): j for j, c in enumerate(close.columns)}
    ad = np.array([None] * close.shape[1], dtype=object)
    for kod, v in zip(df["kod"], df["industry"]):
        j = kol.get(str(kod))
        if j is not None and isinstance(v, str) and v.strip():
            ad[j] = v.strip()
    adlar = sorted({a for a in ad if a is not None})
    ids = np.full(close.shape[1], -1)
    for k, a in enumerate(adlar):
        ids[ad == a] = k
    return ids, adlar, ad


def sektor_gucu(R: np.ndarray, grup: np.ndarray, min_uye: int) -> np.ndarray:
    """Gün x hisse: hissenin grubunun (hissenin kendisi HARİÇ) ortalama getirisi - piyasa ortalaması."""
    T, S = R.shape
    gecerli = np.isfinite(R)
    Rz = np.where(gecerli, R, 0.0)
    pazar = Rz.sum(1) / np.maximum(gecerli.sum(1), 1)
    guc = np.full((T, S), np.nan)
    for g in np.unique(grup[grup >= 0]):
        uye = np.where(grup == g)[0]
        if len(uye) < min_uye:
            continue
        s = Rz[:, uye].sum(1, keepdims=True)
        c = gecerli[:, uye].sum(1, keepdims=True)
        loo_s = s - Rz[:, uye]
        loo_c = c - gecerli[:, uye]
        loo = np.where(loo_c >= 3, loo_s / np.maximum(loo_c, 1), np.nan)
        guc[:, uye] = loo - pazar[:, None]
    return guc


def buyuk3_block(close: pd.DataFrame, volume: pd.DataFrame, now=None, P: dict = PARAM, idx: pd.Series = None) -> dict:
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    ist = now.tz_convert("Europe/Istanbul")
    dropped = False
    # Gün henüz bitmediyse (borsa 18:00'de kapanır, Yahoo gecikmeli) son mum yarım olabilir: sinyal hesabına katma
    if pd.Timestamp(close.index[-1]).date() == ist.date() and (ist.hour, ist.minute) < (18, 40):
        close, volume = close.iloc[:-1], volume.iloc[:-1]
        dropped = True
    if idx is None:
        idx = B2.fetch_index(close)
    dn, dr = B2.dayaniklilik(close, idx, P["window"])
    ids, adlar, ad = sektor_haritasi(close)
    R = close.pct_change(P["sektor_lb"], fill_method=None).clip(-0.5, 1.0).to_numpy()
    G = sektor_gucu(R, ids, P["min_uye"])

    T = close.shape[0]
    C = close.to_numpy()
    DR = dr.to_numpy()
    cols = list(close.columns)
    L = BY.likit_maske(close, volume)
    bh = BY.ozellikler(close, volume)
    dn_ok = np.nan_to_num(DR, nan=0.0) >= P["dn_min"]
    mask2 = bh & dn_ok                 # Büyük Hareket 2 şartları
    sek_ok = ~(G <= 0)                 # zayıf alt sektör elenir; bilinmeyen (NaN) tutulur
    mask = mask2 & sek_ok
    first = T - 1 - P["track"]
    ii2, jj2 = BY.sinyaller(mask2, P["cooldown"])
    n_bh2 = int(sum(1 for p in ii2 if p >= first))
    ii, jj = BY.sinyaller(mask, P["cooldown"])
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
        b = BY.kiyas(C, L, p, int(k))
        d = f"{close.index[p]:%Y-%m-%d}"
        g = G[p, j]
        entries.append({
            "symbol": cols[j].replace(".IS", ""), "date": d, "entry": round(entry, 2),
            "peak_px": round(entry * (1 + peak / 100), 2),
            "stop_px": round(entry * (1 + peak / 100) * (1 - P["sl"] / 100), 2),
            "st": st, "days": int(k), "pnl": round(float(cur), 2), "max": max(vals), "min": min(vals),
            "xdate": f"{close.index[p + k]:%Y-%m-%d}" if st != "açık" else None,
            "bench": round(b, 2) if b is not None else None,
            "diff": round(float(cur) - b, 2) if b is not None else None,
            "live": d >= P["live_from"], "dnr": int(round(float(DR[p, j]) * 100)), "path": seen,
            "ind": ad[j], "sg": round(float(g) * 100, 1) if np.isfinite(g) else None,
        })
    # Yeni tarih önce; aynı gün içinde dayanıklılık sırası (dnr) yüksek olan üstte
    entries.sort(key=lambda e: (e["date"], e["dnr"], e["symbol"]), reverse=True)

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
    BY.kiyas_istat(done, opn, stats)
    return {"params": P, "last_bar": f"{close.index[-1]:%Y-%m-%d}", "dropped_partial": dropped,
            "universe": int(close.shape[1]), "n": len(entries), "n_live": sum(e["live"] for e in entries),
            "n_bh2": n_bh2, "sektor_bilinen": int((ids >= 0).sum()),
            "stats": stats, "entries": entries[:300]}
