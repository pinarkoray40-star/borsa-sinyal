"""
ayakta.py - "Ayakta" sinyali (günlük veri). signals.py bu dosyayı içe aktarır; tek başına çalıştırılmaz.

Fikir: endeks zayıfken kendi başına yükselen hisseyi bulmak.
Sinyal (hepsi aynı gün kapanışında sağlanmalı):
  endeks zayıf : BIST 100, son 60 işlem gününde en az %3 düşmüş (endeks zayıf değilse hiç sinyal çıkmaz)
  ayakta       : hisse aynı 60 günde en az %10 yükselmiş (mutlak getiri)
  likit        : 60 günlük ortalama işlem hacmi (TL) en düşük %30 hisse hariç
Aynı hissede sinyalden sonra 84 gün yeni sinyal sayılmaz (tek pozisyon).
İşlem kuralı (takip): giriş = sinyal günü kapanışı; 84. işlem günü kapanışıyla SÜRE DOLDU. STOP YOK
(geçmiş testte stop denenmedi, test 84 gün tutma üzerine yapıldı). Hepsi kapanış fiyatıyla.
Geçmiş test (ayakta_test.py, 5 yıl, 525 hisse, W=60, grup B): 84 gün sonra aynı günün tüm likit hisse ortalamasına göre +5,5 puan.
Kıyas: her işlem için, aynı giriş gününde likit hisselerin aynı sürede (işlemin çıkış gününe ya da bugüne kadar) ortalama getirisi.
"Bugün şartı sağlayanlar": cooldown'a takılsa da takılmasa da, son kapanışta şartları tutan hisseler.
buyuk.py ve buyuk2.py'ye dokunmaz; onların takip fonksiyonunu ve endeks indirmesini içe aktarır.
Kayıt tutmaz: sinyaller ve sonuçlar her seferinde günlük veriden yeniden hesaplanır.
"""
import numpy as np
import pandas as pd

import buyuk as BY
import buyuk2 as BY2

PARAM = {"w": 60, "weak": -3.0, "min_sr": 10.0, "sl": 100.0, "hold": 84, "track": 126, "cooldown": 84,
         "live_from": "2026-11-20"}


def hizala_endeks(close: pd.DataFrame, idx: pd.Series) -> pd.Series:
    """Endeks serisini hisse tarihlerine hizalar (en fazla 3 gün ileri doldurur)."""
    cidx = pd.to_datetime(close.index).tz_localize(None)
    ix = idx.copy()
    ix.index = pd.to_datetime(ix.index).tz_localize(None)
    ix = ix.reindex(cidx).ffill(limit=3)
    ix.index = close.index
    return ix


def maske(close: pd.DataFrame, volume: pd.DataFrame, ix: pd.Series, P: dict = PARAM):
    """Dönüş: (sinyal maskesi np.ndarray, hisse W günlük getirisi, endeks W günlük getirisi, likit maskesi)."""
    W = P["w"]
    sr = close / close.shift(W) - 1
    ir = ix / ix.shift(W) - 1
    weak = (ir <= P["weak"] / 100).to_numpy()
    ciro = (close * volume).rolling(60, min_periods=30).mean()
    likit = ciro.rank(axis=1, pct=True) >= 0.30
    m = (sr > P["min_sr"] / 100) & likit & close.notna() & close.shift(W).notna()
    m = (m & weak[:, None]).fillna(False).to_numpy()
    return m, sr.to_numpy(), ir.to_numpy(), likit.to_numpy()


def ayakta_block(close: pd.DataFrame, volume: pd.DataFrame, now=None, P: dict = PARAM, idx: pd.Series = None) -> dict:
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    ist = now.tz_convert("Europe/Istanbul")
    dropped = False
    # Gün henüz bitmediyse (borsa 18:00'de kapanır, Yahoo gecikmeli) son mum yarım olabilir: sinyal hesabına katma
    if pd.Timestamp(close.index[-1]).date() == ist.date() and (ist.hour, ist.minute) < (18, 40):
        close, volume = close.iloc[:-1], volume.iloc[:-1]
        dropped = True
    if idx is None:
        idx = BY2.fetch_index(close)
    ix = hizala_endeks(close, idx)

    mask, SR, IR, L = maske(close, volume, ix, P)
    T = close.shape[0]
    C = close.to_numpy()
    cols = list(close.columns)
    ii, jj = BY.sinyaller(mask, P["cooldown"])
    last_sig = {}
    for p, j in zip(ii, jj):
        last_sig[j] = max(last_sig.get(j, -1), p)

    def kiyas(p, k):
        """Giriş gününde likit hisselerin k gün sonraki ortalama getirisi (%)."""
        if k <= 0 or p + k >= T:
            return None
        a, b = C[p], C[p + k]
        ok = L[p] & np.isfinite(a) & np.isfinite(b) & (a > 0)
        return float(np.mean(b[ok] / a[ok] - 1) * 100) if ok.any() else None

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
        b = kiyas(p, int(k))
        d = f"{close.index[p]:%Y-%m-%d}"
        entries.append({
            "symbol": cols[j].replace(".IS", ""), "date": d, "entry": round(entry, 2),
            "st": st, "days": int(k), "pnl": round(float(cur), 2), "max": max(vals), "min": min(vals),
            "xdate": f"{close.index[p + k]:%Y-%m-%d}" if st != "açık" else None,
            "live": d >= P["live_from"], "sr": round(float(SR[p, j]) * 100, 1),
            "ir": round(float(IR[p]) * 100, 1) if np.isfinite(IR[p]) else None,
            "bench": round(b, 2) if b is not None else None,
            "diff": round(float(cur) - b, 2) if b is not None else None, "path": seen,
        })
    entries.sort(key=lambda e: (e["date"], e["symbol"]), reverse=True)

    done = [e for e in entries if e["st"] != "açık"]
    opn = [e for e in entries if e["st"] == "açık"]
    stats = {"n": len(entries), "n_open": len(opn), "n_done": len(done)}
    if done:
        r = np.array([e["pnl"] for e in done])
        stats.update({
            "avg": round(float(r.mean()), 2), "median": round(float(np.median(r)), 2),
            "best": round(float(r.max()), 1), "worst": round(float(r.min()), 1),
            "win": round(float((r > 0).mean()) * 100, 1)})
        dd = [e for e in done if e["diff"] is not None]
        if dd:
            stats.update({"bench_avg": round(float(np.mean([e["bench"] for e in dd])), 2),
                          "diff_avg": round(float(np.mean([e["diff"] for e in dd])), 2),
                          "beat": round(float(np.mean([e["diff"] > 0 for e in dd])) * 100, 1)})
    if opn:
        c = np.array([e["pnl"] for e in opn])
        stats["open_avg"] = round(float(c.mean()), 2)
        stats["open_win"] = round(float((c > 0).mean()) * 100, 1)
        od = [e for e in opn if e["diff"] is not None]
        if od:
            stats.update({"open_bench": round(float(np.mean([e["bench"] for e in od])), 2),
                          "open_diff": round(float(np.mean([e["diff"] for e in od])), 2)})

    # bugün şartı sağlayanlar (cooldown'a bakmadan)
    bugun = []
    for j in np.flatnonzero(mask[T - 1]):
        sp = last_sig.get(int(j))
        bugun.append({"symbol": cols[j].replace(".IS", ""), "sr": round(float(SR[T - 1, j]) * 100, 1),
                      "px": round(float(C[T - 1, j]), 2),
                      "last": f"{close.index[sp]:%Y-%m-%d}" if sp is not None else None,
                      "since": int(T - 1 - sp) if sp is not None else None,
                      "new": bool(sp is not None and sp == T - 1)})
    bugun.sort(key=lambda e: -e["sr"])

    last_ir = float(IR[-1]) if np.isfinite(IR[-1]) else None
    return {"params": P, "last_bar": f"{close.index[-1]:%Y-%m-%d}", "dropped_partial": dropped,
            "universe": int(close.shape[1]), "n": len(entries), "n_live": sum(e["live"] for e in entries),
            "index": {"ret": round(last_ir * 100, 1) if last_ir is not None else None,
                      "weak": bool(last_ir is not None and last_ir <= P["weak"] / 100), "w": P["w"]},
            "today": bugun[:60], "stats": stats, "entries": entries[:300]}

