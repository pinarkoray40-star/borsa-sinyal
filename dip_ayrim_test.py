"""
dip_ayrim_test.py - "Büyük Hareket" sinyalinde HEDEFE gidenleri STOP olanlardan ayıran özellik var mı?

buyuk.py ile aynı 4 şart (hareketli + 20g kırılım + hacim kuruması + zirveye yakın), +%30 hedef / -%15 stop / 42 gün,
%0,4 komisyon. dip_buyuk_test.py, dip_dar_test.py, dip_gunluk_test.py, dip_islem_test.py ve signals.py aynı klasörde olmalı.
Sitedeki hiçbir dosyaya dokunmaz.
Bölüm 1: işlemler 6'şar aylık dönemlere bölünür (strateji her dönemde aynı mı çalışıyor?).
Bölüm 2: sinyal gününde ölçülen ~20 özellik, her biri 3 dilime bölünüp hedef/stop/getiri karşılaştırılır.
Bölüm 3: ilk yarıda seç, ikinci yarıda sına (ve tersi): hangi filtreler gerçekten tutuyor?
"""
import os

import numpy as np
import pandas as pd

import dip_buyuk_test as B
import dip_dar_test as D
import dip_gunluk_test as G
import dip_islem_test as I
import signals as S

TP, SL = 30.0, 15.0
MIN_N = 100
ADIM = 3


def yollar2(close, ii, jj):
    C = close.to_numpy()
    T = C.shape[0]
    ii, jj = np.array(ii, dtype=int), np.array(jj, dtype=int)
    ok = ii + I.H < T
    ii, jj = ii[ok], jj[ok]
    idx = ii[:, None] + np.arange(1, I.H + 1)[None, :]
    P = (C[idx, jj[:, None]] / C[ii, jj][:, None] - 1) * 100
    good = np.isfinite(P).all(axis=1)
    return ii[good], jj[good], P[good]


def ozellikler(close, volume, raw_count):
    T, N = close.shape
    r = close.pct_change()
    r20 = close / close.shift(20) - 1
    r60 = close / close.shift(60) - 1
    ciro = (close * volume).rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    mk20 = r20.median(axis=1)
    genis = (close > sma50).astype(float).where(close.notna()).mean(axis=1) * 100
    bc = lambda s: np.repeat(np.asarray(s, dtype=float)[:, None], N, axis=1)
    hi15, lo15 = close.rolling(15).max(), close.rolling(15).min()
    F = {
        "Sinyal günü getirisi %": r * 100,
        "5 günlük getiri %": (close / close.shift(5) - 1) * 100,
        "20 günlük getiri %": r20 * 100,
        "60 günlük getiri %": r60 * 100,
        "20g getiri - piyasa medyanı %": r20.sub(mk20, axis=0) * 100,
        "60 gün günlük oynaklık %": r.rolling(60).std() * 100,
        "Kırılım gücü (kapanış/önceki 20g zirve) %": (close / close.shift(1).rolling(20).max() - 1) * 100,
        "Zirve oranı (kapanış/60g zirve) %": close / close.rolling(60).max() * 100,
        "Bugünkü hacim / 20g ort.": volume / volume.rolling(20).mean(),
        "Hacim kuruması (5g/60g ort.)": volume.rolling(5).mean() / volume.rolling(60).mean(),
        "Ciro sırası (likidite) %": ciro.rank(axis=1, pct=True) * 100,
        "Fiyat (TL)": close,
        "15g bant genişliği %": (hi15 - lo15) / lo15 * 100,
        "SMA50'ye uzaklık %": (close / sma50 - 1) * 100,
        "SMA200'e uzaklık %": (close / sma200 - 1) * 100,
        "Son 10 günde yükselen gün oranı %": (r > 0).astype(float).rolling(10).mean() * 100,
        "PİYASA: 20g medyan getiri %": bc(mk20 * 100),
        "PİYASA: SMA50 üstü hisse oranı %": bc(genis),
        "PİYASA: o gün sinyal sayısı": bc(raw_count),
    }
    return {k: (v.to_numpy() if hasattr(v, "to_numpy") else v) for k, v in F.items()}


def ist(r, cu, cd, ti, sel, y):
    n = int(sel.sum())
    if n == 0:
        return {"n": 0}
    x = r[sel]
    a, b = r[sel & (ti < y)], r[sel & (ti >= y)]
    return {"n": n, "hedef": cu[sel].mean() * 100, "stop": cd[sel].mean() * 100, "ort": float(x.mean()),
            "med": float(np.median(x)), "y1": float(a.mean()) if len(a) >= 15 else np.nan,
            "y2": float(b.mean()) if len(b) >= 15 else np.nan}


def calistir(close, volume):
    T = close.shape[0]
    y = T // 2
    feats, gecerli = D.genis(close, volume)
    B.ekle(close, volume, feats)
    mask = gecerli & feats["hareketli"] & feats["kirilim"] & feats["kuruma"] & feats["yakin"]
    raw_count = mask.sum(axis=1)
    ii, jj = G.sinyaller(mask)
    ii, jj, P = yollar2(close, ii, jj)
    r, cu, cd = I.islem(P, TP, SL)
    allsel = np.ones(len(r), dtype=bool)
    F = ozellikler(close, volume, raw_count)
    V = {k: arr[ii, jj] for k, arr in F.items()}
    mk42 = (close.shift(-I.H) / close - 1).median(axis=1).to_numpy() * 100

    f = lambda x, d=1: "—" if (x is None or pd.isna(x)) else f"{x:+.{d}f}"
    g = lambda x, d=0: "—" if (x is None or pd.isna(x)) else f"{x:.{d}f}"
    base = ist(r, cu, cd, ii, allsel, y)
    L = ["# Büyük Hareket: hedefe gidenleri stop olanlardan ayıran ne?\n",
         f"{close.shape[1]} hisse, {close.index[0]:%d.%m.%Y} - {close.index[-1]:%d.%m.%Y}. 4 şartlı sinyal, +%{TP:.0f} hedef / -%{SL:.0f} stop / {I.H} gün, "
         f"%{I.KOM} komisyon dahil. Toplam {base['n']} işlem: hedef %{g(base['hedef'])}, stop %{g(base['stop'])}, ortalama {f(base['ort'])}, "
         f"medyan {f(base['med'])}, yarı1 {f(base['y1'])}, yarı2 {f(base['y2'])}.\n"]

    # --- 1) dönemler
    yr = pd.Series(close.index[ii].year.astype(str) + "-" + np.where(close.index[ii].month <= 6, "1.yarı", "2.yarı"))
    L += ["## 1) 6'şar aylık dönemler\n",
          "| Dönem | İşlem | Hedef % | Stop % | Süre % | Ort % | Medyan % | O dönemde piyasa 42g (medyan hisse) |",
          "|---|---|---|---|---|---|---|---|"]
    for key in sorted(yr.unique()):
        sel = (yr == key).to_numpy()
        if sel.sum() == 0:
            continue
        a = ist(r, cu, cd, ii, sel, y)
        mm = np.nanmean(mk42[ii[sel]])
        L.append("| %s | %d | %s | %s | %s | %s | %s | %s |" % (
            key, a["n"], g(a["hedef"]), g(a["stop"]), g(100 - a["hedef"] - a["stop"]), f(a["ort"]), f(a["med"]), f(mm)))
    L.append("")

    # --- 2) özellikler: 3 dilim
    ranks = []
    det = {}
    for k, v in V.items():
        ok = np.isfinite(v)
        if ok.sum() < 3 * MIN_N // 2:
            continue
        q1, q2 = np.nanpercentile(v[ok], [33.3, 66.7])
        if q1 == q2:
            continue
        sels = [ok & (v <= q1), ok & (v > q1) & (v <= q2), ok & (v > q2)]
        names = ["alt 1/3 (≤ %s)" % g(q1, 2), "orta 1/3", "üst 1/3 (> %s)" % g(q2, 2)]
        st = [ist(r, cu, cd, ii, s, y) for s in sels]
        det[k] = (names, st)
        best = max(range(3), key=lambda i: st[i]["ort"])
        worst = min(range(3), key=lambda i: st[i]["ort"])
        sp = st[best]["ort"] - st[worst]["ort"]
        tut = (st[best]["y1"] > st[worst]["y1"]) and (st[best]["y2"] > st[worst]["y2"]) if not (
            pd.isna(st[best]["y1"]) or pd.isna(st[worst]["y1"]) or pd.isna(st[best]["y2"]) or pd.isna(st[worst]["y2"])) else False
        ranks.append((k, sp, names[best], names[worst], tut))
    ranks.sort(key=lambda t: (-t[4], -t[1]))
    L += ["## 2) Özellikler: en iyi dilim ile en kötü dilim arasındaki fark\n",
          "Tut = iki yarı dönemde de aynı sıralama (en iyi dilim en kötüden iyi) çıktı mı. Fark = ortalama getiri farkı (puan).\n",
          "| Özellik | Fark | En iyi dilim | En kötü dilim | Tut |", "|---|---|---|---|---|"]
    for k, sp, bn, wn, tut in ranks:
        L.append("| %s | %s | %s | %s | %s |" % (k, f(sp), bn, wn, "EVET" if tut else "hayır"))
    L.append("")
    L += ["## 3) En ayıran 8 özelliğin dilimleri\n",
          "| Özellik / dilim | İşlem | Hedef % | Stop % | Ort % | Medyan % | Yarı1 | Yarı2 |", "|---|---|---|---|---|---|---|---|"]
    for k, sp, bn, wn, tut in ranks[:8]:
        names, st = det[k]
        for nm, a in zip(names, st):
            L.append("| %s: %s | %d | %s | %s | %s | %s | %s | %s |" % (
                k, nm, a["n"], g(a["hedef"]), g(a["stop"]), f(a["ort"]), f(a["med"]), f(a["y1"]), f(a["y2"])))
    L.append("")

    # --- 4) ilk yarıda seç, ikinci yarıda sına
    def zincir(egit_yarim, baslik):
        ei = (ii < y) if egit_yarim == 1 else (ii >= y)
        ti_ = ~ei
        rows = ["## %s\n" % baslik,
                "Eşikler ve seçim yalnızca eğitim yarısından; test yarısı hiç kullanılmaz.\n",
                "| Adım | Eklenen filtre | Eğitim işlem | Eğitim hedef/stop % | Eğitim ort % | Test işlem | Test hedef/stop % | Test ort % | Test medyan % |",
                "|---|---|---|---|---|---|---|---|---|"]
        cur = np.ones(len(r), dtype=bool)
        used = []
        b0 = ist(r, cu, cd, ii, ei, y)
        t0 = ist(r, cu, cd, ii, ti_, y)
        rows.append("| 0 | (hiç filtre yok) | %d | %s/%s | %s | %d | %s/%s | %s | %s |" % (
            b0["n"], g(b0["hedef"]), g(b0["stop"]), f(b0["ort"]), t0["n"], g(t0["hedef"]), g(t0["stop"]), f(t0["ort"]), f(t0["med"])))
        for adim in range(1, ADIM + 1):
            en, ens, entx = None, -1e9, ""
            for k, v in V.items():
                if k in used:
                    continue
                trv = v[ei & cur]
                trv = trv[np.isfinite(trv)]
                if len(trv) < 3 * MIN_N // 2:
                    continue
                q1, q2 = np.nanpercentile(trv, [33.3, 66.7])
                for lab, cond in (("alt 1/3 (≤ %s)" % g(q1, 2), v <= q1), ("üst 1/3 (> %s)" % g(q2, 2), v > q2),
                                  ("alt 2/3 (≤ %s)" % g(q2, 2), v <= q2), ("üst 2/3 (> %s)" % g(q1, 2), v > q1)):
                    s = ei & cur & np.isfinite(v) & cond
                    n = int(s.sum())
                    if n < MIN_N:
                        continue
                    x = r[s]
                    sc = x.mean() - x.std(ddof=1) / np.sqrt(n)      # ortalama - 1 standart hata
                    if sc > ens:
                        en, ens, entx, ecnd = k, sc, lab, cond
            if en is None:
                break
            used.append(en)
            cur = cur & np.isfinite(V[en]) & ecnd
            a, b = ist(r, cu, cd, ii, ei & cur, y), ist(r, cu, cd, ii, ti_ & cur, y)
            rows.append("| %d | %s: %s | %d | %s/%s | %s | %d | %s/%s | %s | %s |" % (
                adim, en, entx, a["n"], g(a["hedef"]), g(a["stop"]), f(a["ort"]), b["n"], g(b.get("hedef")), g(b.get("stop")),
                f(b.get("ort")), f(b.get("med"))))
        rows.append("")
        return rows

    L += zincir(1, "4) İlk yarıda seç, ikinci yarıda sına")
    L += zincir(2, "5) Ters yön: ikinci yarıda seç, ilk yarıda sına")
    L += ["Nasıl okunur: bir filtre ancak iki yönde de test ortalaması filtresiz test satırından (adım 0) belirgin iyiyse ve "
          "işlem sayısı 100'ün altına düşmediyse ciddiye alınır. ~20 özellik denendi, tesadüfen iyi görünen çıkması beklenir; "
          "tek yönde tutan filtreye güvenme. Güncel hisse listesi (batmışlar yok) ve kapanış fiyatı varsayımları iyimserdir."]
    return "\n".join(L)


def main():
    import yfinance  # noqa: F401
    symbols = S.read_symbols("bist_symbols.txt")
    print(f"{len(symbols)} sembol, 5 yıllık günlük veri indiriliyor...")
    close, volume = S.fetch_all(symbols, "5y")
    txt = calistir(close, volume)
    print(txt)
    ozet = os.environ.get("GITHUB_STEP_SUMMARY")
    if ozet:
        with open(ozet, "a", encoding="utf-8") as fh:
            fh.write(txt + "\n")


if __name__ == "__main__":
    main()
