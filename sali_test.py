"""
sali_test.py - Piyasa şalteri (genişlik) şimdiki sinyal kurallarında işe yarıyor mu?
Hiçbir dosyayı değiştirmez, siteye dokunmaz; sadece rapor yazar (Actions logu + sali_rapor.txt).

Şalter: genişlik = o gün 50 günlük ortalamasının üstünde kapanan hisselerin oranı (%).
  "Şalter eşik E": yalnızca genişlik >= E iken alınan sinyaller sayılır.
Sinyaller, sitedekiyle AYNI kurallarla 5 yıllık veride yeniden üretilir:
  BH   : Büyük Hareket,  izleyen stop %25, 84 gün
  BH2  : Büyük Hareket 2 (BH + dayanıklılık sırası >= %70), izleyen stop %25, 84 gün
  AYK  : Ayakta (endeks 60g <= -%3, hisse 60g > +%10), 84 gün, stop yok
Kıyas: aynı giriş gününde likit hisselerin, işlemin çıkış gününe kadar ortalama getirisi (fark = işlem - kıyas, puan).
t = aylık ortalama farkların t değeri (aynı aydaki işlemler tek gözlem sayılır).
Her tabloda "elenen" sütunu, şalterin ELEDİĞİ işlemlerin farkıdır: şalter işe yarıyorsa eleneni kötü, kalanı iyi olmalı.
"""
import numpy as np
import pandas as pd

import ayakta as AY
import buyuk as BY
import buyuk2 as BY2
import signals as S

LINES = []
ESIKLER = [0, 30, 40, 45, 50, 55, 60, 65, 70]
HOLD = 84


def out(s=""):
    print(s)
    LINES.append(s)


def f(x, nd=1, sign=True):
    if x is None or not np.isfinite(x):
        return "  -  "
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def tay(rows):
    """rows: DataFrame (ay, diff). Aylık ortalamaların t değeri."""
    if len(rows) < 3:
        return float("nan")
    m = rows.groupby("ay")["diff"].mean().to_numpy()
    if len(m) < 3 or m.std(ddof=1) == 0:
        return float("nan")
    return float(m.mean() / (m.std(ddof=1) / np.sqrt(len(m))))


def islemler(close, ii, jj, sl, L, genis):
    """Her sinyal için tamamlanmış işlem: (p, pnl, kıyas, genişlik). Veri sonuna taşanlar atılır."""
    T = close.shape[0]
    C = close.to_numpy()
    res = []
    for p, j in zip(ii, jj):
        if p + HOLD >= T:
            continue
        entry = C[p, j]
        if not np.isfinite(entry) or entry <= 0:
            continue
        path = (C[p + 1:p + HOLD + 1, j] / entry - 1) * 100
        st, k, xr, peak = BY.takip(path, sl, HOLD)
        if xr is None:
            continue
        a, b = C[p], C[p + k]
        ok = L[p] & np.isfinite(a) & np.isfinite(b) & (a > 0)
        if not ok.any():
            continue
        bench = float(np.mean(b[ok] / a[ok] - 1) * 100)
        res.append((p, float(xr), bench, float(genis[p])))
    return res


def tablo(ad, res, dates):
    if not res:
        out(f"-- {ad}: sinyal yok")
        return
    df = pd.DataFrame(res, columns=["p", "pnl", "bench", "genis"])
    df["diff"] = df["pnl"] - df["bench"]
    df["ay"] = [f"{dates[p]:%Y-%m}" for p in df["p"]]
    half = df["p"].median()
    out(f"-- {ad}: toplam {len(df)} işlem, ort {f(df['pnl'].mean())}% kıyas {f(df['bench'].mean())}% fark {f(df['diff'].mean())}")
    out("   eşik | n    | ort    medyan  kazanan | kıyas  fark   t    | yarı1  yarı2 | elenen n / fark")
    for e in ESIKLER:
        k = df[df["genis"] >= e]
        el = df[df["genis"] < e]
        if len(k) < 5:
            out(f"   >={e:<3} | {len(k):<4} | yeterli işlem yok")
            continue
        y1 = k[k["p"] <= half]["diff"].mean()
        y2 = k[k["p"] > half]["diff"].mean()
        out(f"   >={e:<3} | {len(k):<4} | {f(k['pnl'].mean()):>6} {f(k['pnl'].median()):>6} {(k['pnl'] > 0).mean() * 100:>5.0f}%  | "
            f"{f(k['bench'].mean()):>6} {f(k['diff'].mean()):>6} {f(tay(k)):>5} | {f(y1):>6} {f(y2):>6} | "
            f"{len(el)} / {f(el['diff'].mean()) if len(el) else '  -  '}")
    out("   yıllara göre (şalter >=50 ile hepsi):")
    df["yil"] = [f"{dates[p]:%Y}" for p in df["p"]]
    for y, g in df.groupby("yil"):
        a = g[g["genis"] >= 50]
        out(f"     {y}: hepsi n={len(g)} fark {f(g['diff'].mean())} | >=50: n={len(a)} fark {f(a['diff'].mean()) if len(a) else '  -  '}")
    out("")


def run(close, volume, idx):
    dates = close.index
    T = close.shape[0]
    sma50 = close.rolling(50).mean()
    genis = ((close > sma50).astype(float).where(close.notna()).mean(axis=1) * 100).to_numpy()
    ix = AY.hizala_endeks(close, idx)
    ciro = (close * volume).rolling(60, min_periods=30).mean()
    L = (ciro.rank(axis=1, pct=True) >= 0.30).to_numpy()

    out(f"Son mum {dates[-1]:%d.%m.%Y}, evren {close.shape[1]} hisse, dönem {dates[0]:%d.%m.%Y} - {dates[-1]:%d.%m.%Y}")
    out(f"Bugün genişlik (SMA50 üstü hisse oranı): %{genis[-1]:.1f}")
    out("Not: işlemler 84 gün (ya da stop) sonuna kadar tamamlanmış olanlardır; son ~4 ayın sinyalleri dahil değil.")
    out("")

    bh_mask = BY.ozellikler(close, volume)
    ii, jj = BY.sinyaller(bh_mask, 21)
    tablo("BH Büyük Hareket (stop %25, 84g)", islemler(close, ii, jj, 25.0, L, genis), dates)

    dn, dr = BY2.dayaniklilik(close, idx, 120)
    dn_ok = np.nan_to_num(dr.to_numpy(), nan=0.0) >= 0.70
    ii, jj = BY.sinyaller(bh_mask & dn_ok, 21)
    tablo("BH2 Büyük Hareket 2 (stop %25, 84g)", islemler(close, ii, jj, 25.0, L, genis), dates)

    P = AY.PARAM
    m, _, _, _ = AY.maske(close, volume, ix, P)
    ii, jj = BY.sinyaller(m, P["cooldown"])
    tablo("AYK Ayakta (stop yok, 84g)", islemler(close, ii, jj, 100.0, L, genis), dates)

    out("Nasıl okunur: şalter ancak (a) eşik yükseldikçe 'fark' düzenli artıyorsa, (b) 'elenen' farkı kalandan belirgin kötüyse, "
        "(c) iki yarıda da aynı yönde ve (d) kalan işlem sayısı yeterliyse (en az ~50) işe yarıyordur. Kapanış fiyatı, komisyonsuz.")


def main():
    symbols = S.read_symbols("bist_symbols.txt")
    print(f"{len(symbols)} sembol indiriliyor (5y)...")
    close, volume = S.fetch_all(symbols, "5y")
    idx = BY2.fetch_index(close)
    run(close, volume, idx)
    with open("sali_rapor.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES))


if __name__ == "__main__":
    main()
