"""
dip_rejim_test.py - "Büyük Hareket" sinyali piyasa zayıfken zarar ediyor mu? Piyasa sağlığı şalteri işe yarar mı?

Aynı 4 şartlı sinyal, +%30 hedef / -%15 stop / 42 gün, %0,4 komisyon dahil (dip_ayrim_test.py ile aynı işlemler).
Piyasa sağlığı iki ölçüyle bakılır:
  genişlik = o gün 50 günlük ortalamasının üstünde kapanan hisselerin oranı (%)
  piyasa 20g = tüm hisselerin 20 günlük getirisinin medyanı (%)
dip_ayrim_test.py, dip_buyuk_test.py, dip_dar_test.py, dip_gunluk_test.py, dip_islem_test.py, signals.py aynı klasörde olmalı.
Sitedeki hiçbir dosyaya dokunmaz.
"""
import os

import numpy as np
import pandas as pd

import dip_ayrim_test as A
import dip_buyuk_test as B
import dip_dar_test as D
import dip_gunluk_test as G
import dip_islem_test as I
import signals as S

TP, SL = A.TP, A.SL
MIN_N = 100


def calistir(close, volume):
    T = close.shape[0]
    y = T // 2
    feats, gecerli = D.genis(close, volume)
    B.ekle(close, volume, feats)
    mask = gecerli & feats["hareketli"] & feats["kirilim"] & feats["kuruma"] & feats["yakin"]
    ii, jj = G.sinyaller(mask)
    ii, jj, P = A.yollar2(close, ii, jj)
    r, cu, cd = I.islem(P, TP, SL)
    sma50 = close.rolling(50).mean()
    genis = (close > sma50).astype(float).where(close.notna()).mean(axis=1) * 100
    mk20 = (close / close.shift(20) - 1).median(axis=1) * 100
    mk42 = (close.shift(-I.H) / close - 1).median(axis=1).to_numpy() * 100
    gv = genis.to_numpy()[ii]
    mv = mk20.to_numpy()[ii]
    idx = close.index
    yr = pd.Series(idx[ii].year.astype(str) + "-" + np.where(idx[ii].month <= 6, "1.yarı", "2.yarı")).to_numpy()
    donemler = sorted(set(yr))
    son4 = donemler[-4:]

    f = lambda x, d=1: "—" if (x is None or pd.isna(x)) else f"{x:+.{d}f}"
    g = lambda x, d=0: "—" if (x is None or pd.isna(x)) else f"{x:.{d}f}"
    allsel = np.ones(len(r), dtype=bool)
    base = A.ist(r, cu, cd, ii, allsel, y)

    L = ["# Büyük Hareket: piyasa sağlığı şalteri\n",
         f"{close.shape[1]} hisse, {idx[0]:%d.%m.%Y} - {idx[-1]:%d.%m.%Y}. Toplam {base['n']} işlem, filtresiz: hedef %{g(base['hedef'])}, "
         f"stop %{g(base['stop'])}, ortalama {f(base['ort'])}, medyan {f(base['med'])}, yarı1 {f(base['y1'])}, yarı2 {f(base['y2'])}.\n",
         "## 0) Piyasa bugün nasıl?\n",
         f"Son veri günü {idx[-1]:%d.%m.%Y}: SMA50 üstü hisse oranı %{g(genis.iloc[-1], 1)}, piyasa 20 günlük medyan getiri {f(mk20.iloc[-1])}%. "
         f"Bir ay önce (22 gün): oran %{g(genis.iloc[-23], 1)}, 20g getiri {f(mk20.iloc[-23])}%. "
         f"Tüm dönem ortalaması: oran %{g(genis.mean(), 1)}.\n"]

    # --- 1) dönem özeti
    L += ["## 1) 6'şar aylık dönemler: piyasa sağlığı ve sonuç\n",
          "| Dönem | İşlem | Ort. genişlik % | Piyasa 20g (ort.) | Hedef % | Stop % | Ort % | Medyan % | Piyasa 42g (medyan hisse) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k in donemler:
        s = yr == k
        a = A.ist(r, cu, cd, ii, s, y)
        L.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s |" % (
            k, a["n"], g(np.nanmean(gv[s]), 1), f(np.nanmean(mv[s])), g(a["hedef"]), g(a["stop"]), f(a["ort"]), f(a["med"]),
            f(np.nanmean(mk42[ii[s]]))))
    L.append("")

    # --- 2) genişlik dilimleri
    L += ["## 2) Piyasa genişliğine göre dilimler (SMA50 üstü hisse oranı)\n",
          "| Genişlik | İşlem | Hedef % | Stop % | Ort % | Medyan % | Yarı1 | Yarı2 |", "|---|---|---|---|---|---|---|---|"]
    kenar = [-1, 30, 40, 50, 60, 70, 101]
    for lo, hi in zip(kenar[:-1], kenar[1:]):
        s = np.isfinite(gv) & (gv > lo) & (gv <= hi)
        a = A.ist(r, cu, cd, ii, s, y)
        ad = ("≤ %d" % hi) if lo < 0 else (("> %d" % lo) if hi > 100 else "%d - %d" % (lo, hi))
        L.append("| %s | %d | %s | %s | %s | %s | %s | %s |" % (
            ad, a["n"], g(a["hedef"]), g(a["stop"]), f(a["ort"]), f(a["med"]), f(a["y1"]), f(a["y2"])))
    L.append("")

    # --- 3) şalter eşikleri
    def izgara(ad, v, esikler, birim):
        sat = ["## %s\n" % ad,
               "Geçen = şart sağlanınca alınan işlemler, Elenen = alınmayanlar. Sağdaki 4 sütun son 4 dönemin ortalaması (işlem sayısı).\n",
               "| Eşik | Geçen işlem | Hedef % | Stop % | Ort % | Medyan % | Yarı1 | Yarı2 | Elenen ort % | " + " | ".join(son4) + " |",
               "|---|---|---|---|---|---|---|---|---|" + "---|" * len(son4)]
        for e in esikler:
            ok = np.isfinite(v) & (v >= e)
            a = A.ist(r, cu, cd, ii, ok, y)
            el = A.ist(r, cu, cd, ii, np.isfinite(v) & ~ok, y)
            cols = []
            for k in son4:
                s = ok & (yr == k)
                n = int(s.sum())
                cols.append("%s (%d)" % (f(r[s].mean()) if n else "—", n))
            sat.append("| ≥ %s%s | %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                g(e, 0 if birim == "%" else 1), birim, a["n"], g(a["hedef"]), g(a["stop"]), f(a["ort"]), f(a["med"]), f(a["y1"]),
                f(a["y2"]), f(el["ort"]), " | ".join(cols)))
        sat.append("")
        return sat

    L += izgara("3) Şalter: SMA50 üstü hisse oranı ≥ eşik", gv, [0, 30, 40, 45, 50, 55, 60, 65, 70], "%")
    L += izgara("4) Şalter: piyasa 20 günlük medyan getiri ≥ eşik", mv, [-100, -5, -2, 0, 2, 4, 6], "%")

    # --- 5) tek parametre: bir yarıda seç, diğerinde sına
    def sec(egit_yarim, baslik, v, esikler):
        ei = (ii < y) if egit_yarim == 1 else (ii >= y)
        ti = ~ei
        en, ens = None, -1e9
        for e in esikler:
            s = ei & np.isfinite(v) & (v >= e)
            n = int(s.sum())
            if n < MIN_N:
                continue
            x = r[s]
            sc = x.mean() - x.std(ddof=1) / np.sqrt(n)
            if sc > ens:
                en, ens = e, sc
        if en is None:
            return ["%s: uygun eşik yok." % baslik]
        b0, t0 = A.ist(r, cu, cd, ii, ei, y), A.ist(r, cu, cd, ii, ti, y)
        a = A.ist(r, cu, cd, ii, ei & np.isfinite(v) & (v >= en), y)
        t = A.ist(r, cu, cd, ii, ti & np.isfinite(v) & (v >= en), y)
        return ["| %s | ≥ %s | %d | %s → %s | %d | %s → %s | %s/%s |" % (
            baslik, g(en, 0), a["n"], f(b0["ort"]), f(a["ort"]), t["n"], f(t0["ort"]), f(t["ort"]), g(t["hedef"]), g(t["stop"]))]

    L += ["## 5) Eşiği bir yarıda seç, diğer yarıda sına\n",
          "Eşik eğitim yarısında (ortalama − 1 standart hata en yüksek, en az 100 işlem) seçilir; test yarısı hiç kullanılmaz. "
          "Oklar: filtresiz → filtreli ortalama.\n",
          "| Şalter / yön | Seçilen eşik | Eğitim işlem | Eğitim ort % | Test işlem | Test ort % | Test hedef/stop % |",
          "|---|---|---|---|---|---|---|"]
    gz = [30, 40, 45, 50, 55, 60, 65, 70]
    mz = [-5, -2, 0, 2, 4, 6]
    L += sec(1, "Genişlik: ilk yarıda seç, ikincide sına", gv, gz)
    L += sec(2, "Genişlik: ikinci yarıda seç, ilkinde sına", gv, gz)
    L += sec(1, "Piyasa 20g: ilk yarıda seç, ikincide sına", mv, mz)
    L += sec(2, "Piyasa 20g: ikinci yarıda seç, ilkinde sına", mv, mz)
    L.append("")
    L += ["Nasıl okunur: şalter ancak (a) 3. ve 4. tabloda eşik yükseldikçe ortalama düzenli artıyorsa, (b) iki yarıda da filtreli ortalama "
          "filtresizden iyiyse, (c) 5. bölümde seçilen eşik test yarısında da kazandırıyorsa, (d) geçen işlem sayısı çok düşmediyse ciddiye alınır. "
          "Son dönemlerde az işlem olduğundan oradaki sayılar gürültülüdür. Güncel hisse listesi (batmışlar yok) ve kapanış fiyatı varsayımları iyimserdir."]
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
