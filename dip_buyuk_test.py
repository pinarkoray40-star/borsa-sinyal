"""
dip_buyuk_test.py - "büyük yükselişi (%30+) önceden yakalayan şart hangisi?" testi

dip_gunluk_test.py, dip_dar_test.py ve signals.py aynı klasörde olmalı. Sitedeki hiçbir dosyaya dokunmaz.
Hedef: sinyalden sonraki 42 işlem günü içinde kapanış en az bir kez +%30 (ve +%20) görüldü mü?
Oran, her günkü sıradan (likit) hisse oranıyla kıyaslanır (kat = lift).
Şartlar dönemin İLK yarısında seçilir, İKİNCİ yarıda sınanır (tesadüfü elemek için).
"""
import os

import numpy as np
import pandas as pd

import dip_dar_test as D
import dip_gunluk_test as G
import signals as S

H = 42
HEDEFLER = (30, 20)
MIN_N = 100
ADIM = 4

SECIM = ["dar_10_4", "dar_20_8", "dar_30_10", "guc", "guc60", "yakin", "uzak", "birikim", "hacim", "kirilim",
         "kuruma", "hareketli", "sma50", "donus", "dipte", "potansiyel"]
AD = {"dar_10_4": "dar 10g %4", "dar_20_8": "dar 20g %8", "dar_30_10": "dar 30g %10",
      "guc": "güç 20g", "guc60": "güç 60g", "yakin": "zirveye yakın", "uzak": "dipten uzak",
      "birikim": "birikim (OBV)", "hacim": "hacim artışı", "kirilim": "20g zirve kırılımı",
      "kuruma": "hacim kuruması", "hareketli": "hareketli hisse (60g oynaklık üst %40)",
      "sma50": "50g ort. üstü", "donus": "dönüş", "dipte": "dibe yakın", "potansiyel": "zirveden %20+ aşağıda"}


def ekle(close, volume, feats):
    r = close.pct_change()
    vol60 = r.rolling(60).std()
    feats["hareketli"] = (vol60.rank(axis=1, pct=True) >= 0.6).fillna(False).to_numpy()
    feats["kuruma"] = (volume.rolling(5).mean() / volume.rolling(60).mean() < 0.7).fillna(False).to_numpy()
    feats["kirilim"] = (close >= close.shift(1).rolling(20).max()).fillna(False).to_numpy()
    r60 = close / close.shift(60) - 1
    feats["guc60"] = (r60.sub(r60.median(axis=1), axis=0) > 0).fillna(False).to_numpy()
    feats["sma50"] = (close > close.rolling(50).mean()).fillna(False).to_numpy()


def hedef(close):
    fm = close.iloc[::-1].rolling(H, min_periods=H).max().iloc[::-1].shift(-1)   # t+1..t+H en yüksek kapanış
    mx = (fm / close - 1).to_numpy() * 100
    son = (close.shift(-H) / close - 1).to_numpy() * 100
    return mx, son


def wilson(k, n, z=1.64):
    if n == 0:
        return 0.0
    p = k / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


class Olcer:
    def __init__(self, gecerli, mx, son):
        self.g, self.mx, self.son = gecerli, mx, son
        self.T = mx.shape[0]
        self.yarim = self.T // 2

    def taban(self, lo, hi):
        sub = self.mx[lo:hi][self.g[lo:hi]]
        sub = sub[np.isfinite(sub)]
        return {t: float((sub >= t).mean() * 100) if len(sub) else np.nan for t in HEDEFLER}

    def sinyal(self, mask, lo, hi):
        m = mask.copy()
        m[:lo] = False
        m[hi:] = False
        ii, jj = G.sinyaller(m)
        v = self.mx[ii, jj]
        ok = np.isfinite(v)
        return ii[ok], jj[ok]

    def ozet(self, mask, lo, hi):
        ii, jj = self.sinyal(mask, lo, hi)
        v = self.mx[ii, jj]
        s = self.son[ii, jj]
        r = {"n": int(len(v))}
        for t in HEDEFLER:
            r[f"h{t}"] = float((v >= t).mean() * 100) if len(v) else np.nan
            r[f"k{t}"] = int((v >= t).sum())
        r["ort"] = float(np.nanmean(s)) if len(s) else np.nan
        r["med"] = float(np.nanmedian(s)) if len(s) else np.nan
        return r


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


def calistir(close, volume):
    feats, gecerli = D.genis(close, volume)
    ekle(close, volume, feats)
    mx, son = hedef(close)
    O = Olcer(gecerli, mx, son)
    T, y = O.T, O.yarim
    b_all, b1, b2 = O.taban(0, T), O.taban(0, y), O.taban(y, T)
    f = lambda x, d=1: "—" if pd.isna(x) else f"{x:.{d}f}"
    L = ["# Büyük yükseliş testi (42 işlem günü içinde +%30 / +%20)\n",
         f"{close.shape[1]} hisse, {close.index[0]:%d.%m.%Y} - {close.index[-1]:%d.%m.%Y}. "
         "Hit = sinyalden sonraki 42 işlem günü içinde kapanış en az bir kez hedefe değdi (tepeden satılabildiği varsayımı iyimserdir). "
         "Kat = hit oranı / aynı dönemdeki tüm likit hisse-günlerinin oranı. Ort/Medyan = 42. gün kapanış getirisi.\n",
         "Taban (sıradan hisse-gün): +%30: tümü %" + f(b_all[30]) + ", yarı1 %" + f(b1[30]) + ", yarı2 %" + f(b2[30]) +
         " | +%20: tümü %" + f(b_all[20]) + "\n"]

    # 1) tek tek şartlar
    L += ["## 1) Tek tek şartlar (tüm dönem)\n",
          "| Şart | Sinyal | Hit%30 | Kat | Yarı1 kat | Yarı2 kat | Hit%20 | Ort 42g | Medyan 42g |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k in SECIM:
        m = gecerli & feats[k]
        a, a1, a2 = O.ozet(m, 0, T), O.ozet(m, 0, y), O.ozet(m, y, T)
        L.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s |" % (
            AD[k], a["n"], f(a["h30"]), f(a["h30"] / b_all[30], 2), f(a1["h30"] / b1[30], 2), f(a2["h30"] / b2[30], 2),
            f(a["h20"]), f(a["ort"]), f(a["med"])))
    L.append("")

    # 2) ilk yarıda ardışık seçim, ikinci yarıda sınav
    L += ["## 2) Şartları ilk yarıda seç, ikinci yarıda sına\n",
          "Her adımda ilk yarıda (Wilson alt sınırına göre) en iyi şartı ekler. Test = seçimde hiç kullanılmayan ikinci yarı. "
          "Test katı 1'in altındaysa seçilen şart tesadüftü.\n",
          "| Adım | Eklenen şart | Eğitim sinyal | Eğitim hit%30 | Eğitim kat | Test sinyal | Test hit%30 | Test kat | Test hit%20 | Test ort 42g | Test medyan |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    mask = gecerli.copy()
    secilen = []
    for adim in range(1, ADIM + 1):
        en, enk = None, -1
        for k in SECIM:
            if k in secilen:
                continue
            r = O.ozet(mask & feats[k], 0, y)
            if r["n"] < MIN_N:
                continue
            sk = wilson(r["k30"], r["n"])
            if sk > enk:
                en, enk = k, sk
        if en is None:
            break
        secilen.append(en)
        mask = mask & feats[en]
        tr, te = O.ozet(mask, 0, y), O.ozet(mask, y, T)
        L.append("| %d | %s | %d | %s | %s | %d | %s | %s | %s | %s | %s |" % (
            adim, " + ".join(AD[s] for s in secilen), tr["n"], f(tr["h30"]), f(tr["h30"] / b1[30], 2),
            te["n"], f(te["h30"]), f(te["h30"] / b2[30], 2), f(te["h20"]), f(te["ort"]), f(te["med"])))
    L.append("")

    # 3) ters yön: ikinci yarıda seç, ilkinde sına
    L += ["## 3) Ters yön: ikinci yarıda seç, ilk yarıda sına\n",
          "| Adım | Eklenen şart | Eğitim sinyal | Eğitim kat | Test sinyal | Test hit%30 | Test kat | Test ort 42g | Test medyan |",
          "|---|---|---|---|---|---|---|---|---|"]
    mask = gecerli.copy()
    secilen = []
    for adim in range(1, ADIM + 1):
        en, enk = None, -1
        for k in SECIM:
            if k in secilen:
                continue
            r = O.ozet(mask & feats[k], y, T)
            if r["n"] < MIN_N:
                continue
            sk = wilson(r["k30"], r["n"])
            if sk > enk:
                en, enk = k, sk
        if en is None:
            break
        secilen.append(en)
        mask = mask & feats[en]
        tr, te = O.ozet(mask, y, T), O.ozet(mask, 0, y)
        L.append("| %d | %s | %d | %s | %d | %s | %s | %s | %s |" % (
            adim, " + ".join(AD[s] for s in secilen), tr["n"], f(tr["h30"] / b2[30], 2),
            te["n"], f(te["h30"]), f(te["h30"] / b1[30], 2), f(te["ort"]), f(te["med"])))
    L += ["", "Uyarılar: güncel hisse listesi (batmışlar yok, iyimser), komisyon yok, +%30'a 'değdi' demek satabildin demek değildir, "
          "sinyaller üst üste biner. Hem bölüm 2 hem bölüm 3'te testte kat 1,3'ün üstünde kalan şartlar ciddiye alınır."]
    return "\n".join(L)


if __name__ == "__main__":
    main()
