"""
dip_cikis_test.py - "Büyük Hareket" sinyalinde ÇIKIŞ kuralını değiştirmek (büyük hedef, izleyen stop, daha uzun süre) sonucu iyileştirir mi?

Aynı sinyaller (4 şart), giriş sinyal günü kapanışı, %0,4 komisyon dahil, hepsi kapanış fiyatıyla. Sadece çıkış kuralı değişir:
  A) Sabit stop -%15 ile hedef: 30, 40, 50, 60, 80 ve hedefsiz
  B) İzleyen stop: kapanış, girişten beri görülen en yüksek kapanışın %X altına inince sat (hedef yok)
  C) Bekleme süresi: 42, 63 ve 84 işlem günü
Adil karşılaştırma için hepsi aynı işlem kümesinde ölçülür (girişten sonra en az 84 işlem günü verisi olanlar).
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

KOM = I.KOM
HH = 84
SURELER = [42, 63, 84]
HEDEFLER = [30.0, 40.0, 50.0, 60.0, 80.0, np.inf]
IZLEYEN = [10.0, 15.0, 20.0, 25.0, 30.0]
BIG = 10 ** 6
MIN_N = 100


def yollar(close, ii, jj, hh):
    C = close.to_numpy()
    T = C.shape[0]
    ii, jj = np.array(ii, dtype=int), np.array(jj, dtype=int)
    ok = ii + hh < T
    ii, jj = ii[ok], jj[ok]
    idx = ii[:, None] + np.arange(1, hh + 1)[None, :]
    P = (C[idx, jj[:, None]] / C[ii, jj][:, None] - 1) * 100
    good = np.isfinite(P).all(axis=1)
    return ii[good], jj[good], P[good]


def sabit(P, tp, sl):
    up, dn = P >= tp, P <= -sl
    iu = np.where(up.any(1), up.argmax(1), BIG)
    idn = np.where(dn.any(1), dn.argmax(1), BIG)
    rows = np.arange(len(P))
    ret = P[:, -1].copy()
    cu, cd = iu < idn, idn < iu
    ret[cu] = P[rows[cu], iu[cu]]
    ret[cd] = P[rows[cd], idn[cd]]
    gun = np.where(cu, iu + 1, np.where(cd, idn + 1, P.shape[1]))
    return ret - KOM, cu, cd, gun


def izleyen(P, t):
    R = 1 + P / 100
    Rm = np.maximum.accumulate(np.concatenate([np.ones((len(P), 1)), R], axis=1), axis=1)[:, 1:]
    trig = R <= Rm * (1 - t / 100)
    k = np.where(trig.any(1), trig.argmax(1), BIG)
    rows = np.arange(len(P))
    cd = k < BIG
    ret = P[:, -1].copy()
    ret[cd] = P[rows[cd], k[cd]]
    gun = np.where(cd, k + 1, P.shape[1])
    return ret - KOM, np.zeros(len(P), dtype=bool), cd, gun


def calistir(close, volume):
    T = close.shape[0]
    y = T // 2
    feats, gecerli = D.genis(close, volume)
    B.ekle(close, volume, feats)
    mask = gecerli & feats["hareketli"] & feats["kirilim"] & feats["kuruma"] & feats["yakin"]
    ii, jj = G.sinyaller(mask)
    ii, jj, P84 = yollar(close, ii, jj, HH)
    n = len(ii)
    sma50 = close.rolling(50).mean()
    gv = ((close > sma50).astype(float).where(close.notna()).mean(axis=1).to_numpy() * 100)[ii]
    saglikli = np.isfinite(gv) & (gv >= 50)
    idx = pd.DatetimeIndex(close.index[ii])
    donem = pd.Series(idx.year.astype(str) + "-" + np.where(idx.month <= 6, "1.yarı", "2.yarı")).to_numpy()
    allsel = np.ones(n, dtype=bool)

    f = lambda x, d=1: "—" if (x is None or pd.isna(x)) else f"{x:+.{d}f}"
    g = lambda x, d=0: "—" if (x is None or pd.isna(x)) else f"{x:.{d}f}"

    def satir(ad, h, res, hedefvar=True):
        r, cu, cd, gun = res
        a, b = r[ii < y], r[ii >= y]
        sg = r[saglikli]
        return dict(ad=ad, h=h, r=r, cu=cu, cd=cd, gun=gun,
                    n=len(r), hedef=cu.mean() * 100, stop=cd.mean() * 100, sure=(~cu & ~cd).mean() * 100,
                    ort=r.mean(), med=float(np.median(r)), win=(r > 0).mean() * 100,
                    y1=a.mean() if len(a) >= 15 else np.nan, y2=b.mean() if len(b) >= 15 else np.nan,
                    sag=sg.mean() if len(sg) >= 15 else np.nan, gun_ort=gun.mean(), hedefvar=hedefvar)

    def md(s):
        return "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            s["ad"], s["h"], g(s["hedef"]) if s["hedefvar"] else "—", g(s["stop"]), g(s["sure"]), f(s["ort"]), f(s["med"]),
            g(s["win"]), f(s["y1"]), f(s["y2"]), f(s["sag"]), g(s["gun_ort"]))

    BASL = "| Kural | Bekleme (gün) | Hedef % | Stop % | Süre bitti % | Ort % | Medyan % | Kazanan % | Yarı1 | Yarı2 | Sağl. ort % | Ort. gün |"
    AYR = "|---|---|---|---|---|---|---|---|---|---|---|---|"

    tum = []
    base = satir("sabit +30/-15", 42, sabit(P84[:, :42], 30.0, 15.0))
    L = ["# Büyük Hareket: çıkış kuralı testi (büyük hedef, izleyen stop, uzun bekleme)\n",
         f"{close.shape[1]} hisse, {close.index[0]:%d.%m.%Y} - {close.index[-1]:%d.%m.%Y}. Girişten sonra en az {HH} işlem günü verisi olan {n} işlem (hepsi aynı işlemler). "
         f"%{KOM} komisyon dahil, kapanış fiyatlarıyla. Sağl. = yalnızca piyasa sağlıklıyken (SMA50 üstü oran ≥ %50) ortalama. "
         "Kazanan = net getirisi artı olan işlem oranı. Ort. gün = işlemde geçen ortalama işlem günü.\n",
         "Kıyas noktası (şu anki kural): ortalama %s, medyan %s, yarı1 %s, yarı2 %s.\n" % (f(base["ort"]), f(base["med"]), f(base["y1"]), f(base["y2"]))]

    # A) sabit stop + hedef grid
    L += ["## 1) Sabit stop -%15, hedef büyüyor\n", BASL, AYR]
    for h in SURELER:
        for tp in HEDEFLER:
            s = satir("stop -15 / hedef %s" % ("yok" if np.isinf(tp) else "+%d" % tp), h, sabit(P84[:, :h], tp, 15.0), hedefvar=not np.isinf(tp))
            tum.append(s)
            L.append(md(s))
    L.append("")

    # B) izleyen stop
    L += ["## 2) İzleyen stop (zirve kapanıştan %X düşünce sat, hedef yok)\n",
          "Stop % sütunu = izleyen stopla çıkılan işlem oranı.\n", BASL, AYR]
    for h in SURELER:
        for t in IZLEYEN:
            s = satir("izleyen -%d" % t, h, izleyen(P84[:, :h], t), hedefvar=False)
            tum.append(s)
            L.append(md(s))
    L.append("")

    # 3) bir yarıda seç, diğerinde sına
    def sec(egit):
        ei = (ii < y) if egit == 1 else (ii >= y)
        ti = ~ei
        en, ens = None, -1e9
        for s in tum:
            x = s["r"][ei]
            if len(x) < MIN_N:
                continue
            sc = x.mean() - x.std(ddof=1) / np.sqrt(len(x))
            if sc > ens:
                en, ens = s, sc
        b = base["r"]
        return ("| %s | %s (%d gün) | %s → %s | %s → %s | %s |" % (
            "İlk yarıda seç, ikincide sına" if egit == 1 else "İkinci yarıda seç, ilkinde sına", en["ad"], en["h"],
            f(b[ei].mean()), f(en["r"][ei].mean()), f(b[ti].mean()), f(en["r"][ti].mean()), f(np.median(en["r"][ti]))))

    L += ["## 3) Kuralı bir yarıda seç, diğer yarıda sına\n",
          f"{len(tum)} kural arasından eğitim yarısında (ortalama − 1 standart hata en yüksek) seçilir. Oklar: şu anki kuralın ortalaması → seçilen kuralın ortalaması.\n",
          "| Yön | Seçilen kural | Eğitim yarısı | Test yarısı | Test medyan % |", "|---|---|---|---|---|", sec(1), sec(2), ""]

    # 4) dönemler
    en_iyi = max(tum, key=lambda s: s["ort"])
    en_iyi_iz = max([s for s in tum if s["ad"].startswith("izleyen")], key=lambda s: s["ort"])
    donemler = sorted(set(donem))[-4:]
    L += ["## 4) Son dönemler: şu anki kural ve en yüksek ortalamalı kurallar (tüm veride seçildi, iyimser)\n",
          "| Dönem | İşlem | Şu anki kural ort %% | %s (%d gün) | %s (%d gün) |" % (en_iyi["ad"], en_iyi["h"], en_iyi_iz["ad"], en_iyi_iz["h"]),
          "|---|---|---|---|---|"]
    for d in donemler:
        s = donem == d
        L.append("| %s | %d | %s | %s | %s |" % (d, s.sum(), f(base["r"][s].mean()), f(en_iyi["r"][s].mean()), f(en_iyi_iz["r"][s].mean())))
    L += ["",
          "Nasıl okunur: yeni kural ancak (a) şu anki kuraldan belirgin iyiyse (ortalama ve medyan), (b) Yarı1 ve Yarı2'nin ikisinde de iyiyse, "
          "(c) 3. bölümde bir yarıda seçilen kural diğer yarıda da kazandırıyorsa ciddiye alınır. Uzun bekleme sermayeyi daha uzun bağlar ve "
          "ortalama gün sütununa bakmadan karşılaştırma yapma. Çok kural denendiği için tesadüfen iyi görünen çıkması beklenir. "
          "Güncel hisse listesi (batmışlar yok), kapanış fiyatları ve gap'siz stop varsayımları iyimserdir."]
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
