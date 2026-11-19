"""
dip_oynak_test.py - "Büyük Hareket" sinyalinde stop ve hedefi hissenin OYNAKLIĞINA göre ayarlamak sonucu değiştirir mi?

Sabit kural: +%30 hedef / -%15 stop. Oynaklığa göre kural: stop = sl_kat x (hissenin 60 günlük günlük oynaklığı),
hedef = tp_kat x aynı oynaklık. Oynak hissede stop/hedef geniş, sakin hissede dar olur.
Aynı sinyaller, aynı 42 gün, %0,4 komisyon dahil; sadece çıkış kuralı değişir.
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

SL_MIN, SL_MAX = 8.0, 30.0       # stop genişliği sınırı (%)
TP_MIN, TP_MAX = 15.0, 70.0      # hedef sınırı (%)
SL_KAT = [2.5, 3.0, 3.75, 5.0, 6.0]
TP_KAT = [5.0, 7.5, 10.0]
MIN_N = 100


def kural(P, sig, sl_kat, tp_kat):
    sl = np.clip(sl_kat * sig, SL_MIN, SL_MAX)
    tp = np.clip(tp_kat * sig, TP_MIN, TP_MAX)
    r, cu, cd = I.islem(P, tp[:, None], sl[:, None])
    return r, cu, cd, sl, tp


def calistir(close, volume):
    T = close.shape[0]
    y = T // 2
    feats, gecerli = D.genis(close, volume)
    B.ekle(close, volume, feats)
    mask = gecerli & feats["hareketli"] & feats["kirilim"] & feats["kuruma"] & feats["yakin"]
    ii, jj = G.sinyaller(mask)
    ii, jj, P = A.yollar2(close, ii, jj)
    sig = (close.pct_change().rolling(60).std() * 100).to_numpy()[ii, jj]
    ok = np.isfinite(sig)
    ii, jj, P, sig = ii[ok], jj[ok], P[ok], sig[ok]
    n = len(ii)
    sma50 = close.rolling(50).mean()
    gv = ((close > sma50).astype(float).where(close.notna()).mean(axis=1).to_numpy() * 100)[ii]
    saglikli = np.isfinite(gv) & (gv >= 50)
    r0, cu0, cd0 = I.islem(P, A.TP, A.SL)

    f = lambda x, d=1: "—" if (x is None or pd.isna(x)) else f"{x:+.{d}f}"
    g = lambda x, d=0: "—" if (x is None or pd.isna(x)) else f"{x:.{d}f}"

    def ozet(r, cu, cd, sel):
        k = int(sel.sum())
        if k == 0:
            return dict(n=0, hedef=np.nan, stop=np.nan, ort=np.nan, med=np.nan, y1=np.nan, y2=np.nan)
        a, b = r[sel & (ii < y)], r[sel & (ii >= y)]
        return dict(n=k, hedef=cu[sel].mean() * 100, stop=cd[sel].mean() * 100, ort=float(r[sel].mean()),
                    med=float(np.median(r[sel])), y1=float(a.mean()) if len(a) >= 15 else np.nan,
                    y2=float(b.mean()) if len(b) >= 15 else np.nan)

    allsel = np.ones(n, dtype=bool)
    b0 = ozet(r0, cu0, cd0, allsel)
    L = ["# Büyük Hareket: oynaklığa göre stop ve hedef\n",
         f"{close.shape[1]} hisse, {close.index[0]:%d.%m.%Y} - {close.index[-1]:%d.%m.%Y}. {n} işlem. Sabit kural (+%{A.TP:.0f}/-%{A.SL:.0f}): "
         f"hedef %{g(b0['hedef'])}, stop %{g(b0['stop'])}, ortalama {f(b0['ort'])}, medyan {f(b0['med'])}, yarı1 {f(b0['y1'])}, yarı2 {f(b0['y2'])}. "
         f"İşlem başına %{I.KOM} komisyon dahil. Oynaklık = 60 günlük günlük getiri std (medyan %{g(np.median(sig), 2)}).\n"]

    # --- 1) sabit kural, oynaklık dilimlerine göre
    q1, q2 = np.percentile(sig, [33.3, 66.7])
    L += ["## 1) Sabit kural (+30/-15) oynaklık dilimlerine göre\n",
          "| Oynaklık dilimi | İşlem | Hedef % | Stop % | Süre % | Ort % | Medyan % | Yarı1 | Yarı2 |", "|---|---|---|---|---|---|---|---|---|"]
    for ad, s in (("düşük (≤ %.2f)" % q1, sig <= q1), ("orta", (sig > q1) & (sig <= q2)), ("yüksek (> %.2f)" % q2, sig > q2)):
        a = ozet(r0, cu0, cd0, s)
        L.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s |" % (
            ad, a["n"], g(a["hedef"]), g(a["stop"]), g(100 - a["hedef"] - a["stop"]), f(a["ort"]), f(a["med"]), f(a["y1"]), f(a["y2"])))
    L.append("")

    # --- 2) ızgara
    L += ["## 2) Oynaklığa göre stop/hedef kuralları\n",
          f"Stop = stop katı x oynaklık (en az %{SL_MIN:.0f}, en çok %{SL_MAX:.0f}); hedef = hedef katı x oynaklık (en az %{TP_MIN:.0f}, en çok %{TP_MAX:.0f}). "
          "Ort. stop/hedef = o kuralın işlemlerdeki ortalama genişliği. Sağl. = yalnızca piyasa sağlıklıyken (SMA50 üstü oran ≥ %50) ortalama.\n",
          "| Stop katı | Hedef katı | Ort. stop % | Ort. hedef % | Hedef % | Stop % | Süre % | Ort % | Medyan % | Yarı1 | Yarı2 | Sağl. ort % |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    sa = ozet(r0, cu0, cd0, saglikli)
    L.append("| sabit | sabit | 15 | 30 | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        g(b0["hedef"]), g(b0["stop"]), g(100 - b0["hedef"] - b0["stop"]), f(b0["ort"]), f(b0["med"]), f(b0["y1"]), f(b0["y2"]), f(sa["ort"])))
    sonuc = {}
    for sk in SL_KAT:
        for tk in TP_KAT:
            r, cu, cd, sl, tp = kural(P, sig, sk, tk)
            sonuc[(sk, tk)] = (r, cu, cd)
            a = ozet(r, cu, cd, allsel)
            s2 = ozet(r, cu, cd, saglikli)
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                g(sk, 2), g(tk, 1), g(sl.mean(), 1), g(tp.mean(), 1), g(a["hedef"]), g(a["stop"]), g(100 - a["hedef"] - a["stop"]),
                f(a["ort"]), f(a["med"]), f(a["y1"]), f(a["y2"]), f(s2["ort"])))
    L.append("")

    # --- 3) bir yarıda seç, diğerinde sına
    def sec(egit):
        ei = (ii < y) if egit == 1 else (ii >= y)
        ti = ~ei
        en, ens = None, -1e9
        for key, (r, cu, cd) in sonuc.items():
            x = r[ei]
            if len(x) < MIN_N:
                continue
            sc = x.mean() - x.std(ddof=1) / np.sqrt(len(x))
            if sc > ens:
                en, ens = key, sc
        r, cu, cd = sonuc[en]
        return ("| %s | %s x %s | %s → %s | %s → %s | %s/%s |" % (
            "İlk yarıda seç, ikincide sına" if egit == 1 else "İkinci yarıda seç, ilkinde sına", g(en[0], 2), g(en[1], 1),
            f(r0[ei].mean()), f(r[ei].mean()), f(r0[ti].mean()), f(r[ti].mean()), g(cu[ti].mean() * 100), g(cd[ti].mean() * 100)))

    L += ["## 3) Kuralı bir yarıda seç, diğer yarıda sına\n",
          "Seçim: eğitim yarısında ortalama − 1 standart hata en yüksek olan kural. Oklar: sabit kuralın ortalaması → seçilen kuralın ortalaması.\n",
          "| Yön | Seçilen kural (stop katı x hedef katı) | Eğitim yarısı | Test yarısı | Test hedef/stop % |", "|---|---|---|---|---|",
          sec(1), sec(2), ""]

    # --- 4) son dönemler
    yr = np.where(pd.DatetimeIndex(close.index[ii]).month <= 6, "1.yarı", "2.yarı")
    yr = pd.Series(pd.DatetimeIndex(close.index[ii]).year.astype(str) + "-" + yr).to_numpy()
    donemler = sorted(set(yr))[-4:]
    ek = max(sonuc, key=lambda k: sonuc[k][0].mean())
    L += ["## 4) Son 4 dönem: sabit kural ile en yüksek ortalamalı oynaklık kuralı (%s x %s, tüm veride seçildi, iyimser)\n" % (g(ek[0], 2), g(ek[1], 1)),
          "| Dönem | İşlem | Sabit kural ort % | Oynaklık kuralı ort % |", "|---|---|---|---|"]
    for d in donemler:
        s = yr == d
        L.append("| %s | %d | %s | %s |" % (d, s.sum(), f(r0[s].mean()), f(sonuc[ek][0][s].mean())))
    L += ["",
          "Nasıl okunur: oynaklık kuralı ancak (a) 2. tabloda sabit kuraldan belirgin iyiyse, (b) iki yarıda da (Yarı1, Yarı2) sabitten iyiyse, "
          "(c) 3. bölümde bir yarıda seçilen kural diğer yarıda da kazandırıyorsa ciddiye alınır. 15 kural denendi; "
          "tesadüfen iyi görünen çıkması beklenir. Geniş stop daha büyük tek işlem zararı demektir (Ort. stop sütununa bak). "
          "Güncel hisse listesi ve kapanış fiyatı varsayımları iyimserdir."]
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
