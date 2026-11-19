"""
dip_kiyas_test.py - Geniş izleyen stop + uzun bekleme gerçekten SİNYALDEN mi geliyor, yoksa piyasanın genel yükselişinden mi?

Her sinyal işlemi için aynı gün, aynı çıkış kuralıyla 30 rastgele "likit hisse" işlemi simüle edilir (kıyas).
Fark = sinyal işleminin getirisi - o günkü rastgele hisse ortalaması. Fark pozitifse sinyal gerçekten bir şey katıyordur.
Ayrıca: aykırı değer kontrolü (en iyi 3 işlem ve en iyi %5 çıkarılınca), dönem dönem sonuç, en kötü işlem ve ardışık zarar serisi.
dip_cikis_test.py, dip_ayrim_test.py, dip_buyuk_test.py, dip_dar_test.py, dip_gunluk_test.py, dip_islem_test.py, signals.py aynı klasörde olmalı.
Sitedeki hiçbir dosyaya dokunmaz.
"""
import os

import numpy as np
import pandas as pd

import dip_buyuk_test as B
import dip_cikis_test as K
import dip_dar_test as D
import dip_gunluk_test as G
import signals as S

SURELER = [63, 84, 126]
IZLEYEN = [15.0, 20.0, 25.0, 30.0]
NK = 30                      # her işlem için rastgele kıyas hisse sayısı


def kurallar():
    kl = [("mevcut: +30 hedef/-15 stop, 42 gün", 42, lambda P: K.sabit(P[:, :42], 30.0, 15.0)),
          ("hedefsiz, -15 stop", None, lambda P: K.sabit(P, np.inf, 15.0))]
    for t in IZLEYEN:
        kl.append(("izleyen -%d" % t, None, (lambda tt: (lambda P: K.izleyen(P, tt)))(t)))
    return kl


def kiyas_yollari(close, gecerli_np, ii, hh, rng):
    """Her işlem için NK rastgele likit hissenin hh günlük getiri yolu. Dönüş: (n*NK, hh), her işlem NK satır."""
    C = close.to_numpy()
    out = np.full((len(ii), NK, hh), np.nan)
    for a, i in enumerate(ii):
        cand = np.flatnonzero(gecerli_np[i] & np.isfinite(C[i]))
        if len(cand) < 5:
            continue
        sec = rng.choice(cand, size=min(NK * 3, len(cand)), replace=False)
        yol = (C[i + 1:i + hh + 1][:, sec] / C[i, sec][None, :] - 1) * 100
        ok = np.isfinite(yol).all(axis=0)
        yol = yol[:, ok].T[:NK]
        out[a, :len(yol)] = yol
    return out


def calistir(close, volume):
    T = close.shape[0]
    y = T // 2
    feats, gecerli = D.genis(close, volume)
    B.ekle(close, volume, feats)
    mask = gecerli & feats["hareketli"] & feats["kirilim"] & feats["kuruma"] & feats["yakin"]
    gec_np = np.asarray(gecerli, dtype=bool)
    ii0, jj0 = G.sinyaller(mask)
    sma50 = close.rolling(50).mean()
    genis = (close > sma50).astype(float).where(close.notna()).mean(axis=1).to_numpy() * 100
    rng = np.random.default_rng(2026)

    f = lambda x, d=1: "—" if (x is None or pd.isna(x)) else f"{x:+.{d}f}"
    g = lambda x, d=0: "—" if (x is None or pd.isna(x)) else f"{x:.{d}f}"
    L = ["# Çıkış kuralı kıyas testi: sinyal mi, piyasa mı?\n",
         f"{close.shape[1]} hisse, {close.index[0]:%d.%m.%Y} - {close.index[-1]:%d.%m.%Y}. Her işlem için aynı gün {NK} rastgele likit hisse, aynı çıkış kuralı, "
         f"%{K.KOM} komisyon dahil. Fark = sinyal getirisi - kıyas ortalaması. t = farkın ortalaması / standart hatası (işlemler yakın tarihli olduğundan iyimser). "
         "Top3 çıkar = en iyi 3 işlem atılınca ortalama. %5 kırp = en iyi ve en kötü %5 atılınca ortalama.\n"]
    ozel = {}

    for hh in SURELER:
        ii, jj, P = K.yollar(close, ii0, jj0, hh)
        n = len(ii)
        if n < 50:
            continue
        Bp = kiyas_yollari(close, gec_np, ii, hh, rng)
        saglikli = np.isfinite(genis[ii]) & (genis[ii] >= 50)
        idx = pd.DatetimeIndex(close.index[ii])
        donem = pd.Series(idx.year.astype(str) + "-" + np.where(idx.month <= 6, "1.yarı", "2.yarı")).to_numpy()
        L += ["## Bekleme %d gün (%d işlem; girişten sonra en az %d işlem günü verisi olanlar)\n" % (hh, n, hh),
              "| Kural | Ort % | Medyan % | Kıyas ort % | Fark % | t | Yarı1 fark | Yarı2 fark | Fark>0 işlem % | Sağl. fark | Top3 çıkar | %5 kırp | Ort. gün |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for ad, sabit_h, fn in kurallar():
            r, cu, cd, gun = fn(P)
            # kıyas: her işlem için NK yol; satır sayısı n*NK
            flat = Bp.reshape(n * NK, hh)
            ok = np.isfinite(flat).all(axis=1)
            fl = np.where(ok[:, None], flat, 0.0)
            rb = fn(fl)[0].reshape(n, NK)
            okm = ok.reshape(n, NK)
            kiyas = np.where(okm.sum(axis=1) > 0, (rb * okm).sum(axis=1) / np.maximum(okm.sum(axis=1), 1), np.nan)
            gecer = np.isfinite(kiyas)
            fark = (r - kiyas)[gecer]
            rr = r[gecer]
            ei = ii[gecer] < y
            t = fark.mean() / (fark.std(ddof=1) / np.sqrt(len(fark))) if len(fark) > 5 else np.nan
            srt = np.sort(rr)
            k5 = max(int(len(srt) * 0.05), 1)
            sag = saglikli[gecer]
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                ad, f(rr.mean()), f(np.median(rr)), f(kiyas[gecer].mean()), f(fark.mean()), g(t, 1),
                f(fark[ei].mean()) if ei.sum() > 15 else "—", f(fark[~ei].mean()) if (~ei).sum() > 15 else "—",
                g((fark > 0).mean() * 100), f(fark[sag].mean()) if sag.sum() > 15 else "—",
                f(srt[:-3].mean()), f(srt[k5:-k5].mean()), g(gun.mean())))
            if hh == 84:
                ozel[ad] = (r, kiyas, ii, donem, gecer)
        L.append("")

    # --- dönemler (84 gün)
    if ozel:
        L += ["## Dönem dönem (bekleme 84 gün): sinyal ort % / kıyas ort % / işlem\n"]
        ad_list = [a for a in ("mevcut: +30 hedef/-15 stop, 42 gün", "izleyen -20", "izleyen -25", "izleyen -30") if a in ozel]
        L += ["| Dönem | " + " | ".join(ad_list) + " |", "|---|" + "---|" * len(ad_list)]
        donemler = sorted(set(next(iter(ozel.values()))[3]))
        for d in donemler:
            cells = []
            for a in ad_list:
                r, kiyas, ii, donem, gecer = ozel[a]
                s = (donem == d) & gecer
                cells.append("%s / %s / %d" % (f(r[s].mean()), f(kiyas[s].mean()), s.sum()) if s.sum() else "—")
            L.append("| %s | %s |" % (d, " | ".join(cells)))
        L.append("")

        # --- risk
        L += ["## Risk (bekleme 84 gün)\n",
              "| Kural | Kazanan % | Kazanan ort % | Kaybeden ort % | En kötü işlem % | En iyi işlem % | %5 persentil | En uzun ardışık zarar (işlem) |",
              "|---|---|---|---|---|---|---|---|"]
        for a in ad_list:
            r, kiyas, ii, donem, gecer = ozel[a]
            rr, ti = r[gecer], ii[gecer]
            rr = rr[np.argsort(ti, kind="stable")]
            seri = cur = 0
            for v in rr:
                cur = cur + 1 if v <= 0 else 0
                seri = max(seri, cur)
            w, l = rr[rr > 0], rr[rr <= 0]
            L.append("| %s | %s | %s | %s | %s | %s | %s | %d |" % (
                a, g((rr > 0).mean() * 100), f(w.mean()) if len(w) else "—", f(l.mean()) if len(l) else "—",
                f(rr.min()), f(rr.max()), f(np.percentile(rr, 5)), seri))
        L.append("")

    L += ["Nasıl okunur: izleyen stop kuralı ancak (a) Fark sütunu belirgin pozitifse (ve t yaklaşık 2'nin üzerindeyse), (b) Yarı1 ve Yarı2 farkı ikisinde de pozitifse, "
          "(c) Top3 çıkar ve %5 kırp ortalamaları hâlâ iyiyse, (d) dönem tablosunda zayıf dönemde kıyastan çok geride kalmıyorsa ciddiye alınır. "
          "Kıyas, o günkü likit hisselerden rastgele çekilir; enflasyon ve genel yükseliş her iki tarafta da vardır. "
          "Güncel hisse listesi (batmışlar yok) uzun beklemede getiriyi iyimser gösterir, kıyası da aynı yönde etkiler. "
          "Gerçekte stop gap'le daha kötü dolabilir ve her işlemde alım-satım farkı komisyondan büyük olabilir."]
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
