"""
dayanikli_test.py - "Kötü piyasada bile giden, iyi piyasada uçan" hisse bulunabiliyor mu?

İki ölçü (son 120 işlem günü, XU100'e göre günlük fazla getiri):
  dn  = endeksin DÜŞTÜĞÜ günlerde hissenin ortalama fazla getirisi   (yüksek = düşüşte dayanıklı)
  up  = endeksin YÜKSELDİĞİ günlerde hissenin ortalama fazla getirisi (yüksek = yükselişte uçan)
Her ikisi de gün bazında tüm hisseler arasında yüzdelik sıraya çevrilir.

Sinyal varyantları (aynı gün kapanışında, işlem hacmi en düşük %30 hariç, aynı hissede 21 gün bekleme):
  A) dayanıklı        : dn sırası >= %80
  B) uçan             : up sırası >= %80
  C) çift güç         : dn ve up sırası ikisi de >= %70
  D) çift güç + zirveye yakın (120 gün zirvenin %85'i üstünde)
  E) Büyük Hareket    : sitedeki mevcut sinyal (buyuk.py) = referans
  F) Büyük Hareket + çift güç
  G) Büyük Hareket + dayanıklı

İşlem kuralı sitedekiyle aynı: giriş = sinyal günü kapanışı, izleyen stop -%25, 84 işlem günü süre, komisyon dahil.
Kıyas: aynı gün, aynı kuralla 30 rastgele likit hisse. Fark = sinyal - kıyas.
Asıl soru: ileride endeks DÜŞERKEN (84 günde <= 0) ve YÜKSELİRKEN (> 0) sinyal ne yapıyor, endeks ve kıyas ne yapıyor?
Sitedeki hiçbir dosyaya dokunmaz. dip_cikis_test.py, dip_kiyas_test.py, buyuk.py, signals.py aynı klasörde olmalı.
Yatırım tavsiyesi değildir.
"""
import os

import numpy as np
import pandas as pd

import buyuk as BY
import dip_cikis_test as K
import dip_kiyas_test as Q
import signals as S

INDEX = "XU100.IS"
W = 120          # dn/up ölçüm penceresi
HH = 84          # bekleme
STOP = 25.0      # izleyen stop %
COOL = 21
MAXN = 5000      # kıyas için en fazla işlem (rastgele örneklenir)


def ozellikler(close, volume, idx):
    r = close.pct_change(fill_method=None)
    ri = idx.pct_change(fill_method=None)
    e = r.sub(ri, axis=0)
    dn_gun = (ri < 0)
    up_gun = (ri > 0)
    dn = e.where(dn_gun, np.nan).rolling(W, min_periods=30).mean()
    up = e.where(up_gun, np.nan).rolling(W, min_periods=30).mean()
    dn_r = dn.rank(axis=1, pct=True)
    up_r = up.rank(axis=1, pct=True)
    ciro = (close * volume).rolling(20).mean()
    likit = ciro.rank(axis=1, pct=True) >= 0.30
    gecerli = (likit & close.notna() & close.shift(W).notna()).fillna(False)
    yakin120 = close >= 0.85 * close.rolling(120).max()

    A = (dn_r >= 0.80)
    B = (up_r >= 0.80)
    C = (dn_r >= 0.70) & (up_r >= 0.70)
    D = C & yakin120
    E = pd.DataFrame(BY.ozellikler(close, volume), index=close.index, columns=close.columns)
    F = E & C
    G = E & A
    V = {
        "A dayanıklı": A, "B uçan": B, "C çift güç": C, "D çift+zirve": D,
        "E Büyük Hareket (ref)": E, "F BH + çift": F, "G BH + dayanıklı": G,
    }
    V = {k: (v & gecerli).fillna(False).to_numpy() for k, v in V.items()}
    return V, gecerli.to_numpy(), dn, up, dn_r, up_r


def sinyal_idx(mask):
    out_i, out_j = [], []
    for j in range(mask.shape[1]):
        last = -10 ** 9
        for i in np.flatnonzero(mask[:, j]):
            if i - last >= COOL:
                out_i.append(int(i))
                out_j.append(j)
                last = i
    return np.array(out_i, dtype=int), np.array(out_j, dtype=int)


def kural(P):
    return K.izleyen(P, STOP)[0]


def calistir(close, volume, idx):
    T = close.shape[0]
    V, gec_np, dn, up, dn_r, up_r = ozellikler(close, volume, idx)
    ix = idx.to_numpy()
    ixf = np.full(T, np.nan)
    ixf[:T - HH] = (ix[HH:] / ix[:T - HH] - 1) * 100     # ileri 84 gün endeks getirisi
    rng = np.random.default_rng(2026)
    f = lambda x, d=1: "—" if (x is None or pd.isna(x)) else f"{x:+.{d}f}"
    g = lambda x, d=0: "—" if (x is None or pd.isna(x)) else f"{x:.{d}f}"
    y = T // 2

    L = ["# Dayanıklı güç testi: kötü piyasada giden, iyi piyasada uçan hisse var mı?\n",
         f"{close.shape[1]} hisse, {close.index[0]:%d.%m.%Y} - {close.index[-1]:%d.%m.%Y}. Kural: izleyen stop -%{STOP:.0f}, {HH} gün, komisyon %{K.KOM} dahil. "
         f"Kıyas = aynı gün aynı kuralla {Q.NK} rastgele likit hisse. Fark = sinyal - kıyas. Endeks = o işlemlerin {HH} gün sonraki XU100 getirisi (ortalama). "
         "Kötü piyasa = o 84 günde endeks <= 0, iyi piyasa = > 0.\n"]
    sonuc = {}
    for ad, mask in V.items():
        ii, jj = sinyal_idx(mask)
        if len(ii) == 0:
            continue
        ii, jj, P = K.yollar(close, ii, jj, HH)
        if len(ii) > MAXN:
            s = np.sort(rng.choice(len(ii), MAXN, replace=False))
            ii, jj, P = ii[s], jj[s], P[s]
        n = len(ii)
        if n < 40:
            sonuc[ad] = None
            continue
        Bp = Q.kiyas_yollari(close, gec_np, ii, HH, rng)
        flat = Bp.reshape(n * Q.NK, HH)
        ok = np.isfinite(flat).all(axis=1)
        rb = kural(np.where(ok[:, None], flat, 0.0)).reshape(n, Q.NK)
        okm = ok.reshape(n, Q.NK)
        kiyas = np.where(okm.sum(1) > 0, (rb * okm).sum(1) / np.maximum(okm.sum(1), 1), np.nan)
        r = kural(P)
        gecer = np.isfinite(kiyas) & np.isfinite(ixf[ii])
        sonuc[ad] = dict(ii=ii[gecer], jj=jj[gecer], r=r[gecer], k=kiyas[gecer], x=ixf[ii][gecer])

    def sat(d, m):
        if m.sum() < 15:
            return None
        fark = d["r"][m] - d["k"][m]
        t = fark.mean() / (fark.std(ddof=1) / np.sqrt(len(fark))) if len(fark) > 5 else np.nan
        return m.sum(), d["r"][m].mean(), d["k"][m].mean(), fark.mean(), t, d["x"][m].mean()

    def tablo(baslik, secici, not_=""):
        out = [f"## {baslik}\n", "| Sinyal | n | Sinyal % | Kıyas % | Fark % | t | Endeks % |", "|---|---|---|---|---|---|---|"]
        for ad, d in sonuc.items():
            if d is None:
                out.append(f"| {ad} | az | | | | | |")
                continue
            s = sat(d, secici(d))
            out.append(f"| {ad} | " + (" | ".join([str(s[0]), f(s[1]), f(s[2]), f(s[3]), g(s[4], 1), f(s[5])]) if s else "az | | | | |") + " |")
        out.append("")
        if not_:
            out.append(not_ + "\n")
        return out

    L += tablo("1) Hepsi birlikte", lambda d: np.ones(len(d["r"]), bool))
    L += tablo("2) KÖTÜ piyasada (ileri 84g endeks <= 0)", lambda d: d["x"] <= 0,
               "Burada Sinyal % > Endeks % ve Fark > 0 ise: endeks düşerken/yatay iken bu hisseler ayakta kalıyor demektir.")
    L += tablo("3) İYİ piyasada (ileri 84g endeks > 0)", lambda d: d["x"] > 0,
               "Burada Sinyal % >> Endeks % ise: yükselişte endeksten fazla koşuyor demektir.")
    L += tablo("4) Güçlü yükseliş piyasası (ileri 84g endeks > +%15)", lambda d: d["x"] > 15)
    L += tablo("5) Güçlü düşüş piyasası (ileri 84g endeks < -%5)", lambda d: d["x"] < -5)

    L += ["## 6) İki yarı (fark %)\n", "| Sinyal | Yarı1 fark | Yarı2 fark | Yarı1 kötü pzr fark | Yarı2 kötü pzr fark |", "|---|---|---|---|---|"]
    for ad, d in sonuc.items():
        if d is None:
            continue
        e = d["ii"] < y
        cells = []
        for m in (e, ~e, e & (d["x"] <= 0), ~e & (d["x"] <= 0)):
            s = sat(d, m)
            cells.append(f(s[3]) + " (n=%d)" % s[0] if s else "—")
        L.append(f"| {ad} | " + " | ".join(cells) + " |")
    L.append("")

    # --- bugün şartı tutanlar
    L += ["## 7) Bugün şartı tutanlar (son kapanış: %s)\n" % close.index[-1].strftime("%d.%m.%Y")]
    son = T - 1
    sma50 = idx.rolling(50).mean().to_numpy()
    L.append("Endeks son kapanış SMA50'nin %s.\n" % ("ÜSTÜNDE" if ix[son] > sma50[son] else "ALTINDA"))
    cols = list(close.columns)
    sat_ = [("C çift güç", V["C çift güç"][son]), ("A dayanıklı", V["A dayanıklı"][son]), ("F BH + çift", V["F BH + çift"][son])]
    for ad, m in sat_:
        js = np.flatnonzero(m)
        if len(js) == 0:
            L.append(f"**{ad}**: bugün tutan yok.\n")
            continue
        js = js[np.argsort(-(dn_r.to_numpy()[son, js] + up_r.to_numpy()[son, js]))][:20]
        L.append(f"**{ad}** ({len(np.flatnonzero(m))} hisse, ilk {len(js)}):\n")
        L += ["| Hisse | Düşüşte fazla %/gün | Yükselişte fazla %/gün | dn sıra | up sıra |", "|---|---|---|---|---|"]
        for j in js:
            L.append("| %s | %s | %s | %s | %s |" % (cols[j].replace(".IS", ""), f(dn.iloc[son, j] * 100, 2), f(up.iloc[son, j] * 100, 2),
                                                     g(dn_r.iloc[son, j] * 100), g(up_r.iloc[son, j] * 100)))
        L.append("")

    L += ["Nasıl okunur: bir varyant ancak (a) 2) KÖTÜ piyasa tablosunda Sinyal % hem Endeks %'in hem Kıyas %'in belirgin üstündeyse (Fark > 0, t > 2), "
          "(b) 3) İYİ piyasada da kıyasın altına düşmüyorsa, (c) 6) iki yarıda da farkı pozitifse ciddiye alınır. "
          "Kötü piyasa ileriye bakılarak ayrıldığından o anda bilinemez; test sadece 'bu hisseler böyle davranıyor mu' sorusunu yanıtlar. "
          "İşlemler yakın tarihli olduğundan t iyimserdir. Güncel hisse listesi (batmışlar yok) getiriyi iyimser gösterir, kıyası da aynı yönde etkiler. "
          "Yatırım tavsiyesi değildir."]
    return "\n".join(L)


def main():
    import yfinance  # noqa: F401
    symbols = S.read_symbols("bist_symbols.txt")
    print(f"{len(symbols)} sembol + endeks, 5 yıllık günlük veri indiriliyor...")
    close, volume = S.fetch_all(symbols + [INDEX], "5y")
    if INDEX not in close.columns:
        raise SystemExit("Endeks (XU100.IS) verisi alınamadı.")
    idx = close[INDEX].ffill(limit=3)
    close = close.drop(columns=[INDEX])
    volume = volume.drop(columns=[INDEX])
    txt = calistir(close, volume, idx)
    print(txt)
    ozet = os.environ.get("GITHUB_STEP_SUMMARY")
    if ozet:
        with open(ozet, "a", encoding="utf-8") as fh:
            fh.write(txt + "\n")


if __name__ == "__main__":
    main()
