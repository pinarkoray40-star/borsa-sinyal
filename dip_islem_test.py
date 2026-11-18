"""
dip_islem_test.py - gerçek işlem simülasyonu: hedefte sat, stopta çık, 42 günde çık

dip_buyuk_test.py, dip_dar_test.py, dip_gunluk_test.py ve signals.py aynı klasörde olmalı.
Sitedeki hiçbir dosyaya dokunmaz.
Kural: giriş = sinyal günü kapanışı. Sonraki 42 işlem gününde kapanış önce +HEDEF% olursa o günkü kapanıştan sat,
önce -STOP% olursa o günkü kapanıştan çık, ikisi de olmazsa 42. gün kapanışından çık.
Sadece kapanış fiyatı var (gün içi yok): gün içi stop/hedef daha iyi veya kötü olabilir. Her işlemden %0,4 komisyon düşülür.
"""
import os

import numpy as np
import pandas as pd

import dip_buyuk_test as B
import dip_dar_test as D
import dip_gunluk_test as G
import signals as S

H = 42
KOM = 0.4
KURALLAR = ((30, 15), (30, 10), (20, 10), (40, 15))
VARYANT = {
    "tüm likit hisseler (kıyas)": None,
    "hareketli": ["hareketli"],
    "hareketli + kırılım": ["hareketli", "kirilim"],
    "hareketli + kırılım + hacim kuruması": ["hareketli", "kirilim", "kuruma"],
    "4 şart (+ zirveye yakın)": ["hareketli", "kirilim", "kuruma", "yakin"],
    "4 şart + güç 20g": ["hareketli", "kirilim", "kuruma", "yakin", "guc"],
}


def yollar(close, ii, jj):
    C = close.to_numpy()
    T = C.shape[0]
    ok = ii + H < T
    ii, jj = ii[ok], jj[ok]
    idx = ii[:, None] + np.arange(1, H + 1)[None, :]
    P = (C[idx, jj[:, None]] / C[ii, jj][:, None] - 1) * 100
    good = np.isfinite(P).all(axis=1)
    return ii[good], P[good]


def islem(P, tp, sl):
    up = P >= tp
    dn = P <= -sl
    iu = np.where(up.any(1), up.argmax(1), H + 1)
    idn = np.where(dn.any(1), dn.argmax(1), H + 1)
    ret = P[:, -1].copy()
    rows = np.arange(len(P))
    cu = iu < idn
    cd = idn < iu
    ret[cu] = P[rows[cu], iu[cu]]
    ret[cd] = P[rows[cd], idn[cd]]
    return ret - KOM, cu, cd


def calistir(close, volume):
    feats, gecerli = D.genis(close, volume)
    B.ekle(close, volume, feats)
    T = close.shape[0]
    y = T // 2
    veri = {}
    for ad, keys in VARYANT.items():
        mask = gecerli.copy()
        for k in (keys or []):
            mask &= feats[k]
        ii, jj = G.sinyaller(mask)
        veri[ad] = yollar(close, ii, jj)
    f = lambda x, d=2: "—" if pd.isna(x) else f"{x:+.{d}f}"
    L = ["# İşlem simülasyonu (hedefte sat, stopta çık, 42 günde çık)\n",
         f"{close.shape[1]} hisse, {close.index[0]:%d.%m.%Y} - {close.index[-1]:%d.%m.%Y}. "
         f"Giriş sinyal günü kapanışı, çıkışlar kapanış fiyatıyla, işlem başına %{KOM} komisyon düşüldü. "
         "Ort/Medyan = işlem başına net getiri %. Kazanan = net getirisi artı olan işlem oranı. Hedef/Stop/Süre = işlemin nasıl bittiği. "
         "Fark = ortalama - kıyas satırının ortalaması, t = farkın gürültüye oranı. Yarı1/Yarı2 = dönemin ilk ve ikinci yarısındaki ortalama.\n"]
    for tp, sl in KURALLAR:
        L += [f"## Hedef +%{tp}, stop -%{sl}\n",
              "| Varyant | İşlem | Ort % | Medyan % | Kazanan % | Hedef % | Stop % | Süre % | Fark | t | Yarı1 | Yarı2 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        kiyas = None
        for ad in VARYANT:
            ti, P = veri[ad]
            r, cu, cd = islem(P, tp, sl)
            n = len(r)
            ort = float(r.mean()) if n else np.nan
            if kiyas is None:
                kiyas = ort
            fark = ort - kiyas
            tv = fark / (r.std(ddof=1) / np.sqrt(n)) if n > 2 and r.std() > 0 else np.nan
            a, b = r[ti < y], r[ti >= y]
            L.append("| %s | %d | %s | %s | %.0f | %.0f | %.0f | %.0f | %s | %s | %s | %s |" % (
                ad + ("" if n >= B.MIN_N or ad.startswith("tüm") else " ⚠"), n, f(ort),
                f(float(np.median(r))) if n else "—", (r > 0).mean() * 100 if n else 0,
                cu.mean() * 100 if n else 0, cd.mean() * 100 if n else 0, (~cu & ~cd).mean() * 100 if n else 0,
                "—" if ad.startswith("tüm") else f(fark), "—" if ad.startswith("tüm") else f(tv, 1),
                f(float(a.mean())) if len(a) >= 10 else "—", f(float(b.mean())) if len(b) >= 10 else "—"))
        L.append("")
    L += ["Nasıl okunur: kıyas satırı 'rastgele likit hisseyi al, aynı kuralla sat' sonucudur. Varyantın hem ortalaması hem MEDYANI "
          "kıyasın üstündeyse, t en az 3 ve Yarı1/Yarı2 ikisi de artıdaysa değer var demektir. ⚠ = 100'den az işlem.",
          "Uyarılar: güncel hisse listesi (batmış hisseler yok, iyimser); gün içi fiyat yok, stop gerçekte kapanıştan kötü doldurulabilir "
          "(gap); sinyaller üst üste binip aynı güne yığılabilir; çok şart denendi."]
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
