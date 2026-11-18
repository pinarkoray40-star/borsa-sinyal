"""
dip_gunluk_test.py - 1-2 ay tutma için günlük "dipten potansiyel" şart testi

Sitedeki hiçbir dosyaya DOKUNMAZ. Son 5 yıl, tüm BIST, günlük veri.
Her şartı tek başına ve birlikte dener; sinyalden 21 ve 42 işlem günü (yaklaşık 1 ve 2 ay)
sonraki getiriye bakar ve aynı günlerde tüm likit hisselerin ortalamasıyla kıyaslar.

Şartlar:
  birikim     : OBV 40 gün öncesinden yüksek ama fiyat 40 günde %5'ten fazla artmamış (sessiz para girişi)
  dar         : son 15 günde kapanış bandı en çok %6
  hacim       : son 3 günün ortalama hacmi, 20 günlük ortalamanın en az 1,5 katı
  potansiyel  : fiyat 60 günlük zirvenin en az %20 altında (toparlanma payı var)
  dipte       : fiyat 60 günlük dibin en çok %15 üstünde
  dönüş       : kapanış 10 günlük ortalamanın üstünde (düşüş durmuş)
  güç         : son 20 günde piyasadan (medyan hisse) iyi
Her varyantta işlem hacmi en düşük %30 hisse hariçtir. Bir hissede sinyalden sonra 21 gün yeni sinyal sayılmaz.

Sonuç: ekrana, GitHub Actions özet sayfasına ve dip_gunluk_sonuc.csv'ye yazılır.
Gereken: pip install yfinance pandas numpy
"""
import os
import sys

import numpy as np
import pandas as pd

import signals as S

HORIZONS = (21, 42)
COOLDOWN = 21
MIN_N = 30

AD = {"birikim": "birikim", "dar": "dar aralık", "hacim": "hacim artışı", "potansiyel": "zirveden %20+ aşağıda",
      "dipte": "dibe yakın", "donus": "dönüş (10g ort. üstü)", "guc": "piyasadan güçlü"}

VARYANTLAR = {
    "birikim": ["birikim"],
    "dar aralık": ["dar"],
    "hacim artışı": ["hacim"],
    "zirveden %20+ aşağıda": ["potansiyel"],
    "dibe yakın": ["dipte"],
    "dönüş": ["donus"],
    "piyasadan güçlü": ["guc"],
    "birikim + potansiyel": ["birikim", "potansiyel"],
    "birikim + dar": ["birikim", "dar"],
    "birikim + hacim": ["birikim", "hacim"],
    "birikim + dönüş": ["birikim", "donus"],
    "birikim + potansiyel + dönüş": ["birikim", "potansiyel", "donus"],
    "birikim + dar + hacim": ["birikim", "dar", "hacim"],
    "dar + hacim (Hacim takip kuralı)": ["dar", "hacim"],
    "potansiyel + dönüş + hacim": ["potansiyel", "donus", "hacim"],
    "dipte + birikim + dönüş": ["dipte", "birikim", "donus"],
    "birikim + potansiyel + dönüş + güç": ["birikim", "potansiyel", "donus", "guc"],
}


def ozellikler(close, volume):
    """Her şart için (tarih x hisse) True/False tablosu + likit hisse maskesi."""
    sign = np.sign(close.diff()).fillna(0.0)
    obv = (sign * volume.fillna(0.0)).cumsum()
    obv_up = obv > obv.shift(40)
    p40 = close / close.shift(40) - 1
    birikim = obv_up & (p40 < 0.05)

    rng15 = (close.rolling(15).max() - close.rolling(15).min()) / close.rolling(15).min()
    dar = rng15 <= 0.06
    volx = volume.rolling(3).mean() / volume.rolling(20).mean()
    hacim = volx >= 1.5
    hi60 = close.rolling(60).max()
    lo60 = close.rolling(60).min()
    potansiyel = (1 - close / hi60) >= 0.20
    dipte = (close / lo60 - 1) <= 0.15
    donus = close > close.rolling(10).mean()
    r20 = close / close.shift(20) - 1
    guc = r20.sub(r20.median(axis=1), axis=0) > 0

    ciro = (close * volume).rolling(20).mean()
    likit = ciro.rank(axis=1, pct=True) >= 0.30
    gecerli = likit & close.notna() & close.shift(60).notna()
    f = {"birikim": birikim, "dar": dar, "hacim": hacim, "potansiyel": potansiyel,
         "dipte": dipte, "donus": donus, "guc": guc}
    return {k: v.fillna(False).to_numpy() for k, v in f.items()}, gecerli.to_numpy()


def ileri(close, h):
    return (close.shift(-h) / close - 1).to_numpy() * 100


def sinyaller(mask):
    """Her hissede ilk sinyal, sonra COOLDOWN gün bekle. (tarih_idx, hisse_idx) listesi."""
    out_i, out_j = [], []
    for j in range(mask.shape[1]):
        last = -10**9
        for i in np.flatnonzero(mask[:, j]):
            if i - last >= COOLDOWN:
                out_i.append(i)
                out_j.append(j)
                last = i
    return np.array(out_i, dtype=int), np.array(out_j, dtype=int)


def calistir(close, volume):
    feats, gecerli = ozellikler(close, volume)
    T = close.shape[0]
    fwd = {h: ileri(close, h) for h in HORIZONS}
    mk = {}
    for h in HORIZONS:
        a = np.where(gecerli, fwd[h], np.nan)
        with np.errstate(all="ignore"):
            mk[h] = np.nanmean(a, axis=1)               # her günün likit hisse ortalaması (kıyas)
    rows = []
    # kıyas satırı
    for ad, keys in [("TÜM LİKİT HİSSELER (kıyas)", None)] + list(VARYANTLAR.items()):
        if keys is None:
            mask = gecerli.copy()
        else:
            mask = gecerli.copy()
            for k in keys:
                mask &= feats[k]
        ii, jj = sinyaller(mask) if keys is not None else (None, None)
        r = {"varyant": ad}
        for h in HORIZONS:
            if keys is None:
                vals = fwd[h][gecerli]
                vals = vals[np.isfinite(vals)]
                r[f"n{h}"] = int(len(vals))
                r[f"ort{h}"] = float(vals.mean()) if len(vals) else np.nan
                r[f"fark{h}"] = 0.0
                r[f"medf{h}"] = 0.0
                r[f"gec{h}"] = np.nan
                r[f"t{h}"] = np.nan
                r[f"y1_{h}"] = r[f"y2_{h}"] = np.nan
                continue
            ret = fwd[h][ii, jj]
            m = mk[h][ii]
            ok = np.isfinite(ret) & np.isfinite(m)
            ret, m, ti = ret[ok], m[ok], ii[ok]
            fark = ret - m
            r[f"n{h}"] = int(len(ret))
            r[f"ort{h}"] = float(ret.mean()) if len(ret) else np.nan
            r[f"fark{h}"] = float(fark.mean()) if len(ret) else np.nan
            r[f"medf{h}"] = float(np.median(fark)) if len(ret) else np.nan
            r[f"gec{h}"] = float((fark > 0).mean() * 100) if len(ret) else np.nan
            r[f"t{h}"] = float(fark.mean() / (fark.std(ddof=1) / np.sqrt(len(fark)))) if len(fark) > 2 and fark.std() > 0 else np.nan
            ilk = ti < T // 2
            r[f"y1_{h}"] = float(fark[ilk].mean()) if ilk.sum() >= 10 else np.nan
            r[f"y2_{h}"] = float(fark[~ilk].mean()) if (~ilk).sum() >= 10 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def rapor(res, n_hisse, ilk_tarih, son_tarih):
    f = lambda x, d=2: "—" if pd.isna(x) else f"{x:+.{d}f}"
    g = lambda x: "—" if pd.isna(x) else f"{x:.0f}"
    L = ["# Günlük dip testi (1-2 ay tutma)\n",
         f"{n_hisse} hisse, {ilk_tarih} - {son_tarih}. Giriş = sinyal günü kapanışı. "
         "Fark = hissenin getirisi - o gün tüm likit hisselerin ortalama getirisi. "
         "Yarı 1 / Yarı 2 = dönemin ilk ve ikinci yarısındaki ortalama fark (ikisi de artıysa tutarlı).\n"]
    for h, ad in ((21, "21 işlem günü sonra (~1 ay)"), (42, "42 işlem günü sonra (~2 ay)")):
        L += [f"## {ad}\n",
              "| Varyant | Sinyal | Ort. getiri % | Ort. fark | Medyan fark | Piyasayı geçen % | t | Yarı 1 | Yarı 2 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for _, r in res.iterrows():
            kiyas = r["varyant"].startswith("TÜM")
            guv = "" if (kiyas or r[f"n{h}"] >= MIN_N) else " ⚠"
            L.append("| %s%s | %d | %s | %s | %s | %s | %s | %s | %s |" % (
                r["varyant"], guv, int(r[f"n{h}"]), f(r[f"ort{h}"]), "—" if kiyas else f(r[f"fark{h}"]),
                "—" if kiyas else f(r[f"medf{h}"]), "—" if kiyas else g(r[f"gec{h}"]),
                "—" if kiyas else f(r[f"t{h}"], 1),
                "—" if kiyas else f(r[f"y1_{h}"]), "—" if kiyas else f(r[f"y2_{h}"])))
        L.append("")
    L += [f"⚠ = {MIN_N}'dan az sinyal.",
          "Nasıl okunur: t = farkın gürültüye oranı. Sinyalin hiç etkisi olmayan rastgele veride bile t 1-2 çıkabilir; "
          "t en az 3 değilse tesadüf say. Ayrıca MEDYAN fark da artıda (yalnızca ortalama artıysa birkaç büyük yükseliş şişiriyordur), "
          "piyasayı geçen oran %50'nin üstünde ve Yarı 1 / Yarı 2 ikisi de artıda olmalı. Sinyaller üst üste bindiği için gerçek belirsizlik t'nin gösterdiğinden de büyüktür.",
          "Uyarılar: güncel hisse listesi kullanıldığı için bugün delist olmuş/batmış hisseler yok, sonuçlar iyimser olabilir. "
          "Komisyon yok. Çok sayıda şart denendiği için tesadüfen iyi görünen satır çıkabilir; yalnızca tutarlı olanı ciddiye al."]
    return "\n".join(L)


def main():
    import yfinance  # noqa: F401
    symbols = S.read_symbols("bist_symbols.txt")
    print(f"{len(symbols)} sembol, 5 yıllık günlük veri indiriliyor...")
    close, volume = S.fetch_all(symbols, "5y")
    print(f"{close.shape[1]} hisse, {close.shape[0]} gün")
    res = calistir(close, volume)
    res.to_csv("dip_gunluk_sonuc.csv", index=False, encoding="utf-8-sig")
    txt = rapor(res, close.shape[1], f"{close.index[0]:%d.%m.%Y}", f"{close.index[-1]:%d.%m.%Y}")
    print(txt)
    ozet = os.environ.get("GITHUB_STEP_SUMMARY")
    if ozet:
        with open(ozet, "a", encoding="utf-8") as fh:
            fh.write(txt + "\n")


if __name__ == "__main__":
    main()
