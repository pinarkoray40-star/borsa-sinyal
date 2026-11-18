"""
dip_dar_test.py - "dar aralık" şartının detaylı testi (1-2 ay tutma)

dip_gunluk_test.py aynı klasörde olmalı, onun fonksiyonlarını kullanır. Sitedeki hiçbir dosyaya dokunmaz.
Dar aralık = son W günde kapanış bandı (en yüksek/en düşük - 1) en çok %B.
W (gün) ve B (%) değerlerini tek tek dener; sonra üç temel ayarın üstüne ek şart koyar:
  zirveye yakın  : kapanış 60 günlük zirvenin en az %90'ı (kırılıma yakın)
  dipten uzak    : kapanış 60 günlük dibin en az %15 üstünde
  güç            : son 20 günde piyasadan iyi
  dönüş          : kapanış 10 günlük ortalamanın üstünde
Gereken: pip install yfinance pandas numpy
"""
import os

import dip_gunluk_test as G
import signals as S

PENCERELER = (10, 15, 20, 30)
BANTLAR = (3, 4, 6, 8, 10)
TEMEL = ((10, 4), (15, 6), (20, 8))
EKLER = {"zirveye yakın": "yakin", "dipten uzak": "uzak", "güç": "guc", "dönüş": "donus"}

_orijinal = G.ozellikler


def genis(close, volume):
    feats, gecerli = _orijinal(close, volume)
    hi60 = close.rolling(60).max()
    lo60 = close.rolling(60).min()
    for w in PENCERELER:
        rng = (close.rolling(w).max() - close.rolling(w).min()) / close.rolling(w).min()
        for b in BANTLAR:
            feats[f"dar_{w}_{b}"] = (rng <= b / 100.0).fillna(False).to_numpy()
    feats["yakin"] = (close >= 0.90 * hi60).fillna(False).to_numpy()
    feats["uzak"] = (close >= 1.15 * lo60).fillna(False).to_numpy()
    return feats, gecerli


def varyantlar():
    v = {}
    for w in PENCERELER:
        for b in BANTLAR:
            v[f"dar {w} gün, %{b}"] = [f"dar_{w}_{b}"]
    for w, b in TEMEL:
        for ad, key in EKLER.items():
            v[f"dar {w} gün, %{b} + {ad}"] = [f"dar_{w}_{b}", key]
    return v


def main():
    import yfinance  # noqa: F401
    symbols = S.read_symbols("bist_symbols.txt")
    print(f"{len(symbols)} sembol, 5 yıllık günlük veri indiriliyor...")
    close, volume = S.fetch_all(symbols, "5y")
    G.ozellikler = genis
    G.VARYANTLAR = varyantlar()
    res = G.calistir(close, volume)
    res.to_csv("dip_dar_sonuc.csv", index=False, encoding="utf-8-sig")
    txt = G.rapor(res, close.shape[1], f"{close.index[0]:%d.%m.%Y}", f"{close.index[-1]:%d.%m.%Y}")
    txt += ("\n\nBu testte %d satır denendi; bu kadar çok denemede tesadüfen yüksek t çıkabilir, "
            "bu yüzden t en az 3,5 olanı ciddiye al." % len(G.VARYANTLAR))
    print(txt)
    ozet = os.environ.get("GITHUB_STEP_SUMMARY")
    if ozet:
        with open(ozet, "a", encoding="utf-8") as fh:
            fh.write(txt + "\n")


if __name__ == "__main__":
    main()
