"""
hacim_backtest_b.py
"B yatay birikim" sinyalini farklı eşiklerle geçmişe dönük test eder.

Sinyal: son 15 günde fiyat dar bantta (bant <= BANT) VE son 5 günün hacmi
önceki 20 günün ortalamasının en az HACIM katı.

Alış: sinyalden SONRAKİ günün kapanışı. Çıkış: 5, 10 ve 20 işlem günü sonra.
Maliyet: gidiş-dönüş %0.4 düşülür. Aynı hissede sinyal sonrası 10 gün yeni
sinyal sayılmaz.

Dayanıklılık kontrolü: her varyant için verinin ilk yarısı ve ikinci yarısı
ayrı ayrı gösterilir. İki yarıda da kıyastan iyiyse sinyal daha güvenilirdir.

Not: Sadece bugün borsada olan hisseler test edilir (hayatta kalma yanlılığı),
tavan/taban kilitlerinde alamama durumu hesaba katılmaz.

Kullanım: python hacim_backtest_b.py [HISSE ...]
Gereken: pip install borsapy pandas numpy
"""
import sys
import numpy as np
import pandas as pd
import borsapy as bp

UFUKLAR = [5, 10, 20]
MALIYET = 0.004
COOLDOWN = 10
BANTLAR = [0.06, 0.08, 0.10, 0.12]
HACIMLER = [1.5, 2.0, 3.0]


def kolon_bul(df, ad):
    for c in df.columns:
        if str(c).lower() == ad.lower():
            return c
    raise KeyError(ad)


def hisseleri_al():
    if len(sys.argv) > 1:
        return [t.upper() for t in sys.argv[1:]]
    df = bp.companies()
    for c in df.columns:
        if str(c).lower() in ("ticker", "symbol", "kod", "code"):
            return [str(x).strip().upper() for x in df[c].dropna()]
    return [str(x).strip().upper() for x in df.iloc[:, 0].dropna()]


def veri_al(kod):
    for periyot in ("2y", "1y", "6mo"):
        try:
            df = bp.Ticker(kod).history(period=periyot)
            if df is not None and len(df) >= 100:
                return df
        except Exception:
            continue
    return None


def hazirla(df):
    k = df[kolon_bul(df, "Close")].astype(float).reset_index(drop=True)
    h = df[kolon_bul(df, "Volume")].astype(float).reset_index(drop=True)
    d = pd.DataFrame({"k": k, "h": h})
    onceki20 = h.shift(5).rolling(20).mean()
    d["oran"] = h.rolling(5).mean() / onceki20.replace(0, np.nan)
    d["getiri5"] = k / k.shift(5) - 1
    mx15 = k.rolling(15).max()
    mn15 = k.rolling(15).min()
    d["bant"] = (mx15 - mn15) / mn15
    d["zirve"] = k / k.rolling(60).max()
    d["yari"] = (np.arange(len(k)) >= len(k) / 2).astype(int)  # 0 = ilk yarı
    for u in UFUKLAR:
        d[f"f{u}"] = k.shift(-1 - u) / k.shift(-1) - 1 - MALIYET
    return d.dropna(subset=["oran", "getiri5", "bant", "zirve"])


def varyantlari_kur():
    v = {}
    for b in BANTLAR:
        for hc in HACIMLER:
            ad = f"B bant<=%{int(b*100)} hacim>={hc:g}"
            v[ad] = (lambda d, b=b, hc=hc: (d.bant <= b) & (d.oran >= hc))
    v["B bant<=%10 hacim>=2, tepede degil"] = (
        lambda d: (d.bant <= 0.10) & (d.oran >= 2) & (d.zirve < 0.95))
    v["B bant<=%10 hacim>=2, 5g getiri>0"] = (
        lambda d: (d.bant <= 0.10) & (d.oran >= 2) & (d.getiri5 > 0))
    return v


VARYANTLAR = varyantlari_kur()


def cooldown_sec(maske):
    secilen = []
    son = -10**9
    for i in np.flatnonzero(maske.values):
        if i - son > COOLDOWN:
            secilen.append(i)
            son = i
    return secilen


def ozet(a):
    if len(a) == 0:
        return [0, np.nan, np.nan, np.nan]
    a = np.array(a)
    return [len(a), a.mean() * 100, np.median(a) * 100, (a > 0).mean() * 100]


def main():
    kodlar = hisseleri_al()
    # sonuc[varyant][ufuk] = {"hepsi": [], 0: [], 1: []}
    sonuc = {v: {u: {"h": [], 0: [], 1: []} for u in UFUKLAR} for v in VARYANTLAR}
    taban = {u: {"h": [], 0: [], 1: []} for u in UFUKLAR}
    test_edilen = 0

    for kod in kodlar:
        try:
            df = veri_al(kod)
            if df is None:
                continue
            d = hazirla(df)
            if len(d) < 50:
                continue
        except Exception as e:
            print(f"{kod}: atlandı ({e})")
            continue
        test_edilen += 1
        for u in UFUKLAR:
            s = d[[f"f{u}", "yari"]].dropna()
            taban[u]["h"].extend(s[f"f{u}"].tolist())
            for y in (0, 1):
                taban[u][y].extend(s.loc[s.yari == y, f"f{u}"].tolist())
        for ad, fn in VARYANTLAR.items():
            idx = cooldown_sec(fn(d))
            sec = d.iloc[idx]
            for u in UFUKLAR:
                s = sec[[f"f{u}", "yari"]].dropna()
                sonuc[ad][u]["h"].extend(s[f"f{u}"].tolist())
                for y in (0, 1):
                    sonuc[ad][u][y].extend(s.loc[s.yari == y, f"f{u}"].tolist())

    print(f"Test edilen hisse: {test_edilen}\n")
    pd.set_option("display.width", 250)
    tum = []
    for u in UFUKLAR:
        print(f"=== {u} işlem günü sonra (maliyet düşülmüş) ===")
        t_h = ozet(taban[u]["h"])
        t0 = ozet(taban[u][0])
        t1 = ozet(taban[u][1])
        satirlar = [["KIYAS (tum gunler)", t_h[0], t_h[1], t_h[2], t_h[3],
                     t0[1], t1[1]]]
        for ad in VARYANTLAR:
            o = ozet(sonuc[ad][u]["h"])
            o0 = ozet(sonuc[ad][u][0])
            o1 = ozet(sonuc[ad][u][1])
            satirlar.append([ad, o[0], o[1], o[2], o[3], o0[1], o1[1]])
        tablo = pd.DataFrame(
            satirlar,
            columns=["varyant", "adet", "ort_%", "medyan_%", "kazanma_%",
                     "ort_1.yari", "ort_2.yari"])
        print(tablo.round(2).to_string(index=False))
        print()
        tablo.insert(0, "ufuk_gun", u)
        tum.append(tablo)
    pd.concat(tum).to_csv("hacim_backtest_b_sonuc.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
