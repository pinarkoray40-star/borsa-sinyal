"""
hacim_backtest_c.py
Dar aralık (B) sinyalini farklı bant eşikleriyle test eder.

Sinyal (B): son 15 günde fiyat bandı <= BANT VE son 5 günün hacmi,
önceki 20 günün ortalamasının en az HACIM katı.

Bant eşikleri: %3, %4, %5. Hacim eşiği: 1.5. Filtre yok.
Alış: sinyalden sonraki günün kapanışı. Çıkış: 5, 10 ve 20 işlem günü sonra.
Maliyet: gidiş-dönüş %0.4. Aynı hissede sinyal sonrası 10 gün yeni sinyal yok.
Verinin ilk ve ikinci yarısı ayrı ayrı gösterilir (dayanıklılık kontrolü).

Kullanım: python hacim_backtest_c.py [HISSE ...]
Gereken: pip install borsapy pandas numpy
"""
import sys
import numpy as np
import pandas as pd
import borsapy as bp

UFUKLAR = [5, 10, 20]
MALIYET = 0.004
COOLDOWN = 10
BANTLAR = [0.03, 0.04, 0.05]
HACIMLER = [1.5]
FILTRELER = ["yok"]


def kolon_bul(df, ad):
    for c in df.columns:
        if str(c).lower() == ad.lower():
            return c
    raise KeyError(ad)


def tarihler(df):
    """Veri çerçevesinden (saat dilimsiz, günlük) tarih indeksi çıkarır."""
    idx = None
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    else:
        for c in df.columns:
            if str(c).lower() in ("date", "datetime", "tarih", "time"):
                idx = pd.DatetimeIndex(pd.to_datetime(df[c]))
                break
    if idx is None:
        return None
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


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


def endeks_al():
    """BIST 100 kapanış serisi (tarih indeksli). Alınamazsa None."""
    denemeler = [lambda: bp.Index("XU100").history(period="2y"),
                 lambda: bp.Ticker("XU100").history(period="2y")]
    for dene in denemeler:
        try:
            df = dene()
            if df is None or len(df) < 100:
                continue
            idx = tarihler(df)
            if idx is None:
                continue
            s = pd.Series(df[kolon_bul(df, "Close")].astype(float).values, index=idx)
            s = s[~s.index.duplicated(keep="last")].sort_index()
            return s
        except Exception:
            continue
    return None


def hazirla(df, endeks):
    k = df[kolon_bul(df, "Close")].astype(float).reset_index(drop=True)
    h = df[kolon_bul(df, "Volume")].astype(float).reset_index(drop=True)
    d = pd.DataFrame({"k": k, "h": h})
    onceki20 = h.shift(5).rolling(20).mean()
    d["oran"] = h.rolling(5).mean() / onceki20.replace(0, np.nan)
    mx15 = k.rolling(15).max()
    mn15 = k.rolling(15).min()
    d["bant"] = (mx15 - mn15) / mn15
    d["mom60"] = k / k.shift(60) - 1
    d["rs20"] = np.nan
    if endeks is not None:
        dt = tarihler(df)
        if dt is not None:
            e = pd.Series(endeks.reindex(dt, method="ffill").values)
            d["rs20"] = (k / k.shift(20) - 1) - (e / e.shift(20) - 1)
    d["yari"] = (np.arange(len(k)) >= len(k) / 2).astype(int)
    for u in UFUKLAR:
        d[f"f{u}"] = k.shift(-1 - u) / k.shift(-1) - 1 - MALIYET
    return d.dropna(subset=["oran", "bant", "mom60"])


def varyantlari_kur():
    v = {}
    for b in BANTLAR:
        for hc in HACIMLER:
            for f in FILTRELER:
                ad = f"bant<=%{int(round(b*100))} hac>={hc:g} {f}"

                def fn(d, b=b, hc=hc, f=f):
                    m = (d.bant <= b) & (d.oran >= hc)
                    if f in ("guc", "ikisi"):
                        m = m & (d.rs20 > 0)
                    if f in ("mom", "ikisi"):
                        m = m & (d.mom60 > 0)
                    return m
                v[ad] = fn
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
    endeks = endeks_al()
    if endeks is None:
        print("UYARI: BIST 100 verisi alınamadı, göreceli güç varyantları boş çıkacak.\n")
    else:
        print(f"BIST 100 verisi alındı ({len(endeks)} gün).\n")

    kodlar = hisseleri_al()
    sonuc = {v: {u: {"h": [], 0: [], 1: []} for u in UFUKLAR} for v in VARYANTLAR}
    taban = {u: {"h": [], 0: [], 1: []} for u in UFUKLAR}
    test_edilen = 0

    for kod in kodlar:
        try:
            df = veri_al(kod)
            if df is None:
                continue
            d = hazirla(df, endeks)
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
    pd.concat(tum).to_csv("hacim_backtest_c_sonuc.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
