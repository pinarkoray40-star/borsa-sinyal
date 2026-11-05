"""
kazanan_analiz.py
Dar aralık sinyalinin en iyi ve en kötü sonuç verenlerini karşılaştırır.

Sinyal (hacim_backtest_c.py ile aynı): son 15 günde fiyat bandı <= %5 VE
son 5 günün hacmi, önceki 20 günün ortalamasının en az 1.5 katı.
Alış: sinyalden sonraki günün kapanışı. Sonuç: 20 işlem günü sonra, maliyet (%0.4) düşülmüş.
Aynı hissede sinyal sonrası 10 gün yeni sinyal yok.

Amaç: sinyal günündeki hangi özellikler (bant, hacim, getiri, endekse göre güç...)
sonradan en çok kazandıranları ayırıyor? Sadece sinyal günü bilinen şeylere bakılır.

Kullanım: python kazanan_analiz.py [HISSE ...]
Gereken: pip install borsapy pandas numpy
Canlı deftere dokunmaz, sadece geçmişi inceler.
"""
import sys
import numpy as np
import pandas as pd
import borsapy as bp

MALIYET = 0.004
COOLDOWN = 10
UFUK = 20
BANT_MAX = 0.05
HACIM_MIN = 1.5
KAZANAN_ESIK = 0.30

# (kolon, ekranda görünen ad, çarpan)
OZELLIKLER = [
    ("bant", "bant_%", 100),
    ("oran", "hacim5g_x", 1),
    ("oran1", "hacim1g_x", 1),
    ("ret5", "getiri5g_%", 100),
    ("ret20", "getiri20g_%", 100),
    ("mom60", "getiri60g_%", 100),
    ("rs20", "endekseGore20g_%", 100),
    ("konum", "bant_konum", 1),
    ("zirve", "zirveyeUzak_%", 100),
    ("likit", "likidite_mnTL", 1),
    ("fiyat", "fiyat_TL", 1),
    ("end20", "endeks20g_%", 100),
]


def kolon_bul(df, ad):
    for c in df.columns:
        if str(c).lower() == ad.lower():
            return c
    raise KeyError(ad)


def tarihler(df):
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
    d["oran1"] = h / h.shift(1).rolling(20).mean().replace(0, np.nan)
    mx15 = k.rolling(15).max()
    mn15 = k.rolling(15).min()
    d["bant"] = (mx15 - mn15) / mn15
    d["konum"] = (k - mn15) / (mx15 - mn15).replace(0, np.nan)
    d["ret5"] = k / k.shift(5) - 1
    d["ret20"] = k / k.shift(20) - 1
    d["mom60"] = k / k.shift(60) - 1
    d["zirve"] = k / k.rolling(120, min_periods=60).max() - 1
    d["likit"] = (k * h).rolling(20).mean() / 1e6
    d["fiyat"] = k
    d["rs20"] = np.nan
    d["end20"] = np.nan
    d["end_f"] = np.nan
    d["tarih"] = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
    dt = tarihler(df)
    if dt is not None and len(dt) == len(k):
        d["tarih"] = dt.values
        if endeks is not None:
            e = pd.Series(endeks.reindex(dt, method="ffill").values)
            d["end20"] = e / e.shift(20) - 1
            d["rs20"] = d["ret20"] - d["end20"]
            d["end_f"] = e.shift(-1 - UFUK) / e.shift(-1) - 1
    d["f20"] = k.shift(-1 - UFUK) / k.shift(-1) - 1 - MALIYET
    return d.dropna(subset=["oran", "bant", "mom60"])


def cooldown_sec(maske):
    secilen = []
    son = -10**9
    for i in np.flatnonzero(maske.values):
        if i - son > COOLDOWN:
            secilen.append(i)
            son = i
    return secilen


def sinyalleri_topla(kodlar, endeks):
    parcalar = []
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
        maske = (d.bant <= BANT_MAX) & (d.oran >= HACIM_MIN)
        idx = cooldown_sec(maske)
        if not idx:
            continue
        s = d.iloc[idx].copy()
        s["kod"] = kod
        s["yari"] = (s.index.values >= len(df) / 2).astype(int)
        parcalar.append(s)
    if not parcalar:
        return test_edilen, None
    tum = pd.concat(parcalar).reset_index(drop=True)
    return test_edilen, tum


def auc(x, y):
    """y=True olanın değeri, y=False olandan büyük çıkma olasılığı. 0.5 = ayırt etmiyor."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=bool)
    m = ~np.isnan(x)
    x = x[m]
    y = y[m]
    n1 = int(y.sum())
    n0 = int((~y).sum())
    if n1 < 5 or n0 < 5:
        return np.nan
    r = pd.Series(x).rank().values
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def tarih_yaz(t):
    try:
        if pd.isna(t):
            return "-"
        return pd.Timestamp(t).strftime("%Y-%m-%d")
    except Exception:
        return "-"


def sayi(x, nd=1, isaret=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:+.{nd}f}" if isaret else f"{x:.{nd}f}"


def liste_yaz(df, baslik):
    print(baslik)
    print(f"{'kod':<7}{'tarih':<11}{'f20%':>7}{'bant%':>6}{'hac5x':>6}{'g5%':>6}{'rs20%':>7}")
    for _, r in df.iterrows():
        print(f"{r.kod:<7}{tarih_yaz(r.tarih):<11}{r.f20*100:>7.1f}{r.bant*100:>6.1f}"
              f"{r.oran:>6.1f}{r.ret5*100:>6.1f}{(r.rs20*100 if not np.isnan(r.rs20) else np.nan):>7.1f}")
    print()


def main():
    endeks = endeks_al()
    if endeks is None:
        print("UYARI: BIST 100 verisi alınamadı, endeksle ilgili kısımlar boş çıkacak.\n")
    else:
        print(f"BIST 100 verisi alındı ({len(endeks)} gün).\n")

    kodlar = hisseleri_al()
    test_edilen, tum = sinyalleri_topla(kodlar, endeks)
    print(f"Test edilen hisse: {test_edilen}\n")
    if tum is None:
        print("Hiç sinyal bulunamadı.")
        return

    p = tum.dropna(subset=["f20"]).reset_index(drop=True)
    if len(p) < 30:
        print(f"Sadece {len(p)} sinyal var, analiz için az.")
        return
    p.to_csv("kazanan_analiz_sonuc.csv", index=False, encoding="utf-8-sig")

    # 1) Genel dağılım
    print(f"=== 1) GENEL: bant<=%{int(BANT_MAX*100)}, hacim>={HACIM_MIN:g}, {UFUK} gün sonra ===")
    print(f"Sinyal sayısı: {len(p)} (farklı hisse: {p.kod.nunique()})")
    if p.tarih.notna().any():
        gunluk = p.groupby(p.tarih.dt.normalize()).size()
        print(f"Farklı sinyal günü: {len(gunluk)}, aynı günde en çok: {int(gunluk.max())} sinyal "
              f"(sinyaller aynı günlere yığılıyorsa gerçek örnek sayısı bundan azdır)")
    pb4 = p[p.bant <= 0.04]
    if len(pb4):
        print(f"  bant<=%4 alt grubu: {len(pb4)} sinyal, ort {pb4.f20.mean()*100:+.2f}%, "
              f"medyan {pb4.f20.median()*100:+.2f}%, kazanma {(pb4.f20 > 0).mean()*100:.1f}%")
    q = p.f20.quantile([0.1, 0.25, 0.5, 0.75, 0.9]) * 100
    print("Getiri dağılımı % (10/25/50/75/90. yüzdelik):")
    print("  " + " / ".join(f"{v:+.1f}" for v in q.values))
    for esik in (0.30, 0.20, 0.10):
        n = int((p.f20 >= esik).sum())
        print(f"  >= +%{int(esik*100)} yapan: {n} (%{n/len(p)*100:.1f})")
    for esik in (-0.10, -0.20):
        n = int((p.f20 <= esik).sum())
        print(f"  <= -%{int(-esik*100)} yapan: {n} (%{n/len(p)*100:.1f})")
    print()

    # 2) En iyi üçte bir - en kötü üçte bir
    q33, q67 = p.f20.quantile([1 / 3, 2 / 3])
    ust = p.f20 >= q67
    alt = p.f20 <= q33
    sub = p[ust | alt].reset_index(drop=True)
    y = (sub.f20 >= q67).values
    n1 = int(y.sum())
    n0 = int((~y).sum())
    se = np.sqrt((n1 + n0 + 1) / (12 * n1 * n0))
    print("=== 2) EN İYİ 1/3 ile EN KÖTÜ 1/3: sinyal günü neyle ayrışıyor? ===")
    print(f"En iyi 1/3: {n1} sinyal (f20 >= {q67*100:+.1f}%), en kötü 1/3: {n0} sinyal (f20 <= {q33*100:+.1f}%)")
    print("AUC: 0.50 = ayırt etmez; 0.60+ = en iyiler bu özellikte yüksek, 0.40- = düşük.")
    print(f"'*' = fark rastgele sapmanın 2 katından büyük (|AUC-0.5| > {2*se:.2f}).")
    print("tutarlı = iki ayrı yarı dönemde de aynı yönde mi.")
    satirlar = []
    for kol, ad, c in OZELLIKLER:
        if kol not in sub.columns or sub[kol].notna().sum() < 20:
            continue
        a = auc(sub[kol].values, y)
        m0 = (sub.yari == 0).values
        m1 = ~m0
        a0 = auc(sub.loc[m0, kol].values, y[m0])
        a1 = auc(sub.loc[m1, kol].values, y[m1])
        if np.isnan(a):
            continue
        tutarli = "evet" if (not np.isnan(a0) and not np.isnan(a1)
                             and (a0 - 0.5) * (a - 0.5) > 0 and (a1 - 0.5) * (a - 0.5) > 0) else "hayır"
        yuk = sub.loc[y, kol].median() * c
        dus = sub.loc[~y, kol].median() * c
        yildiz = "*" if abs(a - 0.5) > 2 * se else ""
        satirlar.append((abs(a - 0.5), ad, yuk, dus, a, tutarli, yildiz))
    satirlar.sort(reverse=True)
    print(f"{'özellik':<18}{'iyi_med':>9}{'kötü_med':>9}{'AUC':>6}  tutarlı")
    for _, ad, yuk, dus, a, tut, yl in satirlar:
        print(f"{ad:<18}{yuk:>9.2f}{dus:>9.2f}{a:>6.2f}{yl:<2} {tut}")
    print(f"Not: {len(satirlar)} özellik denendi, rastgele de 1 tanesi '*' alabilir.")
    print()

    # 3) Her özellik için alt/orta/üst üçte bir sonuçları
    print("=== 3) Özellik düşük / orta / yüksek olunca sonuç (medyan% / kazanma%) ===")
    print(f"{'özellik':<18}{'sınırlar':<14}{'düşük':>11}{'orta':>11}{'yüksek':>11}")
    for _, ad, *_r in satirlar:
        kol, c = next((k, cc) for k, a_, cc in OZELLIKLER if a_ == ad)
        v = p[kol]
        ok = v.notna()
        if ok.sum() < 30:
            continue
        grup = pd.qcut(v[ok].rank(method="first"), 3, labels=False)
        s33, s67 = v[ok].quantile([1 / 3, 2 / 3]) * c
        hucre = []
        for g in (0, 1, 2):
            f = p.loc[ok & (grup.reindex(p.index) == g), "f20"]
            hucre.append(f"{f.median()*100:+.1f}/{(f > 0).mean()*100:.0f}" if len(f) else "-")
        print(f"{ad:<18}{f'{s33:.1f}|{s67:.1f}':<14}{hucre[0]:>11}{hucre[1]:>11}{hucre[2]:>11}")
    print()

    # 4) Kazananlar listesi
    kaz = p[p.f20 >= KAZANAN_ESIK].sort_values("f20", ascending=False)
    print(f"=== 4) %{int(KAZANAN_ESIK*100)}+ yapanlar: {len(kaz)} sinyal, {kaz.kod.nunique()} farklı hisse ===")
    if len(kaz) < 8:
        print("Çok az, kural çıkarmak için yetersiz. Sadece ipucu olarak bak.")
    if len(kaz):
        liste_yaz(kaz.head(25), "En çok kazandıranlar:")
    liste_yaz(p.sort_values("f20").head(8), "Karşılaştırma için en çok kaybettirenler:")

    # 5) Zaman kümelenmesi
    if p.tarih.notna().any():
        print("=== 5) Aylara göre dağılım (kazananlar belli bir dönemde mi toplanmış?) ===")
        ay = p.tarih.dt.to_period("M").astype(str)
        tab = pd.DataFrame({"ay": ay, "f": p.f20, "k": (p.f20 >= KAZANAN_ESIK).astype(int)})
        g = tab.groupby("ay").agg(adet=("f", "size"), medyan=("f", "median"), kazanan=("k", "sum"))
        print(f"{'ay':<9}{'adet':>5}{'medyan%':>9}{'%30+':>6}")
        for ay_adi, r in g.iterrows():
            print(f"{ay_adi:<9}{int(r.adet):>5}{r.medyan*100:>9.1f}{int(r.kazanan):>6}")
        print()

    # 6) Endeks etkisi
    if p.end_f.notna().any():
        print("=== 6) Endeks etkisi (kazananlar sadece piyasa yükseldiği için mi?) ===")
        e = p.dropna(subset=["end_f"])
        alfa = e.f20 - e.end_f
        print(f"Tüm sinyaller: endeks aynı dönemde medyan {e.end_f.median()*100:+.1f}%, "
              f"hisse endeksi geçen: %{(alfa > 0).mean()*100:.0f}, medyan fark {alfa.median()*100:+.1f} puan")
        ek = e[e.f20 >= KAZANAN_ESIK]
        if len(ek):
            print(f"%{int(KAZANAN_ESIK*100)}+ yapanlar: endeks aynı dönemde medyan {ek.end_f.median()*100:+.1f}% "
                  f"({len(ek)} sinyal)")
        print()

    print("Sonuç dosyası: kazanan_analiz_sonuc.csv")


if __name__ == "__main__":
    main()
