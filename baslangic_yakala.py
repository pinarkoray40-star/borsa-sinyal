"""
baslangic_yakala.py
Yükseliş BAŞLARKEN yakalama kurallarını test eder ve bugün hangi hisselerin
şartları tuttuğunu listeler. Hedef: zirveye değil, gerçekçi getiriye bakmak.

Her kural sadece o güne kadar bilinen veriyle çalışır:
  1 ilk tavan (dipte)   : bugün >=%9.5 yükseldi, önceki 10 günde tavan yok,
                          fiyat 120 günlük dibin en fazla %60 üstünde
  2 kirilim+hacim       : kapanış önceki 20 günün en yükseğini aştı, hacim oranı >= 1.5,
                          dipten en fazla %70 yukarıda, 60 günlük getiri <= %40
  3 dipten cikis        : dipten %20-80 yukarıda, 20g getiri >= %10, endekse göre güçlü (>0),
                          120 günlük zirvenin en fazla %85'inde (zirvede değil)
  4 guclu momentum      : 20g >= %15, 60g >= %20, endekse göre > 5 puan, zirvenin en fazla %90'ında
  5 kontrol: sadece oynak: günlük oynaklık >= %3.5 (kuralların oynaklıktan mı yoksa
                          gerçekten bilgi taşıdığından mı kazandığını ayırmak için)

Gerçekçilik:
  - Alış: sinyalden sonraki günün kapanışı. Maliyet gidiş-dönüş %0.4 düşülür.
  - Ertesi gün tavan kilitli kapanan sinyaller ALINAMAZ sayılır (elenir).
  - 20 günlük ortalama işlem hacmi >= MIN_HACIM_TL milyon TL olmayan hisseler elenir.
  - Aynı hissede sinyal sonrası COOLDOWN gün yeni sinyal yok.
  - Zirve değil, 5/10/20/40 gün sonraki kapanış getirisi ölçülür.
  - Veri tarih olarak ikiye bölünür (ilk yarı / son yarı), dayanıklılık için.

Kullanım: python baslangic_yakala.py [HISSE ...]
Gereken: pip install borsapy pandas numpy
"""
import sys
import numpy as np
import pandas as pd
import borsapy as bp
from concurrent.futures import ThreadPoolExecutor

UFUK = 60
MALIYET = 0.004
COOLDOWN = 20
MIN_HACIM_TL = 10.0     # milyon TL, 20 gün ortalama
TAVAN = 9.5             # % günlük getiri
HORIZON = [5, 10, 20, 40]
BUGUN_GUN = 2           # bugün listesi: son kaç işlem günü
LISTE_SAYISI = 20

KURALLAR = {
    "1 ilk tavan (dipte)": lambda d: (d.getiri >= TAVAN) & (d.onceki10_tavan == 0) & (d.dip120 <= 160),
    "2 kirilim+hacim": lambda d: (d.k > d.max20_onceki) & (d.hacim_oran >= 1.5)
                                  & (d.dip120 <= 170) & (d.mom60 <= 40),
    "3 dipten cikis": lambda d: (d.dip120 >= 120) & (d.dip120 <= 180) & (d.mom20 >= 10)
                                 & (d.rs20 > 0) & (d.zirve120 <= 85),
    "4 guclu momentum": lambda d: (d.mom20 >= 15) & (d.mom60 >= 20) & (d.rs20 > 5) & (d.zirve120 <= 90),
    "5 kontrol: sadece oynak": lambda d: (d.vol20 >= 3.5),
}
CEKIRDEK = ["mom20", "mom60", "dip120", "zirve120", "vol20", "hacim_oran", "hacim_tl",
            "max20_onceki", "onceki10_tavan", "getiri"]


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
    for periyot in ("2y", "1y"):
        try:
            df = bp.Ticker(kod).history(period=periyot)
            if df is not None and len(df) >= 200:
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
            return s[~s.index.duplicated(keep="last")].sort_index()
        except Exception:
            continue
    return None


def hazirla(kod, df, endeks):
    k = df[kolon_bul(df, "Close")].astype(float).reset_index(drop=True)
    h = df[kolon_bul(df, "Volume")].astype(float).reset_index(drop=True)
    dt = tarihler(df)
    getiri = k.pct_change() * 100

    d = pd.DataFrame(index=range(len(k)))
    d["hisse"] = kod
    d["tarih"] = dt.strftime("%Y-%m-%d") if dt is not None else range(len(k))
    d["k"] = k
    d["getiri"] = getiri
    d["onceki10_tavan"] = (getiri >= TAVAN).astype(float).shift(1).rolling(10).sum()
    d["max20_onceki"] = k.shift(1).rolling(20).max()
    onceki20 = h.shift(5).rolling(20).mean()
    d["hacim_oran"] = h.rolling(5).mean() / onceki20.replace(0, np.nan)
    d["mom20"] = (k / k.shift(20) - 1) * 100
    d["mom60"] = (k / k.shift(60) - 1) * 100
    d["rs20"] = np.nan
    if endeks is not None and dt is not None:
        e = pd.Series(endeks.reindex(dt, method="ffill").values)
        d["rs20"] = (k / k.shift(20) - 1) * 100 - (e / e.shift(20) - 1) * 100
    d["zirve120"] = k / k.rolling(120).max() * 100
    d["dip120"] = k / k.rolling(120).min() * 100
    d["vol20"] = getiri.rolling(20).std()
    d["hacim_tl"] = (k * h).rolling(20).mean() / 1e6

    alis = k.shift(-1)
    for u in HORIZON:
        d[f"f{u}"] = (k.shift(-1 - u) / alis - 1 - MALIYET) * 100
    ileri_min = k[::-1].rolling(20).min()[::-1].shift(-2)
    d["kotu20"] = (ileri_min / alis - 1) * 100
    ileri_max = k[::-1].rolling(UFUK).max()[::-1].shift(-2)
    d["oran2"] = ileri_max / alis
    d["fizibil"] = ((k.shift(-1) / k - 1) * 100) < TAVAN

    d = d.dropna(subset=CEKIRDEK).reset_index(drop=True)
    likit = d["hacim_tl"] >= MIN_HACIM_TL
    for ad, fn in KURALLAR.items():
        d[ad] = (fn(d) & likit).fillna(False)
    return d


def cooldown_sec(maske):
    secilen, son = [], -10**9
    for i in np.flatnonzero(np.asarray(maske)):
        if i - son > COOLDOWN:
            secilen.append(i)
            son = i
    return secilen


def ozet_satir(s, yari_tarih):
    if len(s) == 0:
        return None
    ilk = s[s["tarih"] < yari_tarih]
    son = s[s["tarih"] >= yari_tarih]
    r = {"adet": len(s)}
    for u in HORIZON:
        r[f"ort{u}"] = s[f"f{u}"].mean()
        r[f"med{u}"] = s[f"f{u}"].median()
    r["kaz20"] = (s["f20"] > 0).mean() * 100
    r["iki_kat"] = (s["oran2"] >= 2).mean() * 100
    r["kotu20"] = s["kotu20"].median()
    r["ort20_ilk"] = ilk["f20"].mean() if len(ilk) else np.nan
    r["ort20_son"] = son["f20"].mean() if len(son) else np.nan
    return r


def isle(sonuc):
    tum_tarih = sorted(set(t for d in sonuc for t in d["tarih"].tolist()))
    yari_tarih = tum_tarih[len(tum_tarih) // 2]
    global_son = tum_tarih[-1]
    print(f"Veri: {tum_tarih[0]} -> {global_son} | yarı bölme tarihi: {yari_tarih}")
    print(f"Hisse sayısı: {len(sonuc)} | likidite şartı: 20g ort. hacim >= {MIN_HACIM_TL:g} mn TL\n")

    t1, t2, bugun = [], [], []
    # Kıyas: tüm alınabilir günler
    kiyas = pd.concat([d[(d["hacim_tl"] >= MIN_HACIM_TL) & d["fizibil"] & d["f20"].notna()]
                       for d in sonuc], ignore_index=True)
    r = ozet_satir(kiyas, yari_tarih)
    isim = "KIYAS (tum gunler)"
    if r:
        t1.append([isim, "-", "-", r["adet"], r["ort20"], r["med20"], r["kaz20"], r["iki_kat"], r["kotu20"]])
        t2.append([isim, r["ort5"], r["med5"], r["ort10"], r["med10"], r["ort40"], r["med40"],
                   r["ort20_ilk"], r["ort20_son"]])

    for ad in KURALLAR:
        ham = ali = 0
        secilen = []
        for d in sonuc:
            ok = d[ad] & d["f20"].notna()
            ham += int(ok.sum())
            ok2 = ok & d["fizibil"]
            ali += int(ok2.sum())
            idx = cooldown_sec(ok2.values)
            if idx:
                secilen.append(d.iloc[idx])
        s = pd.concat(secilen, ignore_index=True) if secilen else pd.DataFrame()
        r = ozet_satir(s, yari_tarih) if len(s) else None
        if r:
            t1.append([ad, ham, ali, r["adet"], r["ort20"], r["med20"], r["kaz20"], r["iki_kat"], r["kotu20"]])
            t2.append([ad, r["ort5"], r["med5"], r["ort10"], r["med10"], r["ort40"], r["med40"],
                       r["ort20_ilk"], r["ort20_son"]])
        else:
            t1.append([ad, ham, ali, 0] + [np.nan] * 5)
            t2.append([ad] + [np.nan] * 8)
        if ad.startswith("5"):
            continue
        for d in sonuc:
            if d["tarih"].iloc[-1] != global_son:
                continue
            son = d.tail(BUGUN_GUN)
            son = son[son[ad]]
            for _, row in son.iterrows():
                bugun.append({"kural": ad[:1], "hisse": row["hisse"], "tarih": row["tarih"],
                              "kapanis": round(float(row["k"]), 2), "gun_%": round(float(row["getiri"]), 1),
                              "mom20_%": round(float(row["mom20"]), 1), "dip120_%": round(float(row["dip120"]), 0),
                              "hacim_oran": round(float(row["hacim_oran"]), 2)})

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    c1 = ["kural", "ham", "alinabilir", "adet", "ort20_%", "med20_%", "kazan20_%", "2kat_%", "en_kotu20_med_%"]
    c2 = ["kural", "ort5", "med5", "ort10", "med10", "ort40", "med40", "ort20_ilkyari", "ort20_sonyari"]
    print("=== 20 gün sonra getiri (maliyet düşülmüş, alış ertesi gün kapanış) ===")
    print("ham = kural tutan gün | alinabilir = ertesi gün tavan kilitli olmayan | adet = bekleme sonrası")
    print("2kat_% = 60 gün içinde zirvede 2 kata ulaşma oranı | en_kotu20 = 20 günde görülen en düşük (medyan)")
    print(pd.DataFrame(t1, columns=c1).round(1).to_string(index=False))
    print("\n=== Diğer vadeler ve dayanıklılık (ort = ortalama %, med = medyan %) ===")
    print(pd.DataFrame(t2, columns=c2).round(1).to_string(index=False))

    print(f"\n=== Bugün şartları tutanlar (son {BUGUN_GUN} işlem günü, kural numarasıyla) ===")
    print("Not: 1 numaralı kuralda hisse ertesi gün de tavan kilitlenirse alınamayabilir.")
    if bugun:
        b = pd.DataFrame(bugun)
        for kno, g in b.groupby("kural"):
            print(f"\n-- Kural {kno} ({len(g)} hisse) --")
            print(g.sort_values(["tarih", "mom20_%"], ascending=[False, False])
                  .head(LISTE_SAYISI).drop(columns="kural").to_string(index=False))
        b.to_csv("baslangic_adaylar.csv", index=False, encoding="utf-8-sig")
    else:
        print("Şartları tutan hisse yok.")


def guvenli(args):
    kod, endeks = args
    try:
        df = veri_al(kod)
        if df is None:
            return None
        return hazirla(kod, df, endeks)
    except Exception as e:
        print(f"{kod}: hata ({type(e).__name__}: {e})")
        return None


def main():
    endeks = endeks_al()
    print("BIST100 verisi:", "alındı" if endeks is not None else "ALINAMADI (kural 3-4 boş kalır)")
    kodlar = hisseleri_al()
    with ThreadPoolExecutor(max_workers=4) as ex:
        sonuc = [r for r in ex.map(guvenli, [(k, endeks) for k in kodlar]) if r is not None and len(r)]
    if not sonuc:
        print("Veri yok.")
        return
    print(f"Taranan hisse: {len(sonuc)} / {len(kodlar)}")
    isle(sonuc)


if __name__ == "__main__":
    main()
