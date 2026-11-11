"""
yukselis_kural.py
Adim 1: Gecmisten, yukselis oncesi gunlerin ozelliklerini TEK bir kurala birlestirir
        ve kuralin 20 gun sonraki GERCEKCI getirisini olcer.
Adim 2: Kurali bugun saglayan hisseleri listeler.

Dürüstlük için:
  - Veri zamana göre ikiye bölünür: EGITIM (ilk %60) ve TEST (son %40).
    Kural sadece EGITIM'de aranır, TEST'te hiç görülmemiş veride sınanır.
  - Alış: sinyal gününden SONRAKİ günün kapanışı. Satış: alıştan 20 işlem günü sonra.
  - Gidiş-dönüş maliyet düşülür (MALIYET, %).
  - Ertesi gün tavan kilitli olan günler (alınamazdı) dışarıda bırakılır.
  - Düşük hacimli (alınıp satılamayan) hisseler dışarıda bırakılır.
  - Aynı hissede birbirine yakın sinyaller tek sinyal sayılır (ARA gün).
  - Kural, "ortalama getiri"yi (uçlar kırpılarak) en çok artıran koşulları
    en fazla MAX_KOSUL adet seçerek bulunur. TEST sonucu tabandan iyi değilse
    kural çalışmıyor demektir; rapor bunu açıkça yazar.

Kullanım: python yukselis_kural.py [HISSE ...]
Gereken: pip install borsapy pandas numpy
"""
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

import buyuk_yukselis_analiz as by

UFUK = 20               # işlem günü
MALIYET = 0.5           # gidiş-dönüş toplam, %
MIN_HACIM_TL = 5.0      # milyon TL, son 20 gün ort. işlem hacmi
MIN_FIYAT = 1.0         # TL
ARA = 20                # aynı hissede iki sinyal arası en az gün
EGITIM_ORAN = 0.6
MAX_KOSUL = 3
MIN_SINYAL = 150        # EGITIM'de bir kural için en az bağımsız sinyal
MIN_IYILESME = 0.3      # bir koşul eklemek için skorun en az bu kadar artması (puan)
ADAYLAR = ["bant15", "bant30", "hacim_oran", "mom20", "mom60", "rs20",
           "zirve120", "dip120", "vol20", "tavan10"]
KUANTILLER = [0.2, 0.4, 0.6, 0.8]


def hisse_hazirla(args):
    kod, endeks = args
    try:
        df = by.veri_al(kod)
        if df is None:
            return None
        d, _ = by.hazirla(kod, df, endeks)
        if len(d) < 40:
            return None
        k = d["kapanis"]
        alis = k.shift(-1)
        d["ret"] = (k.shift(-1 - UFUK) / alis - 1) * 100 - MALIYET
        d["ertesi_tavan"] = (k.shift(-1) / k - 1) >= 0.095
        d["bugun"] = (k / k.shift(1) - 1) * 100
        d["gun"] = np.arange(len(d))
        return d
    except Exception as e:
        print(f"{kod}: hata ({type(e).__name__}: {e})")
        return None


def kumele(df):
    """Aynı hissede ARA günden yakın sinyalleri tek sinyal say (ilkini tut)."""
    tut = []
    for _, g in df.sort_values(["hisse", "gun"]).groupby("hisse", sort=False):
        son = -10**9
        for idx, gun in zip(g.index, g["gun"].values):
            if gun - son >= ARA:
                tut.append(idx)
                son = gun
    return df.loc[tut]


def ozet(r):
    r = pd.Series(r).dropna()
    if len(r) == 0:
        return {"n": 0, "ort": np.nan, "med": np.nan, "kazanc": np.nan, "p30": np.nan, "kayip20": np.nan}
    return {"n": len(r), "ort": r.mean(), "med": r.median(), "kazanc": (r > 0).mean() * 100,
            "p30": (r >= 30).mean() * 100, "kayip20": (r <= -20).mean() * 100}


def satir(ad, o):
    if o["n"] == 0:
        return f"  {ad:10s} sinyal yok"
    return (f"  {ad:10s} n={o['n']:5d}  ort %{o['ort']:6.1f}  medyan %{o['med']:6.1f}  "
            f"kazanan %{o['kazanc']:4.0f}  >=+30%: %{o['p30']:4.1f}  <=-20%: %{o['kayip20']:4.1f}")


def kosul_maske(df, kosul):
    f, op, esik = kosul
    x = df[f].values
    return (x <= esik) if op == "<=" else (x >= esik)


def kural_maske(df, kural):
    m = np.ones(len(df), bool)
    for kosul in kural:
        m &= kosul_maske(df, kosul)
    return m


def kural_yaz(kural):
    return " VE ".join(f"{f} {op} {esik:g}" for f, op, esik in kural) if kural else "(kural yok)"


def kural_ara(egitim):
    """Greedy: her adımda eğitim ortalama getirisini (kırpılmış) en çok artıran koşulu ekle."""
    egitim = egitim.reset_index(drop=True)
    ret_k = egitim["ret"].clip(-50, 100).values
    hisse_id = pd.factorize(egitim["hisse"])[0].astype(np.int64)
    kova = hisse_id * 100000 + (egitim["gun"].values // ARA)

    kural = []
    maske = np.ones(len(egitim), bool)
    skor = ret_k.mean()
    kullanilan = set()
    for _ in range(MAX_KOSUL):
        en_iyi = None
        for f in ADAYLAR:
            if f in kullanilan:
                continue
            x = egitim[f].values
            if np.isnan(x).all():
                continue
            for q in KUANTILLER:
                esik = float(np.nanquantile(x, q))
                for op in ("<=", ">="):
                    m = maske & ((x <= esik) if op == "<=" else (x >= esik))
                    if m.sum() == 0:
                        continue
                    if len(np.unique(kova[m])) < MIN_SINYAL:
                        continue
                    s = ret_k[m].mean()
                    if en_iyi is None or s > en_iyi[0]:
                        en_iyi = (s, (f, op, round(esik, 2)), m)
        if en_iyi is None or en_iyi[0] - skor < MIN_IYILESME:
            break
        skor, kosul, maske = en_iyi
        kural.append(kosul)
        kullanilan.add(kosul[0])
    return kural


def main():
    endeks = by.endeks_al()
    print("BIST100 verisi:", "alındı" if endeks is not None else "ALINAMADI (rs20 boş)")
    kodlar = by.hisseleri_al()
    with ThreadPoolExecutor(max_workers=4) as ex:
        sonuc = [r for r in ex.map(hisse_hazirla, [(k, endeks) for k in kodlar]) if r is not None]
    if not sonuc:
        print("Veri yok.")
        return
    havuz = pd.concat(sonuc, ignore_index=True)
    havuz["tarih"] = pd.to_datetime(havuz["tarih"])
    print(f"Taranan hisse: {len(sonuc)} / {len(kodlar)}   son tarih: {havuz['tarih'].max().date()}")

    uygun = (havuz["hacim_tl"] >= MIN_HACIM_TL) & (havuz["fiyat"] >= MIN_FIYAT)
    etiketli = havuz[uygun & havuz["ret"].notna() & ~havuz["ertesi_tavan"]].copy()

    gunler = np.sort(etiketli["tarih"].unique())
    kesim = int(len(gunler) * EGITIM_ORAN)
    egitim_son = gunler[max(kesim - UFUK, 0)]      # eğitim ile test arasında UFUK gün boşluk
    test_ilk = gunler[kesim]
    egitim = etiketli[etiketli["tarih"] <= egitim_son]
    test = etiketli[etiketli["tarih"] >= test_ilk]
    print(f"Eğitim: {pd.Timestamp(gunler[0]).date()} → {pd.Timestamp(egitim_son).date()}   "
          f"Test: {pd.Timestamp(test_ilk).date()} → {pd.Timestamp(gunler[-1]).date()}")
    print(f"Alış: ertesi gün kapanış, satış: {UFUK} işlem günü sonra, maliyet %{MALIYET} düşülmüş\n")

    kural = kural_ara(egitim)
    print("=== TABAN (kural yok, uygun tüm hisse-günler, kümelenmiş) ===")
    taban_e = ozet(kumele(egitim)["ret"])
    taban_t = ozet(kumele(test)["ret"])
    print(satir("eğitim", taban_e))
    print(satir("test", taban_t))

    if not kural:
        print("\nEğitimde anlamlı bir kural bulunamadı (hiçbir koşul tabanı yeterince iyileştirmedi).")
        return

    print("\n=== KURAL ADIM ADIM (her satır bir koşul daha ekler) ===")
    for i in range(1, len(kural) + 1):
        alt = kural[:i]
        print(f"\n{i}. adım: {kural_yaz(alt)}")
        e = egitim[kural_maske(egitim, alt)]
        t = test[kural_maske(test, alt)]
        print(satir("eğitim", ozet(kumele(e)["ret"]) if len(e) else ozet([])))
        print(satir("test", ozet(kumele(t)["ret"]) if len(t) else ozet([])))

    son_t = ozet(kumele(test[kural_maske(test, kural)])["ret"]) if kural_maske(test, kural).any() else ozet([])
    print("\n=== HÜKÜM ===")
    if son_t["n"] >= 30 and son_t["ort"] >= taban_t["ort"] + 1.0 and son_t["med"] > taban_t["med"]:
        print(f"Kural TEST'te tabanı geçti (ort %{son_t['ort']:.1f} vs %{taban_t['ort']:.1f}, "
              f"n={son_t['n']}). Yine de bu geçmiş sınamadır, garanti değildir.")
    else:
        print(f"Kural TEST'te tabanı güvenle geçemedi (ort %{son_t['ort']:.1f} vs taban %{taban_t['ort']:.1f}, "
              f"n={son_t['n']}). Bugünkü liste 'şartları tutanlar' olarak okunmalı, "
              f"işe yaradığının kanıtı olarak değil.")

    # --- Bugün şartları tutan hisseler ---
    son_tarih = havuz["tarih"].max()
    bugun = havuz[(havuz["tarih"] == son_tarih)
                  & (havuz["hacim_tl"] >= MIN_HACIM_TL) & (havuz["fiyat"] >= MIN_FIYAT)]
    tutan = bugun[kural_maske(bugun, kural)].copy()
    print(f"\n=== {son_tarih.date()} itibarıyla kuralı tutan hisseler: {len(tutan)} ===")
    if len(tutan):
        tutan["not"] = np.where(tutan["bugun"] >= 9.5, "tavan (alınamaz olabilir)", "")
        kolon = ["hisse", "fiyat", "bugun", "bant15", "hacim_oran", "mom20", "rs20",
                 "zirve120", "hacim_tl", "not"]
        gor = tutan.sort_values("hacim_oran", ascending=False)[kolon].round(1)
        print(gor.head(40).to_string(index=False))
        gor.to_csv("yukselis_kural_bugun.csv", index=False, encoding="utf-8-sig")
    else:
        print("Bugün kuralı tutan hisse yok.")


if __name__ == "__main__":
    main()
