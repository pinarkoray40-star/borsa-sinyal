"""
hacim_birikim.py
Ücretsiz fiyat/hacim verisiyle iki sinyal arar (AKD/takas gerekmez):

  Sinyal A - Yükselirken hacim: son 5 günde fiyat artmış VE hacim, önceki
             20 günün ortalamasının en az HACIM_ORAN_MIN katı.
  Sinyal B - Yatay birikim: son 15 günde fiyat dar bantta (<= %8) VE son 5
             günün hacmi önceki 20 günün ortalamasının en az HACIM_ORAN_MIN katı.

Ayrıca zirveye uzaklık (60 günlük) yazılır; zirveye çok yakın hisseler
"tepede" diye işaretlenir (yükselişi yeni başlayanları istiyorsun).

Kullanım:
  python hacim_birikim.py                  -> tickers.txt varsa onu, yoksa tüm BIST'i tarar
  python hacim_birikim.py THYAO RYGYO ...  -> verdiğin hisseler

tickers.txt: her satıra bir hisse kodu (örn. THYAO).
Gereken: pip install borsapy pandas
"""
import sys
import os
import pandas as pd
import borsapy as bp

# Eşikler (istersen değiştir)
HACIM_ORAN_MIN = 3.0      # son 5g ort hacim / önceki 20g ort hacim
YATAY_BANT_MAX = 0.08     # 15 günlük (max-min)/min
TEPE_ESIK = 0.95          # kapanış / 60g zirve >= bu ise "tepede"


def tum_hisseler():
    """borsapy'den tüm BIST hisse kodlarını çekmeyi dener."""
    df = bp.companies()
    for c in df.columns:
        if str(c).lower() in ("ticker", "symbol", "kod", "code"):
            return [str(x).strip().upper() for x in df[c].dropna()]
    # kolon adı tutmazsa ilk kolonu dene
    return [str(x).strip().upper() for x in df.iloc[:, 0].dropna()]


def hisseleri_al():
    if len(sys.argv) > 1:
        return [t.upper() for t in sys.argv[1:]]
    if os.path.exists("tickers.txt"):
        with open("tickers.txt", encoding="utf-8") as f:
            return [s.strip().upper() for s in f if s.strip()]
    try:
        liste = tum_hisseler()
        if liste:
            print(f"borsapy'den {len(liste)} hisse alındı.")
            return liste
    except Exception as e:
        print(f"Otomatik liste alınamadı ({e}).")
    print("Hisse listesi yok: tickers.txt oluştur veya komuta hisse kodu yaz.")
    sys.exit(1)


def kolon_bul(df, ad):
    for c in df.columns:
        if str(c).lower() == ad.lower():
            return c
    raise KeyError(ad)


def analiz(kod):
    df = bp.Ticker(kod).history(period="6mo")
    if df is None or len(df) < 40:
        return None
    kapanis = df[kolon_bul(df, "Close")].astype(float)
    hacim = df[kolon_bul(df, "Volume")].astype(float)

    son5_hacim = hacim.iloc[-5:].mean()
    onceki20_hacim = hacim.iloc[-25:-5].mean()
    if onceki20_hacim <= 0:
        return None
    hacim_oran = son5_hacim / onceki20_hacim

    getiri5 = kapanis.iloc[-1] / kapanis.iloc[-6] - 1
    son15 = kapanis.iloc[-15:]
    bant = (son15.max() - son15.min()) / son15.min()
    zirve60 = kapanis.iloc[-60:].max()
    zirve_orani = kapanis.iloc[-1] / zirve60

    sinyaller = []
    if getiri5 > 0 and hacim_oran >= HACIM_ORAN_MIN:
        sinyaller.append("A:yukselirken_hacim")
    if bant <= YATAY_BANT_MAX and hacim_oran >= HACIM_ORAN_MIN:
        sinyaller.append("B:yatay_birikim")

    return {
        "hisse": kod,
        "sinyal": ",".join(sinyaller),
        "hacim_orani": round(hacim_oran, 2),
        "getiri_5g_%": round(getiri5 * 100, 1),
        "bant_15g_%": round(bant * 100, 1),
        "zirveye_oran_%": round(zirve_orani * 100, 1),
        "tepede": "EVET" if zirve_orani >= TEPE_ESIK else "hayir",
    }


def main():
    from concurrent.futures import ThreadPoolExecutor

    kodlar = hisseleri_al()

    def guvenli(kod):
        try:
            return analiz(kod)
        except Exception as e:
            print(f"{kod}: atlandı ({e})")
            return None

    with ThreadPoolExecutor(max_workers=4) as ex:
        sonuc = [r for r in ex.map(guvenli, kodlar) if r]

    if not sonuc:
        print("Sonuç yok.")
        return

    tablo = pd.DataFrame(sonuc)
    sinyalli = tablo[tablo["sinyal"] != ""].sort_values("hacim_orani", ascending=False)

    pd.set_option("display.width", 200)
    print(f"Taranan: {len(tablo)} hisse | Sinyal veren: {len(sinyalli)}\n")
    if len(sinyalli):
        print(sinyalli.to_string(index=False))
    else:
        print("Bugün sinyal veren hisse yok.")
    tablo.to_csv("hacim_birikim_sonuc.csv", index=False, encoding="utf-8-sig")
    print("\nTüm sonuçlar hacim_birikim_sonuc.csv dosyasına yazıldı.")


if __name__ == "__main__":
    main()
