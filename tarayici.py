"""Günlük tarayıcı + canlı takip defteri.

Her iş günü kapanıştan sonra tüm BIST hisselerini tarar. Kriterlerin hepsini tutan ve
takipte olmayan hisse, o günün kapanış fiyatıyla deftere girer. Girenler, giriş gününden
itibaren 6 ay boyunca günlük % ve girişe göre toplam % ile izlenir. Stop yoktur: toplam
getiri artıdaysa KÂR, eksideyse ZARAR yazar.

Kriterler (hepsi aynı gün tutmalı):
  1. Zirveden uzak  : kapanış, son 52 haftalık zirvenin en az %25 altında
  2. Geç kalınmamış : son 21 işlem gününde en fazla %20 yükselmiş
  3. Dönüş          : son 5 gün içinde 20 günlük ortalamanın altındaydı, bugün üstünde
  4. Hacim          : son 5 günlük ortalama hacim, 20 günlük ortalamanın en az 1,5 katı
  5. Göreceli güç   : son 20 günde BIST 100'den daha iyi
  (+ likidite)      : 20 günlük ortalama işlem hacmi en az 1 milyon TL

Dosyalar (takip/ klasörü):
  pozisyonlar.csv : her hisse için giriş tarihi, giriş fiyatı, bitiş tarihi, durum
  gunluk.csv      : her pozisyon için gün gün kapanış, günlük %, toplam %, KÂR/ZARAR
  ozet.md         : telefondan bakmak için özet tablo

Fiyatlar bedelsiz/bölünme düzeltmelidir (Yahoo auto_adjust); toplam % bu yüzden doğrudur,
ama bedelsizden sonra kapanış sütunu eski giriş fiyatından düşük görünebilir.

Giriş fiyatı sinyalin çıktığı günün kapanışıdır (gerçekte o fiyattan alınamayabilir).
Yatırım tavsiyesi değildir.
"""
import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from signals import fetch_all, read_symbols

INDEX = "XU100.IS"

P = dict(
    uzak=-0.25,        # zirveye göre en fazla bu kadar uzakta olmalı (<= -%25)
    ay_max=0.20,       # 21 günlük getiri bu değerin altında olmalı
    sma=20,            # ortalama penceresi
    dondu_gun=5,       # kaç gün içinde ortalamanın altından çıkmış olmalı
    hacim_kat=1.5,     # 5g hacim / 20g hacim
    min_tl=1_000_000,  # 20 günlük ortalama günlük işlem hacmi (TL)
    ay=6,              # takip süresi (ay)
)

POZ_COLS = ["sembol", "giris_tarihi", "giris_fiyati", "bitis_tarihi", "durum"]
GUN_COLS = ["sembol", "giris_tarihi", "tarih", "kapanis", "gunluk_pct", "toplam_pct", "sonuc"]


def sonuc(x: float) -> str:
    return "KÂR" if x > 0 else ("ZARAR" if x < 0 else "BAŞABAŞ")


# --------------------------------------------------------------------------- veri

def temizle_indeks(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx.normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


def eksik_gunu_at(close: pd.DataFrame, volume: pd.DataFrame, simdi=None):
    """Seans bitmeden çalışıldıysa bugünün yarım mumunu at (kapanış sayılmaz)."""
    simdi = simdi or datetime.now(ZoneInfo("Europe/Istanbul"))
    bugun = pd.Timestamp(simdi.date())
    seans_bitti = simdi.hour > 18 or (simdi.hour == 18 and simdi.minute >= 20)
    if simdi.weekday() < 5 and not seans_bitti and len(close) and close.index[-1] == bugun:
        return close.iloc[:-1], volume.iloc[:-1]
    return close, volume


# --------------------------------------------------------------------------- tarama

def tara(close: pd.DataFrame, volume: pd.DataFrame, endeks: pd.Series, p: dict = P) -> pd.DataFrame:
    """Son gün için kriterleri tutan hisseleri döndürür (ayrıntılı sütunlarla)."""
    sma = close.rolling(p["sma"]).mean()
    ust = close > sma
    alt = (close <= sma).astype(float)
    dondu = ust & (alt.shift(1).rolling(p["dondu_gun"]).max() >= 1)

    zirve = close.rolling(252, min_periods=130).max()
    uzaklik = close / zirve - 1
    ay_getiri = close / close.shift(21) - 1

    v5 = volume.rolling(5).mean()
    v20 = volume.rolling(20).mean()
    hacim_kat = v5 / v20

    g20 = close / close.shift(20) - 1
    e20 = endeks / endeks.shift(20) - 1
    goreceli = g20.sub(e20, axis=0)

    tl = (close * volume).rolling(20).mean()

    kosul = (
        (uzaklik <= p["uzak"])
        & (ay_getiri <= p["ay_max"])
        & dondu
        & (hacim_kat >= p["hacim_kat"])
        & (goreceli > 0)
        & (tl >= p["min_tl"])
    )
    son = close.index[-1]
    secili = kosul.loc[son]
    secili = secili[secili].index
    out = pd.DataFrame({
        "sembol": list(secili),
        "kapanis": [close.at[son, s] for s in secili],
        "zirveden_uzak_pct": [round(uzaklik.at[son, s] * 100, 1) for s in secili],
        "ay_getiri_pct": [round(ay_getiri.at[son, s] * 100, 1) for s in secili],
        "hacim_kat": [round(hacim_kat.at[son, s], 2) for s in secili],
        "goreceli_pct": [round(goreceli.at[son, s] * 100, 1) for s in secili],
    })
    return out


def tarama_say(close, volume, endeks, p: dict = P) -> dict:
    """Her kriteri tek tek kaç hissenin tuttuğunu sayar (eşikleri ayarlamak için)."""
    sma = close.rolling(p["sma"]).mean()
    ust = close > sma
    alt = (close <= sma).astype(float)
    dondu = ust & (alt.shift(1).rolling(p["dondu_gun"]).max() >= 1)
    zirve = close.rolling(252, min_periods=130).max()
    kos = {
        "zirveden en az %25 uzak": (close / zirve - 1) <= p["uzak"],
        "21g getiri <= %20": (close / close.shift(21) - 1) <= p["ay_max"],
        "ortalamayı yukarı kesmiş": dondu,
        "hacim >= 1,5x": (volume.rolling(5).mean() / volume.rolling(20).mean()) >= p["hacim_kat"],
        "endeksten güçlü (20g)": (close / close.shift(20) - 1).sub(endeks / endeks.shift(20) - 1, axis=0) > 0,
        "likidite": (close * volume).rolling(20).mean() >= p["min_tl"],
    }
    return {ad: int(k.iloc[-1].sum()) for ad, k in kos.items()}


# --------------------------------------------------------------------------- defter

def oku_csv(path: Path, cols: list) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path, dtype={"sembol": str})
    return pd.DataFrame(columns=cols)


def yeni_girisler(poz: pd.DataFrame, sinyal: pd.DataFrame, gun: pd.Timestamp, p: dict = P) -> pd.DataFrame:
    """Takipte olmayan sinyalleri pozisyon listesine ekler."""
    takipte = set(poz.loc[poz["durum"] == "TAKIPTE", "sembol"]) if len(poz) else set()
    satirlar = []
    for _, r in sinyal.iterrows():
        if r["sembol"] in takipte:
            continue
        satirlar.append({
            "sembol": r["sembol"],
            "giris_tarihi": gun.strftime("%Y-%m-%d"),
            "giris_fiyati": round(float(r["kapanis"]), 4),
            "bitis_tarihi": (gun + pd.DateOffset(months=p["ay"])).strftime("%Y-%m-%d"),
            "durum": "TAKIPTE",
        })
    if not satirlar:
        return poz
    yeni = pd.DataFrame(satirlar, columns=POZ_COLS)
    return pd.concat([poz, yeni], ignore_index=True) if len(poz) else yeni


def gunluk_satirlar(poz: pd.Series, seri: pd.Series) -> pd.DataFrame:
    """Bir pozisyonun giriş gününden bitişe kadar gün gün kaydı."""
    giris = pd.Timestamp(poz["giris_tarihi"])
    bitis = pd.Timestamp(poz["bitis_tarihi"])
    s = seri.dropna()
    s = s[(s.index >= giris) & (s.index <= bitis)]
    if giris not in s.index or s.empty:
        return pd.DataFrame(columns=GUN_COLS)
    taban = s.loc[giris]
    toplam = (s / taban - 1) * 100
    gunluk = s.pct_change() * 100
    gunluk.iloc[0] = 0.0
    df = pd.DataFrame({
        "sembol": poz["sembol"],
        "giris_tarihi": poz["giris_tarihi"],
        "tarih": s.index.strftime("%Y-%m-%d"),
        "kapanis": s.round(4).values,
        "gunluk_pct": gunluk.round(2).values,
        "toplam_pct": toplam.round(2).values,
    })
    df["sonuc"] = [sonuc(x) for x in df["toplam_pct"]]
    return df[GUN_COLS]


def defteri_guncelle(poz: pd.DataFrame, gun_eski: pd.DataFrame, close: pd.DataFrame):
    """TAKIPTE pozisyonların günlük kaydını yeniden kurar; süresi dolanları TAMAMLANDI yapar.

    Tamamlananların satırlarına dokunulmaz (veri penceresi dışına çıksalar bile korunur).
    """
    son = close.index[-1]
    poz = poz.copy()
    parcalar = []
    for i, r in poz.iterrows():
        eski = gun_eski[(gun_eski["sembol"] == r["sembol"]) & (gun_eski["giris_tarihi"] == r["giris_tarihi"])]
        if r["durum"] == "TAMAMLANDI":
            parcalar.append(eski)
            continue
        if r["sembol"] not in close.columns:
            print(f"  uyarı: {r['sembol']} veride yok, eski kayıt korundu")
            parcalar.append(eski)
            continue
        yeni = gunluk_satirlar(r, close[r["sembol"]])
        if yeni.empty:
            print(f"  uyarı: {r['sembol']} giriş günü ({r['giris_tarihi']}) veride bulunamadı, eski kayıt korundu")
            parcalar.append(eski)
            continue
        parcalar.append(yeni)
        if son >= pd.Timestamp(r["bitis_tarihi"]):
            poz.at[i, "durum"] = "TAMAMLANDI"
    parcalar = [d for d in parcalar if len(d)]
    gun = pd.concat(parcalar, ignore_index=True) if parcalar else pd.DataFrame(columns=GUN_COLS)
    return poz, gun


# --------------------------------------------------------------------------- özet

def ozet_yaz(path: Path, poz: pd.DataFrame, gun: pd.DataFrame, bugun_giren: list, tarih: pd.Timestamp, say: dict):
    L = [f"# Takip özeti — {tarih.strftime('%d.%m.%Y')}", ""]
    L.append("Bugün giren: " + (", ".join(bugun_giren) if bugun_giren else "yok"))
    L.append("")

    if len(gun):
        son_satir = gun.sort_values("tarih").groupby(["sembol", "giris_tarihi"], as_index=False).tail(1)
        m = poz.merge(son_satir, on=["sembol", "giris_tarihi"], how="left")
    else:
        m = poz.assign(tarih=None, kapanis=None, gunluk_pct=None, toplam_pct=None, sonuc=None)

    takipte = m[m["durum"] == "TAKIPTE"].sort_values("toplam_pct", ascending=False)
    biten = m[m["durum"] == "TAMAMLANDI"].sort_values("toplam_pct", ascending=False)

    if len(takipte):
        kar = int((takipte["toplam_pct"] > 0).sum())
        zarar = int((takipte["toplam_pct"] < 0).sum())
        L.append(f"Takipte: {len(takipte)} hisse | KÂR: {kar} | ZARAR: {zarar} | "
                 f"ortalama: %{takipte['toplam_pct'].mean():.2f}")
    else:
        L.append("Takipte hisse yok.")
    L.append("")

    if len(takipte):
        L += ["| Hisse | Giriş | Giriş ₺ | Son ₺ | Günlük % | Toplam % | Durum | Bitiş |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in takipte.iterrows():
            L.append(f"| {r['sembol']} | {r['giris_tarihi']} | {r['giris_fiyati']:.2f} | "
                     f"{r['kapanis']:.2f} | {r['gunluk_pct']:+.2f} | {r['toplam_pct']:+.2f} | "
                     f"{r['sonuc']} | {r['bitis_tarihi']} |")
        L.append("")

    if len(biten):
        L += ["## Süresi dolanlar", "",
              "| Hisse | Giriş | Bitiş | Toplam % | Sonuç |", "|---|---|---|---|---|"]
        for _, r in biten.iterrows():
            L.append(f"| {r['sembol']} | {r['giris_tarihi']} | {r['bitis_tarihi']} | "
                     f"{r['toplam_pct']:+.2f} | {r['sonuc']} |")
        L.append("")

    L += ["## Bugünkü tarama (hangi kriteri kaç hisse tuttu)", ""]
    L += [f"- {ad}: {n}" for ad, n in say.items()]
    L.append("")
    L.append("Giriş fiyatı sinyal gününün kapanışıdır. Fiyatlar bedelsiz/bölünme düzeltmelidir. "
             "Yatırım tavsiyesi değildir.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- ana

def main():
    ap = argparse.ArgumentParser(description="Günlük tarayıcı ve takip defteri")
    ap.add_argument("--period", default="2y")
    ap.add_argument("--symbols-file", default="bist_symbols.txt")
    ap.add_argument("--dir", default="takip")
    args = ap.parse_args()

    klasor = Path(args.dir)
    klasor.mkdir(parents=True, exist_ok=True)

    symbols = read_symbols(args.symbols_file)
    if INDEX not in symbols:
        symbols.append(INDEX)
    print(f"{len(symbols)} sembol indiriliyor ({args.period})...")
    close, volume = fetch_all(symbols, period=args.period)
    close, volume = temizle_indeks(close), temizle_indeks(volume)
    close, volume = eksik_gunu_at(close, volume)
    if INDEX not in close.columns:
        raise SystemExit("BIST 100 endeksi verisi alınamadı.")

    endeks = close[INDEX]
    close = close.drop(columns=[INDEX])
    volume = volume.drop(columns=[INDEX], errors="ignore")
    close.columns = [c.replace(".IS", "") for c in close.columns]
    volume.columns = [c.replace(".IS", "") for c in volume.columns]
    gun = close.index[-1]
    print(f"Son kapanış günü: {gun.date()}  ({close.shape[1]} hisse)")

    poz = oku_csv(klasor / "pozisyonlar.csv", POZ_COLS)
    gun_eski = oku_csv(klasor / "gunluk.csv", GUN_COLS)

    sinyal = tara(close, volume, endeks)
    say = tarama_say(close, volume, endeks)
    print("Kriter sayıları:", say)
    print(f"Tüm kriterleri tutan: {len(sinyal)}")

    onceki = len(poz)
    poz = yeni_girisler(poz, sinyal, gun)
    yeniler = list(poz["sembol"].iloc[onceki:]) if len(poz) > onceki else []
    poz, gun_df = defteri_guncelle(poz, gun_eski, close)

    poz.to_csv(klasor / "pozisyonlar.csv", index=False, encoding="utf-8")
    gun_df.to_csv(klasor / "gunluk.csv", index=False, encoding="utf-8")
    ozet_yaz(klasor / "ozet.md", poz, gun_df, yeniler, gun, say)
    print(f"Yeni giren: {yeniler or 'yok'} | toplam pozisyon: {len(poz)}")


if __name__ == "__main__":
    main()
