"""'Geri alim haberi cikan gun al' testi (kap_geri_alim_topla.py ciktisini kullanir).

Soru: Bir sirket kendi payini geri almaya basladigini / geri alim yaptigini KAP'ta duyurdugunda
ve o gunden sonra giris yapsaydik, endekse gore ne kazanirdik?

Olay tanimlari (metin ayristirmaya az bagimli, geriye bakis yanliligi yok):
  ilk     : bir hissede, 180 gun sessizlikten sonra gelen ilk geri alim kaydi (yeni geri alim donemi)
  baslat  : baslik/ozetinde "baslat" gecen kayit (program baslatildi), ayni hissede 30 gun arayla
  islem   : gunluk geri alim islemi bildirimi, ayni hissede 60 gun sessizlikten sonra ilk (aktif alim basladi)
  sonlan  : program sonlandirma bildirimi (30 gun arayla)
  elden   : geri alinan paylarin elden cikarilmasi (satis) (30 gun arayla)

Kurallar:
  - Giris: bildirimin yayin gununden SONRAKI ilk islem gununun kapanisi.
  - Cikis: giristen 5 / 10 / 20 / 40 / 60 islem gunu sonra. Sonuc: XU100'e gore fazla getiri.
  - Karsilastirma (taban): ayni hisselerin rastgele gunleri (her 10 gunde bir ornek).
  - Maliyet dusulmemistir. Fiyati alinamayan (ornegin borsadan cikmis) hisseler disarida kalir.

Cikti: kap_geri_alim_rapor.txt, kap_geri_alim_olaylar.csv, kap_geri_alim_guncel.csv
Gereken: pip install borsapy pandas numpy requests
"""
import argparse
import functools
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import kap_probe2 as kp

print = functools.partial(print, flush=True)
warnings.filterwarnings("ignore")

try:
    import borsapy as bp
except ImportError:
    sys.exit("borsapy kurulu degil: pip install borsapy")

HORIZONLAR = (5, 10, 20, 40, 60)
BASELINE_ADIM = 10
ISCI = 6
GUNCEL_GUN = 20          # 'guncel' listede son kac takvim gunu

OLAYLAR = {
    "ilk": "yeni geri alim donemi (180 gun sessizlik sonrasi ilk kayit)",
    "baslat": "program baslatildi (metinde 'baslat')",
    "islem": "aktif alim basladi (60 gun sessizlik sonrasi ilk gunluk islem)",
    "sonlan": "program sonlandirildi",
    "elden": "geri alinan pay elden cikarildi (satis)",
}


def kolon(df, adlar, poz):
    s = None
    for c in adlar:
        if c in df.columns:
            s = df[c].astype(float)
            break
    if s is None:
        s = df.iloc[:, poz].astype(float)
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    return s[~s.index.duplicated()].sort_index().dropna()


def fiyat_al(kod):
    hata = None
    for deneme in range(3):
        try:
            df = bp.Ticker(kod).history(period="max")
            if df is None or len(df) < 100:
                return kod, None, "kisa veri"
            return kod, kolon(df, ("Close", "close"), 3), None
        except Exception as e:  # noqa: BLE001
            hata = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (deneme + 1))
    return kod, None, hata


def giris_pos(index, zaman):
    """Yayin gununden SONRAKI ilk islem gunu."""
    return int(index.searchsorted(pd.Timestamp(zaman).normalize(), side="right"))


def aralikli(zamanlar, bosluk):
    """Sirali zamanlardan, bir oncekinden en az 'bosluk' gun sonra gelenleri (ve ilkini) dondurur."""
    out, son = [], None
    for z in zamanlar:
        if son is None or (z - son).days >= bosluk:
            out.append(z)
        son = z
    return out


def olaylari_cikar(b):
    ol = []
    for kod, g in b.groupby("kod"):
        g = g.sort_values("zaman")
        gerial = g[g["tur2"] != "elden"]
        for z in aralikli(list(gerial["zaman"]), 180):
            ol.append((kod, z, "ilk"))
        for z in aralikli(list(g[g["tur2"] == "baslat"]["zaman"]), 30):
            ol.append((kod, z, "baslat"))
        for z in aralikli(list(g[g["tur2"] == "islem"]["zaman"]), 60):
            ol.append((kod, z, "islem"))
        for z in aralikli(list(g[g["tur2"] == "sonlan"]["zaman"]), 30):
            ol.append((kod, z, "sonlan"))
        for z in aralikli(list(g[g["tur2"] == "elden"]["zaman"]), 30):
            ol.append((kod, z, "elden"))
    return pd.DataFrame(ol, columns=["kod", "zaman", "olay"])


def tur_belirle(r):
    k = kp.norm(r["konu"])
    s = kp.norm(r["baslik"] + " " + r["ad"])
    if "elden cikar" in k:
        return "elden"
    if "sonlandir" in s:
        return "sonlan"
    if "baslat" in s:
        return "baslat"
    if r["tur"] == "gunluk islem":
        return "islem"
    return "diger"


def ozet_satir(ad, x):
    x = pd.Series(list(x), dtype=float).dropna()
    if len(x) == 0:
        return f"  {ad:16s} veri yok"
    return (f"  {ad:16s} n={len(x):4d}  ortanca %{x.median() * 100:6.1f}  ort. %{x.mean() * 100:6.1f}  "
            f"pozitif %{(x > 0).mean() * 100:3.0f}")


def welch_t(a, b):
    a = pd.Series(list(a), dtype=float).dropna().values
    b = pd.Series(list(b), dtype=float).dropna().values
    if len(a) < 5 or len(b) < 5:
        return np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kayit", default="kap_geri_alim_kayit.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.kayit, dtype=str).fillna("")
    df["zaman"] = pd.to_datetime(df["yayin"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    eksik = df["zaman"].isna()
    if eksik.any():
        df.loc[eksik, "zaman"] = pd.to_datetime(df.loc[eksik, "yayin"], dayfirst=True, errors="coerce")
    df["tur2"] = df.apply(tur_belirle, axis=1)
    b = df[df["zaman"].notna()].copy()
    b["kod"] = b["kodlar"].str.split(",")
    b = b.explode("kod")
    b["kod"] = b["kod"].str.strip().str.upper()
    b = b[b["kod"] != ""].drop_duplicates(["kod", "no"])

    out = []

    def yaz(*a):
        s = " ".join(str(x) for x in a)
        out.append(s)
        print(s)

    yaz(f"KAP geri alim kaydi: {len(df)}, hisse: {b['kod'].nunique()}, "
        f"tarih: {b['zaman'].min().date()} - {b['zaman'].max().date()}")
    yaz("Tur sayilari:", {k: int(v) for k, v in b["tur2"].value_counts().items()})
    ol = olaylari_cikar(b)
    yaz("Olay sayilari:", {k: int(v) for k, v in ol["olay"].value_counts().items()})
    if ol.empty:
        yaz("Olay yok, duruyorum.")
        open("kap_geri_alim_rapor.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
        return

    kodlar = sorted(ol["kod"].unique())
    print(f"{len(kodlar)} hisse icin fiyat alinacak...")
    xu = kolon(bp.Index("XU100").history(period="max"), ("Close", "close"), 3)
    fiy, hatalar = {}, []
    with ThreadPoolExecutor(max_workers=ISCI) as ex:
        for kod, f, h in ex.map(fiyat_al, kodlar):
            if f is None:
                hatalar.append((kod, h))
            else:
                fiy[kod] = f
    yaz(f"Fiyati alinan hisse: {len(fiy)} / {len(kodlar)}  (alinamayan {len(hatalar)}: "
        f"{', '.join(k for k, _ in hatalar[:15])})")

    seri, sorunlu = {}, []
    for kod, f in fiy.items():
        ort = f.index.intersection(xu.index)
        if len(ort) < 150:
            continue
        ff, xx = f.loc[ort], xu.loc[ort]
        if ff.pct_change().abs().max() > 0.25:
            sorunlu.append(kod)      # bolunme/bedelsiz duzeltilmemis olabilir
            continue
        seri[kod] = (ff, xx)
    if sorunlu:
        yaz(f"Tek gunde >%25 hareket olan (duzeltilmemis olabilir), disarida: {len(sorunlu)} hisse "
            f"({', '.join(sorunlu[:15])})")
    yaz(f"Teste giren hisse: {len(seri)}")

    def ex_getiri(f, x, i, h):
        if i < 0 or i + h >= len(f):
            return np.nan
        return (f.iloc[i + h] / f.iloc[i] - 1) - (x.iloc[i + h] / x.iloc[i] - 1)

    satirlar = []
    for r in ol.itertuples():
        if r.kod not in seri:
            continue
        f, x = seri[r.kod]
        i0 = giris_pos(f.index, r.zaman)
        sat = {"kod": r.kod, "zaman": r.zaman, "olay": r.olay}
        for h in HORIZONLAR:
            sat[f"ex{h}"] = ex_getiri(f, x, i0, h)
        satirlar.append(sat)
    tabl = pd.DataFrame(satirlar)
    tabl.to_csv("kap_geri_alim_olaylar.csv", index=False, encoding="utf-8-sig")

    # Taban: ayni hisselerin rastgele gunleri
    taban = {h: [] for h in HORIZONLAR}
    for kod, (f, x) in seri.items():
        fa, xa = f.values, x.values
        for h in HORIZONLAR:
            idx = np.arange(120, len(fa) - h, BASELINE_ADIM)
            if len(idx):
                taban[h] += list((fa[idx + h] / fa[idx] - 1) - (xa[idx + h] / xa[idx] - 1))

    yaz("\n=== OLAY GUNU GIRIS (XU100'e gore fazla getiri; giris = yayindan sonraki ilk islem gunu kapanisi) ===")
    yaz("t: olay ortalamasi ile taban ortalamasi farkinin Welch t degeri (|t|>2 anlamli sayilir; "
        "ayni hissede ust uste olaylar t'yi sisirebilir)")
    for olay, aciklama in OLAYLAR.items():
        t_o = tabl[tabl["olay"] == olay] if not tabl.empty else tabl
        yaz(f"\n--- {olay}: {aciklama}  (olay n={len(t_o)}, hisse={t_o['kod'].nunique() if len(t_o) else 0})")
        for h in HORIZONLAR:
            e = t_o[f"ex{h}"] if len(t_o) else pd.Series(dtype=float)
            t = pd.Series(taban[h], dtype=float)
            yaz(f"{h} gun:")
            yaz(ozet_satir("olay", e))
            yaz(ozet_satir("taban", t))
            if e.dropna().size and t.dropna().size:
                yaz(f"  fark: ortanca {(e.median() - t.median()) * 100:+.1f} puan, "
                    f"ortalama {(e.mean() - t.mean()) * 100:+.1f} puan, t={welch_t(e, t):+.1f}")

    # Istikrar: yila ve yariya gore (ilk ve baslat, 20 ve 40 gun)
    yaz("\n=== YILA GORE (olay: ilk) ===")
    if not tabl.empty:
        tabl["yil"] = pd.to_datetime(tabl["zaman"]).dt.year
        for h in (20, 40):
            yaz(f"{h} gun:")
            for yil, g in tabl[tabl["olay"] == "ilk"].groupby("yil"):
                yaz(ozet_satir(str(yil), g[f"ex{h}"]))
        yaz("\n=== YARIYA GORE (olay: ilk, 20 ve 40 gun) ===")
        orta = pd.to_datetime(tabl["zaman"]).sort_values().iloc[len(tabl) // 2]
        for h in (20, 40):
            for ad, g in (("1. yari", tabl[(tabl["olay"] == "ilk") & (tabl["zaman"] < orta)]),
                          ("2. yari", tabl[(tabl["olay"] == "ilk") & (tabl["zaman"] >= orta)])):
                yaz(ozet_satir(f"{h}g {ad}", g[f"ex{h}"]))

    # Guncel: son GUNCEL_GUN gunde yeni geri alim donemi / program baslatan hisseler
    yaz(f"\n=== SON {GUNCEL_GUN} GUNDE YENI GERI ALIM (ilk / baslat) ===")
    bugun = pd.Timestamp.today().normalize()
    gun = []
    for r in ol[ol["olay"].isin(["ilk", "baslat"])].itertuples():
        if r.kod not in seri or (bugun - r.zaman.normalize()).days > GUNCEL_GUN:
            continue
        f, x = seri[r.kod]
        i0 = giris_pos(f.index, r.zaman)
        son = len(f) - 1
        if i0 <= son:
            ham = f.iloc[son] / f.iloc[i0] - 1
            fazla = ham - (x.iloc[son] / x.iloc[i0] - 1)
        else:
            ham = fazla = np.nan
        gun.append({"kod": r.kod, "zaman": r.zaman.date(), "olay": r.olay,
                    "giristen_beri_%": round(ham * 100, 1) if pd.notna(ham) else None,
                    "endekse_gore_%": round(fazla * 100, 1) if pd.notna(fazla) else None})
    if gun:
        gd = pd.DataFrame(gun).sort_values("zaman", ascending=False)
        gd.to_csv("kap_geri_alim_guncel.csv", index=False, encoding="utf-8-sig")
        yaz(gd.head(50).to_string(index=False))
    else:
        yaz("Yok.")

    yaz("\nNot: gecmis test, garanti degildir. Borsadan cikmis hisseler fiyat alinamadigi icin disarida "
        "(hayatta kalma yanliligi olabilir). Yatirim tavsiyesi degildir.")
    open("kap_geri_alim_rapor.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
