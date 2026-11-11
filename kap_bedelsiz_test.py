"""'Haber cikan gun al' bedelsiz testi (kap_bedelsiz_topla.py ciktisini kullanir).

Soru: Bir sirket bedelsiz sermaye artirimi icin yonetim kurulu karari / SPK basvurusu
duyurdugunda (KAP) ve o gunden sonra giris yapsaydik, endekse gore ne kazanirdik?

Kurallar (geriye bakis yanliligi olmasin diye):
  - Giris: bildirimin yayin gununden SONRAKI ilk islem gununun kapanisi.
  - Cikis: giristen 5 / 10 / 20 / 40 / 60 islem gunu sonra. Sonuc: XU100'e gore fazla getiri.
  - Sonuc, SPK onayi gelsin ya da gelmesin hepsini sayar (onay gelmeyenler de dahil).
  - Karsilastirma: ayni hisselerin rastgele gunleri (her 10 gunde bir ornek).
  - Maliyet dusulmemistir.

Ayrica: basvurudan onaya ve onaydan tescile kac gun gectigi, onay haberinden sonrasi,
ve bugun SPK onayi/tescili beklenen bedelsiz listesi.

Cikti: kap_bedelsiz_rapor.txt, kap_bedelsiz_donguler.csv, kap_bedelsiz_bekleyen.csv
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
ONAY_HORIZONLAR = (1, 3, 5, 10, 20)
BASELINE_ADIM = 10
ISCI = 6
ESKI_GUN = 400          # bu kadar gun sessiz kalan surec "eski" sayilir
BEKLEYEN_GUN = 240      # bekleyen listede en fazla bu kadar eski


def asama(baslik):
    t = kp.norm(baslik)
    if "bedelsiz" not in t or "bedelli" in t or "tahsisli" in t:
        return None
    if "duzelt" in t:
        return None
    if "vazgec" in t or "iptal" in t:
        return "iptal"
    if "tescil" in t:
        return "tescil"
    if "onay" in t:
        return "onay"
    if "basvuru" in t:
        return "basvuru"
    if "yonetim kurulu" in t:
        return "yk"
    return "diger"


def donguler(kay):
    out = []
    for kod, g in kay.groupby("kod"):
        cur = None
        for r in g.sort_values("zaman").itertuples():
            a = r.asama
            if cur is not None and (r.zaman - cur["son"]).days > ESKI_GUN:
                cur["bitis"] = "eski"
                out.append(cur)
                cur = None
            if a in ("yk", "basvuru"):
                if cur is None:
                    cur = {"kod": kod, "baslangic": r.zaman, "tur": a, "onay": None, "tescil": None,
                           "son": r.zaman, "bitis": None}
                else:
                    cur["son"] = r.zaman
            elif a == "onay":
                if cur is None:
                    cur = {"kod": kod, "baslangic": r.zaman, "tur": "onay", "onay": r.zaman, "tescil": None,
                           "son": r.zaman, "bitis": None}
                elif cur["onay"] is None:
                    cur["onay"] = r.zaman
                    cur["son"] = r.zaman
            elif a == "tescil":
                if cur is None:
                    cur = {"kod": kod, "baslangic": r.zaman, "tur": "tescil", "onay": None, "tescil": r.zaman,
                           "son": r.zaman, "bitis": None}
                else:
                    cur["tescil"] = r.zaman
                    cur["son"] = r.zaman
                cur["bitis"] = "tescil"
                out.append(cur)
                cur = None
            elif a == "iptal":
                if cur is not None:
                    cur["bitis"] = "iptal"
                    out.append(cur)
                    cur = None
        if cur is not None:
            cur["bitis"] = "acik"
            out.append(cur)
    return pd.DataFrame(out)


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


def ozet_satir(ad, x):
    x = pd.Series(list(x), dtype=float).dropna()
    if len(x) == 0:
        return f"  {ad:14s} veri yok"
    return (f"  {ad:14s} n={len(x):4d}  ortanca %{x.median() * 100:6.1f}  ort. %{x.mean() * 100:6.1f}  "
            f"pozitif %{(x > 0).mean() * 100:3.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kayit", default="kap_bedelsiz_kayit.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.kayit, dtype=str).fillna("")
    df["zaman"] = pd.to_datetime(df["yayin"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    eksik = df["zaman"].isna()
    if eksik.any():
        df.loc[eksik, "zaman"] = pd.to_datetime(df.loc[eksik, "yayin"], dayfirst=True, errors="coerce")
    df["asama"] = df["baslik"].map(asama)
    b = df[df["asama"].notna() & df["zaman"].notna()].copy()
    b["kod"] = b["kodlar"].str.split(",")
    b = b.explode("kod")
    b["kod"] = b["kod"].str.strip().str.upper()
    b = b[b["kod"] != ""].drop_duplicates(["kod", "zaman", "asama"])

    out = []

    def yaz(*a):
        s = " ".join(str(x) for x in a)
        out.append(s)
        print(s)

    yaz(f"KAP kayit: {len(df)} (konuya uyan), bedelsiz asamali: {len(b)}, hisse: {b['kod'].nunique()}")
    yaz("Asama sayilari:", {k: int(v) for k, v in b["asama"].value_counts().items()})
    if b.empty:
        yaz("Bedelsiz kaydi yok, duruyorum.")
        open("kap_bedelsiz_rapor.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
        return

    dg = donguler(b)
    for c in ("baslangic", "onay", "tescil", "son"):
        dg[c] = pd.to_datetime(dg[c])
    yaz(f"Surec (dongu) sayisi: {len(dg)}  bitis: {({k: int(v) for k, v in dg['bitis'].value_counts().items()})}  "
        f"baslangic turu: {({k: int(v) for k, v in dg['tur'].value_counts().items()})}")
    dg["gun_onay"] = (dg["onay"] - dg["baslangic"]).dt.days
    dg["gun_tescil"] = (dg["tescil"] - dg["onay"]).dt.days
    tamam = dg[dg["gun_onay"].notna() & dg["tur"].isin(["yk", "basvuru"])]
    if len(tamam):
        yaz(f"Baslangictan SPK onayina gecen gun: ortanca {tamam['gun_onay'].median():.0f}, "
            f"ceyrekler {tamam['gun_onay'].quantile(.25):.0f}-{tamam['gun_onay'].quantile(.75):.0f} (n={len(tamam)})")
    tc = dg[dg["gun_tescil"].notna()]
    if len(tc):
        yaz(f"SPK onayindan tescile gecen gun: ortanca {tc['gun_tescil'].median():.0f} (n={len(tc)})")

    # Fiyatlar
    kodlar = sorted(dg["kod"].unique())
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
        yaz(f"Fiyat serisinde tek gunde >%25 hareket olan (duzeltilmemis olabilir), disarida: {', '.join(sorunlu)}")

    def ex_getiri(f, x, i, h):
        if i < 0 or i + h >= len(f):
            return np.nan
        return (f.iloc[i + h] / f.iloc[i] - 1) - (x.iloc[i + h] / x.iloc[i] - 1)

    satirlar, ev = [], {h: [] for h in HORIZONLAR}
    onayg = {h: [] for h in ONAY_HORIZONLAR}
    bekleme = []
    for r in dg.itertuples():
        if r.kod not in seri:
            continue
        f, x = seri[r.kod]
        sat = {"kod": r.kod, "baslangic": r.baslangic, "tur": r.tur, "onay": r.onay, "tescil": r.tescil,
               "bitis": r.bitis, "gun_onay": r.gun_onay, "gun_tescil": r.gun_tescil}
        if r.tur in ("yk", "basvuru"):
            i0 = giris_pos(f.index, r.baslangic)
            for h in HORIZONLAR:
                v = ex_getiri(f, x, i0, h)
                sat[f"ex{h}"] = v
                if not np.isnan(v):
                    ev[h].append(v)
            if pd.notna(r.onay):
                j = int(f.index.searchsorted(pd.Timestamp(r.onay).normalize(), side="left"))
                if 0 <= i0 < len(f) and j < len(f) and j > i0:
                    bekleme.append((f.iloc[j] / f.iloc[i0] - 1) - (x.iloc[j] / x.iloc[i0] - 1))
        if pd.notna(r.onay):
            io = giris_pos(f.index, r.onay)
            for h in ONAY_HORIZONLAR:
                v = ex_getiri(f, x, io, h)
                sat[f"onay_ex{h}"] = v
                if not np.isnan(v):
                    onayg[h].append(v)
        satirlar.append(sat)
    pd.DataFrame(satirlar).to_csv("kap_bedelsiz_donguler.csv", index=False, encoding="utf-8-sig")

    # Taban: ayni hisselerin rastgele gunleri
    taban = {h: [] for h in HORIZONLAR}
    for kod, (f, x) in seri.items():
        fa, xa = f.values, x.values
        for h in HORIZONLAR:
            idx = np.arange(120, len(fa) - h, BASELINE_ADIM)
            if len(idx):
                taban[h] += list((fa[idx + h] / fa[idx] - 1) - (xa[idx + h] / xa[idx] - 1))

    yaz("\n=== HABER GUNU GIRIS: yonetim kurulu karari / SPK basvurusu duyuruldu (XU100'e gore fazla getiri) ===")
    yaz("Giris: yayin gununden sonraki ilk islem gunu kapanisi. Onay gelenler de gelmeyenler de dahil.")
    for h in HORIZONLAR:
        e = pd.Series(ev[h], dtype=float).dropna()
        t = pd.Series(taban[h], dtype=float).dropna()
        yaz(f"\n{h} islem gunu sonra:")
        yaz(ozet_satir("bedelsiz haberi", e))
        yaz(ozet_satir("taban (rastgele)", t))
        if len(e) and len(t):
            yaz(f"  fark (ortanca): %{(e.median() - t.median()) * 100:+.1f} puan")

    if bekleme:
        yaz("\n=== Bekleme donemi: giristen SPK onayi gunu kapanisina (sadece onay gelenler) ===")
        yaz(ozet_satir("giris -> onay", bekleme))

    yaz("\n=== SPK onayi haberinden sonra (onay gununden sonraki ilk gun kapanisindan giris) ===")
    for h in ONAY_HORIZONLAR:
        yaz(ozet_satir(f"onaydan +{h} gun", onayg[h]))

    # Istikrar: yila gore, 20 gun
    yaz("\n=== Yila gore (20 islem gunu, haber gunu giris) ===")
    tabl = pd.DataFrame(satirlar)
    if "ex20" in tabl:
        tabl["yil"] = pd.to_datetime(tabl["baslangic"]).dt.year
        for yil, g in tabl[tabl["tur"].isin(["yk", "basvuru"])].groupby("yil"):
            yaz(ozet_satir(str(yil), g["ex20"]))

    # Bugun bekleyenler
    yaz("\n=== BUGUN SPK ONAYI / TESCIL BEKLEYEN bedelsiz sureclari ===")
    bek = []
    for r in dg.itertuples():
        if r.bitis != "acik" or r.tur not in ("yk", "basvuru") or r.kod not in seri:
            continue
        if (pd.Timestamp.today().normalize() - r.baslangic.normalize()).days > BEKLEYEN_GUN:
            continue
        f, x = seri[r.kod]
        i0 = giris_pos(f.index, r.baslangic)
        son = len(f) - 1
        if i0 <= son:
            ham = f.iloc[son] / f.iloc[i0] - 1
            fazla = ham - (x.iloc[son] / x.iloc[i0] - 1)
        else:
            ham, fazla = np.nan, np.nan
        bek.append({"kod": r.kod, "baslangic": r.baslangic.date(), "tur": r.tur,
                    "gun": (pd.Timestamp.today().normalize() - r.baslangic.normalize()).days,
                    "onay_geldi": "evet" if pd.notna(r.onay) else "hayir",
                    "giristen_beri_%": round(ham * 100, 1) if pd.notna(ham) else None,
                    "endekse_gore_%": round(fazla * 100, 1) if pd.notna(fazla) else None})
    if bek:
        bd = pd.DataFrame(bek).sort_values("baslangic", ascending=False)
        bd.to_csv("kap_bedelsiz_bekleyen.csv", index=False, encoding="utf-8-sig")
        yaz(bd.head(60).to_string(index=False))
    else:
        yaz("Bekleyen surec bulunamadi.")

    yaz("\nNot: gecmis test, garanti degildir. Yatirim tavsiyesi degildir.")
    open("kap_bedelsiz_rapor.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
