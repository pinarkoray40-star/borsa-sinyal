"""geri_alim_bh2.py - BH2 sinyali geldiginde son 60 gunde geri alim kaydi var mi?
Soru: geri alim kaydi olan BH2 sinyalleri, olmayanlardan daha iyi mi?
Islem kurali buyuk2 ile ayni (izleyen stop, 84 gun, kapanis). Sadece kapanmis islemler.
Cikti: geri_alim_bh2_rapor.txt, geri_alim_bh2_islemler.csv
Gereken: kap_geri_alim_kayit.csv, bist_symbols.txt
"""
import functools
import sys
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

import buyuk as BY
import buyuk2 as B2
import kap_probe2 as kp

print = functools.partial(print, flush=True)
warnings.filterwarnings("ignore")

PENCERE = 60
BASLANGIC = "2021-01-01"
KAYIT = sys.argv[1] if len(sys.argv) > 1 else "kap_geri_alim_kayit.csv"


def semboller(yol="bist_symbols.txt"):
    out = []
    for s in open(yol, encoding="utf-8").read().replace(",", "\n").split():
        s = s.strip().upper()
        if not s or s.startswith("#"):
            continue
        out.append(s if s.endswith(".IS") else s + ".IS")
    return sorted(set(out))


def tz_sil(x):
    i = pd.to_datetime(x.index)
    if getattr(i, "tz", None) is not None:
        i = i.tz_localize(None)
    x = x.copy()
    x.index = i
    return x


def veri_indir(syms):
    cl, vo = [], []
    for i in range(0, len(syms), 100):
        ch = syms[i:i + 100]
        raw = yf.download(ch, start=BASLANGIC, interval="1d", auto_adjust=True,
                          progress=False, group_by="column", threads=True)
        if raw is None or raw.empty:
            continue
        c, v = raw["Close"], raw["Volume"]
        if isinstance(c, pd.Series):
            c, v = c.to_frame(ch[0]), v.to_frame(ch[0])
        cl.append(tz_sil(c))
        vo.append(tz_sil(v))
        print(f"  indirildi {min(i + 100, len(syms))}/{len(syms)}")
    close = pd.concat(cl, axis=1)
    volume = pd.concat(vo, axis=1)
    close = close.loc[:, ~close.columns.duplicated()].sort_index()
    volume = volume.loc[:, ~volume.columns.duplicated()].sort_index()
    ok = [c for c in close.columns if close[c].notna().sum() >= 130]
    close = close[ok]
    volume = volume.reindex(index=close.index, columns=ok)
    return close, volume


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


def kayitlar(yol):
    df = pd.read_csv(yol, dtype=str).fillna("")
    z = pd.to_datetime(df["yayin"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    e = z.isna()
    if e.any():
        z[e] = pd.to_datetime(df.loc[e, "yayin"], dayfirst=True, errors="coerce")
    df["zaman"] = z
    df["tur2"] = df.apply(tur_belirle, axis=1)
    b = df[df["zaman"].notna()].copy()
    b["kod"] = b["kodlar"].str.split(",")
    b = b.explode("kod")
    b["kod"] = b["kod"].str.strip().str.upper()
    return b[b["kod"] != ""].drop_duplicates(["kod", "no"])


def welch_t(a, b):
    a = pd.Series(a, dtype=float).dropna().values
    b = pd.Series(b, dtype=float).dropna().values
    if len(a) < 5 or len(b) < 5:
        return np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else np.nan


def satir(ad, g, ref=None):
    if len(g) == 0:
        return f"  {ad:34s} n=   0"
    s = (f"  {ad:34s} n={len(g):4d}  ort %{g['pnl'].mean():6.1f}  medyan %{g['pnl'].median():6.1f}  "
         f"kazanan %{(g['pnl'] > 0).mean() * 100:3.0f}")
    d = g["diff"].dropna()
    if len(d):
        s += f"  evrene fark {d.mean():+6.1f}"
    if ref is not None:
        t = welch_t(g["pnl"], ref["pnl"])
        s += f"  t={t:+.1f}"
    return s


def main():
    out = []

    def yaz(*a):
        s = " ".join(str(x) for x in a)
        out.append(s)
        print(s)

    kay = kayitlar(KAYIT)
    kay_hisse = set(kay["kod"])
    z0 = kay["zaman"].min().normalize()
    yaz(f"KAP kaydi: {len(kay)} satir, {len(kay_hisse)} hisse, ilk tarih {z0.date()}")
    zam_hepsi, zam_baslat = {}, {}
    for kod, g in kay.groupby("kod"):
        g = g[g["tur2"] != "elden"]
        zam_hepsi[kod] = np.sort(g["zaman"].dt.normalize().values)
        zam_baslat[kod] = np.sort(g[g["tur2"] == "baslat"]["zaman"].dt.normalize().values)

    syms = semboller()
    print(f"{len(syms)} hisse indirilecek...")
    close, volume = veri_indir(syms)
    yaz(f"Fiyat verisi: {close.shape[1]} hisse, son gun {close.index[-1].date()}")

    P = B2.PARAM
    idx = B2.fetch_index(close)
    _, dr = B2.dayaniklilik(close, idx, P["window"])
    L = BY.likit_maske(close, volume)
    bh = BY.ozellikler(close, volume)
    DR = dr.to_numpy()
    C = close.to_numpy()
    cols = list(close.columns)
    mask = bh & (np.nan_to_num(DR, nan=0.0) >= P["dn_min"])
    ii, jj = BY.sinyaller(mask, P["cooldown"])
    basla = z0 + pd.Timedelta(days=PENCERE)
    yaz(f"BH2 sinyali (tum gecmis): {len(ii)}; kapsam icin sinyal tarihi >= {basla.date()}")

    def var(arr, d):
        if arr is None or len(arr) == 0:
            return False
        lo = np.datetime64(d - pd.Timedelta(days=PENCERE))
        hi = np.datetime64(d)
        k = np.searchsorted(arr, lo, side="right")
        return bool(k < len(arr) and arr[k] <= hi)

    rows, yeni = [], []
    for p, j in zip(ii, jj):
        d = close.index[p]
        if d < basla:
            continue
        kod = cols[j].replace(".IS", "")
        entry = float(C[p, j])
        if not np.isfinite(entry) or entry <= 0:
            continue
        ga = var(zam_hepsi.get(kod), d)
        gb = var(zam_baslat.get(kod), d)
        path = (C[p + 1:p + P["hold"] + 1, j] / entry - 1) * 100
        st, k, xr, peak = BY.takip(path, P["sl"], P["hold"])
        if st == "açık":
            if ga and (close.index[-1] - d).days <= 30:
                yeni.append((kod, d.date(), int(round(float(DR[p, j]) * 100)), gb))
            continue
        k = int(k)
        pnl = xr if xr is not None else (float(path[k - 1]) if k > 0 else 0.0)
        b = BY.kiyas(C, L, p, k)
        rows.append({"kod": kod, "tarih": d, "yil": d.year, "pnl": float(pnl),
                     "diff": (float(pnl) - b) if b is not None else np.nan,
                     "ga60": ga, "baslat60": gb, "kayit_hissesi": kod in kay_hisse,
                     "dnr": int(round(float(DR[p, j]) * 100))})
    t = pd.DataFrame(rows)
    t.to_csv("geri_alim_bh2_islemler.csv", index=False, encoding="utf-8-sig")
    yaz(f"\nKapanmis BH2 islemi (kapsam icinde): {len(t)}")
    if t.empty:
        yaz("Islem yok, duruyorum.")
        open("geri_alim_bh2_rapor.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
        return

    ga_y = t[t["ga60"]]
    ga_n = t[~t["ga60"] & t["kayit_hissesi"]]
    ga_h = t[~t["kayit_hissesi"]]
    yaz("\n=== BH2 + SON 60 GUNDE GERI ALIM KAYDI ===")
    yaz("(t: gruptaki islemlerin ortalamasi, 'geri alim yok (kayit hissesi)' grubuna gore)")
    yaz(satir("TUM BH2", t))
    yaz(satir("geri alim VAR (60g)", ga_y, ga_n))
    yaz(satir("geri alim YOK (kayit hissesi)", ga_n))
    yaz(satir("kayit hissesi degil", ga_h))
    yaz(satir("program BASLATILDI (60g)", t[t["baslat60"]], ga_n))

    yaz("\n=== YILA GORE: geri alim VAR / YOK(kayit hissesi) ===")
    for yil, g in t.groupby("yil"):
        yaz(f"{yil}:")
        yaz(satir("  var", g[g["ga60"]]))
        yaz(satir("  yok", g[~g["ga60"] & g["kayit_hissesi"]]))

    yaz("\n=== SON 30 GUNDE ACIK BH2 SINYALI + GERI ALIM KAYDI VAR ===")
    if yeni:
        for kod, d, dnr, gb in sorted(yeni, key=lambda x: x[1], reverse=True):
            yaz(f"  {kod:8s} {d}  dayaniklilik sirasi %{dnr}  program baslatildi: {'evet' if gb else 'hayir'}")
    else:
        yaz("  yok")

    yaz("\nNot: ornek sayisi az olabilir; t'yi dikkatli oku. Gecmis test, tavsiye degildir.")
    open("geri_alim_bh2_rapor.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
