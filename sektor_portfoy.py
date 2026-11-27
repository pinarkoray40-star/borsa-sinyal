import glob
import warnings

import numpy as np
import pandas as pd

import portfoy2 as P
from sektor_test import grup_idlerini_kur, sektor_gucu, H

warnings.filterwarnings("ignore")

B = P.B
B2 = P.B2

KONF = [
    ("N=10, maliyet %0,8", 10, 0.004),
    ("N=20, maliyet %0,8", 20, 0.004),
    ("N=30, maliyet %0,8", 30, 0.004),
    ("N=20, maliyet %1,5", 20, 0.0075),
]


def kirpik_ort(x, p=0.05):
    x = np.sort(x)
    k = int(len(x) * p)
    return x[:len(x) - k].mean() if len(x) - k > 0 else float("nan")


def yogunlasma(ii, jj, guc60, alt_id, alt_ad, F, evr, yillar):
    ii, jj = np.asarray(ii), np.asarray(jj)
    ok = ii < len(F)
    ii, jj = ii[ok], jj[ok]
    r = np.clip(F[ii, jj], -0.9, 3.0)
    g = guc60[ii, jj]
    iyi = np.isfinite(r) & np.isfinite(evr[ii]) & np.isfinite(g)
    fark = ((r - evr[ii]) * 100)[iyi]
    ii, jj, g = ii[iyi], jj[iyi], g[iyi]
    yil = yillar[ii]
    ad = np.array([alt_ad[alt_id[j]] for j in jj])
    guclu = g > 0
    zayif = ~guclu

    print("=" * 100)
    print(f"A) YOGUNLASMA KONTROLU (BH2, alt sektor gucu 60g, {len(fark)} sinyal)")
    print("-" * 100)
    for baslik, m in (("GUCLU", guclu), ("ZAYIF", zayif)):
        x = fark[m]
        print(f"{baslik:<6} n={len(x):<4} ort {x.mean():+6.1f}  en iyi %5 cikarilinca {kirpik_ort(x):+6.1f}"
              f"  medyan {np.median(x):+6.1f}  kazanan %{(x > 0).mean() * 100:3.0f}")

    print("-" * 100)
    print("YIL YIL (ort fark, puan):")
    for y in sorted(set(yil)):
        a = fark[guclu & (yil == y)]
        b = fark[zayif & (yil == y)]
        sa = f"{a.mean():+7.1f} (n={len(a)})" if len(a) else "      -"
        sb = f"{b.mean():+7.1f} (n={len(b)})" if len(b) else "      -"
        print(f"  {y}  guclu {sa:<18} zayif {sb}")

    print("-" * 100)
    print("GUCLU sinyallerde en kalabalik 10 alt sektor (n, ort fark):")
    seri = pd.Series(ad[guclu])
    top = seri.value_counts()
    for k in top.index[:10]:
        x = fark[guclu & (ad == k)]
        print(f"  {k:<34} n={len(x):<4} ort {x.mean():+7.1f}  medyan {np.median(x):+6.1f}")
    print(f"  toplam {top.shape[0]} farkli alt sektor; en kalabalik 3'un payi %{top.iloc[:3].sum() / top.sum() * 100:.0f}")

    print("-" * 100)
    cikar = set(top.index[:3])
    hr = np.array([a not in cikar for a in ad])
    a, b = fark[guclu & hr], fark[zayif & hr]
    print(f"EN KALABALIK 3 ALT SEKTOR CIKARILINCA: guclu n={len(a)} ort {a.mean():+.1f} "
          f"medyan {np.median(a):+.1f} | zayif n={len(b)} ort {b.mean():+.1f} "
          f"medyan {np.median(b):+.1f}")


def portfoy_testi(close, opn, kume, P_C, sk):
    C = close.to_numpy()
    O = opn.to_numpy()
    idx = close.index
    evs = {}
    for ad, m in kume.items():
        ii, jj = B.sinyaller(m, P.CD)
        evs[ad] = P.olaylar(C, O, ii, jj, P.C1, sk)
    print("=" * 100)
    print("B) PORTFOY TESTI. sonuclanan sinyal:",
          " | ".join(f"{a} {len(e)}" for a, e in evs.items()))
    if any(len(e) < 10 for e in evs.values()):
        print("sinyal cok az, durdu")
        return
    tend = max(v["x"] for e in evs.values() for v in e)
    top3 = {}
    for ad, e in evs.items():
        rets = np.array([v["ret"] for v in e])
        top3[ad] = set(np.argsort(rets)[::-1][:max(1, int(len(e) * 0.03))].tolist())

    rast = dict(tohumlar=P.TOHUMLAR)
    varyant = [
        ("BH2 rastgele (taban)", "BH2"),
        ("BH2 zayifi ele 60g", "Z60"),
        ("BH2 sadece guclu 60g", "G60"),
        ("BH2 zayifi ele 20g", "Z20"),
        ("BH2 sadece guclu 20g", "G20"),
    ]
    gecen = {ad: 0 for ad, _ in varyant[1:]}
    print("(yillik % / max dusus %; d1 = 2021-23; d2 = 2024-26; top3yok = en iyi %3 islem yok)")
    print("(zayifi ele = alt sektoru zayif olanlari eler, alt sektoru bilinmeyenleri tutar; sadece guclu = yalniz guclu alt sektor)")
    for kad, n, c in KONF:
        P.N = n
        P.C1 = c
        print(f"\n=== {kad} ===")
        sonuc = {}
        for ad, ek in varyant:
            e = evs[ek]
            r = P.kos(e, C, O, idx, tend, **rast)
            r3 = P.kos(e, C, O, idx, tend, atla=top3[ek], **rast)
            if r is None or r3 is None:
                continue
            sonuc[ad] = (r, r3)
            print(f"{ad}: tum {r[0]:.0f}/{r[1]:.0f} | d1 {r[2]:.0f} | d2 {r[4]:.0f} | "
                  f"alinan {r[6]:.0f}/{r[7]:.0f} | top3yok {r3[0]:.0f}")
        base = sonuc.get("BH2 rastgele (taban)")
        if base is None:
            continue
        for ad, (r, r3) in sonuc.items():
            if ad.startswith("BH2 rastgele"):
                continue
            ok = (r[2] > base[0][2] and r[4] > base[0][4]
                  and r3[0] > base[1][0])
            gecen[ad] += int(ok)
            print(f"  -> {ad}: d1 {r[2] - base[0][2]:+.0f} d2 {r[4] - base[0][4]:+.0f} "
                  f"top3yok {r3[0] - base[1][0]:+.0f} dusus {r[1] - base[0][1]:+.0f} "
                  f"{'KABUL ADAYI' if ok else 'RED'}")
    print(f"\n=== OZET: kac ayarda KABUL ADAYI ({len(KONF)} ayar) ===")
    for ad, k in gecen.items():
        print(f"{ad}: {k}/{len(KONF)}")
    print("\nbitti")


def main():
    df = pd.read_csv(glob.glob("**/sektor.csv", recursive=True)[0])
    close, opn, vol, xu = P.indir(P.sembol_oku())
    C = close.to_numpy()
    T, S = C.shape
    print("hisse:", S, "| gun:", T)

    bh = B.ozellikler(close, vol)
    sk = P.skorlar(close, xu)
    dn_ok = np.nan_to_num(sk["dn"] / 100.0, nan=0.0) >= B2.PARAM["dn_min"]
    bh2 = bh & dn_ok
    ii2, jj2 = B.sinyaller(bh2, P.CD)

    F = np.clip(C[H:] / C[:-H] - 1, -0.9, 3.0)
    evr = np.concatenate([np.nanmean(F, axis=1), np.full(H, np.nan)])
    F = np.vstack([F, np.full((H, S), np.nan)])

    alt_id, alt_ad = grup_idlerini_kur(close, df, "industry")
    R60 = pd.DataFrame(C).pct_change(60).clip(-0.5, 1.0).to_numpy()
    R20 = pd.DataFrame(C).pct_change(20).clip(-0.5, 1.0).to_numpy()
    guc60 = sektor_gucu(R60, alt_id, 6)
    guc20 = sektor_gucu(R20, alt_id, 6)
    yillar = pd.DatetimeIndex(close.index).year.values

    yogunlasma(ii2, jj2, guc60, alt_id, alt_ad, F, evr, yillar)

    kume = {
        "BH2": bh2,
        "Z60": bh2 & ~(guc60 <= 0),
        "G60": bh2 & (guc60 > 0),
        "Z20": bh2 & ~(guc20 <= 0),
        "G20": bh2 & (guc20 > 0),
    }
    portfoy_testi(close, opn, kume, None, sk)


if __name__ == "__main__":
    main()
