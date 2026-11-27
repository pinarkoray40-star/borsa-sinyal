import glob
import warnings

import numpy as np
import pandas as pd

import portfoy2 as P
from sektor_test import grup_idlerini_kur, sektor_gucu, grup_raporu, H

warnings.filterwarnings("ignore")

B = P.B
B2 = P.B2

PENCERE = 250      # kume cikarmak icin kullanilan gecmis gun sayisi
YENILE = 63        # kumeler kac gunde bir yeniden cikarilir
MIN_GOZLEM = 150   # kumelenecek hissenin pencerede en az gunluk getiri sayisi
K_LER = (30, 60, 100)
K_PORTFOY = 60
MIN_UYE = 6

KONF = [
    ("N=10, maliyet %0,8", 10, 0.004),
    ("N=20, maliyet %0,8", 20, 0.004),
    ("N=30, maliyet %0,8", 30, 0.004),
    ("N=20, maliyet %1,5", 20, 0.0075),
]


def kume_kimlikleri(ret_dilim, K):
    """ret_dilim: (gun x hisse) piyasadan arindirilmis gunluk getiri. Dondurur: hisse basina kume id (-1 = yok)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    S = ret_dilim.shape[1]
    df = pd.DataFrame(ret_dilim)
    kolonlar = np.where(df.notna().sum().to_numpy() >= MIN_GOZLEM)[0]
    ids = np.full(S, -1)
    if len(kolonlar) < K * 2:
        return ids
    corr = df.iloc[:, kolonlar].corr(min_periods=MIN_GOZLEM).fillna(0.0).to_numpy()
    corr = (corr + corr.T) / 2
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, None))
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="ward")
    lab = fcluster(Z, t=K, criterion="maxclust")
    ids[kolonlar] = lab - 1
    return ids


def kume_gucu(ret_adj, R, K):
    """Her gun x hisse: hissenin kumesinin (hisse haric) ortalama getirisi - piyasa. Kumeler sadece GECMIS veriyle cikarilir."""
    T, S = R.shape
    guc = np.full((T, S), np.nan)
    pos = PENCERE
    son_ids = None
    while pos < T:
        son = min(pos + YENILE, T)
        ids = kume_kimlikleri(ret_adj[pos - PENCERE:pos], K)
        guc[pos:son] = sektor_gucu(R[pos:son], ids, MIN_UYE)
        son_ids = ids
        pos = son
    return guc, son_ids


def portfoy_testi(close, opn, kume, varyant, sk):
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
    gecen = {ad: 0 for ad, _ in varyant[1:]}
    print("(yillik % / max dusus %; d1 = 2021-23; d2 = 2024-26; top3yok = en iyi %3 islem yok)")
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
        base = sonuc.get(varyant[0][0])
        if base is None:
            continue
        for ad, (r, r3) in sonuc.items():
            if ad == varyant[0][0]:
                continue
            ok = (r[2] > base[0][2] and r[4] > base[0][4] and r3[0] > base[1][0])
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
    print("BH2 sinyal:", len(ii2))

    F = np.clip(C[H:] / C[:-H] - 1, -0.9, 3.0)
    evr = np.concatenate([np.nanmean(F, axis=1), np.full(H, np.nan)])
    F = np.vstack([F, np.full((H, S), np.nan)])

    r = close.pct_change(fill_method=None).clip(-0.2, 0.2)
    ret_adj = r.sub(r.mean(axis=1), axis=0).to_numpy()
    R20 = pd.DataFrame(C).pct_change(20).clip(-0.5, 1.0).to_numpy()
    R60 = pd.DataFrame(C).pct_change(60).clip(-0.5, 1.0).to_numpy()

    alt_id, _ = grup_idlerini_kur(close, df, "industry")
    gs20 = sektor_gucu(R20, alt_id, MIN_UYE)
    gs60 = sektor_gucu(R60, alt_id, MIN_UYE)
    # adil kiyas: Yahoo alt sektorunu de kumelerin basladigi gunden itibaren degerlendir
    gs20_k, gs60_k = gs20.copy(), gs60.copy()
    gs20_k[:PENCERE] = np.nan
    gs60_k[:PENCERE] = np.nan

    print("\n" + "=" * 100)
    print(f"A) OLAY BAZLI TEST (BH2 sinyalleri, 84 gun fark; kumeler {PENCERE} gunluk gecmisle, {YENILE} gunde bir yenilenir)")
    print("=" * 100)
    grup_raporu("BH2 / YAHOO ALT SEKTOR 20g (kume donemiyle ayni gunler)", ii2, jj2, gs20_k, F, evr)
    grup_raporu("BH2 / YAHOO ALT SEKTOR 60g (kume donemiyle ayni gunler)", ii2, jj2, gs60_k, F, evr)
    kume_sonuc = {}
    for K in K_LER:
        g20, ids = kume_gucu(ret_adj, R20, K)
        g60, _ = kume_gucu(ret_adj, R60, K)
        kume_sonuc[K] = (g20, g60)
        if ids is not None:
            boy = pd.Series(ids[ids >= 0]).value_counts()
            print(f"\n[K={K}] son donem: olculen hisse {int((ids >= 0).sum())}, kume {len(boy)}, "
                  f"medyan boyut {int(boy.median())}, en buyuk {int(boy.max())}, en kucuk {int(boy.min())}")
        grup_raporu(f"BH2 / KUME K={K} 20g", ii2, jj2, g20, F, evr)
        grup_raporu(f"BH2 / KUME K={K} 60g", ii2, jj2, g60, F, evr)

    g20, g60 = kume_sonuc[K_PORTFOY]
    kume = {
        "BH2": bh2,
        "ALT": bh2 & ~(gs20 <= 0),
        "K20": bh2 & ~(g20 <= 0),
        "K60": bh2 & ~(g60 <= 0),
        "IKI": bh2 & ~(gs20 <= 0) & ~(g20 <= 0),
    }
    varyant = [
        ("BH2 rastgele (taban)", "BH2"),
        ("BH3 mevcut: alt sektor zayifi ele 20g", "ALT"),
        (f"KUME K={K_PORTFOY} zayifi ele 20g", "K20"),
        (f"KUME K={K_PORTFOY} zayifi ele 60g", "K60"),
        ("IKISI birden: biri zayifsa ele 20g", "IKI"),
    ]
    portfoy_testi(close, opn, kume, varyant, sk)


if __name__ == "__main__":
    main()
