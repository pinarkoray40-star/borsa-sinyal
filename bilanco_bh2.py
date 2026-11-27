import numpy as np
import pandas as pd

import portfoy2 as P
from bilanco_test import bilanco_yukle, LAG

B = P.B
B2 = P.B2

KONF = [
    ("N=10, maliyet %0,8", 10, 0.004),
    ("N=20, maliyet %0,8", 20, 0.004),
    ("N=30, maliyet %0,8", 30, 0.004),
    ("N=20, maliyet %1,5", 20, 0.0075),
]
# temiz ROE evreninde ROE yuzdeligi esigi: 0.5 = ust yari, 0.7 = ust %30
ESIKLER = (0.5, 0.7)


def bilanco_matrisi(w, close):
    """Her gun x hisse icin: o gun KAMUYA ACIKLANMIS son bilancoda
    'temiz' hissenin ROE yuzdeligi (temiz degilse veya veri yoksa NaN)."""
    idx = pd.DatetimeIndex(close.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    kolon = {str(c).replace(".IS", ""): j for j, c in enumerate(close.columns)}
    mat = np.full(close.shape, np.nan)
    donemler = []
    for (yil, ay), g in sorted(w.groupby(["yil", "ay"]), key=lambda x: x[0]):
        avail = (pd.Timestamp(int(yil), int(ay), 1) + pd.offsets.MonthEnd(0)
                 + pd.Timedelta(days=LAG[int(ay)]))
        pos = idx.searchsorted(avail)
        if pos >= len(idx):
            continue
        donemler.append((pos, g))
    for k, (pos, g) in enumerate(donemler):
        son = donemler[k + 1][0] if k + 1 < len(donemler) else len(idx)
        if son <= pos:
            continue
        g = g.set_index("kod")
        temiz = g[(g["roe"] <= 0.8) & (g["d_opm"] <= 0.5) & (g["d_nm"] <= 0.5)]
        if len(temiz) < 30:
            continue
        pct = temiz["roe"].rank(pct=True)
        for kod, p in pct.items():
            j = kolon.get(kod)
            if j is not None:
                mat[pos:son, j] = p
    return mat


def calistir(close, opn, vol, xu, w):
    print("hisse:", close.shape[1], "| gun:", close.shape[0])
    C = close.to_numpy()
    O = opn.to_numpy()
    idx = close.index
    bil = bilanco_matrisi(w, close)
    print(f"bilanco kapsami: gunluk hisse-gun orani %{np.isfinite(bil).mean() * 100:.0f}")

    bh = B.ozellikler(close, vol)
    sk = P.skorlar(close, xu)
    dn_ok = np.nan_to_num(sk["dn"] / 100.0, nan=0.0) >= B2.PARAM["dn_min"]
    bh2 = bh & dn_ok
    kume = {"BH2": bh2}
    for e in ESIKLER:
        kume[f"B{int(e * 100)}"] = bh2 & (bil >= e)

    evs = {}
    for ad, m in kume.items():
        ii, jj = B.sinyaller(m, P.CD)
        evs[ad] = P.olaylar(C, O, ii, jj, P.C1, sk)
    print("sonuclanan sinyal:", " | ".join(f"{a} {len(e)}" for a, e in evs.items()))
    if len(evs["BH2"]) < 10 or any(len(e) < 10 for e in evs.values()):
        print("sinyal cok az, durdu")
        return

    tend = max(v["x"] for e in evs.values() for v in e)
    top3 = {}
    for ad, e in evs.items():
        rets = np.array([v["ret"] for v in e])
        top3[ad] = set(np.argsort(rets)[::-1][:max(1, int(len(e) * 0.03))].tolist())

    rast = dict(tohumlar=P.TOHUMLAR)
    varyant = [
        ("BH2 rastgele (taban)", "BH2", rast),
        ("BH2+B50 rastgele", "B50", rast),
        ("BH2+B70 rastgele", "B70", rast),
        ("BH2 sirala: dayaniklilik", "BH2", dict(anahtar="dn")),
        ("BH2+B50 sirala: dayaniklilik", "B50", dict(anahtar="dn")),
    ]
    gecen = {ad: 0 for ad, _, _ in varyant[1:]}
    print("(yillik % / max dusus %; d1 = 2021-23; d2 = 2024-26; top3yok = en iyi %3 islem yok)")
    print("(taban = bilancosuz BH2 rastgele; B50/B70 = temiz ROE ust yari / ust %30 sarti)")
    for kad, n, c in KONF:
        P.N = n
        P.C1 = c
        print(f"\n=== {kad} ===")
        sonuc = {}
        for ad, ek, kw in varyant:
            e = evs[ek]
            r = P.kos(e, C, O, idx, tend, **kw)
            r3 = P.kos(e, C, O, idx, tend, atla=top3[ek], **kw)
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
    print("Not: bilanco verisi 2020 ortasindan basliyor; ilk donem sinyalleri filtreyle eleniyor olabilir.")
    print("\nbitti")


def main():
    w = bilanco_yukle()
    close, opn, vol, xu = P.indir(P.sembol_oku())
    calistir(close, opn, vol, xu, w)


if __name__ == "__main__":
    main()
