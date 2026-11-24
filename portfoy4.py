"""
portfoy4.py - BH2'ye ciro (likitlik) filtresi eklemenin portföy etkisi. Siteye dokunmaz.
portfoy2.py'nin portföy motorunu kullanır (portfoy3.py ile aynı yapı).
Filtre sinyal ANINDA uygulanır (cooldown filtreden sonra işler), canlıda olacağı gibi.
Karşılaştırma tabanı: filtresiz BH2 (aynı sıralama yöntemiyle).
"""
import numpy as np

import portfoy2 as P

B = P.B
B2 = P.B2

KONF = [
    ("N=10, maliyet %0,8", 10, 0.004),
    ("N=20, maliyet %0,8", 20, 0.004),
    ("N=30, maliyet %0,8", 30, 0.004),
    ("N=40, maliyet %0,8", 40, 0.004),
    ("N=20, maliyet %1,5", 20, 0.0075),
]
CIROLAR = (50, 60)


def calistir4(close, opn, vol, xu):
    print("hisse:", close.shape[1], "| gun:", close.shape[0])
    C = close.to_numpy()
    O = opn.to_numpy()
    idx = close.index
    bh = B.ozellikler(close, vol)
    sk = P.skorlar(close, xu)
    dn_ok = np.nan_to_num(sk["dn"] / 100.0, nan=0.0) >= B2.PARAM["dn_min"]
    ciro = (close * vol).rolling(20).mean().rank(axis=1, pct=True).to_numpy() * 100
    ciro = np.nan_to_num(ciro, nan=0.0)

    evs = {}
    ii, jj = B.sinyaller(bh & dn_ok, P.CD)
    evs["BH2"] = P.olaylar(C, O, ii, jj, P.C1, sk)
    for th in CIROLAR:
        ii, jj = B.sinyaller(bh & dn_ok & (ciro >= th), P.CD)
        evs[f"BH2+c{th}"] = P.olaylar(C, O, ii, jj, P.C1, sk)
    print("sonuclanan sinyal:", " | ".join(f"{k} {len(v)}" for k, v in evs.items()))
    if any(len(v) < 10 for v in evs.values()):
        print("sinyal cok az, durdu")
        return

    tend = max(v["x"] for e in evs.values() for v in e)
    top3 = {}
    for ad, e in evs.items():
        rets = np.array([v["ret"] for v in e])
        top3[ad] = set(np.argsort(rets)[::-1][:max(1, int(len(e) * 0.03))].tolist())

    V = [
        ("BH2 rastgele", "BH2", dict(tohumlar=P.TOHUMLAR), None),
        ("BH2+c50 rastgele", "BH2+c50", dict(tohumlar=P.TOHUMLAR), "BH2 rastgele"),
        ("BH2+c60 rastgele", "BH2+c60", dict(tohumlar=P.TOHUMLAR), "BH2 rastgele"),
        ("BH2 sirala: dayaniklilik", "BH2", dict(anahtar="dn"), None),
        ("BH2+c50 sirala: dayaniklilik", "BH2+c50", dict(anahtar="dn"), "BH2 sirala: dayaniklilik"),
        ("BH2+c60 sirala: dayaniklilik", "BH2+c60", dict(anahtar="dn"), "BH2 sirala: dayaniklilik"),
    ]
    gecen = {ad: 0 for ad, _, _, b in V if b}
    print("(yillik % / max dusus %; d1 = 2021-23; d2 = 2024-26; top3yok = en iyi %3 islem yok)")
    for kad, n, c in KONF:
        P.N = n
        P.C1 = c
        print(f"\n=== {kad} ===")
        sonuc = {}
        for ad, ek, kw, base in V:
            e = evs[ek]
            r = P.kos(e, C, O, idx, tend, **kw)
            r3 = P.kos(e, C, O, idx, tend, atla=top3[ek], **kw)
            if r is None or r3 is None:
                continue
            sonuc[ad] = (r, r3)
            print(f"{ad}: tum {r[0]:.0f}/{r[1]:.0f} | d1 {r[2]:.0f} | d2 {r[4]:.0f} | "
                  f"alinan {r[6]:.0f}/{r[7]:.0f} | top3yok {r3[0]:.0f}")
        for ad, ek, kw, base in V:
            if not base or ad not in sonuc or base not in sonuc:
                continue
            r, r3 = sonuc[ad]
            b, b3 = sonuc[base]
            ok = r[2] > b[2] and r[4] > b[4] and r3[0] > b3[0]
            gecen[ad] += int(ok)
            print(f"  -> {ad} (taban {base}): yillik {r[0] - b[0]:+.0f} d1 {r[2] - b[2]:+.0f} "
                  f"d2 {r[4] - b[4]:+.0f} top3yok {r3[0] - b3[0]:+.0f} dusus {r[1] - b[1]:+.0f} "
                  f"{'KABUL ADAYI' if ok else 'RED'}")
    print("\n=== OZET: kac ayarda KABUL ADAYI (5 ayar) ===")
    for ad, k in gecen.items():
        print(f"{ad}: {k}/{len(KONF)}")
    print("\nbitti")


def main():
    close, opn, vol, xu = P.indir(P.sembol_oku())
    calistir4(close, opn, vol, xu)


if __name__ == "__main__":
    main()
