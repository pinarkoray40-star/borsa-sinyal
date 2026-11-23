import numpy as np
import portfoy2 as P

B = P.B
B2 = P.B2

KONF = [
    ("N=10, maliyet %0,8", 10, 0.004),
    ("N=30, maliyet %0,8", 30, 0.004),
    ("N=40, maliyet %0,8", 40, 0.004),
    ("N=20, maliyet %1,5", 20, 0.0075),
]


def calistir3(close, opn, vol, xu):
    print("hisse:", close.shape[1], "| gun:", close.shape[0])
    C = close.to_numpy()
    O = opn.to_numpy()
    idx = close.index
    bh = B.ozellikler(close, vol)
    sk = P.skorlar(close, xu)
    dn_ok = np.nan_to_num(sk["dn"] / 100.0, nan=0.0) >= B2.PARAM["dn_min"]
    ii, jj = B.sinyaller(bh, P.CD)
    ii2, jj2 = B.sinyaller(bh & dn_ok, P.CD)
    evs = {"BH": P.olaylar(C, O, ii, jj, P.C1, sk),
           "BH2": P.olaylar(C, O, ii2, jj2, P.C1, sk)}
    print("sonuclanan sinyal: BH", len(evs["BH"]), "| BH2", len(evs["BH2"]))
    if len(evs["BH"]) < 30 or len(evs["BH2"]) < 10:
        print("sinyal cok az, durdu")
        return
    tend = max(v["x"] for e in evs.values() for v in e)
    top3 = {}
    for ad, e in evs.items():
        rets = np.array([v["ret"] for v in e])
        top3[ad] = set(np.argsort(rets)[::-1][:max(1, int(len(e) * 0.03))].tolist())

    varyant = [
        ("BH rastgele (taban)", "BH", dict(tohumlar=P.TOHUMLAR)),
        ("BH2 rastgele", "BH2", dict(tohumlar=P.TOHUMLAR)),
        ("BH2 sirala: dayaniklilik", "BH2", dict(anahtar="dn")),
        ("BH2 sirala: goreceli guc", "BH2", dict(anahtar="rs")),
    ]
    gecen = {ad: 0 for ad, _, _ in varyant[1:]}
    print("(yillik % / max dusus %; d1 = 2021-23; d2 = 2024-26; top3yok = en iyi %3 islem yok)")
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
        base = sonuc.get("BH rastgele (taban)")
        if base is None:
            continue
        for ad, (r, r3) in sonuc.items():
            if ad.startswith("BH rastgele"):
                continue
            ok = (r[2] > base[0][2] and r[4] > base[0][4]
                  and r3[0] > base[1][0])
            gecen[ad] += int(ok)
            print(f"  -> {ad}: d1 {r[2] - base[0][2]:+.0f} d2 {r[4] - base[0][4]:+.0f} "
                  f"top3yok {r3[0] - base[1][0]:+.0f} dusus {r[1] - base[0][1]:+.0f} "
                  f"{'KABUL ADAYI' if ok else 'RED'}")
    print("\n=== OZET: kac ayarda KABUL ADAYI (4 ayar) ===")
    for ad, k in gecen.items():
        print(f"{ad}: {k}/{len(KONF)}")
    print("(onceki testte N=20, maliyet %0,8: hepsi KABUL ADAYI idi)")
    print("\nbitti")


def main():
    close, opn, vol, xu = P.indir(P.sembol_oku())
    calistir3(close, opn, vol, xu)


if __name__ == "__main__":
    main()
