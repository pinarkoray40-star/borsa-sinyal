import itertools
import numpy as np
import pandas as pd

from tavan_aday import sembol_oku, indir, ozellik, TAVAN, MALIYET


def satir(ad, g, base_p, orta_gun):
    n = len(g)
    if n < 100:
        print(f"  {ad:<30} n={n} (az)")
        return
    p = g["tavan_next"].mean() * 100
    brut = g["ret_next"].mean()
    p1 = g[g["gun"] <= orta_gun]["tavan_next"].mean() * 100
    p2 = g[g["gun"] > orta_gun]["tavan_next"].mean() * 100
    print(f"  {ad:<30} n={n:<6} tavan %{p:5.2f} kat {p / base_p:4.1f}x"
          f" | Y1 %{p1:5.2f} Y2 %{p2:5.2f}"
          f" | brut {brut:+.2f}% net {brut - MALIYET:+.2f}%")


def main():
    semboller = sembol_oku()
    print(f"sembol: {len(semboller)}", flush=True)
    veri = indir(semboller)
    df = pd.concat([ozellik(s, d) for s, d in veri.items()])
    df.index.name = "gun"
    df = df.reset_index()
    df = df.replace([np.inf, -np.inf], np.nan)

    df["ciro_sira"] = df.groupby("gun")["ciro"].rank(pct=True) * 100
    ok = (
        (df["ret1"] < TAVAN) & df["ret_next"].notna() & df["hacim_var"]
        & df["ciro"].notna() & (df["ret1"].abs() < 25) & (df["ret_next"].abs() < 25)
        & df["band15"].notna() & df["zirve_uzak"].notna()
        & df["hacim_kat"].notna() & df["tavan_gecmis"].notna()
    )
    df = df[ok].copy()
    df["tavan_next"] = (df["ret_next"] >= TAVAN).astype(int)
    base_p = df["tavan_next"].mean() * 100
    orta_gun = df["gun"].sort_values().iloc[len(df) // 2]
    print(f"hisse-gun: {len(df)}  taban tavan orani: %{base_p:.2f}"
          f"  (Y1/Y2 sinir: {str(orta_gun)[:10]})")
    print("Kar icin kabaca tavan olasiligi %15+ gerekir.")
    print("=" * 100)

    bayrak = {
        "D": df["deneme"] == 1,                 # gun ici tavana degdi
        "T2": df["tavan_gecmis"] >= 2,          # son 20 gunde 2+ tavan
        "G": df["band15"] >= 28,                # genis 15g bant
        "H": df["hacim_kat"] >= 2,              # hacim 2 kat+
        "R5": df["ret5"] >= 5.5,                # 5g guclu
        "K": df["konum"] >= 0.7,                # kapanis gun yuksegine yakin
        "B": df["ret1"] >= 2,                   # bugun +%2 ustu
    }
    print("Bayraklar: D=gun ici tavana degdi, T2=son 20g 2+ tavan, G=15g bant>=28,"
          " H=hacim>=2x, R5=5g>=+5,5, K=kapanis gun yuksegine yakin, B=bugun>=+2%")

    manuel = [
        ("D", ["D"]),
        ("D+T2", ["D", "T2"]),
        ("D+G", ["D", "G"]),
        ("T2+G", ["T2", "G"]),
        ("D+T2+G", ["D", "T2", "G"]),
        ("D+T2+H", ["D", "T2", "H"]),
        ("T2+H", ["T2", "H"]),
        ("T2+R5", ["T2", "R5"]),
        ("D+K", ["D", "K"]),
        ("T2+B+H", ["T2", "B", "H"]),
    ]
    for ad_set, maske_ek in (("TUM EVREN", None), ("LIKIT (ciro sirasi >= 40)", df["ciro_sira"] >= 40)):
        print("=" * 100)
        print(ad_set)
        for ad, anahtar in manuel:
            m = pd.Series(True, index=df.index)
            for a in anahtar:
                m &= bayrak[a]
            if maske_ek is not None:
                m &= maske_ek
            satir(ad, df[m], base_p, orta_gun)

    print("=" * 100)
    print("OTOMATIK: 2-3 bayrak kombinasyonlari, n>=200, tavan olasiligina gore ilk 10")
    print("(cok deneme yapildi, en iyi gorunenler sans eseri de olabilir; Y1/Y2 tutarli mi bak)")
    sonuc = []
    for r in (2, 3):
        for kombi in itertools.combinations(bayrak.keys(), r):
            m = pd.Series(True, index=df.index)
            for a in kombi:
                m &= bayrak[a]
            g = df[m]
            if len(g) >= 200:
                sonuc.append(("+".join(kombi), g["tavan_next"].mean() * 100, g))
    sonuc.sort(key=lambda x: -x[1])
    for ad, _, g in sonuc[:10]:
        satir(ad, g, base_p, orta_gun)


if __name__ == "__main__":
    main()
