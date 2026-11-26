import numpy as np
import pandas as pd

from tavan_aday import sembol_oku, indir, ozellik, TAVAN, MALIYET


def hazirla():
    semboller = sembol_oku()
    print(f"sembol: {len(semboller)}", flush=True)
    veri = indir(semboller)
    parcalar = []
    for s, d in veri.items():
        f = ozellik(s, d)
        f["ac_next"] = (d["Open"].shift(-1) / d["Close"] - 1) * 100
        parcalar.append(f)
    df = pd.concat(parcalar)
    df.index.name = "gun"
    df = df.reset_index()
    df = df.replace([np.inf, -np.inf], np.nan)
    df["ciro_sira"] = df.groupby("gun")["ciro"].rank(pct=True) * 100
    ok = (
        (df["ret1"] < TAVAN) & df["ret_next"].notna() & df["ac_next"].notna()
        & df["hacim_var"] & df["ciro"].notna()
        & (df["ret1"].abs() < 25) & (df["ret_next"].abs() < 25)
        & (df["ac_next"].abs() < 25)
        & df["band15"].notna() & df["zirve_uzak"].notna()
        & df["hacim_kat"].notna() & df["tavan_gecmis"].notna()
    )
    df = df[ok].copy()
    df["tavan_next"] = (df["ret_next"] >= TAVAN).astype(int)
    return df


def stratejiler(g):
    ac, kap = g["ac_next"], g["ret_next"]
    return {
        "A kapanis": kap,
        "B acilis": ac,
        "E acilis>=2 tut": pd.Series(np.where(ac >= 2, kap, ac), index=g.index),
        "C acilis>=5 tut": pd.Series(np.where(ac >= 5, kap, ac), index=g.index),
        "D acilis>=9 tut": pd.Series(np.where(ac >= 9, kap, ac), index=g.index),
    }


def satir(ad, g, orta_gun):
    n = len(g)
    if n < 100:
        print(f"  {ad:<10} n={n} (az)")
        return
    p = g["tavan_next"].mean() * 100
    po = (g["ac_next"] >= 9).mean() * 100
    st = stratejiler(g)
    netler = {k: v.mean() - MALIYET for k, v in st.items()}
    print(f"  {ad:<10} n={n:<5} tavan %{p:4.1f} acilista tavan %{po:4.1f}")
    print("      net: " + "  ".join(f"{k[0]} {v:+.2f}" for k, v in netler.items()))
    en = max(netler, key=netler.get)
    s = st[en]
    y1 = s[g["gun"] <= orta_gun].mean() - MALIYET
    y2 = s[g["gun"] > orta_gun].mean() - MALIYET
    kaz = ((s - MALIYET) > 0).mean() * 100
    print(f"      en iyi: {en}  net {netler[en]:+.2f}  Y1 {y1:+.2f} Y2 {y2:+.2f}"
          f"  kazanan %{kaz:.0f}")


def main():
    df = hazirla()
    orta_gun = df["gun"].sort_values().iloc[len(df) // 2]
    print(f"hisse-gun: {len(df)}  (Y1/Y2 sinir: {str(orta_gun)[:10]})"
          f"  maliyet %{MALIYET}")
    print("Harf: A kapanis, B acilis, E/C/D = acilis >= +2/+5/+9 ise kapanisa tut, degilse acilista sat")

    bayrak = {
        "D": df["deneme"] == 1,
        "T2": df["tavan_gecmis"] >= 2,
        "G": df["band15"] >= 28,
        "H": df["hacim_kat"] >= 2,
        "R5": df["ret5"] >= 5.5,
        "K": df["konum"] >= 0.7,
    }
    kombiler = [
        ("D+T2+K", ["D", "T2", "K"]),
        ("D+T2+R5", ["D", "T2", "R5"]),
        ("D+T2+G", ["D", "T2", "G"]),
        ("D+T2+H", ["D", "T2", "H"]),
        ("D+K", ["D", "K"]),
        ("D+T2", ["D", "T2"]),
    ]
    for ad_set, ek in (("TUM EVREN", None),
                       ("LIKIT (ciro sirasi >= 40)", df["ciro_sira"] >= 40)):
        print("=" * 90)
        print(ad_set)
        for ad, anahtar in kombiler:
            m = pd.Series(True, index=df.index)
            for a in anahtar:
                m &= bayrak[a]
            if ek is not None:
                m &= ek
            satir(ad, df[m], orta_gun)
    print("=" * 90)
    print("Not: net = %0,8 maliyet dusulmus; giris bugunun kapanisi.")


if __name__ == "__main__":
    main()
