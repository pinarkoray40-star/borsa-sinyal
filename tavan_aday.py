import re
import time
import numpy as np
import pandas as pd
import yfinance as yf

MALIYET = 0.8
TAVAN = 9.5
CHUNK = 40


def sembol_oku():
    metin = open("bist_symbols.txt", encoding="utf-8").read().upper()
    metin = "\n".join(
        l for l in metin.splitlines() if not l.strip().startswith("#")
    )
    out = []
    for t in re.split(r"[\s,;]+", metin):
        if re.fullmatch(r"[A-Z0-9]{3,6}(\.IS)?", t):
            t = t.replace(".IS", "") + ".IS"
            if t not in out:
                out.append(t)
    return out


def indir(semboller):
    veri = {}
    for i in range(0, len(semboller), CHUNK):
        grup = semboller[i:i + CHUNK]
        df = None
        for _ in range(3):
            try:
                df = yf.download(
                    grup, period="5y", interval="1d",
                    group_by="ticker", auto_adjust=True,
                    progress=False, threads=True,
                )
                break
            except Exception as e:
                print("hata:", e, flush=True)
                time.sleep(5)
        if df is None or df.empty:
            continue
        for s in grup:
            try:
                d = df[s] if isinstance(df.columns, pd.MultiIndex) else df
                d = d.dropna(subset=["Close", "High", "Low"])
                if len(d) > 150:
                    veri[s] = d
            except KeyError:
                pass
        time.sleep(1)
        print(f"indirildi {min(i + CHUNK, len(semboller))}/{len(semboller)}"
              f" (veri var: {len(veri)})", flush=True)
    return veri


def ozellik(s, d):
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"].fillna(0)
    f = pd.DataFrame(index=d.index)
    f["sym"] = s
    f["ret1"] = (c / c.shift(1) - 1) * 100
    f["ret5"] = (c / c.shift(5) - 1) * 100
    f["ret20"] = (c / c.shift(20) - 1) * 100
    f["hacim_kat"] = v / v.shift(1).rolling(20).mean()
    f["band15"] = (h.rolling(15).max() - l.rolling(15).min()) / c * 100
    f["zirve_uzak"] = (c / h.rolling(60).max() - 1) * 100
    aralik = (h - l).replace(0, np.nan)
    f["konum"] = (c - l) / aralik
    tavan_bugun = f["ret1"] >= TAVAN
    f["deneme"] = (((h / c.shift(1) - 1) * 100 >= 9) & ~tavan_bugun).astype(int)
    f["tavan_gecmis"] = tavan_bugun.astype(int).shift(1).rolling(20).sum()
    f["ciro"] = (c * v).rolling(20).mean()
    f["fiyat"] = c
    f["hacim_var"] = v > 0
    f["ret_next"] = (c.shift(-1) / c - 1) * 100
    return f


def satir(ad, g, base_p):
    n = len(g)
    if n < 30:
        return
    p = (g["tavan_next"].mean()) * 100
    brut = g["ret_next"].mean()
    med = g["ret_next"].median()
    lift = p / base_p if base_p > 0 else 0
    print(f"  {ad:<18} n={n:<7} tavan %{p:5.2f}  kat {lift:4.1f}x"
          f" | brut {brut:+.2f}% med {med:+.2f}% net {brut - MALIYET:+.2f}%")


def kova(df, col, base_p, q=5):
    print(f"[{col}]")
    try:
        b = pd.qcut(df[col], q, duplicates="drop")
    except Exception:
        return
    for k, g in df.groupby(b, observed=True):
        satir(f"{k.left:.1f}..{k.right:.1f}", g, base_p)


def main():
    semboller = sembol_oku()
    print(f"sembol: {len(semboller)}", flush=True)
    veri = indir(semboller)
    df = pd.concat([ozellik(s, d) for s, d in veri.items()])
    df.index.name = "gun"
    df = df.reset_index()

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
    print(f"hisse-gun: {len(df)}  ertesi gun tavan: {df['tavan_next'].sum()}"
          f"  taban oran: %{base_p:.2f}")
    print(f"ortalama ertesi gun getiri (evren): {df['ret_next'].mean():+.2f}%")
    print("=" * 95)

    print("TAVAN ARIFESI vs DIGERLERI (ortanca)")
    cols = ["ret1", "ret5", "ret20", "hacim_kat", "band15", "zirve_uzak",
            "konum", "tavan_gecmis", "ciro_sira", "fiyat"]
    t = df[df["tavan_next"] == 1][cols].median()
    d0 = df[df["tavan_next"] == 0][cols].median()
    for k in cols:
        print(f"  {k:<14} tavan arifesi {t[k]:9.2f}   digerleri {d0[k]:9.2f}")
    print("=" * 95)

    for col in ["ret1", "ret5", "ret20", "hacim_kat", "band15",
                "zirve_uzak", "konum", "ciro_sira", "fiyat"]:
        kova(df, col, base_p)
    print("[deneme] (gun ici tavana degdi ama kapanista tavan degil)")
    for k in (0, 1):
        satir(f"deneme={k}", df[df["deneme"] == k], base_p)
    print("[tavan_gecmis] (son 20 gunde tavan sayisi)")
    df["tg"] = df["tavan_gecmis"].clip(upper=2).astype(int)
    for k in (0, 1, 2):
        satir(f"tavan_gecmis={k}" + ("+" if k == 2 else ""), df[df["tg"] == k], base_p)
    print("=" * 95)
    print("Not: tavan = ertesi gun kapanis >= +%9,5; giris bugunun kapanisi;"
          " net = %0,8 maliyet dusulmus. Bugun tavan olanlar hesaba katilmadi.")


if __name__ == "__main__":
    main()
