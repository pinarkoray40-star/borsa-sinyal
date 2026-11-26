import re
import time
import numpy as np
import pandas as pd
import yfinance as yf

MALIYET = 0.8      # gidis-donus yuzde
TAVAN_ESIK = 9.0   # %9 ve ustu hareketler tavan/taban kilidi sayilir, alinamaz
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
                    grup, period="60d", interval="5m",
                    group_by="ticker", auto_adjust=False,
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
                d = d.dropna(subset=["Open", "Close"])
                if len(d) > 100:
                    veri[s] = d
            except KeyError:
                pass
        time.sleep(1)
        print(f"indirildi {min(i + CHUNK, len(semboller))}/{len(semboller)}"
              f" (veri var: {len(veri)})", flush=True)
    return veri


def olaylar(veri):
    satirlar = []
    for s, d in veri.items():
        idx = d.index
        if idx.tz is None:
            idx = idx.tz_localize("Europe/Istanbul")
        else:
            idx = idx.tz_convert("Europe/Istanbul")
        d = d.copy()
        d.index = idx
        d = d.sort_index()
        d["gun"] = d.index.date
        d["dk"] = d.index.hour * 60 + d.index.minute
        for gun, g in d.groupby("gun"):
            ilk = g[(g.dk >= 600) & (g.dk < 630)]      # 10:00-10:30
            giris = g[(g.dk >= 630) & (g.dk < 640)]    # 10:30 mumu
            if len(ilk) < 5 or giris.empty:
                continue
            if (ilk["Volume"] > 0).sum() < 5:
                continue
            if g["dk"].iloc[-1] < 1050:                # yarim gun / eksik
                continue
            o = ilk["Open"].iloc[0]
            c = ilk["Close"].iloc[-1]
            e = giris["Open"].iloc[0]
            x = g["Close"].iloc[-1]
            if min(o, c, e, x) <= 0:
                continue
            satirlar.append((gun, s, (c / o - 1) * 100, (x / e - 1) * 100))
    return pd.DataFrame(satirlar, columns=["gun", "sym", "hareket", "getiri"])


def grup_rapor(ad, g, evren_gun):
    if len(g) < 20:
        print(f"{ad:<24} n={len(g)} (az)")
        return
    gunluk = g.groupby("gun")["getiri"].mean()
    fark = gunluk - evren_gun.reindex(gunluk.index)
    nd = len(fark)
    t = fark.mean() / (fark.std() / np.sqrt(nd)) if nd > 2 and fark.std() > 0 else 0
    yar = nd // 2
    f1 = fark.iloc[:yar].mean()
    f2 = fark.iloc[yar:].mean()
    brut = g["getiri"].mean()
    med = g["getiri"].median()
    net = brut - MALIYET
    kaz = ((g["getiri"] - MALIYET) > 0).mean() * 100
    print(f"{ad:<24} n={len(g):<5} gun={nd:<3} brut {brut:+.2f}% med {med:+.2f}%"
          f" | net {net:+.2f}% kaz %{kaz:.0f}"
          f" | fark {fark.mean():+.2f} t={t:.1f} (Y1 {f1:+.2f} Y2 {f2:+.2f})")


def main():
    semboller = sembol_oku()
    print(f"sembol: {len(semboller)}  maliyet: %{MALIYET} gidis-donus", flush=True)
    veri = indir(semboller)
    df = olaylar(veri)
    df = df[df["hareket"].abs() < TAVAN_ESIK].copy()
    print(f"veri olan hisse: {len(veri)}  gun: {df['gun'].nunique()}  olay: {len(df)}")
    if df.empty:
        print("olay yok")
        return

    evren_gun = df.groupby("gun")["getiri"].mean()
    print(f"EVREN (10:30 giris, kapanis cikis) brut ort: {df['getiri'].mean():+.2f}%"
          f"  medyan: {df['getiri'].median():+.2f}%")
    print("-" * 100)

    print("YUKSELENLER (ilk 30 dk hareketi, sonrasi long):")
    for esik in (2, 3, 5):
        grup_rapor(f"hareket >= +{esik}%", df[df["hareket"] >= esik], evren_gun)
    df["sira_yuk"] = df.groupby("gun")["hareket"].rank(ascending=False, method="first")
    df["sira_dus"] = df.groupby("gun")["hareket"].rank(ascending=True, method="first")
    grup_rapor("gunun en iyi 10'u", df[df["sira_yuk"] <= 10], evren_gun)

    print("-" * 100)
    print("DUSENLER (ilk 30 dk dususu, sonrasi long = toparlanma):")
    for esik in (2, 3, 5):
        grup_rapor(f"hareket <= -{esik}%", df[df["hareket"] <= -esik], evren_gun)
    grup_rapor("gunun en kotu 10'u", df[df["sira_dus"] <= 10], evren_gun)

    print("-" * 100)
    print("Not: brut = maliyet oncesi, net = %0,8 dusulmus. fark = grubun gunluk"
          " ortalamasi - evren ortalamasi (brut puan).")
    print("Dusenlerde brut getiri negatifse 'devam ediyor' demek (short zor, sadece bilgi).")


if __name__ == "__main__":
    main()
