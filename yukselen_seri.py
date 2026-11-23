import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

UFUK = 120
STOP = 0.75
TUT = 84


def sembol_oku():
    out = []
    for ln in open("bist_symbols.txt", encoding="utf-8"):
        for t in ln.replace(",", " ").split():
            t = t.strip().upper()
            if not t or t.startswith("#"):
                continue
            if not t.endswith(".IS"):
                t += ".IS"
            out.append(t)
    return sorted(set(out))


def indir(syms):
    data = {}
    for k in range(0, len(syms), 40):
        grp = syms[k:k + 40]
        try:
            df = yf.download(grp, period="6y", interval="1d",
                             auto_adjust=True, group_by="ticker",
                             progress=False, threads=True)
        except Exception as e:
            print("indirme hatasi", k, e)
            continue
        for s in grp:
            try:
                d = df[s][["High", "Low", "Close", "Volume"]].dropna()
                if len(d) > 300:
                    data[s] = d
            except Exception:
                pass
    return data


def sonuc(c, t):
    run = c[t]
    ex = None
    for k in range(1, TUT + 1):
        x = c[t + k]
        if x > run:
            run = x
        if x <= run * STOP:
            ex = x
            break
    if ex is None:
        ex = c[t + TUT]
    mx = c[t + 1:t + UFUK + 1].max()
    return ((ex / c[t] - 1) * 100,
            int(mx >= c[t] * 1.667),
            int(mx >= c[t] * 2.0))


def tara(s, d):
    c = d["Close"].values
    n = len(c)
    ordn = np.array([x.toordinal() for x in d.index])
    rmin = pd.Series(c).rolling(91).min().values
    with np.errstate(invalid="ignore"):
        flag = (c / rmin >= 2.0).astype(int)
    cs = np.cumsum(flag)
    hi250 = pd.Series(c).rolling(251).max().values
    out = []
    for t in range(260, n - UFUK - 1):
        if ordn[t] % 3 != 0:
            continue
        a250 = cs[t] - cs[t - 251] > 0
        a60 = cs[t] - cs[t - 60] > 0
        dist = (c[t] / hi250[t] - 1) * 100
        ts, w167, w2 = sonuc(c, t)
        out.append(dict(
            sym=s.replace(".IS", ""), tarih=d.index[t],
            ts=ts, win=w167, w2=w2,
            A=bool(a250), B=bool(a60), C=bool(a250 and not a60),
            D=bool(a250 and dist <= -30), N=bool(not a250)))
    return out


def kiyas(e, bm, ad):
    if len(e) < 100:
        print(f"{ad}: n={len(e)} cok az")
        return
    g = e.groupby("ay")[["ts", "win", "w2"]].mean()
    cnt = e.groupby("ay").size()
    g = g[cnt >= 20]
    g = g[g.index.isin(bm.index)]
    M = len(g)
    if M < 4:
        print(f"{ad}: n={len(e)} ay sayisi az")
        return
    b = bm.loc[g.index]
    dd = g - b
    dd["win"] *= 100
    dd["w2"] *= 100
    res = []
    for k in ("ts", "win", "w2"):
        t = dd[k].mean() / (dd[k].std() / np.sqrt(M))
        res.append((dd[k].mean(), t))
    print(f"{ad}: n={len(e)} ay={M}")
    print(f"   stop ort: grup {g['ts'].mean():.1f} evren {b['ts'].mean():.1f} "
          f"fark {res[0][0]:+.1f} t={res[0][1]:.1f}")
    print(f"   +%67: grup %{g['win'].mean() * 100:.1f} evren "
          f"%{b['win'].mean() * 100:.1f} fark {res[1][0]:+.1f} t={res[1][1]:.1f}")
    print(f"   2 kat: grup %{g['w2'].mean() * 100:.1f} evren "
          f"%{b['w2'].mean() * 100:.1f} fark {res[2][0]:+.1f} t={res[2][1]:.1f}")


def main():
    syms = sembol_oku()
    data = indir(syms)
    print("veri gelen:", len(data), "/", len(syms))
    rows = []
    for s, d in data.items():
        rows += tara(s, d)
    bs = pd.DataFrame(rows)
    bs["ay"] = bs["tarih"].dt.to_period("M")
    bs["yil"] = bs["tarih"].dt.year
    bm = bs.groupby("ay")[["ts", "win", "w2"]].mean()
    print(f"\nornek: {len(bs)} (3 gunde bir, her hisse)")
    print("A: son 250g'de 90g icinde 2 kat yapmis | B: son 60g'de (taze)")
    print("C: 61-250g once yapmis, son 60g yok (sogumus)")
    print("D: A ve zirveden >=%30 asagida | N: son 250g'de hic yapmamis")
    print("(fark = grup - ayni ay tum evren; t = ay bazli)")

    ad = {"A": "A seri 250g", "B": "B taze 60g", "C": "C sogumus",
          "D": "D seri+dipte", "N": "N hic yapmamis"}
    print("\n=== GENEL ===")
    for k in "ABCDN":
        print(f"[{k} paylasim %{bs[k].mean() * 100:.0f}]")
        kiyas(bs[bs[k]], bm, ad[k])

    med = bs["tarih"].median()
    print("\n=== YARILAR ===")
    for k in "ABC":
        e = bs[bs[k]]
        kiyas(e[e["tarih"] <= med], bm, ad[k] + " 1.yari")
        kiyas(e[e["tarih"] > med], bm, ad[k] + " 2.yari")

    print("\n=== YILA GORE (A seri) ===")
    e = bs[bs["A"]]
    for y in sorted(e["yil"].unique()):
        kiyas(e[e["yil"] == y], bm, str(y))
    print("\nbitti")


main()
