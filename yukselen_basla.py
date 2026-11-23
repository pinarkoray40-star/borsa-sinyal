import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

TETIK = 1.2       # 20g dipten +%20
BEKLE = 20        # ayni hissede tetikler arasi
UFUK = 120        # kazanan: dibin 2 katina 120g icinde
STOP = 0.75       # izleyen stop -%25
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


def endeks():
    x = yf.download("XU100.IS", period="6y", interval="1d",
                    auto_adjust=True, progress=False)["Close"]
    return x.squeeze()


def olaylar(s, d):
    c = d["Close"].values
    h = d["High"].values
    l = d["Low"].values
    v = d["Volume"].values
    xr = d["xr60"].values
    n = len(c)
    mn = pd.Series(c).rolling(20).min().values
    ratio = c / mn
    out = []
    son = -999
    for t in range(260, n - UFUK - 1):
        if ratio[t] >= TETIK and ratio[t - 1] < TETIK and t - son >= BEKLE:
            son = t
            dip = mn[t]
            win = int(c[t + 1:t + UFUK + 1].max() >= 2 * dip)
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
            a = int(np.argmin(c[t - 19:t + 1]))
            tur = c[t - 249:t + 1] * v[t - 249:t + 1]
            tur20 = c[t - 19:t + 1] * v[t - 19:t + 1]
            out.append(dict(
                sym=s.replace(".IS", ""),
                tarih=d.index[t],
                win=win,
                f42=(c[t + 42] / c[t] - 1) * 100,
                f84=(c[t + 84] / c[t] - 1) * 100,
                ts=(ex / c[t] - 1) * 100,
                hiz=19 - a,
                r5=(c[t] / c[t - 5] - 1) * 100,
                r60=(c[t] / c[t - 60] - 1) * 100,
                rs60=((c[t] / c[t - 60] - 1) - xr[t]) * 100,
                dist=(c[t] / c[t - 250:t + 1].max() - 1) * 100,
                vr=v[t - 4:t + 1].mean() / max(v[t - 54:t - 4].mean(), 1),
                vtrig=v[t] / max(v[t - 20:t].mean(), 1),
                likrel=np.median(tur20) / max(np.median(tur), 1),
                gecmis=(c[t - 250:t - 19].max() / c[t - 250:t - 19].min() - 1) * 100,
                kap=(c[t] - l[t]) / max(h[t] - l[t], 1e-9),
                piyasa60=xr[t] * 100,
            ))
    return out


FEATS = [("hiz", "dipten tetige gun"), ("r5", "5g getiri %"),
         ("r60", "60g getiri %"), ("rs60", "60g goreceli guc"),
         ("dist", "250g zirveye %"), ("vr", "hacim orani 5/50"),
         ("vtrig", "tetik gunu hacim kati"),
         ("likrel", "islem hacmi/kendi 250g"),
         ("gecmis", "onceki 1y hareket %"), ("kap", "gun ici kapanis (0-1)"),
         ("piyasa60", "endeks 60g %")]


def main():
    syms = sembol_oku()
    data = indir(syms)
    print("veri gelen:", len(data), "/", len(syms))
    xu = endeks()
    xr60 = xu / xu.shift(60) - 1
    rows = []
    for s, d in data.items():
        d = d.copy()
        d["xr60"] = xr60.reindex(d.index, method="ffill").fillna(0)
        rows += olaylar(s, d)
    ev = pd.DataFrame(rows)
    ev.to_csv("yukselen_basla.csv", index=False)
    ev["yari"] = (ev["tarih"] > ev["tarih"].median()).astype(int)

    print("\n=== GENEL ===")
    print(f"olay: {len(ev)} | hisse: {ev['sym'].nunique()}")
    print(f"tarih araligi: {ev['tarih'].min().date()} - {ev['tarih'].max().date()}")
    print(f"kazanan (dibin 2 kati, {UFUK}g): %{ev['win'].mean() * 100:.1f}")
    print(f"42g medyan %{ev['f42'].median():.1f} | 84g medyan %{ev['f84'].median():.1f}")
    print(f"izleyen stop -%25 (max {TUT}g): ort %{ev['ts'].mean():.1f} "
          f"medyan %{ev['ts'].median():.1f} kazanan(+) %{(ev['ts'] > 0).mean() * 100:.0f}")
    print("\nyila gore kazanan orani:")
    g = ev.groupby(ev["tarih"].dt.year)["win"].agg(["mean", "size"])
    for y, r in g.iterrows():
        print(f"{y}: %{r['mean'] * 100:.1f} (n={int(r['size'])})")

    print("\n=== OZELLIK -> KAZANAN ORANI (alt/orta/ust 1/3) ===")
    print("(esikler e1/e2 | y1,y2 = ust-alt fark iki yarida | ts = izleyen stop ort alt/ust)")
    res = []
    for k, ad in FEATS:
        e = ev[k].dropna()
        if e.empty:
            continue
        q1, q2 = e.quantile([1 / 3, 2 / 3])
        lo = ev[ev[k] <= q1]
        mi = ev[(ev[k] > q1) & (ev[k] <= q2)]
        hi = ev[ev[k] > q2]

        def w(x):
            return x["win"].mean() * 100 if len(x) else np.nan

        y = []
        for j in (0, 1):
            y.append(w(hi[hi["yari"] == j]) - w(lo[lo["yari"] == j]))
        line = (f"{ad} ({q1:.1f}/{q2:.1f}): alt %{w(lo):.1f} | orta %{w(mi):.1f} | "
                f"ust %{w(hi):.1f} | y1 {y[0]:+.1f} y2 {y[1]:+.1f} | "
                f"ts {lo['ts'].mean():.1f}/{hi['ts'].mean():.1f}")
        res.append((abs(w(hi) - w(lo)), line))
    for _, line in sorted(res, reverse=True):
        print(line)
    print("\nyukselen_basla.csv yazildi")


main()
