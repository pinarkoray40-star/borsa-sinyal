import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

TETIK = 1.2
BEKLE = 20
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


def endeks():
    x = yf.download("XU100.IS", period="6y", interval="1d",
                    auto_adjust=True, progress=False)["Close"]
    return x.squeeze()


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
    return ((ex / c[t] - 1) * 100,
            (c[t + 84] / c[t] - 1) * 100,
            int(c[t + 1:t + UFUK + 1].max() >= c[t] * 1.667))


def tara(s, d):
    c = d["Close"].values
    xr = d["xr60"].values
    n = len(c)
    mn = pd.Series(c).rolling(20).min().values
    ratio = c / mn
    ordn = np.array([x.toordinal() for x in d.index])
    ev, bs = [], []
    son = -999
    for t in range(260, n - UFUK - 1):
        tetik = (ratio[t] >= TETIK and ratio[t - 1] < TETIK
                 and t - son >= BEKLE)
        ornek = (ordn[t] % 3 == 0)
        if not tetik and not ornek:
            continue
        ts, f84, win = sonuc(c, t)
        row = dict(sym=s.replace(".IS", ""), tarih=d.index[t],
                   ts=ts, f84=f84, win=win, piyasa60=xr[t] * 100)
        if tetik:
            son = t
            ev.append(row)
        if ornek:
            bs.append(row)
    return ev, bs


def kiyas(e, b, ad):
    if len(e) < 30 or len(b) < 30:
        print(f"{ad}: n={len(e)} cok az")
        return
    bm = b.groupby("ay")[["ts", "win"]].mean()
    e = e.join(bm, on="ay", rsuffix="_b").dropna(subset=["ts_b"])
    m = e.assign(dts=e["ts"] - e["ts_b"],
                 dwin=(e["win"] - e["win_b"]) * 100) \
         .groupby("ay")[["dts", "dwin"]].mean()
    M = len(m)
    if M < 3:
        print(f"{ad}: n={len(e)} ay sayisi az")
        return
    t1 = m["dts"].mean() / (m["dts"].std() / np.sqrt(M))
    t2 = m["dwin"].mean() / (m["dwin"].std() / np.sqrt(M))
    print(f"{ad}: n={len(e)} ay={M}\n"
          f"   stop ort: olay {e['ts'].mean():.1f} evren {e['ts_b'].mean():.1f} "
          f"fark {m['dts'].mean():+.1f} t={t1:.1f}\n"
          f"   kazanan: olay %{e['win'].mean() * 100:.1f} "
          f"evren %{e['win_b'].mean() * 100:.1f} "
          f"fark {m['dwin'].mean():+.1f} t={t2:.1f}")


def main():
    syms = sembol_oku()
    data = indir(syms)
    print("veri gelen:", len(data), "/", len(syms))
    xu = endeks()
    xr60 = xu / xu.shift(60) - 1
    E, B = [], []
    for s, d in data.items():
        d = d.copy()
        d["xr60"] = xr60.reindex(d.index, method="ffill").fillna(0)
        e, b = tara(s, d)
        E += e
        B += b
    ev = pd.DataFrame(E)
    bs = pd.DataFrame(B)
    for x in (ev, bs):
        x["ay"] = x["tarih"].dt.to_period("M")
        x["yil"] = x["tarih"].dt.year
    print(f"\nolay {len(ev)} | evren ornegi {len(bs)}")
    print("(fark = olay - ayni ay evren; t = ay bazli)")

    print("\n=== GENEL ===")
    kiyas(ev, bs, "tum donem")

    print("\n=== YARILAR ===")
    med = ev["tarih"].median()
    kiyas(ev[ev["tarih"] <= med], bs[bs["tarih"] <= med], "1. yari")
    kiyas(ev[ev["tarih"] > med], bs[bs["tarih"] > med], "2. yari")

    print("\n=== YILA GORE ===")
    for y in sorted(ev["yil"].unique()):
        kiyas(ev[ev["yil"] == y], bs[bs["yil"] == y], str(y))

    print("\n=== ENDEKS 60g DILIMI (evren terciles) ===")
    q1, q2 = bs["piyasa60"].quantile([1 / 3, 2 / 3])
    print(f"esikler: {q1:.1f} / {q2:.1f}")
    kiyas(ev[ev["piyasa60"] <= q1], bs[bs["piyasa60"] <= q1], "endeks zayif")
    kiyas(ev[(ev["piyasa60"] > q1) & (ev["piyasa60"] <= q2)],
          bs[(bs["piyasa60"] > q1) & (bs["piyasa60"] <= q2)], "endeks orta")
    kiyas(ev[ev["piyasa60"] > q2], bs[bs["piyasa60"] > q2], "endeks guclu")
    print("\nbitti")


main()
