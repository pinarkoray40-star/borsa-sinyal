import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

SEEDS = ["ANELE", "KTLEV", "GUNDG", "ASTOR", "SELEC"]
HEDEF = 2.0      # en az 2 kat
UFUK = 90        # +%100'e en geç 90 islem gunu
PEAKWIN = 120    # tepe aramasi


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
    for s in SEEDS:
        out.append(s + ".IS")
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
        print("indirildi", min(k + 40, len(syms)), "/", len(syms), flush=True)
    return data


def endeks():
    x = yf.download("XU100.IS", period="6y", interval="1d",
                    auto_adjust=True, progress=False)["Close"]
    return x.squeeze()


def bul(c):
    ep = []
    i = 0
    n = len(c)
    while i < n - 5:
        lim = min(n, i + UFUK + 1)
        hit = np.where(c[i + 1:lim] >= c[i] * HEDEF)[0]
        if len(hit) == 0:
            i += 1
            continue
        j = i + 1 + int(hit[0])
        s = i + int(np.argmin(c[i:j + 1]))
        e = min(n, s + PEAKWIN + 1)
        p = s + int(np.argmax(c[s:e]))
        if c[p] / c[s] >= HEDEF:
            ep.append((s, p))
            i = p + 1
        else:
            i += 1
    return ep


def onceki(d, i):
    c = d["Close"].values
    h = d["High"].values
    l = d["Low"].values
    v = d["Volume"].values
    if i < 260:
        return None
    return dict(
        r20=(c[i] / c[i - 20] - 1) * 100,
        r60=(c[i] / c[i - 60] - 1) * 100,
        rs60=((c[i] / c[i - 60] - 1) - d["xr60"].values[i]) * 100,
        dist=(c[i] / c[i - 250:i + 1].max() - 1) * 100,
        band=(h[i - 14:i + 1].max() / l[i - 14:i + 1].min() - 1) * 100,
        vr=v[i - 4:i + 1].mean() / max(v[i - 54:i - 4].mean(), 1),
        lik=float(np.median(c[i - 19:i + 1] * v[i - 19:i + 1])) / 1e6,
    )


def seyir(d, s, p):
    c = d["Close"].values
    v = d["Volume"].values
    seg = c[s:p + 1]
    run = np.maximum.accumulate(seg)
    n = len(c)
    return dict(
        gain=(c[p] / c[s] - 1) * 100,
        gun=p - s,
        mdd=(seg / run - 1).min() * 100,
        d50=int(np.argmax(seg >= c[s] * 1.5)),
        up=(np.diff(seg) > 0).mean() * 100 if len(seg) > 1 else np.nan,
        vdur=v[s:p + 1].mean() / max(v[max(0, s - 50):s].mean(), 1),
        post20=(c[p + 20] / c[p] - 1) * 100 if p + 20 < n else np.nan,
        post60=(c[p + 60] / c[p] - 1) * 100 if p + 60 < n else np.nan,
    )


def main():
    syms = sembol_oku()
    print("sembol:", len(syms))
    data = indir(syms)
    print("veri gelen:", len(data))
    xu = endeks()
    xr60 = xu / xu.shift(60) - 1

    base, eps = [], []
    for s, d in data.items():
        d = d.copy()
        d["xr60"] = xr60.reindex(d.index, method="ffill").fillna(0)
        c = d["Close"].values
        n = len(c)
        for i in range(260, n - 5, 5):
            f = onceki(d, i)
            if f:
                base.append(f)
        for (a, p) in bul(c):
            row = dict(sym=s.replace(".IS", ""),
                       bas=str(d.index[a].date()),
                       tepe=str(d.index[p].date()))
            row.update(seyir(d, a, p))
            f = onceki(d, a)
            if f:
                row.update(f)
            eps.append(row)

    base = pd.DataFrame(base)
    eps = pd.DataFrame(eps)
    eps.to_csv("yukselenler.csv", index=False)

    print("\n=== TOHUM HISSELER ===")
    for sd in SEEDS:
        k = eps[eps["sym"] == sd]
        if k.empty:
            if sd + ".IS" in data:
                c = data[sd + ".IS"]["Close"].values
                best = max(c[i + 1:i + 91].max() / c[i] for i in range(len(c) - 91))
                print(f"{sd}: episode yok (en iyi 90g: %{(best - 1) * 100:.0f})")
            else:
                print(f"{sd}: veri yok")
        else:
            for _, r in k.iterrows():
                print(f"{sd}: {r['bas']} -> {r['tepe']}  +%{r['gain']:.0f}  {int(r['gun'])}g")

    print("\n=== GENEL ===")
    print(f"episode: {len(eps)} | hisse: {eps['sym'].nunique()} / {len(data)}")
    print("yillara gore baslangic:")
    print(eps.groupby(eps["bas"].str[:4]).size().to_string())

    print("\n=== YUKSELIS SIRASINDA (medyan) ===")
    for k, ad in [("gain", "kazanc %"), ("gun", "tepe gunu"),
                  ("d50", "+%50 gunu"), ("mdd", "en derin dusus %"),
                  ("up", "yesil gun %"), ("vdur", "hacim kati"),
                  ("post20", "tepeden 20g sonra %"),
                  ("post60", "tepeden 60g sonra %")]:
        print(f"{ad}: {eps[k].median():.1f}")

    print("\n=== YUKSELIS OLNCESI: episode vs evren ===")
    print("(alt/ust 1/3 = episode'larin kacinin evrenin alt/ust ucte oldugu; %33 = fark yok)")
    for k, ad in [("r20", "20g getiri %"), ("r60", "60g getiri %"),
                  ("rs60", "60g goreceli guc"), ("dist", "250g zirveye %"),
                  ("band", "15g bant %"), ("vr", "hacim orani"),
                  ("lik", "islem hacmi mn TL")]:
        e = eps[k].dropna()
        b = base[k].dropna()
        q33, q67 = b.quantile([1 / 3, 2 / 3])
        print(f"{ad}: ep {e.median():.1f} | evren {b.median():.1f} | "
              f"alt %{(e < q33).mean() * 100:.0f} ust %{(e > q67).mean() * 100:.0f}")
    print("\nyukselenler.csv yazildi")


main()
