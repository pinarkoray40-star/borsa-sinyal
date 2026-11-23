import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import buyuk as B

warnings.filterwarnings("ignore")

HOLD = B.PARAM["hold"]
SL = B.PARAM["sl"]
CD = B.PARAM["cooldown"]


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
    cl, op, vo = [], [], []
    for k in range(0, len(syms), 40):
        grp = syms[k:k + 40]
        try:
            df = yf.download(grp, period="6y", interval="1d",
                             auto_adjust=True, progress=False, threads=True)
            tmp = []
            for nm in ("Close", "Open", "Volume"):
                x = df[nm]
                if isinstance(x, pd.Series):
                    x = x.to_frame(grp[0])
                tmp.append(x)
            cl.append(tmp[0])
            op.append(tmp[1])
            vo.append(tmp[2])
        except Exception as e:
            print("indirme hatasi", k, e)
    close = pd.concat(cl, axis=1).sort_index()
    close = close.loc[close.notna().any(axis=1)]
    close = close.loc[:, close.notna().sum() >= 300]
    opn = pd.concat(op, axis=1).sort_index().reindex(
        index=close.index, columns=close.columns)
    vol = pd.concat(vo, axis=1).sort_index().reindex(
        index=close.index, columns=close.columns)
    return close, opn, vol


def olaylar(C, O, ii, jj, c):
    T = C.shape[0]
    ev = []
    for p, j in zip(ii, jj):
        if p + HOLD + 2 >= T:
            continue
        e = p + 1
        E = O[e, j]
        if not (np.isfinite(E) and E > 0 and np.isfinite(C[p, j])):
            continue
        path = (C[p + 1:p + HOLD + 1, j] / E - 1) * 100
        st, k, xr, _ = B.takip(path, SL, HOLD)
        if xr is None:
            continue
        x = p + k + 1
        px = O[x, j]
        if not (np.isfinite(px) and px > 0):
            px = C[p + k, j]
        ret = (px / E) * (1 - c) ** 2 - 1
        ev.append((e, x, j, ret))
    ev.sort(key=lambda a: (a[0], a[2]))
    return ev


def simule(ev, C, O, N, c, skip=None):
    T = C.shape[0]
    Cf = pd.DataFrame(C).ffill().to_numpy()
    byE = {}
    for idx, (e, x, j, r) in enumerate(ev):
        if skip and idx in skip:
            continue
        byE.setdefault(e, []).append((x, j))
    if not byE:
        return None
    t0 = min(byE)
    cash = 100.0
    pos = []
    eq = np.full(T, np.nan)
    taken = seen = 0
    occ = []
    for t in range(t0, T):
        keep = []
        for x, j, sh in pos:
            if x == t:
                px = O[t, j]
                if not (np.isfinite(px) and px > 0):
                    px = Cf[t, j]
                cash += sh * px * (1 - c)
            else:
                keep.append((x, j, sh))
        pos = keep
        for x, j in byE.get(t, []):
            seen += 1
            if len(pos) >= N:
                continue
            E = O[t, j]
            if not (np.isfinite(E) and E > 0):
                continue
            tot = cash + sum(sh * Cf[t - 1, jj] for _, jj, sh in pos)
            alloc = min(tot / N, cash)
            if alloc <= 0:
                continue
            cash -= alloc
            pos.append((x, j, alloc * (1 - c) / E))
            taken += 1
        eq[t] = cash + sum(sh * Cf[t, jj] for _, jj, sh in pos)
        occ.append(len(pos))
    return eq, taken, seen, float(np.mean(occ))


def aylik(s):
    try:
        m = s.resample("ME").last()
    except Exception:
        m = s.resample("M").last()
    return m.pct_change().dropna() * 100


def rapor(ad, s, taken=0, seen=0, occ=0.0):
    s = s.dropna()
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    tot = s.iloc[-1] / s.iloc[0]
    cagr = (tot ** (1 / yrs) - 1) * 100
    dd = s / s.cummax() - 1
    best = 0
    start = None
    for d, v in dd.items():
        if v < 0:
            if start is None:
                start = d
            best = max(best, (d - start).days)
        else:
            start = None
    m = aylik(s)
    line = (f"{ad}: x{tot:.2f} | yillik %{cagr:.0f} | max dusus %{dd.min() * 100:.0f}"
            f" | en uzun sudan cikamama {best}g | poz ay %{(m > 0).mean() * 100:.0f}"
            f" | en kotu ay %{m.min():.0f}")
    if taken:
        line += f" | poz ort {occ:.1f} | alinan {taken}/{seen}"
    print(line)


def yillik(s):
    s = s.dropna()
    ye = s.groupby(s.index.year).last()
    prev = s.iloc[0]
    out = {}
    for y, v in ye.items():
        out[y] = (v / prev - 1) * 100
        prev = v
    return out


def main():
    close, opn, vol = indir(sembol_oku())
    print("hisse:", close.shape[1], "| gun:", close.shape[0])
    C = close.to_numpy()
    O = opn.to_numpy()
    mask = B.ozellikler(close, vol)
    ii, jj = B.sinyaller(mask, CD)
    c1 = 0.004      # tek yon %0,4 = gidis-donus %0,8
    c2 = 0.0075     # gidis-donus %1,5
    ev = olaylar(C, O, ii, jj, c1)
    print("sonuclanan sinyal:", len(ev))
    tend = max(e[1] for e in ev)
    rets = np.array([e[3] for e in ev])
    top3 = set(np.argsort(rets)[::-1][:max(1, int(len(ev) * 0.03))].tolist())

    xu = yf.download("XU100.IS", period="6y", interval="1d",
                     auto_adjust=True, progress=False)["Close"].squeeze()
    xu = xu.reindex(close.index).ffill()

    sims = [("N=10", 10, c1, None), ("N=20", 20, c1, None),
            ("N=40", 40, c1, None),
            ("N=20 maliyet %1,5", 20, c2, None),
            ("N=20 en iyi %3 islem yok", 20, c1, top3)]
    res = {}
    print("\n=== PORTFOY (esit pay = ozkaynak/N, maliyet gidis-donus %0,8) ===")
    for ad, N, c, skip in sims:
        out = simule(ev, C, O, N, c, skip)
        if out is None:
            continue
        eq, taken, seen, occ = out
        s = pd.Series(eq, index=close.index).loc[:close.index[tend]].dropna()
        res[ad] = s
        rapor(ad, s, taken, seen, occ)

    s20 = res["N=20"]
    xs = xu.loc[s20.index[0]:close.index[tend]]
    print()
    rapor("XU100 ayni donem", xs)

    print("\n=== YIL YIL (N=20) ===")
    a = yillik(s20)
    b = yillik(xs)
    for y in sorted(a):
        print(f"{y}: strateji %{a[y]:.0f} | XU100 %{b.get(y, float('nan')):.0f}")

    print("\n=== PAY BUYUKLUGU ETKISI ===")
    print("tek pozisyonun -%37'si (en kotu islem) ozkaynaktan: "
          f"N=10 %{37 / 10:.1f} | N=20 %{37 / 20:.1f} | N=40 %{37 / 40:.1f}")
    print("\nbitti")


main()
