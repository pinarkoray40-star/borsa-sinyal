import warnings
import numpy as np
import pandas as pd
import buyuk as B

warnings.filterwarnings("ignore")

HOLD = B.PARAM["hold"]
SL = B.PARAM["sl"]
CD = B.PARAM["cooldown"]
N = 20
C1 = 0.004                      # tek yon %0,4 = gidis-donus %0,8
BOLUM = pd.Timestamp("2024-01-01")
DN_ESIK = 70
TOHUMLAR = [0, 1, 2, 3, 4]


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
    import yfinance as yf
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
    xu = yf.download("XU100.IS", period="6y", interval="1d",
                     auto_adjust=True, progress=False)["Close"].squeeze()
    xu = xu.reindex(close.index).ffill()
    return close, opn, vol, xu


def skorlar(close, xu):
    """dn: endeks dusen gunlerinde son 120 gunde hissenin endekse gore
    ortalama fazla getirisi, hisseler arasi yuzdelik (0-100).
    rs: 60 gun getiri - endeks 60 gun getiri. dv: dusuk oynaklik (60g std, eksi)."""
    r = close.pct_change()
    ri = xu.pct_change()
    down = (ri < 0).astype(float)
    ok = r.notna().astype(float).mul(down, axis=0)
    num = r.fillna(0).mul(down, axis=0).rolling(120, min_periods=60).sum()
    den = ok.rolling(120, min_periods=60).sum()
    avg = (num / den).where(den >= 30)
    inum = (ri.fillna(0) * down).rolling(120, min_periods=60).sum()
    iden = down.rolling(120, min_periods=60).sum()
    exc = avg.sub(inum / iden, axis=0)
    dn = exc.rank(axis=1, pct=True) * 100
    rs = close.pct_change(60).sub(xu.pct_change(60), axis=0)
    dv = -(r.rolling(60).std())
    return {"dn": dn.to_numpy(), "rs": rs.to_numpy(), "dv": dv.to_numpy()}


def olaylar(C, O, ii, jj, c, sk):
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
        if not np.isfinite(px):
            continue
        ret = (px / E) * (1 - c) ** 2 - 1
        ev.append(dict(e=e, x=x, j=int(j), ret=float(ret), p=int(p),
                       dn=float(sk["dn"][p, j]), rs=float(sk["rs"][p, j]),
                       dv=float(sk["dv"][p, j])))
    return ev


def simule(ev, C, O, c, anahtar=None, tohum=None, atla=None, filtre=None):
    T = C.shape[0]
    Cf = pd.DataFrame(C).ffill().to_numpy()
    rng = np.random.default_rng(tohum) if tohum is not None else None
    byE = {}
    for i, v in enumerate(ev):
        if atla is not None and i in atla:
            continue
        if filtre is not None and not filtre(v):
            continue
        byE.setdefault(v["e"], []).append(v)
    if not byE:
        return None
    t0 = min(byE)
    cash = 100.0
    pos = []
    eq = np.full(T, np.nan)
    taken = seen = 0
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
        cand = list(byE.get(t, []))
        if anahtar is not None:
            cand.sort(key=lambda v: -(v[anahtar] if np.isfinite(v[anahtar]) else -1e18))
        elif rng is not None and len(cand) > 1:
            cand = [cand[i] for i in rng.permutation(len(cand))]
        for v in cand:
            seen += 1
            if len(pos) >= N:
                continue
            E = O[t, v["j"]]
            if not (np.isfinite(E) and E > 0):
                continue
            tot = cash + sum(sh * Cf[t - 1, jj] for _, jj, sh in pos)
            alloc = min(tot / N, cash)
            if alloc <= 0:
                continue
            cash -= alloc
            pos.append((v["x"], v["j"], alloc * (1 - c) / E))
            taken += 1
        eq[t] = cash + sum(sh * Cf[t, jj] for _, jj, sh in pos)
    return eq, taken, seen


def donem(s):
    s = s.dropna()
    if len(s) < 60:
        return np.nan, np.nan
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    if yrs < 0.3:
        return np.nan, np.nan
    cagr = ((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1) * 100
    dd = (s / s.cummax() - 1).min() * 100
    return float(cagr), float(dd)


def kos(ev, C, O, idx, tend, tohumlar=None, **kw):
    rows = []
    for sd in (tohumlar if tohumlar is not None else [None]):
        out = simule(ev, C, O, C1, tohum=sd, **kw)
        if out is None:
            continue
        eq, taken, seen = out
        s = pd.Series(eq, index=idx).loc[:idx[tend]].dropna()
        a = donem(s)
        p1 = donem(s.loc[:BOLUM - pd.Timedelta(days=1)])
        p2 = donem(s.loc[BOLUM:])
        rows.append([a[0], a[1], p1[0], p1[1], p2[0], p2[1], taken, seen])
    if not rows:
        return None
    return np.nanmean(np.array(rows, dtype=float), axis=0)


def calistir(close, opn, vol, xu):
    print("hisse:", close.shape[1], "| gun:", close.shape[0])
    C = close.to_numpy()
    O = opn.to_numpy()
    idx = close.index
    mask = B.ozellikler(close, vol)
    ii, jj = B.sinyaller(mask, CD)
    sk = skorlar(close, xu)
    ev = olaylar(C, O, ii, jj, C1, sk)
    print("sonuclanan sinyal:", len(ev))
    if len(ev) < 30:
        print("sinyal cok az, durdu")
        return
    tend = max(v["x"] for v in ev)
    rets = np.array([v["ret"] for v in ev])
    top3 = set(np.argsort(rets)[::-1][:max(1, int(len(ev) * 0.03))].tolist())
    dnv = np.array([v["dn"] for v in ev])
    gec = int(np.sum(np.isfinite(dnv) & (dnv >= DN_ESIK)))
    print(f"dn>={DN_ESIK} sartini gecen sinyal: {gec}/{len(ev)}")

    f_dn = lambda v: np.isfinite(v["dn"]) and v["dn"] >= DN_ESIK
    varyant = [
        ("BH rastgele sira (taban)", dict(tohumlar=TOHUMLAR)),
        ("BH2* (dn>=70) rastgele", dict(tohumlar=TOHUMLAR, filtre=f_dn)),
        ("BH sirala: dayaniklilik", dict(anahtar="dn")),
        ("BH sirala: goreceli guc", dict(anahtar="rs")),
        ("BH sirala: dusuk oynaklik", dict(anahtar="dv")),
        ("BH2* + sirala dayaniklilik", dict(anahtar="dn", filtre=f_dn)),
    ]
    sonuc = {}
    print("\n=== PORTFOY N=20, ertesi acilis girisi, maliyet gidis-donus %0,8 ===")
    print("(yillik % / max dusus %; tum = 2021-2026; d1 = 2021-23; d2 = 2024-26;")
    print(" top3yok = en iyi %3 islem cikarilinca tum donem yillik %)")
    for ad, kw in varyant:
        r = kos(ev, C, O, idx, tend, **kw)
        r3 = kos(ev, C, O, idx, tend, atla=top3, **kw)
        if r is None:
            continue
        sonuc[ad] = (r, r3)
        print(f"{ad}\n   tum {r[0]:.0f}/{r[1]:.0f} | d1 {r[2]:.0f}/{r[3]:.0f} | "
              f"d2 {r[4]:.0f}/{r[5]:.0f} | alinan {r[6]:.0f}/{r[7]:.0f} | "
              f"top3yok {r3[0]:.0f}")

    base = sonuc.get("BH rastgele sira (taban)")
    if base is None:
        return
    print("\n=== KARAR (taban = rastgele sira) ===")
    print("kabul adayi: d1 ve d2 yillik tabandan yuksek VE top3yok tabandan yuksek")
    for ad, (r, r3) in sonuc.items():
        if ad.startswith("BH rastgele"):
            continue
        k1 = r[2] > base[0][2] and r[4] > base[0][4]
        k2 = r3[0] > base[1][0]
        dd = r[1] - base[0][1]
        print(f"{ad}: d1 {r[2] - base[0][2]:+.0f} d2 {r[4] - base[0][4]:+.0f} "
              f"top3yok {r3[0] - base[1][0]:+.0f} dusus {dd:+.0f} -> "
              f"{'KABUL ADAYI' if (k1 and k2) else 'RED'}")
    print("\nbitti")


def main():
    close, opn, vol, xu = indir(sembol_oku())
    calistir(close, opn, vol, xu)


if __name__ == "__main__":
    main()
