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


def bench_acilis(O, L, p, e, x):
    a = O[e]
    b = O[x]
    ok = L[p] & np.isfinite(a) & np.isfinite(b) & (a > 0)
    return float(np.mean(b[ok] / a[ok] - 1) * 100) if ok.any() else np.nan


def simule(C, O, L, ii, jj, idx):
    T = C.shape[0]
    rows = []
    for p, j in zip(ii, jj):
        if p + HOLD + 2 >= T:
            continue
        e = p + 1
        if not (np.isfinite(O[e, j]) and O[e, j] > 0
                and np.isfinite(C[p, j]) and C[p, j] > 0):
            continue
        # 1) ideal: buyuk.py aynen (giris sinyal kapanisi, cikis kapanis)
        pathA = (C[p + 1:p + HOLD + 1, j] / C[p, j] - 1) * 100
        stA, kA, xrA, _ = B.takip(pathA, SL, HOLD)
        # 2) gercekci: giris ertesi acilis, cikis tetik sonrasi ertesi acilis
        E = O[e, j]
        pathR = (C[p + 1:p + HOLD + 1, j] / E - 1) * 100
        stR, kR, xrR, _ = B.takip(pathR, SL, HOLD)
        if xrA is None or xrR is None:
            continue
        x = p + kR + 1
        px = O[x, j]
        if not (np.isfinite(px) and px > 0):
            px = C[p + kR, j]
        ret = (px / E - 1) * 100
        rows.append(dict(
            tarih=idx[p], sym=j, gun=kR, durum=stR,
            A=xrA, bA=B.kiyas(C, L, p, int(kA)),
            R=ret, bR=bench_acilis(O, L, p, e, x),
            gap_in=(E / C[p, j] - 1) * 100,
            gap_out=((px / C[p + kR, j] - 1) * 100) if stR == "stop" else np.nan,
        ))
    return pd.DataFrame(rows)


def stat(e, rc, bc, ad):
    e = e.dropna(subset=[rc, bc])
    if len(e) < 20:
        print(f"{ad}: n={len(e)} cok az")
        return
    d = e[rc] - e[bc]
    m = d.groupby(e["ay"]).mean()
    M = len(m)
    t = m.mean() / (m.std() / np.sqrt(M)) if M >= 4 else float("nan")
    print(f"{ad}: n={len(e)} ort {e[rc].mean():.1f} med {e[rc].median():.1f} "
          f"kaz %{(e[rc] > 0).mean() * 100:.0f} | evren {e[bc].mean():.1f} "
          f"fark {d.mean():+.1f} t={t:.1f}")


def main():
    syms = sembol_oku()
    close, opn, vol = indir(syms)
    print("hisse:", close.shape[1], "/", len(syms), "| gun:", close.shape[0])
    C = close.to_numpy()
    O = opn.to_numpy()
    L = B.likit_maske(close, vol)
    mask = B.ozellikler(close, vol)
    ii, jj = B.sinyaller(mask, CD)
    print("ham sinyal:", len(ii))
    ev = simule(C, O, L, ii, jj, close.index)
    ev["hisse"] = [close.columns[j].replace(".IS", "") for j in ev["sym"]]
    ev["ay"] = ev["tarih"].dt.to_period("M")
    ev["yil"] = ev["tarih"].dt.year
    for c in (0.8, 1.5):
        ev[f"R{c}"] = ev["R"] - c
    ev.to_csv("gercekcilik.csv", index=False)
    print(f"sonuclanan islem: {len(ev)} | {ev['tarih'].min().date()} - "
          f"{ev['tarih'].max().date()}")

    print("\n=== KAYMA (%) ===")
    print(f"giris acilis - sinyal kapanisi: ort {ev['gap_in'].mean():+.2f} "
          f"med {ev['gap_in'].median():+.2f}")
    s = ev["gap_out"].dropna()
    print(f"stop cikisi acilis - tetik kapanisi: n={len(s)} ort {s.mean():+.2f} "
          f"med {s.median():+.2f} en kotu {s.min():.1f}")
    print(f"stop %{(ev['durum'] == 'stop').mean() * 100:.0f} | "
          f"sure %{(ev['durum'] == 'süre').mean() * 100:.0f}")

    print("\n=== GENEL (fark = islem - ayni zamanlamada evren, evren maliyetsiz) ===")
    stat(ev, "A", "bA", "1 ideal (buyuk.py)      ")
    stat(ev, "R", "bR", "2 gercekci, maliyet 0    ")
    stat(ev, "R0.8", "bR", "3 gercekci, maliyet %0,8 ")
    stat(ev, "R1.5", "bR", "4 gercekci, maliyet %1,5 ")

    med = ev["tarih"].median()
    print("\n=== YARILAR (senaryo 3: maliyet %0,8) ===")
    stat(ev[ev["tarih"] <= med], "R0.8", "bR", "1. yari")
    stat(ev[ev["tarih"] > med], "R0.8", "bR", "2. yari")

    print("\n=== YILA GORE (senaryo 3) ===")
    for y in sorted(ev["yil"].unique()):
        stat(ev[ev["yil"] == y], "R0.8", "bR", str(y))
    print("\nyillara gore ideal:")
    for y in sorted(ev["yil"].unique()):
        stat(ev[ev["yil"] == y], "A", "bA", str(y))

    print("\nen iyi 5 / en kotu 5 (gercekci, brut %):")
    r = ev.sort_values("R")
    print(r.tail(5)[["hisse", "tarih", "R"]].to_string(index=False))
    print(r.head(5)[["hisse", "tarih", "R"]].to_string(index=False))
    print("\nbitti")


main()
