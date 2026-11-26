import numpy as np
import pandas as pd
import yfinance as yf

from bilanco_test import bilanco_yukle, fiyatlar, LAG
from tavan_aday import sembol_oku

MALIYET = 0.008


def skorlar(sub):
    rk = sub[["d_opm", "d_nm", "d_roe", "roe"]].rank(pct=True)
    return rk.mean(axis=1)


def sec(sub):
    out = {}
    ik = skorlar(sub).sort_values(ascending=False)
    out["ik_top20"] = list(ik.index[:20])
    out["ik_top30"] = list(ik.index[:30])
    out["roe_top20"] = list(sub["roe"].sort_values(ascending=False).index[:20])
    temiz = sub[(sub["roe"] <= 0.8) & (sub["d_opm"] <= 0.5) & (sub["d_nm"] <= 0.5)]
    if len(temiz) >= 40:
        ikt = skorlar(temiz).sort_values(ascending=False)
        out["temiz_top20"] = list(ikt.index[:20])
    else:
        out["temiz_top20"] = []
    ikr = ik.rank(pct=True)
    out["eleme_Q1"] = list(ikr[ikr > 0.2].index)
    out["evren"] = list(sub.index)
    return out


def simule(portfoyler, pozlar, closes):
    idx = closes.index[pozlar[0]:]
    equity = pd.Series(np.nan, index=idx)
    eq = 1.0
    onceki = set()
    for i, pos in enumerate(pozlar):
        bitis = pozlar[i + 1] if i + 1 < len(pozlar) else len(closes) - 1
        tk = portfoyler[i]
        if not tk:
            equity.loc[closes.index[pos:bitis + 1]] = eq
            continue
        yeni = len(set(tk) - onceki) / len(tk)
        eq *= (1 - MALIYET * yeni)
        px = closes[tk].iloc[pos:bitis + 1].ffill()
        rel = (px / px.iloc[0]).mean(axis=1)
        seg = eq * rel
        equity.loc[seg.index] = seg.values
        eq = float(seg.iloc[-1])
        onceki = set(tk)
    return equity.ffill()


def ozet(eq):
    eq = eq.dropna()
    eq = eq / eq.iloc[0]
    yil = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] ** (1 / yil) - 1) * 100
    dd = (eq / eq.cummax() - 1).min() * 100
    return cagr, dd, eq.iloc[-1]


def yillik(eq):
    eq = eq.dropna()
    eq = eq / eq.iloc[0]
    son = eq.groupby(eq.index.year).last()
    onc = son.shift(1)
    onc.iloc[0] = 1.0
    return (son / onc - 1) * 100


def donem_getiri(eq, pozlar, closes):
    tarih = [closes.index[p] for p in pozlar] + [closes.index[-1]]
    v = eq.reindex(tarih).ffill()
    return v.pct_change().dropna()


def main():
    w = bilanco_yukle()
    closes, tl = fiyatlar(sembol_oku())
    med60 = tl.rolling(60, min_periods=30).median()
    lik_rank = med60.rank(axis=1, pct=True)

    olaylar = []
    for (yil, ay), g in sorted(w.groupby(["yil", "ay"]), key=lambda x: x[0]):
        avail = (pd.Timestamp(int(yil), int(ay), 1) + pd.offsets.MonthEnd(0)
                 + pd.Timedelta(days=LAG[int(ay)]))
        pos = closes.index.searchsorted(avail)
        if pos >= len(closes) - 1:
            continue
        olaylar.append((pos, int(yil), int(ay), g.set_index("kod")))

    pozlar, port, etiket = [], {}, []
    for pos, yil, ay, g in olaylar:
        lik = lik_rank.iloc[pos]
        px0 = closes.iloc[pos]
        ok = [k for k in g.index
              if k in lik.index and lik[k] >= 0.3 and pd.notna(px0.get(k))]
        if len(ok) < 100:
            continue
        sub = g.loc[ok]
        pozlar.append(pos)
        etiket.append((yil, ay))
        for k, v in sec(sub).items():
            port.setdefault(k, []).append(v)
    print(f"rebalance sayisi: {len(pozlar)}  ilk giris: {closes.index[pozlar[0]].date()}"
          f"  son veri: {closes.index[-1].date()}", flush=True)

    sonuc = {}
    for ad, lst in port.items():
        sonuc[ad] = simule(lst, pozlar, closes)
    idx = closes.index[pozlar[0]:]
    try:
        xu = yf.download("XU100.IS", start="2019-06-01", interval="1d",
                         auto_adjust=True, progress=False)
        xc = xu["Close"].iloc[:, 0] if isinstance(xu.columns, pd.MultiIndex) else xu["Close"]
        xc.index = pd.to_datetime(xc.index).tz_localize(None).normalize()
        xc = xc[~xc.index.duplicated()].reindex(idx).ffill()
        sonuc["XU100"] = xc / xc.iloc[0]
    except Exception as e:
        print("XU100 alinamadi:", e)

    print("=" * 96)
    print(f"{'STRATEJI':<12}{'YILLIK%':>9}{'MAKS DUSUS%':>13}{'SON(x)':>9}"
          f"{'evreni gecen donem%':>22}")
    evren_r = donem_getiri(sonuc["evren"], pozlar, closes)
    for ad, eq in sonuc.items():
        c, d, s = ozet(eq)
        if ad in ("evren", "XU100"):
            hit = ""
        else:
            r = donem_getiri(eq, pozlar, closes)
            hit = f"{(r > evren_r).mean() * 100:.0f}"
        print(f"{ad:<12}{c:>9.1f}{d:>13.1f}{s:>9.2f}{hit:>22}")

    print("=" * 96)
    print("YIL YIL GETIRI (%):")
    tablo = pd.DataFrame({ad: yillik(eq) for ad, eq in sonuc.items()})
    print("yil   " + " ".join(f"{c[:9]:>9}" for c in tablo.columns))
    for y, satir in tablo.iterrows():
        print(f"{y}  " + " ".join(f"{v:9.1f}" for v in satir.values))

    print("=" * 96)
    yil, ay = etiket[-1]
    print(f"GUNCEL PORTFOY (donem {yil}/{ay}, rebalance: {closes.index[pozlar[-1]].date()}):")
    for ad in ("ik_top20", "temiz_top20"):
        print(f"  {ad}: " + ", ".join(port[ad][-1]))
    print("Sonraki rebalance: yillik bilanco sonrasi, yaklasik Mart 2027 ortasi.")


if __name__ == "__main__":
    main()
