"""Sağlamlık testi: "6 ay en iyi %10 momentum + sıkışma" kuralı gerçekten tutarlı mı?

Sıkışma = son 20 günün oynaklığı, son 120 günün oynaklığının yarısından az.
Kontroller: yıllara/aylara bölünme, tek aya bağımlılık, birbirine bindirmeyen örnekler, portföy simülasyonu.
Yatırım tavsiyesi değildir.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from signals import fetch_all, read_symbols

COST = 0.4


def pr(df):
    return df.rank(axis=1, pct=True)


def summarize(label, ret, base2, h2):
    if len(ret) < 20:
        return f"- **{label}**: yetersiz gözlem ({len(ret)})"
    return (f"- **{label}**\n"
            f"  - {len(ret):,} gözlem · ortalama {ret.mean() * 100:+.0f}% · medyan {np.median(ret) * 100:+.0f}% · "
            f"zararda %{(ret < 0).mean() * 100:.0f} · 2x yapan %{(ret >= 1).mean() * 100:.1f} "
            f"(genel %{base2 * 100:.1f})").replace(",", ".")


def run_rule(name, mask, close, fwd126, valid, fwd21, base2, L, liquid_note):
    L.append(f"## {name}\n")
    m = mask & valid
    sig_days = m.values.sum()
    if sig_days < 50:
        L.append("- Yeterli sinyal yok.\n")
        return None
    r = fwd126.values[m.values]
    L.append(summarize("Tüm sinyal günleri (126 gün sonra)", r, base2, 126))
    r21 = fwd21.values[m.values & fwd21.notna().values]
    if len(r21):
        L.append(f"- 21 gün sonra: ortalama {r21.mean() * 100:+.1f}% · medyan {np.median(r21) * 100:+.1f}% · "
                 f"kazanan %{(r21 > 0).mean() * 100:.0f}")
    stocks = int(m.any(axis=0).sum())
    months = int(m.any(axis=1).groupby(m.index.to_period("M")).sum().gt(0).sum())
    L.append(f"- {stocks} farklı hissede, {months} farklı ayda sinyal çıkmış\n")

    # birbirine bindirmeyen: hisse başına ilk sinyal, sonra 21 gün bekle
    eps = []
    mv = m.values
    for j in range(mv.shape[1]):
        i = 0
        col = mv[:, j]
        idxs = np.where(col)[0]
        last = -999
        for i in idxs:
            if i - last >= 21:
                eps.append((close.index[i], j, float(fwd126.iat[i, j])))
                last = i
    if eps:
        er = np.array([e[2] for e in eps])
        L.append(summarize("Birbirine bindirmeyen örnekler (hisse başına 21 günde bir)", er, base2, 126))
        L.append("")

    # aylara bağımlılık
    df = pd.DataFrame({"d": [e[0] for e in eps], "r": [e[2] for e in eps]})
    if len(df) > 30:
        df["ym"] = df["d"].dt.to_period("M")
        g = df.groupby("ym")["r"].agg(["count", "mean", lambda x: (x >= 1).sum()])
        g.columns = ["n", "mean", "w2"]
        tot2 = g["w2"].sum()
        top = g.sort_values("w2", ascending=False).head(4)
        if tot2 > 0:
            L.append("**2x yapanların en çok toplandığı aylar:** " + " · ".join(
                f"{p.strftime('%m.%Y')}: {int(r.w2)} ({r.w2 / tot2 * 100:.0f}%)" for p, r in top.iterrows()))
            top3 = set(g.sort_values("w2", ascending=False).head(3).index)
            rest = df[~df["ym"].isin(top3)]["r"].values
            L.append(summarize("En iyi 3 ay çıkarılınca", rest, base2, 126))
        L.append("")
        L.append("**Yıllara göre (bindirmeyen örnekler, 126 gün sonra):**")
        for y, gg in df.groupby(df["d"].dt.year):
            if len(gg) >= 10:
                L.append(f"- {y}: {len(gg)} örnek · ortalama {gg['r'].mean() * 100:+.0f}% · "
                         f"medyan {gg['r'].median() * 100:+.0f}% · zararda %{(gg['r'] < 0).mean() * 100:.0f} · "
                         f"2x %{(gg['r'] >= 1).mean() * 100:.0f}")
        L.append("")
    return eps


def portfolio(close, score, elig, step=21, topn=10, cost=COST):
    idx = close.index
    rets, dates, held = [], [], []
    for p in range(130, len(idx) - step, step):
        ok = elig.iloc[p]
        sc = score.iloc[p].where(ok).dropna()
        if sc.empty:
            rets.append(0.0); dates.append(idx[p + step]); held.append(0)
            continue
        pick = sc.nlargest(topn).index
        fw = (close.iloc[p + step][pick] / close.iloc[p][pick] - 1).dropna()
        if fw.empty:
            rets.append(0.0); dates.append(idx[p + step]); held.append(0)
            continue
        # boş kalan pozisyonlar nakitte (getiri 0)
        rets.append(fw.sum() / topn - cost / 100 * len(fw) / topn)
        dates.append(idx[p + step]); held.append(len(fw))
    return pd.Series(rets, index=dates), held


def pstats(label, rets, held=None):
    if len(rets) < 6:
        return f"- **{label}**: yetersiz veri"
    eq = (1 + rets).cumprod()
    yrs = len(rets) * 21 / 252
    dd = (eq / eq.cummax() - 1).min()
    s = (f"- **{label}**\n"
         f"  - 1 TL → **{eq.iloc[-1]:.2f} TL** ({yrs:.1f} yılda) · yıllık ≈ %{(eq.iloc[-1] ** (1 / yrs) - 1) * 100:.0f}\n"
         f"  - en kötü düşüş %{dd * 100:.0f} · en kötü ay %{rets.min() * 100:.0f} · aylık kazanma %{(rets > 0).mean() * 100:.0f}")
    if held:
        h = np.array(held)
        s += f"\n  - ortalama {h.mean():.1f} hisse tutuldu · hiç sinyal olmayan ay: {(h == 0).sum()}/{len(h)}"
    ym = rets.groupby(rets.index.year).apply(lambda x: (1 + x).prod() - 1)
    s += "\n  - yıllara göre: " + " · ".join(f"{y}: {v * 100:+.0f}%" for y, v in ym.items())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="5y")
    ap.add_argument("--symbols-file", default="bist_symbols.txt")
    ap.add_argument("--out", default="squeeze.md")
    args = ap.parse_args()

    symbols = read_symbols(args.symbols_file)
    print(f"{len(symbols)} sembol indiriliyor ({args.period})...")
    close, volume = fetch_all(symbols, args.period)
    if close.shape[1] < 20:
        raise SystemExit(f"Yeterli veri yok ({close.shape[1]} hisse).")

    chg = close.pct_change(fill_method=None)
    r126 = close / close.shift(126) - 1
    vol20 = chg.rolling(20).std()
    vol120 = chg.rolling(120).std()
    turn = (close * volume).rolling(20).median()
    liquid = pr(turn) >= 0.30
    fwd126 = close.shift(-126) / close - 1
    fwd21 = close.shift(-21) / close - 1
    valid = fwd126.notna() & r126.notna() & vol120.notna() & turn.notna()
    base2 = float((fwd126 >= 1)[valid].values[valid.values].mean())
    mom = pr(r126.where(liquid)) >= 0.90
    mom_all = pr(r126) >= 0.90
    squeeze = vol20 < 0.5 * vol120

    L = ["# Sağlamlık testi: momentum + sıkışma",
         f"{close.shape[1]} hisse · {close.index[0].date()} → {close.index[-1].date()} · "
         f"rastgele bir hissenin 126 günde 2x yapma ihtimali %{base2 * 100:.1f}.\n",
         "**Kural:** son 6 ay getirisi en iyi %10 **ve** son 20 gün oynaklığı, 120 günlük oynaklığın yarısından az.\n"]

    run_rule("A) Tüm hisseler (önceki testteki gibi)", mom_all & squeeze, close, fwd126, valid, fwd21, base2, L, "")
    run_rule("B) Sadece likit hisseler (işlem hacmi en düşük %30 hariç)", mom & squeeze & liquid, close, fwd126, valid,
             fwd21, base2, L, "")
    run_rule("C) Kıyas: sadece momentum (sıkışma şartı yok), likit", mom & liquid, close, fwd126, valid, fwd21, base2, L, "")

    L.append("## Portföy simülasyonu\n")
    L.append("Her 21 işlem gününde kurala uyan en fazla 10 hisseye eşit pay (boş pozisyon nakit), "
             "bir sonraki yenilemede sat, %0,4 maliyet dahil.\n")
    sc = r126
    r1, h1 = portfolio(close, sc, mom_all & squeeze)
    r2, h2 = portfolio(close, sc, mom & squeeze & liquid)
    r3, h3 = portfolio(close, sc, mom & liquid)
    rnd = pd.DataFrame(np.random.default_rng(0).random(close.shape), index=close.index, columns=close.columns)
    r4, h4 = portfolio(close, rnd, liquid)
    L.append(pstats("Momentum + sıkışma, tüm hisseler", r1, h1))
    L.append(pstats("Momentum + sıkışma, sadece likit", r2, h2))
    L.append(pstats("Sadece momentum (en iyi 10), likit", r3, h3))
    L.append(pstats("Kıyas: rastgele 10 likit hisse", r4, h4))
    L.append("")
    L.append("> Uyarı: Veri bugünkü hisse listesidir (batanlar yok), sonuçlar iyimsernir. "
             "Spread/kayma tam yansıtılmamıştır, küçük hisselerde gerçek alım satım daha pahalıdır. "
             "Geçmiş gelecek için garanti değildir. Yatırım tavsiyesi değildir.")

    text = "\n".join(L)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
