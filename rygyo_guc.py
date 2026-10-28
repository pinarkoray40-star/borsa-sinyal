"""Endeks düşerken ayakta kalan hisse var mı? (göreceli güç) — RYGYO test vakası.

Soru 1: BIST 100 düşerken kendi fiyatı artıda kalan / endeksten belirgin iyi giden hisseler
        sonraki günlerde de endeksi yenmeye devam ediyor mu?
Soru 2: RYGYO bu özelliği ne zaman, ne kadar süre taşıdı ve geçmişte bu işaret ne getirdi?

Tanımlar (hepsi işlem günü):
- Düşen piyasa  : endeksin son 20 günlük getirisi <= %-5
- Göreceli güç  : hissenin 20 günlük getirisi - endeksin 20 günlük getirisi
- Sıra          : göreceli gücün tüm hisseler arasındaki yüzdelik sırası (1 = en güçlü)
- Başarı        : işaretten sonraki 20 günde hissenin endeksi yenmesi

Her işaret için başarı oranı, "düşen piyasadaki rastgele hisse"nin başarı oranına bölünür (lift).
Fiyat/hacim verisiyle çalışır; yatırımcı sayısı, kurumsal oran ve fiili dolaşım verisi YOK.
Yatırım tavsiyesi değildir.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from signals import fetch_all, read_symbols

INDEX = "XU100.IS"
WIN = 20            # göreceli gücün ölçüldüğü pencere
AHEAD = 20          # sonucun bakıldığı ileri pencere
DROP = -0.05        # endeks WIN günde bu kadar düştüyse "düşen piyasa"
STRONG = 0.90       # göreceli güç sırası eşiği (ilk %10)


def runs(flag: np.ndarray):
    """True olan ardışık günlerin (başlangıç, bitiş) indeksleri."""
    out, start = [], None
    for i, f in enumerate(flag):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flag) - 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="5y")
    ap.add_argument("--symbols-file", default="bist_symbols.txt")
    ap.add_argument("--symbol", default="RYGYO", help="incelenecek hisse (ör. RYGYO, DOGUB)")
    ap.add_argument("--out", default="rygyo_guc.md")
    args = ap.parse_args()

    name = args.symbol.upper().replace(".IS", "")
    target = name + ".IS"
    symbols = read_symbols(args.symbols_file)
    if target not in symbols:
        symbols.append(target)
    print(f"{len(symbols)} sembol + endeks indiriliyor ({args.period})...")
    close, volume = fetch_all(symbols + [INDEX], args.period)
    if INDEX not in close.columns:
        raise SystemExit("Endeks (XU100.IS) verisi alınamadı.")
    if target not in close.columns:
        raise SystemExit(f"{name} için yeterli veri yok.")
    idx = close[INDEX].ffill(limit=3)
    close = close.drop(columns=[INDEX])
    volume = volume.drop(columns=[INDEX])
    if close.shape[1] < 20:
        raise SystemExit(f"Yeterli veri yok ({close.shape[1]} hisse).")

    cols = list(close.columns)
    j = cols.index(target)

    r20 = close.pct_change(WIN, fill_method=None)
    ix20 = idx.pct_change(WIN, fill_method=None)
    rs = r20.sub(ix20, axis=0)
    rank = rs.rank(axis=1, pct=True)
    falling = (ix20 <= DROP).fillna(False)
    fall_df = pd.DataFrame(np.repeat(falling.values[:, None], close.shape[1], axis=1),
                           index=close.index, columns=cols)

    fut_r = close.shift(-AHEAD) / close - 1
    fut_ix = idx.shift(-AHEAD) / idx - 1
    fut_ex = fut_r.sub(fut_ix, axis=0)
    turn = (close * volume).rolling(20).median()
    hi252 = close.rolling(252, min_periods=200).max()

    valid = r20.notna() & fut_ex.notna() & turn.notna() & (close > 0)

    # ham işaretler (son gün için de çalışır) ve gözlem için geçerli olanlar
    strong = (rank >= STRONG)
    R = {
        "A) Endeks düşerken hisse artıda (20g getiri > 0)": fall_df & (r20 > 0),
        "B) Endeks düşerken göreceli güç sırası ilk %10": fall_df & strong,
        "C) B + hisse artıda": fall_df & strong & (r20 > 0),
        "D) C + 252 gün zirvesinin %5 yakınında": fall_df & strong & (r20 > 0) & (close >= hi252 * 0.95),
    }
    R = {k: v.fillna(False) for k, v in R.items()}
    keyC = "C) B + hisse artıda"
    S = {k: (v & valid) for k, v in R.items()}

    FE, FR = fut_ex.values, fut_r.values
    vm = valid.values
    base_mask = fall_df.values & vm
    n_base = int(base_mask.sum())
    if n_base < 500:
        raise SystemExit(f"Düşen piyasa gözlemi az ({n_base}); --period 5y ile deneyin.")
    base_hit = float((FE[base_mask] > 0).mean())
    base_med = float(np.median(FE[base_mask]))

    days_ok = close.index[base_mask.any(axis=1)]
    mid = days_ok[len(days_ok) // 2]
    first = (close.index <= mid)[:, None]

    def stats(mask, sub=None):
        m = mask.values if hasattr(mask, "values") else mask
        m = m & vm
        if sub is not None:
            m = m & sub
        n = int(m.sum())
        if n == 0:
            return dict(n=0, hit=np.nan, med_ex=np.nan, med_r=np.nan)
        return dict(n=n, hit=float((FE[m] > 0).mean()), med_ex=float(np.median(FE[m])),
                    med_r=float(np.median(FR[m])))

    b1 = stats(fall_df, first)
    b2 = stats(fall_df, ~first)

    L = []
    L.append(f"# Göreceli güç testi — {name}")
    L.append(f"{close.shape[1]} hisse · {close.index[0].date()} → {close.index[-1].date()} · "
             f"düşen piyasa günü: {int(falling.sum())} (endeks {WIN} günde ≤ %{DROP * 100:.0f}) · "
             f"{n_base:,} (hisse-gün) gözlemi".replace(",", "."))
    L.append(f"**Genel olasılık:** düşen piyasada rastgele bir hissenin sonraki {AHEAD} günde endeksi yenme "
             f"ihtimali **%{base_hit * 100:.1f}** (medyan fark {base_med * 100:+.1f}%).\n")
    L.append("**Okuma:** *Lift* = işaret varken başarı ÷ genel başarı. 1 = fark yok. "
             "Tutarlılık = ilk yarı / ikinci yarı başarı oranı, ikisi de genel orandan yüksekse işaret güvenilir.\n")

    L.append("## İşaretler\n")
    for k, m in S.items():
        s = stats(m)
        if s["n"] < 200:
            L.append(f"- **{k}** · yeterli gözlem yok ({s['n']})")
            continue
        s1, s2 = stats(m, first), stats(m, ~first)
        ok = (s1["n"] > 0 and s2["n"] > 0 and s1["hit"] > b1["hit"] and s2["hit"] > b2["hit"])
        L.append(f"- **{k}**\n"
                 f"  - başarı %{s['hit'] * 100:.1f} (lift {s['hit'] / base_hit:.2f}x) · gözlem {s['n']:,}\n"
                 f"  - sonraki {AHEAD} gün medyan: hisse {s['med_r'] * 100:+.1f}% · endekse göre {s['med_ex'] * 100:+.1f}% "
                 f"(genel {base_med * 100:+.1f}%)\n"
                 f"  - tutarlılık {s1['hit'] * 100:.0f}% / {s2['hit'] * 100:.0f}% "
                 f"(genel {b1['hit'] * 100:.0f}% / {b2['hit'] * 100:.0f}%) "
                 f"{'✔ tutarlı' if ok else '✖ tutarsız'}".replace(",", "."))
    L.append("")

    # --- Hedef hisse ---
    L.append(f"## {name}: geçmiş seriler (C işareti: düşen piyasada ilk %10 + artıda)\n")
    sj = S[keyC].iloc[:, j].values
    eps = runs(sj)
    if not eps:
        L.append(f"- {name} bu işareti geçmişte hiç vermemiş.")
    else:
        exs = []
        for a, b in eps:
            e = fut_ex.iat[a, j]
            exs.append(e)
        hits = sum(1 for e in exs if e > 0)
        L.append(f"- {len(eps)} ayrı seri, ilk gününden {AHEAD} gün sonra endeksi yenen: {hits} / {len(eps)} "
                 f"· medyan fark {np.median(exs) * 100:+.1f}%")
        for (a, b), e in list(zip(eps, exs))[-8:]:
            L.append(f"  - {close.index[a]:%d.%m.%Y} → {close.index[b]:%d.%m.%Y} ({b - a + 1} gün) · "
                     f"ilk günden {AHEAD} gün sonra endekse göre {e * 100:+.1f}%")
    L.append("")

    L.append(f"## {name}: son 100 gün (5 günde bir)\n")
    n_idx = len(close.index)
    for i in sorted(range(n_idx - 1, max(n_idx - 101, 0), -5)):
        rk = rank.iat[i, j]
        flag = "✔C" if R[keyC].iat[i, j] else ("↓" if falling.iat[i] else "")
        L.append(f"- {close.index[i]:%d.%m.%Y} · {close.iat[i, j]:.2f} TL · endeks {WIN}g {ix20.iat[i] * 100:+.1f}% · "
                 f"hisse {r20.iat[i, j] * 100:+.1f}% · fark {rs.iat[i, j] * 100:+.1f}% · "
                 f"sıra %{rk * 100:.0f} {flag}")
    cur = runs(R[keyC].iloc[:, j].values)
    if cur and cur[-1][1] == n_idx - 1:
        a, b = cur[-1]
        L.append(f"\n- **Şu an C işareti açık**: {close.index[a]:%d.%m.%Y}'den beri {b - a + 1} işlem günü.")
    else:
        L.append("\n- Şu an C işareti kapalı.")
    L.append("")

    # --- bugünün durumu ---
    L.append("## Bugün\n")
    L.append(f"- Endeks {WIN} gün: {ix20.iloc[-1] * 100:+.1f}% → düşen piyasa: "
             f"{'evet' if falling.iloc[-1] else 'hayır'} · {close.index[-1]:%d.%m.%Y}")
    L.append(f"- {name}: sıra %{rank.iloc[-1, j] * 100:.0f} · fark {rs.iloc[-1, j] * 100:+.1f}%")
    top = rs.iloc[-1].where(r20.iloc[-1] > 0).dropna().sort_values(ascending=False).head(10)
    L.append("- Göreceli güç ilk 10 (hisse artıda olanlar): "
             + ", ".join(f"{c.replace('.IS', '')} ({v * 100:+.0f}%)" for c, v in top.items()))
    L.append("")

    L.append("> Uyarı: Sadece fiyat ve hacimle çalışır; yatırımcı sayısı, kurumsal oran ve fiili dolaşım yok. "
             "Veri bugünkü hisse listesidir (işlemden kalkanlar yok). Gözlemler üst üste bindiği için sayılar "
             "göründüğünden daha az bağımsızdır. Düşen piyasa günlerinin sayısı azdır, sonuçlar birkaç dönemin "
             "etkisini taşıyabilir. Yatırım tavsiyesi değildir.")

    text = "\n".join(L)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
