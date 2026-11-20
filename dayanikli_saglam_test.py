"""
dayanikli_saglam_test.py - Dayanıklı güç sinyali sağlam mı, yoksa belirli bir ayara / döneme mi bağlı?

dayanikli_test.py'deki dn (endeks düşen günlerde fazla getiri) ve up (endeks yükselen günlerde fazla getiri) ölçüsünü
farklı pencere (60 / 120 / 250 gün) ve eşiklerle dener. Hepsinde zirveye yakın şartı var (kapanış, 120 gün zirvenin %85'i üstünde).
Kural sitedekiyle aynı: giriş = sinyal günü kapanışı, izleyen stop -%25, 84 işlem günü, komisyon dahil.
Kıyas: aynı gün, aynı kuralla 30 rastgele likit hisse. Fark = sinyal - kıyas.

Ek kontroller:
  - t (ay bazlı): her sinyal ayının ortalama farkı bir gözlem sayılır (sinyaller aynı günlere yığıldığı için işlem bazlı t iyimser).
  - Yıl yıl fark: sinyal tarihinin yılına göre.
  - Referans "düşük beta": dayanıklılık sadece düşük beta mı? Aynı pencerede beta sırası en düşük %30 + zirveye yakın.
Sitedeki hiçbir dosyaya dokunmaz. dip_cikis_test.py, dip_kiyas_test.py, signals.py aynı klasörde olmalı.
Yatırım tavsiyesi değildir.
"""
import os

import numpy as np
import pandas as pd

import dip_cikis_test as K
import dip_kiyas_test as Q
import signals as S

INDEX = "XU100.IS"
HH = 84
STOP = 25.0
COOL = 21
MAXN = 4000


def sinyal_idx(mask):
    out_i, out_j = [], []
    for j in range(mask.shape[1]):
        last = -10 ** 9
        for i in np.flatnonzero(mask[:, j]):
            if i - last >= COOL:
                out_i.append(int(i))
                out_j.append(j)
                last = i
    return np.array(out_i, dtype=int), np.array(out_j, dtype=int)


def kural(P):
    return K.izleyen(P, STOP)[0]


def konfigler(close, volume, idx):
    r = close.pct_change(fill_method=None)
    ri = idx.pct_change(fill_method=None)
    e = r.sub(ri, axis=0)
    ciro = (close * volume).rolling(20).mean()
    likit = ciro.rank(axis=1, pct=True) >= 0.30
    yakin = close >= 0.85 * close.rolling(120).max()
    C = {}
    gec = {}
    for w in (60, 120, 250):
        mp = max(20, w // 4)
        dn = e.where(ri < 0, np.nan).rolling(w, min_periods=mp).mean()
        up = e.where(ri > 0, np.nan).rolling(w, min_periods=mp).mean()
        dr, ur = dn.rank(axis=1, pct=True), up.rank(axis=1, pct=True)
        g = (likit & close.notna() & close.shift(w).notna()).fillna(False)
        gec[w] = g
        C[f"W{w} dn>=70 up>=70"] = ((dr >= 0.70) & (ur >= 0.70) & yakin & g, w)
        C[f"W{w} dn>=80 up>=60"] = ((dr >= 0.80) & (ur >= 0.60) & yakin & g, w)
        C[f"W{w} dn>=90"] = ((dr >= 0.90) & yakin & g, w)
        if w == 120:
            beta = r.rolling(w, min_periods=mp).cov(ri) / ri.rolling(w, min_periods=mp).var()
            br = beta.rank(axis=1, pct=True)
            C["REF düşük beta<=30 (W120)"] = ((br <= 0.30) & yakin & g, w)
    return {k: (v.fillna(False).to_numpy(), w) for k, (v, w) in C.items()}


def calistir(close, volume, idx):
    T = close.shape[0]
    cfg = konfigler(close, volume, idx)
    gec_np = (close.notna() & ((close * volume).rolling(20).mean().rank(axis=1, pct=True) >= 0.30)).fillna(False).to_numpy()
    ix = idx.to_numpy()
    ixf = np.full(T, np.nan)
    ixf[:T - HH] = (ix[HH:] / ix[:T - HH] - 1) * 100
    rng = np.random.default_rng(2026)
    y = T // 2
    f = lambda x, d=1: "—" if (x is None or pd.isna(x)) else f"{x:+.{d}f}"
    g = lambda x, d=1: "—" if (x is None or pd.isna(x)) else f"{x:.{d}f}"

    sonuc = {}
    for ad, (mask, w) in cfg.items():
        ii, jj = sinyal_idx(mask)
        if len(ii) == 0:
            sonuc[ad] = None
            continue
        ii, jj, P = K.yollar(close, ii, jj, HH)
        if len(ii) > MAXN:
            s = np.sort(rng.choice(len(ii), MAXN, replace=False))
            ii, jj, P = ii[s], jj[s], P[s]
        n = len(ii)
        if n < 40:
            sonuc[ad] = None
            continue
        Bp = Q.kiyas_yollari(close, gec_np, ii, HH, rng)
        flat = Bp.reshape(n * Q.NK, HH)
        ok = np.isfinite(flat).all(axis=1)
        rb = kural(np.where(ok[:, None], flat, 0.0)).reshape(n, Q.NK)
        okm = ok.reshape(n, Q.NK)
        kiyas = np.where(okm.sum(1) > 0, (rb * okm).sum(1) / np.maximum(okm.sum(1), 1), np.nan)
        r = kural(P)
        gecer = np.isfinite(kiyas) & np.isfinite(ixf[ii])
        sonuc[ad] = dict(ii=ii[gecer], r=r[gecer], k=kiyas[gecer], x=ixf[ii][gecer])

    def t_islem(fark):
        return fark.mean() / (fark.std(ddof=1) / np.sqrt(len(fark))) if len(fark) > 5 else np.nan

    def t_ay(d, m):
        idxs = pd.DatetimeIndex(close.index[d["ii"][m]])
        ay = pd.Series(d["r"][m] - d["k"][m]).groupby([idxs.year, idxs.month]).mean()
        if len(ay) < 6:
            return np.nan, len(ay)
        return ay.mean() / (ay.std(ddof=1) / np.sqrt(len(ay))), len(ay)

    def fm(d, m, k=15):
        return (d["r"][m] - d["k"][m]).mean() if m.sum() >= k else np.nan

    L = ["# Dayanıklı güç sağlamlık testi\n",
         f"{close.shape[1]} hisse, {close.index[0]:%d.%m.%Y} - {close.index[-1]:%d.%m.%Y}. Hepsinde zirveye yakın şartı. Kural: izleyen stop -%{STOP:.0f}, {HH} gün, "
         f"komisyon %{K.KOM} dahil. Fark = sinyal - {Q.NK} rastgele likit hisse kıyası. t(ay) = her sinyal ayı tek gözlem sayılarak.\n"]

    L += ["## 1) Genel\n", "| Ayar | n | Fark % | t | t(ay) | ay sayısı |", "|---|---|---|---|---|---|"]
    for ad, d in sonuc.items():
        if d is None:
            L.append(f"| {ad} | az | | | | |")
            continue
        m = np.ones(len(d["r"]), bool)
        ta, na = t_ay(d, m)
        L.append(f"| {ad} | {m.sum()} | {f((d['r'] - d['k']).mean())} | {g(t_islem(d['r'] - d['k']))} | {g(ta)} | {na} |")
    L.append("")

    L += ["## 2) Piyasa ve yarılar (fark %)\n", "| Ayar | Kötü pzr | İyi pzr | Yarı1 | Yarı2 | Yarı1 kötü |", "|---|---|---|---|---|---|"]
    for ad, d in sonuc.items():
        if d is None:
            continue
        e = d["ii"] < y
        L.append(f"| {ad} | {f(fm(d, d['x'] <= 0))} | {f(fm(d, d['x'] > 0))} | {f(fm(d, e))} | {f(fm(d, ~e))} | {f(fm(d, e & (d['x'] <= 0)))} |")
    L.append("")

    yillar = sorted(set(close.index.year))
    L += ["## 3) Yıl yıl fark % (sinyal yılına göre; en az 15 işlem)\n", "| Ayar | " + " | ".join(str(a) for a in yillar) + " |", "|---|" + "---|" * len(yillar)]
    for ad, d in sonuc.items():
        if d is None:
            continue
        yy = np.asarray(close.index.year)[d["ii"]]
        L.append(f"| {ad} | " + " | ".join(f(fm(d, yy == a)) for a in yillar) + " |")
    L.append("")

    L += ["Nasıl okunur: ayar sağlamsa (a) tüm W/eşik kombinasyonlarında Fark pozitif ve t(ay) yaklaşık 2'nin üzerinde olur, "
          "(b) yıl yıl tabloda çoğu yıl pozitiftir (sadece son yıl değil), (c) Yarı1 de pozitiftir, "
          "(d) düşük beta referansından belirgin iyidir (yoksa dayanıklılık sadece 'düşük riskli hisse' demektir). "
          "Tek bir ayar diğerlerinden çok iyiyse onu seçme: ayar oynatınca kaybolan etki sağlam değildir. "
          "Batmış hisseler listede yok (iyimser). Kötü/iyi piyasa ileriye bakılarak ayrıldığı için o anda bilinemez. Yatırım tavsiyesi değildir."]
    return "\n".join(L)


def main():
    import yfinance  # noqa: F401
    symbols = S.read_symbols("bist_symbols.txt")
    print(f"{len(symbols)} sembol + endeks, 5 yıllık günlük veri indiriliyor...")
    close, volume = S.fetch_all(symbols + [INDEX], "5y")
    if INDEX not in close.columns:
        raise SystemExit("Endeks (XU100.IS) verisi alınamadı.")
    idx = close[INDEX].ffill(limit=3)
    close = close.drop(columns=[INDEX])
    volume = volume.drop(columns=[INDEX])
    txt = calistir(close, volume, idx)
    print(txt)
    ozet = os.environ.get("GITHUB_STEP_SUMMARY")
    if ozet:
        with open(ozet, "a", encoding="utf-8") as fh:
            fh.write(txt + "\n")


if __name__ == "__main__":
    main()
