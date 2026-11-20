"""
dayanikli_bh_test.py - Dayanıklılık, düşük betanın ÜSTÜNE ek bilgi veriyor mu?

Beta'yı sabit tutup dayanıklılığa (dn) bakar. İki evrende, her biri 2x2 = 4 hücreye bölünür:
  beta düşük  : W120 beta sırası en düşük %30      | beta yüksek : geri kalan
  dn yüksek   : W120 düşüşte fazla getiri sırası >= %70 | dn düşük : geri kalan
Evren 1: Büyük Hareket sinyali (sitedeki mevcut sinyal, buyuk.py).
Evren 2: Zirveye yakın hisseler (kapanış, 120 gün zirvenin %85'i üstünde, likit).
Kural sitedekiyle aynı: giriş = sinyal günü kapanışı, izleyen stop -%25, 84 işlem günü, komisyon dahil.
Kıyas: aynı gün, aynı kuralla 30 rastgele likit hisse. Fark = sinyal - kıyas.
Soru: beta düşükken dn yüksek ile dn düşük arasında fark var mı? (ve tersi: dn yüksekken beta düşük/yüksek)
Sitedeki hiçbir dosyaya dokunmaz. buyuk.py, dip_cikis_test.py, dip_kiyas_test.py, signals.py aynı klasörde olmalı.
Yatırım tavsiyesi değildir.
"""
import os

import numpy as np
import pandas as pd

import buyuk as BY
import dip_cikis_test as K
import dip_kiyas_test as Q
import signals as S

INDEX = "XU100.IS"
W = 120
HH = 84
STOP = 25.0
COOL = 21
MAXN = 4000
BETA_ESIK = 0.30
DN_ESIK = 0.70


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


def np_bool(df):
    return np.nan_to_num(df.to_numpy(dtype=float), nan=0.0).astype(bool)


def hazirla(close, volume, idx):
    r = close.pct_change(fill_method=None)
    ri = idx.pct_change(fill_method=None)
    e = r.sub(ri, axis=0)
    mp = max(20, W // 4)
    dn = e.where(ri < 0, np.nan).rolling(W, min_periods=mp).mean()
    dr = dn.rank(axis=1, pct=True)
    beta = r.rolling(W, min_periods=mp).cov(ri).div(ri.rolling(W, min_periods=mp).var(), axis=0)
    br = beta.rank(axis=1, pct=True)
    ciro = (close * volume).rolling(20).mean()
    likit = ciro.rank(axis=1, pct=True) >= 0.30
    g = (likit & close.notna() & close.shift(W).notna()).fillna(False)
    yakin = close >= 0.85 * close.rolling(120).max()

    bl = np_bool(br <= BETA_ESIK)
    bh = np_bool((br > BETA_ESIK) & br.notna())
    dh = np_bool(dr >= DN_ESIK)
    dl = np_bool((dr < DN_ESIK) & dr.notna())
    bh_sinyal = BY.ozellikler(close, volume)
    zirve = np_bool(yakin & g)
    gec = np_bool(g)
    return dict(bl=bl, bh=bh, dh=dh, dl=dl, BH=bh_sinyal, zirve=zirve, gec=gec)


def calistir(close, volume, idx):
    T = close.shape[0]
    H = hazirla(close, volume, idx)
    ix = idx.to_numpy()
    ixf = np.full(T, np.nan)
    ixf[:T - HH] = (ix[HH:] / ix[:T - HH] - 1) * 100
    rng = np.random.default_rng(2026)
    y = T // 2
    f = lambda x, d=1: "—" if (x is None or pd.isna(x)) else f"{x:+.{d}f}"
    g = lambda x, d=1: "—" if (x is None or pd.isna(x)) else f"{x:.{d}f}"

    def sim(mask):
        ii, jj = sinyal_idx(mask)
        if len(ii) == 0:
            return None
        ii, jj, P = K.yollar(close, ii, jj, HH)
        if len(ii) > MAXN:
            s = np.sort(rng.choice(len(ii), MAXN, replace=False))
            ii, jj, P = ii[s], jj[s], P[s]
        n = len(ii)
        if n < 40:
            return None
        Bp = Q.kiyas_yollari(close, H["gec"], ii, HH, rng)
        flat = Bp.reshape(n * Q.NK, HH)
        ok = np.isfinite(flat).all(axis=1)
        rb = kural(np.where(ok[:, None], flat, 0.0)).reshape(n, Q.NK)
        okm = ok.reshape(n, Q.NK)
        kiyas = np.where(okm.sum(1) > 0, (rb * okm).sum(1) / np.maximum(okm.sum(1), 1), np.nan)
        r = kural(P)
        gecer = np.isfinite(kiyas) & np.isfinite(ixf[ii])
        return dict(ii=ii[gecer], r=r[gecer], k=kiyas[gecer], x=ixf[ii][gecer])

    def fm(d, m, k=15):
        return (d["r"][m] - d["k"][m]).mean() if m.sum() >= k else np.nan

    def t_ay(d):
        idxs = pd.DatetimeIndex(close.index[d["ii"]])
        ay = pd.Series(d["r"] - d["k"]).groupby([idxs.year, idxs.month]).mean()
        if len(ay) < 6:
            return np.nan
        return ay.mean() / (ay.std(ddof=1) / np.sqrt(len(ay)))

    hucreler = [("beta düşük, dn yüksek", H["bl"] & H["dh"]), ("beta düşük, dn düşük", H["bl"] & H["dl"]),
                ("beta yüksek, dn yüksek", H["bh"] & H["dh"]), ("beta yüksek, dn düşük", H["bh"] & H["dl"])]
    evrenler = [("Büyük Hareket sinyali", H["BH"]), ("Zirveye yakın hisseler", H["zirve"])]

    L = ["# Dayanıklılık düşük betanın üstüne ek bilgi mi?\n",
         f"{close.shape[1]} hisse, {close.index[0]:%d.%m.%Y} - {close.index[-1]:%d.%m.%Y}. Kural: izleyen stop -%{STOP:.0f}, {HH} gün, komisyon %{K.KOM} dahil. "
         f"Beta düşük = W{W} beta sırası en düşük %{int(BETA_ESIK * 100)}. dn yüksek = düşüşte fazla getiri sırası >= %{int(DN_ESIK * 100)}. "
         f"Fark = sinyal - {Q.NK} rastgele likit hisse kıyası. t(ay) = her sinyal ayı tek gözlem.\n"]
    sonuc = {}
    for evad, evmask in evrenler:
        L += [f"## {evad}\n", "| Hücre | n | Fark % | t(ay) | Kötü pzr | İyi pzr | Yarı1 | Yarı2 |", "|---|---|---|---|---|---|---|---|"]
        tum = sim(evmask)
        satirlar = [("TÜMÜ", tum)] + [(ad, sim(evmask & m)) for ad, m in hucreler]
        for ad, d in satirlar:
            sonuc[(evad, ad)] = d
            if d is None:
                L.append(f"| {ad} | az | | | | | | |")
                continue
            e = d["ii"] < y
            allm = np.ones(len(d["r"]), bool)
            L.append(f"| {ad} | {len(d['r'])} | {f(fm(d, allm))} | {g(t_ay(d))} | {f(fm(d, d['x'] <= 0))} | {f(fm(d, d['x'] > 0))} | {f(fm(d, e))} | {f(fm(d, ~e))} |")
        L.append("")

    yillar = sorted(set(close.index.year))
    L += ["## Yıl yıl fark % (en az 15 işlem)\n", "| Evren / hücre | " + " | ".join(str(a) for a in yillar) + " |", "|---|" + "---|" * len(yillar)]
    yil_np = np.asarray(close.index.year)
    for (evad, ad), d in sonuc.items():
        if d is None:
            continue
        yy = yil_np[d["ii"]]
        etiket = ("BH" if evad.startswith("Büyük") else "Zirve") + " / " + ad
        L.append(f"| {etiket} | " + " | ".join(f(fm(d, yy == a)) for a in yillar) + " |")
    L.append("")

    L += ["Nasıl okunur: dayanıklılık betadan bağımsız bilgi veriyorsa, beta düşük satırlarında 'dn yüksek' 'dn düşük'ten belirgin iyi olmalı "
          "(ve beta yüksek satırlarında da aynı yönde). Beta düşük olup dn düşükken de fark yüksekse, asıl etki beta'dandır. "
          "Hücreler küçüldükçe sayılara güvenme (n < 150 zayıf). Önce 'TÜMÜ' satırı ile hücreleri kıyasla, sonra yıl yıl tabloya bak: "
          "sadece son 1-2 yılda çalışan bir hücre sağlam sayılmaz. Batmış hisseler listede yok (iyimser). "
          "Kötü/iyi piyasa ileriye bakılarak ayrıldığı için o anda bilinemez. Yatırım tavsiyesi değildir."]
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
