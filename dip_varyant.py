"""
dip_varyant.py - Dip OBV (5 dakikalık) şart testi

Sitedeki hiçbir dosyaya DOKUNMAZ, sadece okur (signals.py'deki fonksiyonları kullanır).
Yaptığı iş: son 60 günlük 5 dakikalık veride
  1) tabanı: ayar seti 2 ile sekiz şartın hepsi (Dip OBV 3 VWAP'sız hali),
  2) her şartı tek tek ÇIKARINCA ne olduğunu,
  3) VWAP, seans saati ve göreceli güç (hisse / XU100) şartlarını ekleyince ne olduğunu
ölçer. Her sinyalden sonra aynı gün kapanışı ve ertesi gün kapanışı getirisine bakar.
Aynı hissede aynı gün tek sinyal sayılır (ilk mum).

Kullanım: python dip_varyant.py
Sonuç: ekrana yazılır, GitHub Actions özet sayfasına ve dip_varyant_sonuc.csv'ye kaydedilir.
Gereken: pip install yfinance pandas numpy
"""
import os
import sys

import numpy as np
import pandas as pd

import signals as S

P = dict(S.DIP3)                       # ayar seti 2 (RelVol 14, OBV 98, RSI 3, 1.1, 41)
MIN_N = 30                             # bundan az sinyalde sonuç güvenilmez
RS_BARS = 12                           # göreceli güç: son 1 saatlik getiri kıyası

BASE_KEYS = ["narrow", "obv", "volbuild", "rsi", "cmf", "flow", "relvol", "atr"]
BASE_AD = {"narrow": "dar aralık", "obv": "OBV", "volbuild": "hacim artışı", "rsi": "RSI 46-63",
           "cmf": "CMF", "flow": "net para akışı", "relvol": "göreceli hacim", "atr": "volatilite (ATR)"}


def kosullar(h, l, c, v, P=P):
    """signals.dip_components ile aynı mantık, ama her şart ayrı ayrı döner."""
    avg_vol = v.rolling(30).mean()
    rel_vol = v / avg_vol
    lo15 = l.rolling(15).min()
    narrow = ((h.rolling(15).max() - lo15) / lo15 * 100) < P["narrow"]

    d = c.diff()
    signed = pd.Series(np.where(d > 0, v, np.where(d < 0, -v, 0.0)), index=c.index)
    obv = signed.rolling(P["obv_lb"]).sum() > 0

    up = lambda k: (v.shift(k) > v.shift(k + 1)).astype(int)
    volbuild = (up(0) + up(1) + up(2) + up(3)) >= 2

    ch = c.diff()
    gain = S._rma(ch.clip(lower=0), P["rsi_len"])
    loss = S._rma((-ch).clip(lower=0), P["rsi_len"])
    rs = gain / loss
    rsi_v = pd.Series(np.where(loss == 0, 100.0, np.where(gain == 0, 0.0, 100 - 100 / (1 + rs))), index=c.index)
    rsi_v = rsi_v.where(gain.notna() & loss.notna())
    rsi = (rsi_v >= 46) & (rsi_v <= 63)

    rng = h - l
    mfm = pd.Series(np.where(rng != 0, ((c - l) - (h - c)) / rng.replace(0, np.nan), 0.0), index=c.index)
    mfv = mfm * v
    cmfv = mfv.rolling(21).mean() / v.rolling(21).mean()
    cmf = (cmfv > 0) & (cmfv > cmfv.shift(1))

    nmf = mfv.rolling(P["flow_lb"]).sum()
    flow = (nmf > 0) & (nmf > nmf.shift(P["flow_lb"]))

    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr_s = S._rma(tr, 14)
    atr = atr_s > atr_s.rolling(20).mean() * P["vol_mult"]

    relvol = rel_vol >= P["rel_vol"]
    return {"narrow": narrow, "obv": obv, "volbuild": volbuild, "rsi": rsi, "cmf": cmf,
            "flow": flow, "relvol": relvol, "atr": atr}


def ekler(h, l, c, v, xu_ret):
    """Denenecek ek şartlar."""
    vw = S.vwap_series(h, l, c, v)
    vwap = (c >= vw * (1 - P["vwap_tol"] / 100.0)).fillna(False)
    t = h.index
    dk = np.asarray(t.hour) * 60 + np.asarray(t.minute)
    saat = pd.Series((dk >= 10 * 60 + 30) & (dk <= 17 * 60 + 30), index=t)   # ilk ve son yarım saat dışarıda
    rs = pd.Series(False, index=t)
    if xu_ret is not None:
        own = c / c.shift(RS_BARS) - 1
        rs = (own > xu_ret.reindex(t)).fillna(False)
    return {"vwap": vwap, "saat": saat, "gucluluk": rs}


def ileri_getiri(c):
    """Her mum için: aynı gün kapanışına ve ertesi gün kapanışına getiri (%)."""
    gun = pd.Series(np.asarray(c.index.date), index=c.index)
    son = c.groupby(gun).transform("last")
    gunluk_son = c.groupby(gun).last()
    ertesi = gunluk_son.shift(-1)
    ert = gun.map(ertesi)
    return pd.DataFrame({"gun": (son / c - 1) * 100, "ertesi": (ert / c - 1) * 100, "tarih": gun})


def varyantlar(ekler_var):
    """Ad -> hangi şartlar zorunlu."""
    v = {"TABAN (8 şart)": list(BASE_KEYS)}
    for k in BASE_KEYS:
        v[f"- {BASE_AD[k]} çıkar"] = [x for x in BASE_KEYS if x != k]
    v["+ VWAP"] = BASE_KEYS + ["vwap"]
    v["+ seans saati (10:30-17:30)"] = BASE_KEYS + ["saat"]
    if ekler_var.get("gucluluk"):
        v["+ göreceli güç (XU100'den güçlü)"] = BASE_KEYS + ["gucluluk"]
    v["+ VWAP + saat"] = BASE_KEYS + ["vwap", "saat"]
    return v


def calistir(frames, xu_close=None):
    H, L, C, V = frames["High"], frames["Low"], frames["Close"], frames["Volume"]
    xu_ret = None
    if xu_close is not None:
        xu_ret = xu_close / xu_close.shift(RS_BARS) - 1
    var = varyantlar({"gucluluk": xu_ret is not None})
    sinyal = {ad: [] for ad in var}
    tum = []                                            # kıyas: tüm mumların ileri getirisi
    for sym in C.columns:
        d = pd.DataFrame({"h": H[sym], "l": L[sym], "c": C[sym], "v": V[sym]}).dropna()
        if len(d) < 140:
            continue
        k = kosullar(d.h, d.l, d.c, d.v)
        k.update(ekler(d.h, d.l, d.c, d.v, xu_ret))
        fr = ileri_getiri(d.c)
        gecerli = fr["gun"].notna()
        tum.append(fr.loc[gecerli, ["gun", "ertesi"]])
        for ad, keys in var.items():
            m = pd.Series(True, index=d.index)
            for key in keys:
                m &= k[key].fillna(False)
            m &= gecerli
            hit = fr[m]
            if hit.empty:
                continue
            first = hit.groupby("tarih").head(1)        # günde tek sinyal
            first = first.assign(sym=sym)
            sinyal[ad].append(first[["sym", "tarih", "gun", "ertesi"]])
    tum = pd.concat(tum) if tum else pd.DataFrame(columns=["gun", "ertesi"])
    kiyas = {"gun": float(tum["gun"].mean()) if len(tum) else np.nan,
             "ertesi": float(tum["ertesi"].mean()) if len(tum) else np.nan}

    rows = []
    for ad in var:
        df = pd.concat(sinyal[ad]) if sinyal[ad] else pd.DataFrame(columns=["gun", "ertesi"])
        r = {"varyant": ad, "n": len(df)}
        for col, kis in (("gun", "gun"), ("ertesi", "ertesi")):
            x = df[col].dropna()
            r[f"{col}_ort"] = float(x.mean()) if len(x) else np.nan
            r[f"{col}_med"] = float(x.median()) if len(x) else np.nan
            r[f"{col}_kaz"] = float((x > 0).mean() * 100) if len(x) else np.nan
            r[f"{col}_fark"] = r[f"{col}_ort"] - kiyas[kis] if len(x) else np.nan
        rows.append(r)
    return pd.DataFrame(rows), kiyas


def rapor(res, kiyas, n_hisse):
    f = lambda x, d=2: "—" if pd.isna(x) else f"{x:+.{d}f}"
    L = ["# Dip OBV şart testi (5 dakikalık, son ~60 gün)\n",
         f"{n_hisse} hisse. Her sinyalde giriş = sinyal mumunun kapanışı. Günde hisse başına tek sinyal. "
         "Fark = sinyalin getirisi - aynı günlerde tüm mumların ortalaması (kıyas).\n",
         f"Kıyas (tüm mumlar): aynı gün kapanışı {f(kiyas['gun'])}%, ertesi gün kapanışı {f(kiyas['ertesi'])}%\n",
         "| Varyant | Sinyal | Gün sonu ort % | Fark | Kazanan % | Ertesi gün ort % | Fark | Kazanan % |",
         "|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        guv = "" if r["n"] >= MIN_N else " ⚠"
        k1 = "—" if pd.isna(r["gun_kaz"]) else "%.0f" % r["gun_kaz"]
        k2 = "—" if pd.isna(r["ertesi_kaz"]) else "%.0f" % r["ertesi_kaz"]
        L.append("| %s%s | %d | %s | %s | %s | %s | %s | %s |" % (
            r["varyant"], guv, int(r["n"]), f(r["gun_ort"]), f(r["gun_fark"]), k1,
            f(r["ertesi_ort"]), f(r["ertesi_fark"]), k2))
    L += ["", f"⚠ = {MIN_N}'dan az sinyal, sonuca güvenme.",
          "Dikkat: 60 günlük veride çok şartı denemek tesadüfen iyi görünen bir sonuç bulmayı kolaylaştırır. "
          "Bir şart ancak hem gün sonu hem ertesi gün sonuçlarında, yeterli sinyalle ve kıyasın belirgin üstünde ise işe yarıyor sayılır."]
    return "\n".join(L)


def main():
    import yfinance  # noqa: F401  (yoksa hemen anlaşılsın)
    symbols = S.read_symbols("bist_symbols.txt")
    print(f"{len(symbols)} sembol, 5 dakikalık 60 gün indiriliyor...")
    frames = S.fetch_intraday(symbols, "60d")
    if frames is None:
        sys.exit("5 dakikalık veri alınamadı.")
    xu = None
    try:
        xf = S.fetch_intraday(["XU100.IS"], "60d")
        if xf is not None and "XU100.IS" in xf["Close"].columns:
            xu = xf["Close"]["XU100.IS"].dropna()
    except Exception as exc:
        print("XU100 alınamadı, göreceli güç atlanıyor:", exc)
    if xu is None:
        print("UYARI: XU100 5 dakikalık verisi yok, göreceli güç testi yapılmayacak.")
    res, kiyas = calistir(frames, xu)
    res.to_csv("dip_varyant_sonuc.csv", index=False, encoding="utf-8-sig")
    txt = rapor(res, kiyas, frames["Close"].shape[1])
    print(txt)
    ozet = os.environ.get("GITHUB_STEP_SUMMARY")
    if ozet:
        with open(ozet, "a", encoding="utf-8") as fh:
            fh.write(txt + "\n")


if __name__ == "__main__":
    main()
