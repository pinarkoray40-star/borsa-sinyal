"""
dip_obv.py
TradingView "Dip OBV Sıkı v1.6" indikatörünün Python karşılığı (günlük grafik, BIST).

İki modu var:
  python dip_obv.py              -> tüm BIST'i tarar, SON GÜN AL sinyali verenleri yazar
  python dip_obv.py --backtest   -> geçmişteki tüm sinyalleri bulur, 5/10/20 gün getiri,
                                    isabet oranı ve stop/hedef sonucunu raporlar
  python dip_obv.py THYAO ASELS  -> sadece verdiğin hisseler (iki modda da çalışır)

Çıktılar: dip_obv_sonuc.csv (tarama) veya dip_obv_backtest.csv (backtest)
Not: Canlı takip defterine DOKUNMAZ. Önce backtest ile bakılır, sonra istenirse deftere bağlanır.
Gereken: pip install borsapy pandas numpy
"""
import sys
import os
import numpy as np
import pandas as pd
import borsapy as bp

# === PARAMETRELER (Pine'daki varsayılanlarla aynı) ===
REL_VOL = 2.0
SQUEEZE_ATR = 5.0
MIN_TL = 5.0            # milyon TL, 30g ort. hacim x kapanış
DD_MIN, DD_MAX = 12.0, 50.0
BASE_NEAR_LOW = 12.0
CLOSE_POS_MIN = 0.6
RSI_LOW, RSI_HIGH = 45, 65
MAX_CHG, MAX_GAP = 7.0, 4.0
MIN_SCORE = 60
COOLDOWN = 10
VOL_MULT = 1.15
OBV_LB = 15
NETFLOW_LB = 10
STOP_MULT = 1.5
RR = 2.0
BASE_LEN = 15
USE_BREAKOUT, USE_MARKET, USE_RS = True, True, True


def kolon_bul(df, ad):
    for c in df.columns:
        if str(c).lower() == ad.lower():
            return c
    raise KeyError(ad)


def tum_hisseler():
    df = bp.companies()
    for c in df.columns:
        if str(c).lower() in ("ticker", "symbol", "kod", "code"):
            return [str(x).strip().upper() for x in df[c].dropna()]
    return [str(x).strip().upper() for x in df.iloc[:, 0].dropna()]


def hisseleri_al(args):
    if args:
        return [t.upper() for t in args]
    if os.path.exists("tickers.txt"):
        with open("tickers.txt", encoding="utf-8") as f:
            return [s.strip().upper() for s in f if s.strip()]
    try:
        liste = tum_hisseler()
        if liste:
            print(f"borsapy'den {len(liste)} hisse alındı.")
            return liste
    except Exception as e:
        print(f"Otomatik liste alınamadı ({e}).")
    print("Hisse listesi yok: tickers.txt oluştur veya komuta hisse kodu yaz.")
    sys.exit(1)


def ohlcv(df):
    d = pd.DataFrame({
        "open": df[kolon_bul(df, "Open")].astype(float),
        "high": df[kolon_bul(df, "High")].astype(float),
        "low": df[kolon_bul(df, "Low")].astype(float),
        "close": df[kolon_bul(df, "Close")].astype(float),
        "volume": df[kolon_bul(df, "Volume")].astype(float),
    })
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        for c in df.columns:
            if str(c).lower() in ("date", "tarih", "datetime"):
                idx = pd.DatetimeIndex(pd.to_datetime(df[c]))
                break
    d.index = pd.DatetimeIndex(pd.to_datetime(idx)).tz_localize(None).normalize()
    d = d[~d.index.duplicated(keep="last")].sort_index()
    return d.dropna()


def xu100_al():
    denemeler = [lambda: bp.Index("XU100").history(period="2y"),
                 lambda: bp.Ticker("XU100").history(period="2y")]
    for f in denemeler:
        try:
            x = ohlcv(f())
            if len(x) > 60:
                return x["close"]
        except Exception:
            continue
    return None


# --- TradingView uyumlu göstergeler ---
def rma(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    tr.iloc[0] = d["high"].iloc[0] - d["low"].iloc[0]
    return rma(tr, n)


def rsi(close, n=14):
    ch = close.diff()
    up = rma(ch.clip(lower=0), n)
    dn = rma((-ch).clip(lower=0), n)
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0)


def obv(close, volume):
    return (np.sign(close.diff()).fillna(0) * volume).cumsum()


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def sma(s, n):
    return s.rolling(n).mean()


def hesapla(d, xclose):
    """Her bar için sinyal bileşenlerini hesaplar, DataFrame döner."""
    c, o, h, l, v = d["close"], d["open"], d["high"], d["low"], d["volume"]
    atr14 = atr(d)
    atr_avg = sma(atr14, 20)

    avg_vol = sma(v, 30).shift(1)
    rel_vol = (v / avg_vol).where(avg_vol > 0, 0.0)
    liquid = avg_vol * c >= MIN_TL * 1e6

    hh = h.rolling(BASE_LEN).max().shift(1)
    ll = l.rolling(BASE_LEN).min().shift(1)
    range_atr = ((hh - ll) / atr14.shift(1)).where(atr14.shift(1) > 0, 999.0)
    is_narrow = range_atr <= SQUEEZE_ATR

    hi60 = h.rolling(60).max()
    lo60 = l.rolling(60).min()
    dd = (hi60 - c) / hi60 * 100
    dd_ok = (dd >= DD_MIN) & (dd <= DD_MAX)
    near_low = ((ll - lo60) / lo60 * 100) <= BASE_NEAR_LOW

    obv_v = obv(c, v)
    obv_rising = (obv_v > obv_v.shift(OBV_LB)) & (obv_v > sma(obv_v, 10))

    rng = h - l
    mfm = ((c - l) - (h - c)) / rng.where(rng != 0)
    mfm = mfm.fillna(0.0)
    mfv = mfm * v
    cmf = sma(mfv, 21) / sma(v, 21)
    cmf_pos = (cmf > 0) & (cmf > cmf.shift(1))
    net = mfv.rolling(NETFLOW_LB).sum()
    net_pos = (net > 0) & (net > net.shift(NETFLOW_LB))

    vb = ((v > v.shift(1)).astype(int) + (v.shift(1) > v.shift(2)).astype(int)
          + (v.shift(2) > v.shift(3)).astype(int) + (v.shift(3) > v.shift(4)).astype(int))
    vol_build = vb >= 2

    rsi_v = rsi(c)
    rsi_ok = (rsi_v >= RSI_LOW) & (rsi_v <= RSI_HIGH)

    close_pos = ((c - l) / rng).where(rng > 0, 0.0)
    candle_ok = (c > o) & (c > c.shift(1)) & (close_pos >= CLOSE_POS_MIN)
    breakout_ok = (c > c.rolling(10).max().shift(1)) if USE_BREAKOUT else pd.Series(True, index=d.index)

    ema20 = ema(c, 20)
    trend_ok = (c > ema20) & (ema20 > ema20.shift(3))

    if xclose is not None:
        x = xclose.reindex(d.index).ffill()
        market_ok = (x > ema(x, 20)) if USE_MARKET else pd.Series(True, index=d.index)
        rs_ok = ((c.pct_change(20) > x.pct_change(20)) if USE_RS else pd.Series(True, index=d.index))
    else:
        market_ok = pd.Series(True, index=d.index)
        rs_ok = pd.Series(True, index=d.index)

    chg = (c / c.shift(1) - 1) * 100
    gap = (o / c.shift(1) - 1) * 100
    not_limit = (chg < MAX_CHG) & (gap < MAX_GAP)

    vol_exp = atr14 > atr_avg * VOL_MULT

    # Haftalık OBV: son tamamlanmış hafta (haftanın içindeki günler bir önceki haftaların değerini görür)
    hafta = d.index.to_period("W-FRI")
    w = d.groupby(hafta).agg(close=("close", "last"), volume=("volume", "sum"))
    w_obv = obv(w["close"], w["volume"])
    w_cond = (w_obv.shift(1) > w_obv.shift(5))
    w_obv_ok = pd.Series(w_cond.reindex(hafta).values, index=d.index).fillna(False).astype(bool)

    score = (vol_build.astype(int) * 25 + cmf_pos.astype(int) * 20 + rsi_ok.astype(int) * 15
             + vol_exp.astype(int) * 15 + w_obv_ok.astype(int) * 25)

    structure = is_narrow & dd_ok & near_low
    flow = obv_rising & net_pos & (rel_vol >= REL_VOL)
    confirm = candle_ok & breakout_ok & trend_ok
    filters = liquid & not_limit & market_ok & rs_ok
    core = (structure & flow & confirm & filters & (score >= MIN_SCORE)).fillna(False)

    out = pd.DataFrame({"core": core, "score": score, "atr": atr14, "rel_vol": rel_vol, "dd": dd})
    return out


def cooldown_uygula(core):
    """Pine'daki gibi: önceki sinyalden en az COOLDOWN bar sonra."""
    sinyal = np.zeros(len(core), dtype=bool)
    son = None
    for i, val in enumerate(core.values):
        if val and (son is None or (i - son) > COOLDOWN):
            sinyal[i] = True
            son = i
    return pd.Series(sinyal, index=core.index)


def sonuc_simule(d, i, stop, hedef, bar=20):
    """Sinyal sonrası 20 bar içinde stop/hedef hangisine önce değdi? Aynı barda ikisi de değerse stop sayılır."""
    for j in range(i + 1, min(i + 1 + bar, len(d))):
        if d["low"].iloc[j] <= stop:
            return "stop"
        if d["high"].iloc[j] >= hedef:
            return "hedef"
    return "açık" if i + bar >= len(d) else "süre doldu"


def analiz(kod, xclose, backtest):
    df = bp.Ticker(kod).history(period="2y")
    if df is None or len(df) < 100:
        return []
    d = ohlcv(df)
    if len(d) < 100:
        return []
    r = hesapla(d, xclose)
    sinyal = cooldown_uygula(r["core"])
    c = d["close"]
    satirlar = []
    idxler = range(len(d)) if backtest else [len(d) - 1]
    for i in idxler:
        if not sinyal.iloc[i]:
            continue
        px = c.iloc[i]
        stop = px - STOP_MULT * r["atr"].iloc[i]
        hedef = px + RR * (px - stop)
        satir = {
            "hisse": kod, "tarih": d.index[i].strftime("%Y-%m-%d"), "fiyat": round(px, 2),
            "skor": int(r["score"].iloc[i]), "hacim_orani": round(r["rel_vol"].iloc[i], 2),
            "zirveden_dusus_%": round(r["dd"].iloc[i], 1),
            "stop": round(stop, 2), "hedef": round(hedef, 2),
        }
        for g in (5, 10, 20):
            satir[f"g{g}_%"] = round((c.iloc[i + g] / px - 1) * 100, 2) if i + g < len(d) else None
        satir["stop_hedef"] = sonuc_simule(d, i, stop, hedef)
        satirlar.append(satir)
    return satirlar


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    backtest = "--backtest" in sys.argv
    kodlar = hisseleri_al(args)

    xclose = xu100_al()
    if xclose is None:
        print("UYARI: XU100 verisi alınamadı, piyasa ve göreceli güç filtreleri devre dışı!")

    from concurrent.futures import ThreadPoolExecutor

    def guvenli(kod):
        try:
            return analiz(kod, xclose, backtest)
        except Exception as e:
            print(f"{kod}: atlandı ({e})")
            return []

    with ThreadPoolExecutor(max_workers=4) as ex:
        sonuc = [s for liste in ex.map(guvenli, kodlar) for s in liste]

    pd.set_option("display.width", 220)
    if not backtest:
        print(f"Taranan: {len(kodlar)} hisse | Bugün AL sinyali veren: {len(sonuc)}\n")
        if sonuc:
            t = pd.DataFrame(sonuc).sort_values("skor", ascending=False)
            print(t.drop(columns=["g5_%", "g10_%", "g20_%", "stop_hedef"]).to_string(index=False))
            t.to_csv("dip_obv_sonuc.csv", index=False, encoding="utf-8-sig")
        else:
            print("Bugün sinyal yok.")
        return

    if not sonuc:
        print("Geçmişte hiç sinyal çıkmadı.")
        return
    t = pd.DataFrame(sonuc).sort_values("tarih")
    t.to_csv("dip_obv_backtest.csv", index=False, encoding="utf-8-sig")
    print(f"Taranan: {len(kodlar)} hisse | Toplam geçmiş sinyal: {len(t)}\n")
    for g in (5, 10, 20):
        s = t[f"g{g}_%"].dropna()
        if len(s):
            print(f"{g:>2} gün: n={len(s):>3} | ortalama %{s.mean():+.1f} | ortanca %{s.median():+.1f} | "
                  f"isabet %{(s > 0).mean() * 100:.0f} | en kötü %{s.min():+.1f} | en iyi %{s.max():+.1f}")
    print("\nStop/hedef (20 bar içinde, 1.5 ATR stop, 2R hedef):")
    print(t["stop_hedef"].value_counts().to_string())
    print("\nSon 25 sinyal:")
    print(t.tail(25).to_string(index=False))


if __name__ == "__main__":
    main()
