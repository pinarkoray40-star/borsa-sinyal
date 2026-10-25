"""Teşhis: backtest stratejileri neden zayıf kalıyor?

Dört şeyi ölçer:
  1. VERİ KALİTESİ: BIST'te günlük fiyat limiti ±%10. Daha büyük günlük hareket =
     düzeltilmemiş bedelsiz/bölünme ya da hatalı veri (Yahoo'da sık görülür).
  2. İŞLEM İSTATİSTİĞİ: kaç işlem, kazanma oranı, ortalama kazanç/kayıp, kaç gün tutuluyor,
     ortalama kasa/nakit oranı, zarar-kes dışına kaç işlem taştı.
  3. EN KÖTÜ GÜNLER: portföyün en çok düştüğü günlerde hangi hisse vardı, hareket gerçek mi veri hatası mı.
  4. VARYANTLAR: aynı stratejiler (a) ham veri / temizlenmiş veri, (b) nakit faizsiz / faizli,
     (c) likidite filtresiz / filtreli. Kıyas olarak BIST 100 ve eşit ağırlık.

Kullanım (GitHub'da "Analiz calistir" butonu, dosya adı: diagnose.py):
    python diagnose.py --period 5y

Sonuç iş günlüğünde ve "Summary" sekmesinde görünür. Yatırım tavsiyesi değildir.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import COST, N_SLOTS, WINDOW, indicators, metrics, momentum_rank, run, strat_signals
from signals import fetch_all, read_symbols

LIMIT = 0.12      # günlük limit ±%10; yuvarlama payıyla %12 üstü = şüpheli
WARMUP = 130      # göstergelerin ısınması için baştaki günler değerlendirmeye alınmaz


# ---------- yardımcılar ----------

def clean_prices(px: pd.DataFrame):
    """%12'den büyük günlük hareketleri 0 sayıp fiyatı yeniden kurar (ham veri hatalarını siler)."""
    r = px.pct_change(fill_method=None)
    bad = r.abs() > LIMIT
    rc = r.where(~bad, 0.0)
    pxc = (1 + rc.fillna(0.0)).cumprod().where(px.notna())
    return pxc, rc, bad, r


def strat_mom(px, ind, liquid=None, trend=False) -> pd.DataFrame:
    """backtest.py'deki momentum stratejileri; istenirse likidite filtreli."""
    mom = momentum_rank(px)
    if liquid is not None:
        mom = mom.where(liquid)
    ok = (px > ind["sma50"]) & (ind["rsi"] < 75)
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    cur = pd.Series(0.0, index=px.columns)
    for i in range(len(px)):
        if i % WINDOW == 0 and mom.iloc[i].notna().any():
            r = mom.iloc[i].dropna()
            if trend:
                pool = [s for s in r.nlargest(15).index if ok.iloc[i][s]][:N_SLOTS]
            else:
                pool = list(r.nlargest(N_SLOTS).index)
            cur = pd.Series(0.0, index=px.columns)
            if pool:
                cur[pool] = 1.0 / N_SLOTS
        w.iloc[i] = cur
    return w


def evaluate(w: pd.DataFrame, rets: pd.DataFrame, cash_rate: float = 0.0) -> dict:
    daily = run(w, rets)
    if cash_rate:
        idle = (1.0 - w.shift(1).fillna(0.0).sum(axis=1)).clip(lower=0.0)
        daily = daily + idle * ((1 + cash_rate) ** (1 / 252) - 1)
    m = metrics(daily.iloc[WARMUP:])
    m["ort_kasa_%"] = round(float(w.shift(1).fillna(0.0).sum(axis=1).iloc[WARMUP:].mean()) * 100, 0)
    return m


def evaluate_series(daily: pd.Series) -> dict:
    m = metrics(daily.fillna(0.0).iloc[WARMUP:])
    m["ort_kasa_%"] = 100
    return m


def trade_table(w: pd.DataFrame, px: pd.DataFrame):
    """Ağırlık matrisinden işlemleri çıkarır: giriş günü kapanışı -> son tutulan günün ertesi kapanışı."""
    W = (w.values > 0)
    P = px.values
    n = len(px)
    rets, holds = [], []
    for j in range(W.shape[1]):
        col = W[:, j]
        if not col.any():
            continue
        d = np.diff(np.concatenate(([0], col.astype(int), [0])))
        for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1):
            e = min(b + 1, n - 1)
            p0, p1 = P[a, j], P[e, j]
            if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                rets.append(p1 / p0 - 1)
                holds.append(e - a)
    return np.array(rets), np.array(holds)


def md_table(rows: dict, cols: list) -> str:
    out = ["| Varyant | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for name, m in rows.items():
        out.append(f"| {name} | " + " | ".join(str(m.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


# ---------- ana teşhis ----------

def diagnose(close: pd.DataFrame, volume: pd.DataFrame, xu100: pd.Series | None, cash_rate: float) -> str:
    L = []
    px = close.sort_index().ffill(limit=5)
    vol = volume.reindex(index=px.index, columns=px.columns)
    rets = px.pct_change(fill_method=None)
    pxc, rc, bad, r_raw = clean_prices(px)
    turn = (px * vol).rolling(20).median()
    liquid = turn.rank(axis=1, pct=True) >= 0.30

    L.append(f"# Teşhis raporu\n\n{px.shape[1]} hisse, {px.index[0].date()} -> {px.index[-1].date()} "
             f"({len(px)} gün). Nakit faizi varsayımı: yıllık %{cash_rate * 100:.0f}.\n")

    # 1) veri kalitesi
    n_bad = int(bad.values.sum())
    n_obs = int(r_raw.notna().values.sum())
    per_sym = bad.sum().sort_values(ascending=False)
    L.append("## 1) Veri kalitesi\n")
    L.append(f"- Günlük hareketi %{LIMIT * 100:.0f}'yi aşan gözlem: **{n_bad}** "
             f"(toplam {n_obs:,} gözlemin %{n_bad / max(n_obs, 1) * 100:.3f}'i)".replace(",", "."))
    L.append(f"- Etkilenen hisse: **{int((per_sym > 0).sum())}** / {px.shape[1]}")
    L.append("- En çok şüpheli günü olan hisseler (hisse: adet, en uç hareket):")
    for sym in per_sym.head(10).index:
        if per_sym[sym] == 0:
            break
        s = r_raw[sym]
        day = s.abs().idxmax()
        L.append(f"  - {sym}: {int(per_sym[sym])} gün, en uç {s[day] * 100:+.0f}% ({day.date()})")
    L.append("")

    # 2) mevcut sinyal stratejisinin işlem istatistiği
    ind = indicators(px)
    w_sig = strat_signals(px, ind)
    expo = w_sig.shift(1).fillna(0.0).sum(axis=1).iloc[WARMUP:]
    tr, hold = trade_table(w_sig, px)
    L.append("## 2) 'Sinyal (RSI+MACD+SMA)' stratejisi, ham veri\n")
    L.append(f"- Ortalama kasa doluluğu: **%{expo.mean() * 100:.0f}** (kalan nakitte %0 getiriyor). "
             f"Hiç pozisyon olmayan gün: %{(expo == 0).mean() * 100:.0f}")
    if len(tr):
        wins, loss = tr[tr > 0], tr[tr <= 0]
        L.append(f"- İşlem sayısı: **{len(tr)}**, kazanma oranı **%{(tr > 0).mean() * 100:.0f}**, "
                 f"ortalama kazanç {wins.mean() * 100 if len(wins) else 0:+.1f}%, "
                 f"ortalama kayıp {loss.mean() * 100 if len(loss) else 0:+.1f}%")
        L.append(f"- Medyan tutma süresi: **{int(np.median(hold))} gün** (hedef pencere {WINDOW}). "
                 f"%{(hold <= 3).mean() * 100:.0f}'i 3 gün içinde kapanmış")
        L.append(f"- Zarar-kes %8 iken -%12'den kötü kapanan işlem: **%{(tr < -0.12).mean() * 100:.0f}** "
                 f"(en kötü {tr.min() * 100:+.0f}%)")
        years = max(len(px) / 252, 1e-9)
        wl = w_sig.shift(1).fillna(0.0)
        turnover = float((wl - wl.shift(1).fillna(0.0)).abs().sum().sum())
        L.append(f"- Yıllık işlem maliyeti yükü: yaklaşık **%{turnover * COST / years * 100:.1f}**")
    L.append("")

    # 3) en kötü günler
    contrib = w_sig.shift(1).fillna(0.0) * rets.fillna(0.0)
    port = contrib.sum(axis=1)
    L.append("## 3) Portföyün en kötü 6 günü (ham veri)\n")
    for day in port.nsmallest(6).index:
        sym = contrib.loc[day].idxmin()
        r = rets.loc[day, sym]
        flag = "VERİ HATASI OLABİLİR" if abs(r) > LIMIT else "gerçek hareket"
        L.append(f"- {day.date()}: portföy {port[day] * 100:+.1f}%, en kötü {sym} {r * 100:+.0f}% ({flag})")
    L.append("")

    # 4) varyantlar
    ind_c = indicators(pxc)
    cols = ["yillik_%", "toplam_getiri_%", "max_dusus_%", "40g_medyan_%", "40g_hedefi_tutan_%",
            "40g_en_kotu_%", "ort_kasa_%"]
    rows = {}
    rows["Sinyal: ham veri"] = evaluate(w_sig, rets)
    rows["Sinyal: ham + nakit faizi"] = evaluate(w_sig, rets, cash_rate)
    w_sig_c = strat_signals(pxc, ind_c)
    rows["Sinyal: temiz veri"] = evaluate(w_sig_c, rc)
    rows["Sinyal: temiz + nakit faizi"] = evaluate(w_sig_c, rc, cash_rate)
    rows["Momentum: ham veri"] = evaluate(strat_mom(px, ind), rets)
    rows["Momentum: ham + likit"] = evaluate(strat_mom(px, ind, liquid), rets)
    rows["Momentum: temiz veri"] = evaluate(strat_mom(pxc, ind_c), rc)
    rows["Momentum: temiz + likit"] = evaluate(strat_mom(pxc, ind_c, liquid), rc)
    rows["Mom+trend: ham veri"] = evaluate(strat_mom(px, ind, trend=True), rets)
    rows["Mom+trend: temiz + likit"] = evaluate(strat_mom(pxc, ind_c, liquid, trend=True), rc)
    rows["Mom+trend: temiz + likit + nakit faizi"] = evaluate(
        strat_mom(pxc, ind_c, liquid, trend=True), rc, cash_rate)
    rows["Kıyas: eşit ağırlık, ham"] = evaluate_series(rets.mean(axis=1))
    rows["Kıyas: eşit ağırlık, temiz"] = evaluate_series(rc.mean(axis=1))
    rows["Kıyas: eşit ağırlık, temiz + sadece likit"] = evaluate_series(rc.where(liquid).mean(axis=1))
    if xu100 is not None and len(xu100) > WARMUP:
        bx = xu100.reindex(px.index).ffill().pct_change(fill_method=None).fillna(0.0)
        rows["Kıyas: BIST 100 endeksi"] = evaluate_series(bx)
    L.append("## 4) Varyantlar (baştaki 130 gün hariç, maliyet dahil)\n")
    L.append(md_table(rows, cols))
    L.append("\nNasıl okunur: 'ham' ile 'temiz' arasındaki fark büyükse sonuçları veri hataları bozuyor. "
             "'nakit faizi' farkı boş kasanın maliyetini, 'likit' farkı küçük hisse etkisini gösterir. "
             "Eşit ağırlık ham ile temiz arasında büyük fark varsa o kıyas şişiktir.")
    L.append("\n> Uyarı: Veri bugünkü hisse listesidir (batan şirketler yok, sonuçlar iyimser çıkar). "
             "%12 üstü hareketleri silmek gerçek (limitsiz) hareketleri de siler, yaklaşık bir temizliktir. "
             "Yatırım tavsiyesi değildir.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols-file", default="bist_symbols.txt")
    ap.add_argument("--period", default="10y")
    ap.add_argument("--cash-rate", type=float, default=0.30, help="boş kasaya yıllık faiz varsayımı (0.30 = %%30)")
    ap.add_argument("--out", default="diagnose.md")
    args = ap.parse_args()

    symbols = read_symbols(args.symbols_file)
    print(f"{len(symbols)} sembol indiriliyor ({args.period})...")
    close, volume = fetch_all(symbols, args.period)

    xu100 = None
    try:
        import yfinance as yf
        raw = yf.download("XU100.IS", period=args.period, auto_adjust=True, progress=False)
        c = raw["Close"]
        xu100 = (c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c).dropna()
    except Exception as exc:  # endeks alınamazsa kıyas satırı atlanır
        print("Endeks alınamadı:", exc)

    text = diagnose(close, volume, xu100, args.cash_rate)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
