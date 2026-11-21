"""
ayakta_test.py - "Endeks zayıfken GERÇEKTEN yükselen (ayakta kalan) hisse sonra ne yapıyor?" testi.
Hiçbir dosyayı değiştirmez, siteye dokunmaz; sadece rapor yazar (Actions logu + ayakta_rapor.txt).

Fikir: önceki dayanıklılık ölçümü ortalama göreceli getiriye bakıyordu. Burada doğrudan:
  Zayıf gün  : son W günde XU100 en az %3 düşmüş (W = 20 ve 60)
  Ayakta     : aynı W gününde hisse MUTLAK olarak yükselmiş (sr > 0)
Gruplar (hepsi zayıf günlerde, işlem hacmi en düşük %30 hariç):
  A  ayakta            : sr > 0
  B  güçlü ayakta      : sr > +%10
  C  güçlü + zirveye yakın : sr > +%10 ve kapanış 60 günlük zirvenin >= %90'ı
  D  (kıyas) göreceli güç : sr - endeks getirisi, gün içinde en yüksek %20 (mutlak yükseliş şartı YOK)
Sonuç: sinyalden 21 / 42 / 84 işlem günü sonraki getiri, aynı günün tüm (likit) hisse ortalamasına göre FARK (puan).
Ayrıca fark, sonraki dönemde endeks düşmeye devam ettiyse / toparlandıysa ayrı ayrı gösterilir ve iki yarı ayrı verilir.
Gözlemler her 10 günde bir alınır (üst üste binmeyi azaltmak için). t = günler arası ortalama fark / standart hata.
En sonda: BUGÜN (son mum) endeks zayıfsa ayakta kalan hisselerin güncel listesi.
"""
import numpy as np
import pandas as pd

import buyuk2 as BY2
import signals as S

LINES = []
HORIZONS = (21, 42, 84)
WINDOWS = (20, 60)
WEAK = -0.03
STEP = 10
PERIOD = "5y"


def out(s=""):
    print(s)
    LINES.append(s)


def tstat(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def fmt(x, nd=1, sign=True):
    if x is None or not np.isfinite(x):
        return "  -  "
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def groups_for(close, idx, W, liquid):
    sr = close / close.shift(W) - 1
    ir = idx / idx.shift(W) - 1
    zirve = close / close.rolling(60, min_periods=40).max()
    rel = sr.sub(ir, axis=0)
    relrank = rel.where(liquid).rank(axis=1, pct=True)
    g = {
        "A ayakta (sr>0)": (sr > 0) & liquid,
        "B güçlü ayakta (sr>+10%)": (sr > 0.10) & liquid,
        "C güçlü+zirveye yakın": (sr > 0.10) & (zirve >= 0.90) & liquid,
        "D göreceli güç top20% (kıyas)": (relrank >= 0.80) & liquid,
    }
    return sr, ir, g


def evaluate(close, idx, liquid, W, gname, mask, ir, h, dates_ok):
    fwd = close.shift(-h) / close - 1
    idxf = idx.shift(-h) / idx - 1
    uni = fwd.where(liquid).mean(axis=1)
    rows = []
    for d in dates_ok:
        m = mask.loc[d]
        if m.sum() < 3:
            continue
        f = fwd.loc[d][m]
        f = f.dropna()
        if len(f) < 3 or not np.isfinite(uni.loc[d]):
            continue
        rows.append((d, float(f.mean() - uni.loc[d]) * 100, float(f.median() - uni.loc[d]) * 100,
                     float((f > uni.loc[d]).mean()) * 100, float(f.mean()) * 100, len(f), float(idxf.loc[d])))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["d", "ex", "exmed", "hit", "abs", "n", "idxf"]).set_index("d")
    half = len(df) // 2
    res = {
        "dates": len(df), "n": int(df["n"].sum()), "ex": df["ex"].mean(), "exmed": df["exmed"].mean(),
        "hit": df["hit"].mean(), "abs": df["abs"].mean(), "t": tstat(df["ex"]),
        "y1": df["ex"].iloc[:half].mean() if half else np.nan, "y2": df["ex"].iloc[half:].mean() if half else np.nan,
    }
    dn = df[df["idxf"] < 0]
    up = df[df["idxf"] >= 0]
    res["ex_dn"], res["n_dn"] = (dn["ex"].mean() if len(dn) else np.nan), len(dn)
    res["ex_up"], res["n_up"] = (up["ex"].mean() if len(up) else np.nan), len(up)
    res["abs_dn"] = dn["abs"].mean() if len(dn) else np.nan
    return res


def run(close, volume, idx):
    cidx = pd.to_datetime(close.index).tz_localize(None)
    ix = idx.copy()
    ix.index = pd.to_datetime(ix.index).tz_localize(None)
    ix = ix.reindex(cidx).ffill(limit=3)
    ix.index = close.index

    liq = (close * volume).rolling(60, min_periods=30).mean()
    liquid = liq.rank(axis=1, pct=True) >= 0.30

    out(f"Son mum: {close.index[-1]:%Y-%m-%d}   evren: {close.shape[1]} hisse   dönem: {close.index[0]:%Y-%m-%d} -> {close.index[-1]:%Y-%m-%d}")
    out(f"Zayıf gün = son W günde XU100 <= {WEAK * 100:.0f}%. Fark = grup ortalaması - aynı günün tüm likit hisse ortalaması (puan).")
    out("")

    today = {}
    for W in WINDOWS:
        sr, ir, groups = groups_for(close, ix, W, liquid)
        weak = (ir <= WEAK)
        sample = np.zeros(len(close), dtype=bool)
        sample[W + 60::STEP] = True
        dates_all = close.index[sample & weak.to_numpy()]
        out(f"==== W={W} gün: zayıf gün sayısı (örneklenen) {len(dates_all)} ====")
        for gname, mask in groups.items():
            out(f"-- {gname}")
            for h in HORIZONS:
                dates_ok = [d for d in dates_all if close.index.get_loc(d) + h < len(close)]
                r = evaluate(close, ix, liquid, W, gname, mask, ir, h, dates_ok)
                if r is None:
                    out(f"   {h:>2}g: yeterli gözlem yok")
                    continue
                out(f"   {h:>2}g: fark {fmt(r['ex'])} (medyan {fmt(r['exmed'])}) t={fmt(r['t'], 1)} | grup mutlak {fmt(r['abs'])}% | "
                    f"hisse>evren %{r['hit']:.0f} | yarı1 {fmt(r['y1'])} yarı2 {fmt(r['y2'])} | gün {r['dates']}, gözlem {r['n']}")
                out(f"        endeks düşmeye DEVAM etti (n={r['n_dn']}): fark {fmt(r['ex_dn'])}, mutlak {fmt(r['abs_dn'])}% | "
                    f"toparlandı (n={r['n_up']}): fark {fmt(r['ex_up'])}")
        out("")
        today[W] = (sr, ir, groups)

    out("==== BUGÜN ====")
    last = close.index[-1]
    for W in WINDOWS:
        sr, ir, groups = today[W]
        irv = float(ir.loc[last]) if np.isfinite(ir.loc[last]) else float("nan")
        weak_now = irv <= WEAK
        out(f"W={W}: endeks son {W} günde {irv * 100:+.1f}%  -> {'ZAYIF' if weak_now else 'zayıf değil'}")
        for gname in ("B güçlü ayakta (sr>+10%)", "C güçlü+zirveye yakın"):
            m = groups[gname].loc[last]
            names = sr.loc[last][m].sort_values(ascending=False)
            out(f"   {gname}: {len(names)} hisse" + ("" if weak_now else " (endeks zayıf olmadığı için sinyal sayılmaz, bilgi amaçlı)"))
            line = ", ".join(f"{k.replace('.IS', '')} {v * 100:+.0f}%" for k, v in names.head(30).items())
            if line:
                out("      " + line)
    out("")
    out("Not: Hepsi kapanış fiyatıyla, işlem maliyeti yok. Fark pozitif ve t>2 ise grup evrenden sistematik iyi; "
        "'düşmeye devam' satırı asıl hedefi (endeks düşerken ayakta kalma) ölçer.")


def main():
    symbols = S.read_symbols("bist_symbols.txt")
    print(f"{len(symbols)} sembol indiriliyor ({PERIOD})...")
    close, volume = S.fetch_all(symbols, PERIOD)
    idx = BY2.fetch_index(close)
    run(close, volume, idx)
    with open("ayakta_rapor.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LINES))


if __name__ == "__main__":
    main()
