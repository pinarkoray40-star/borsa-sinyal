"""
farklar.py - Büyük Hareket (BH) ile Büyük Hareket 2 (BH2) arasındaki FARKLI sinyalleri listeler.
Hiçbir dosyayı değiştirmez, siteye dokunmaz; sadece rapor yazar (Actions logunda ve farklar_rapor.txt).

Çıktı:
  1) Ortak sinyaller (ikisinde de var)
  2) Sadece BH'de olanlar  = dayanıklılık şartının ELEDİĞİ sinyaller
  3) Sadece BH2'de olanlar (cooldown kayması yüzünden çıkabilir)
Her grup için: sayı, ortalama, medyan, kazanan oranı, stop oranı. Süresi dolmamış (açık) işlemler ayrıca gösterilir.
"""
import numpy as np
import pandas as pd

import buyuk as BY
import buyuk2 as BY2
import signals as S

LINES = []


def out(s=""):
    print(s)
    LINES.append(s)


def stat(name, es):
    done = [e for e in es if e["st"] != "açık"]
    opn = [e for e in es if e["st"] == "açık"]
    out(f"{name}: {len(es)} sinyal ({len(done)} kapanmış, {len(opn)} açık)")
    if done:
        r = np.array([e["pnl"] for e in done])
        stop = sum(e["st"] == "stop" for e in done) / len(done) * 100
        out(f"   kapanmış -> ort {r.mean():+.1f}%  medyan {np.median(r):+.1f}%  kazanan {(r > 0).mean() * 100:.0f}%  stop {stop:.0f}%")
    if opn:
        c = np.array([e["pnl"] for e in opn])
        out(f"   açık      -> ort {c.mean():+.1f}%  medyan {np.median(c):+.1f}%  artıda {(c > 0).mean() * 100:.0f}%")


def liste(es):
    for e in sorted(es, key=lambda x: x["date"]):
        dnr = e.get("dnr")
        dn = f"dn%{dnr}" if dnr is not None else "dn?"
        out(f"   {e['date']}  {e['symbol']:<7} {e['st']:<5} {e['days']:>3} gün  {e['pnl']:+6.1f}%  {dn}")


def main():
    symbols = S.read_symbols("bist_symbols.txt")
    print(f"{len(symbols)} sembol indiriliyor...")
    close, volume = S.fetch_all(symbols, "3y")
    a = BY.buyuk_block(close, volume)
    b = BY2.buyuk2_block(close, volume)
    # BH girişlerine dayanıklılık sırasını da ekle (karşılaştırma için)
    idx = BY2.fetch_index(close)
    _, dr = BY2.dayaniklilik(close, idx, BY2.PARAM["window"])
    for e in a["entries"]:
        try:
            v = dr.loc[pd.Timestamp(e["date"]), e["symbol"] + ".IS"]
            e["dnr"] = int(round(float(v) * 100)) if np.isfinite(v) else None
        except Exception:
            e["dnr"] = None

    ka = {(e["symbol"], e["date"]): e for e in a["entries"]}
    kb = {(e["symbol"], e["date"]): e for e in b["entries"]}
    ortak = [ka[k] for k in ka if k in kb]
    sadece_a = [ka[k] for k in ka if k not in kb]
    sadece_b = [kb[k] for k in kb if k not in ka]

    out(f"Son mum: {a['last_bar']}   BH: {a['n']} sinyal   BH2: {b['n']} sinyal")
    out("")
    stat("ORTAK (ikisinde de)", ortak)
    stat("SADECE BH (dayanıklılık şartı eledi)", sadece_a)
    stat("SADECE BH2", sadece_b)
    out("")
    out("--- SADECE BH (BH2'nin eledikleri) ---")
    liste(sadece_a)
    out("")
    out("--- SADECE BH2 ---")
    liste(sadece_b)
    out("")
    out("Not: pnl, süresi dolmuş işlemde çıkış getirisi; açık işlemde bugünkü getiri. Hepsi kapanış fiyatıyla.")
    with open("farklar_rapor.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LINES))


if __name__ == "__main__":
    main()
