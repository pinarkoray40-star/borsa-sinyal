import glob
import warnings

import numpy as np
import pandas as pd

import portfoy2 as P

warnings.filterwarnings("ignore")

B = P.B
B2 = P.B2
H = 84
LBLER = (20, 60)


def grup_idlerini_kur(close, df, kolon):
    kod_kol = {str(c).replace(".IS", ""): j for j, c in enumerate(close.columns)}
    ad = np.array([None] * close.shape[1], dtype=object)
    for kod, v in zip(df["kod"], df[kolon]):
        j = kod_kol.get(str(kod))
        if j is not None and isinstance(v, str) and v.strip():
            ad[j] = v.strip()
    ids = np.full(close.shape[1], -1)
    adlar = sorted({a for a in ad if a is not None})
    for k, a in enumerate(adlar):
        ids[[j for j in range(len(ad)) if ad[j] == a]] = k
    return ids, adlar


def sektor_gucu(Rc, grup, min_uye):
    """Her gun x hisse: hissenin grubunun (hissenin kendisi HARIC) ortalama
    getirisi - tum piyasa ortalamasi. Grup yoksa/kucukse NaN."""
    T, S = Rc.shape
    gecerli = np.isfinite(Rc)
    Rz = np.where(gecerli, Rc, 0.0)
    pazar = Rz.sum(1) / np.maximum(gecerli.sum(1), 1)
    guc = np.full((T, S), np.nan)
    for g in np.unique(grup[grup >= 0]):
        uye = np.where(grup == g)[0]
        if len(uye) < min_uye:
            continue
        s = Rz[:, uye].sum(1, keepdims=True)
        c = gecerli[:, uye].sum(1, keepdims=True)
        loo_s = s - Rz[:, uye]
        loo_c = c - gecerli[:, uye]
        loo = np.where(loo_c >= 3, loo_s / np.maximum(loo_c, 1), np.nan)
        guc[:, uye] = loo - pazar[:, None]
    return guc


def bugun_siralama(Rc, grup, adlar, min_uye, baslik):
    gecerli = np.isfinite(Rc[-1])
    pazar = np.nanmean(Rc[-1])
    sat = []
    for g, a in enumerate(adlar):
        uye = np.where((grup == g) & gecerli)[0]
        if len(uye) >= min_uye:
            sat.append((np.mean(Rc[-1, uye]) - pazar, a, len(uye)))
    sat.sort(reverse=True)
    print(f"\n{baslik} (piyasaya gore fark, puan):")
    for f, a, n in sat[:6]:
        print(f"  EN GUCLU  {a:<30} {f * 100:+6.1f}  (n={n})")
    for f, a, n in sat[-4:]:
        print(f"  EN ZAYIF  {a:<30} {f * 100:+6.1f}  (n={n})")


def yaz(ad, fark, gun):
    n = len(fark)
    if n < 15:
        print(f"  {ad:<26} n={n:<4} (az)")
        return
    ort = fark.mean()
    med = np.median(fark)
    hit = (fark > 0).mean() * 100
    sd = fark.std(ddof=1)
    t = ort / (sd / np.sqrt(n)) if sd > 0 else 0.0
    orta = np.median(gun)
    a, b = fark[gun <= orta], fark[gun > orta]
    y1 = a.mean() if len(a) else float("nan")
    y2 = b.mean() if len(b) else float("nan")
    print(f"  {ad:<26} n={n:<4} ort {ort:+6.1f}  medyan {med:+6.1f}  "
          f"kazanan %{hit:3.0f}  t={t:4.1f}  Y1 {y1:+6.1f}  Y2 {y2:+6.1f}")


def grup_raporu(baslik, ii, jj, guc, F, evr):
    ii, jj = np.asarray(ii), np.asarray(jj)
    ok = ii < len(F)
    ii, jj = ii[ok], jj[ok]
    r = np.clip(F[ii, jj], -0.9, 3.0)
    g = guc[ii, jj]
    iyi = np.isfinite(r) & np.isfinite(evr[ii])
    fark = (r - evr[ii]) * 100
    print(f"\n{baslik}  (sinyal {len(ii)}, getirisi olan {int(iyi.sum())}, "
          f"sektoru olan {int((iyi & np.isfinite(g)).sum())})")
    var = iyi & np.isfinite(g)
    yaz("TUMU (sektoru olan)", fark[var], ii[var])
    for ad, m in (("GUCLU (>0)", g > 0), ("COK GUCLU (>+5p)", g > 0.05),
                  ("ZIRVE GUCLU (>+10p)", g > 0.10),
                  ("ZAYIF (<=0)", g <= 0), ("COK ZAYIF (<-5p)", g < -0.05)):
        s = var & m
        yaz(ad, fark[s], ii[s])


def main():
    df = pd.read_csv(glob.glob("**/sektor.csv", recursive=True)[0])
    print("sektor tablosu:", len(df), "satir")
    close, opn, vol, xu = P.indir(P.sembol_oku())
    C = close.to_numpy()
    T, S = C.shape
    print("hisse:", S, "| gun:", T)

    bh = B.ozellikler(close, vol)
    sk = P.skorlar(close, xu)
    dn_ok = np.nan_to_num(sk["dn"] / 100.0, nan=0.0) >= B2.PARAM["dn_min"]
    ii_bh, jj_bh = B.sinyaller(bh, P.CD)
    ii_b2, jj_b2 = B.sinyaller(bh & dn_ok, P.CD)
    print("sinyal: BH", len(ii_bh), "| BH2", len(ii_b2))

    F = np.clip(C[H:] / C[:-H] - 1, -0.9, 3.0)
    evr = np.nanmean(F, axis=1)
    evr = np.concatenate([evr, np.full(H, np.nan)])
    F = np.vstack([F, np.full((H, S), np.nan)])

    sek_id, sek_ad = grup_idlerini_kur(close, df, "sector")
    alt_id, alt_ad = grup_idlerini_kur(close, df, "industry")
    print("sektor:", len(sek_ad), "| alt sektor:", len(alt_ad))

    print(f"\n(Fark = {H} gun sonraki getiri - ayni gun evren ortalamasi, puan; "
          "Y1/Y2 = sinyal gunlerinin ilk/ikinci yarisi; guc = grubun hisse haric "
          "ortalama getirisi - piyasa)")
    for lb in LBLER:
        Rc = pd.DataFrame(C).pct_change(lb).clip(-0.5, 1.0).to_numpy()
        for ad, ids, adlar, mu in (("SEKTOR", sek_id, sek_ad, 5),
                                   ("ALT SEKTOR", alt_id, alt_ad, 6)):
            guc = sektor_gucu(Rc, ids, mu)
            print("\n" + "=" * 100)
            print(f"{ad} gucu, bakis {lb} gun")
            grup_raporu(f"BH2 / {ad} {lb}g", ii_b2, jj_b2, guc, F, evr)
            grup_raporu(f"BH  / {ad} {lb}g", ii_bh, jj_bh, guc, F, evr)
            if ad == "SEKTOR":
                bugun_siralama(Rc, ids, adlar, mu, f"BUGUN SEKTOR GUCU ({lb}g)")
    print("\nNot: sinyaller ve 84 gunluk pencereler birbirine biner, t degerleri sisik olabilir.")
    print("bitti")


if __name__ == "__main__":
    main()
