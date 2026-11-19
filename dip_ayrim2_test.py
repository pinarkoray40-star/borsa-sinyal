"""
dip_ayrim2_test.py - "Büyük Hareket" sinyalinde HEDEFE gidenleri STOP olanlardan ayıran özellik var mı? (2. çalışma)

Fark: ~45 özellik (19 eski + 26 yeni: uzun vade, hacim birikimi, oynaklık daralması, göreli güç, trend...).
Her özellik için hedefe giden ve stop olan işlemler karşılaştırılır (AUC: 0,5 = ayıramaz, 1,0 = mükemmel ayırır).
Tesadüfü ayıklamak için etiketler 1000 kez karıştırılır; "düzeltilmiş p" = tüm özellikler arasında en iyisi
bu kadar iyi görünme ihtimali. 0,05'in altı ciddiye alınır, altında kalanlar şans olabilir.
Önce tüm işlemlerde, sonra yalnızca piyasa sağlıklıyken (SMA50 üstü hisse oranı >= %50) bakılır.
dip_ayrim_test.py, dip_buyuk_test.py, dip_dar_test.py, dip_gunluk_test.py, dip_islem_test.py, signals.py aynı klasörde olmalı.
Sitedeki hiçbir dosyaya dokunmaz.
"""
import os

import numpy as np
import pandas as pd

import dip_ayrim_test as A
import dip_buyuk_test as B
import dip_dar_test as D
import dip_gunluk_test as G
import dip_islem_test as I
import signals as S

TP, SL = A.TP, A.SL
NPERM = 1000


def yeni_ozellikler(close, volume):
    r = close.pct_change()
    r60 = close / close.shift(60) - 1
    r120 = close / close.shift(120) - 1
    sma50, sma200 = close.rolling(50).mean(), close.rolling(200).mean()
    obv = (np.sign(r).fillna(0) * volume).cumsum()
    kir = (close >= close.shift(1).rolling(20).max()).astype(float)
    ciro = close * volume
    up_v = (volume * (r > 0)).rolling(20).sum()
    dn_v = (volume * (r < 0)).rolling(20).sum()
    F = {
        "120g getiri %": r120 * 100,
        "250g getiri %": (close / close.shift(250) - 1) * 100,
        "52 hafta zirvesine uzaklık %": (close / close.rolling(250).max() - 1) * 100,
        "52 hafta dibinden yükseliş %": (close / close.rolling(250).min() - 1) * 100,
        "120g en büyük düşüş %": (close / close.rolling(120).max() - 1).rolling(120).min() * 100,
        "Oynaklık daralması (10g/60g std)": r.rolling(10).std() / r.rolling(60).std(),
        "14g ort. mutlak getiri %": r.abs().rolling(14).mean() * 100,
        "Yukarı/aşağı gün hacmi (20g)": up_v / dn_v.replace(0, np.nan),
        "OBV eğimi (60g, hacme oranlı)": (obv - obv.shift(60)) / volume.rolling(60).sum(),
        "Son 60g'de 20g-kırılım sayısı": kir.rolling(60).sum(),
        "Son 60g'de +%7 üstü gün sayısı": (r > 0.07).astype(float).rolling(60).sum(),
        "Son 60g'de -%7 altı gün sayısı": (r < -0.07).astype(float).rolling(60).sum(),
        "60g göreli güç (hisse - piyasa medyanı) %": r60.sub(r60.median(axis=1), axis=0) * 100,
        "120g göreli güç (hisse - piyasa medyanı) %": r120.sub(r120.median(axis=1), axis=0) * 100,
        "60g getiri sıralaması (yüzdelik)": r60.rank(axis=1, pct=True) * 100,
        "Ciro trendi (20g/120g)": ciro.rolling(20).mean() / ciro.rolling(120).mean(),
        "SMA50 eğimi (20g) %": (sma50 / sma50.shift(20) - 1) * 100,
        "SMA50 / SMA200 %": (sma50 / sma200 - 1) * 100,
        "Son 5g fiyat aralığı %": (close.rolling(5).max() / close.rolling(5).min() - 1) * 100,
        "60g getiri çarpıklığı": r.rolling(60).skew(),
        "Hacim dalgalanması (60g std/ort)": volume.rolling(60).std() / volume.rolling(60).mean(),
        "En büyük günlük artış 60g %": r.rolling(60).max() * 100,
        "En büyük günlük düşüş 60g %": r.rolling(60).min() * 100,
        "Yeni zirve günü sayısı (son 20g)": (close >= close.expanding().max()).astype(float).rolling(20).sum(),
        "Kapanış / 20g ortalama %": (close / close.rolling(20).mean() - 1) * 100,
        "10g getiri %": (close / close.shift(10) - 1) * 100,
    }
    return {k: v.replace([np.inf, -np.inf], np.nan).to_numpy() for k, v in F.items()}


def auc(x, lab):
    """x: özellik, lab: True=hedef, False=stop. Hedefte yüksek olma olasılığı."""
    ok = np.isfinite(x)
    x, lab = x[ok], lab[ok]
    nh, ns = int(lab.sum()), int((~lab).sum())
    if nh < 10 or ns < 10:
        return np.nan
    rk = pd.Series(x).rank(method="average").to_numpy()
    return (rk[lab].sum() - nh * (nh + 1) / 2) / (nh * ns)


def analiz(V, ii, hedef, stop, y, baslik, ad_sirasi, L, rng):
    hs = hedef | stop
    n = int(hs.sum())
    L += ["## %s\n" % baslik,
          "%d işlem: %d hedefe giden, %d stop olan (süre dolanlar bu karşılaştırmada yok).\n" % (n, hedef.sum(), stop.sum())]
    if hedef.sum() < 30 or stop.sum() < 30:
        L += ["Karşılaştırma için yeterli işlem yok.\n"]
        return
    lab = hedef[hs]
    idx = ii[hs]
    names = [k for k in ad_sirasi if np.isfinite(V[k][hs]).sum() > 0.7 * n]
    X = np.vstack([V[k][hs] for k in names])
    # NaN -> özelliğin medyanı (yalnız sıralama için)
    med = np.nanmedian(X, axis=1)
    Xf = np.where(np.isfinite(X), X, med[:, None])
    R = np.vstack([pd.Series(row).rank(method="average").to_numpy() for row in Xf])
    nh, ns = int(lab.sum()), int((~lab).sum())
    base = nh * (nh + 1) / 2
    obs = (R[:, lab].sum(axis=1) - base) / (nh * ns)
    # etiket karıştırma
    perm = np.argsort(rng.random((NPERM, n)), axis=1)[:, :nh]       # her satırda nh rastgele işlem = "hedef"
    nullmax = np.empty(NPERM)
    for b in range(NPERM):
        a = (R[:, perm[b]].sum(axis=1) - base) / (nh * ns)
        nullmax[b] = np.abs(a - 0.5).max()
    padj = np.array([(nullmax >= abs(o - 0.5)).mean() for o in obs])
    ei = idx < y
    rows = []
    for k, o, p, row in zip(names, obs, padj, Xf):
        a1 = auc(V[k][hs][ei], lab[ei])
        a2 = auc(V[k][hs][~ei], lab[~ei])
        tut = (not np.isnan(a1)) and (not np.isnan(a2)) and (a1 - 0.5) * (a2 - 0.5) > 0 and (o - 0.5) * (a1 - 0.5) > 0
        mh = np.nanmedian(V[k][hs][lab])
        ms = np.nanmedian(V[k][hs][~lab])
        rows.append((k, o, mh, ms, a1, a2, p, tut))
    rows.sort(key=lambda t: -abs(t[1] - 0.5))
    f2 = lambda x: "—" if pd.isna(x) else f"{x:.2f}"
    f3 = lambda x: "—" if pd.isna(x) else f"{x:.3g}"
    L += ["| Özellik | AUC | Hedefe gidenlerde medyan | Stop olanlarda medyan | Yarı1 AUC | Yarı2 AUC | Düzeltilmiş p | İki yarıda aynı yön |",
          "|---|---|---|---|---|---|---|---|"]
    for k, o, mh, ms, a1, a2, p, tut in rows[:15]:
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (k, f2(o), f3(mh), f3(ms), f2(a1), f2(a2), f2(p), "EVET" if tut else "hayır"))
    L.append("")
    en_iyi = rows[0]
    L += ["Şans seviyesi: 1000 karıştırmada en iyi özelliğin AUC'si 0,5'ten ortalama %.3f uzaklaştı, "
          "en uç %%95'lik değeri %.3f. Gerçek en iyi: %.3f (%s).\n" % (
              nullmax.mean(), np.percentile(nullmax, 95), abs(en_iyi[1] - 0.5), en_iyi[0])]
    return rows


def calistir(close, volume):
    T = close.shape[0]
    y = T // 2
    feats, gecerli = D.genis(close, volume)
    B.ekle(close, volume, feats)
    mask = gecerli & feats["hareketli"] & feats["kirilim"] & feats["kuruma"] & feats["yakin"]
    raw_count = mask.sum(axis=1)
    ii, jj = G.sinyaller(mask)
    ii, jj, P = A.yollar2(close, ii, jj)
    r, cu, cd = I.islem(P, TP, SL)
    hedef = np.asarray(cu, dtype=bool)
    stop = np.asarray(cd, dtype=bool)
    F = A.ozellikler(close, volume, raw_count)
    F.update(yeni_ozellikler(close, volume))
    V = {k: arr[ii, jj] for k, arr in F.items()}
    sma50 = close.rolling(50).mean()
    genis = (close > sma50).astype(float).where(close.notna()).mean(axis=1).to_numpy() * 100
    gv = genis[ii]
    rng = np.random.default_rng(12345)
    ad = list(F.keys())
    L = ["# Büyük Hareket: hedefe gidenleri stop olanlardan ayıran ne? (2. çalışma)\n",
         f"{close.shape[1]} hisse, {close.index[0]:%d.%m.%Y} - {close.index[-1]:%d.%m.%Y}. Toplam {len(r)} işlem, {len(ad)} özellik. "
         f"+%{TP:.0f} hedef / -%{SL:.0f} stop / {I.H} gün, %{I.KOM} komisyon dahil.\n"]
    analiz(V, ii, hedef, stop, y, "1) Tüm işlemler", ad, L, rng)
    saglikli = np.isfinite(gv) & (gv >= 50)
    analiz(V, ii, hedef & saglikli, stop & saglikli, y, "2) Yalnızca piyasa sağlıklıyken (SMA50 üstü hisse oranı ≥ %50)", ad, L, rng)
    L += ["Nasıl okunur: bir özellik ancak düzeltilmiş p < 0,05 ise, iki yarıda da aynı yönde (AUC 0,5'in aynı tarafında) ve "
          "medyan farkı anlamlı büyüklükteyse ciddiye alınır. AUC 0,55-0,60 zayıf, 0,60-0,65 orta, 0,65+ güçlü ayırımdır. "
          "İşlemler arka arkaya ve birbirine yakın tarihlerde olduğundan gerçek p değeri biraz daha kötü olabilir. "
          "Güncel hisse listesi ve kapanış varsayımları iyimserdir."]
    return "\n".join(L)


def main():
    import yfinance  # noqa: F401
    symbols = S.read_symbols("bist_symbols.txt")
    print(f"{len(symbols)} sembol, 5 yıllık günlük veri indiriliyor...")
    close, volume = S.fetch_all(symbols, "5y")
    txt = calistir(close, volume)
    print(txt)
    ozet = os.environ.get("GITHUB_STEP_SUMMARY")
    if ozet:
        with open(ozet, "a", encoding="utf-8") as fh:
            fh.write(txt + "\n")


if __name__ == "__main__":
    main()
