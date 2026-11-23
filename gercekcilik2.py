import numpy as np
import pandas as pd

ev = pd.read_csv("gercekcilik.csv")
ev["tarih"] = pd.to_datetime(ev["tarih"])
ev["ay"] = ev["tarih"].dt.to_period("M").astype(str)
ev["yil"] = ev["tarih"].dt.year
ev["N"] = ev["R"] - 0.8          # senaryo 3: maliyet %0,8
ev["gap_in"] = ev["gap_in"].fillna(0)


def stat(e, ad):
    e = e.dropna(subset=["N", "bR"])
    if len(e) < 20:
        print(f"{ad}: n={len(e)} cok az")
        return
    d = e["N"] - e["bR"]
    m = d.groupby(e["ay"]).mean()
    M = len(m)
    t = m.mean() / (m.std() / np.sqrt(M)) if M >= 4 else float("nan")
    print(f"{ad}: n={len(e)} ort {e['N'].mean():.1f} med {e['N'].median():.1f} "
          f"kaz %{(e['N'] > 0).mean() * 100:.0f} | evren {e['bR'].mean():.1f} "
          f"fark {d.mean():+.1f} t={t:.1f}")


print(f"islem: {len(ev)} | senaryo: ertesi acilis giris, maliyet %0,8")
stat(ev, "TEMEL")

print("\n=== GIRIS KAYMASI DAGILIMI (acilis - sinyal kapanisi, %) ===")
g = ev["gap_in"]
for th in (0, 2, 5, 9):
    print(f"> %{th}: {int((g > th).sum())} islem (%{(g > th).mean() * 100:.1f})")

print("\n=== ALINAMAZ ISLEMLER CIKARILINCA (acilis kaymasi esigi) ===")
for th in (3, 5, 9):
    sub = ev[ev["gap_in"] <= th]
    out = ev[ev["gap_in"] > th]
    print(f"\nkayma > %{th} cikarildi: {len(out)} islem atildi, "
          f"atilanlarin ort getirisi %{out['N'].mean():.1f} "
          f"(med %{out['N'].median():.1f})")
    stat(sub, f"  kalan")

print("\n=== EN IYI ISLEMLER CIKARILINCA ===")
srt = ev.sort_values("N", ascending=False)
tot = ev["N"].sum()
for p in (1, 3, 5):
    k = max(1, int(len(ev) * p / 100))
    pay = srt.head(k)["N"].sum() / tot * 100
    print(f"\nen iyi %{p} ({k} islem) toplam getirinin %{pay:.0f}'ini yapiyor")
    stat(srt.iloc[k:], "  kalan")

print("\n=== GETIRI +%100 ILE KIRPILIRSA (winsorize) ===")
w = ev.copy()
w["N"] = w["N"].clip(upper=100)
stat(w, "kirpilmis")

print("\n=== SERT SENARYO: kayma <= %5 ve en iyi %3 cikar ===")
sub = ev[ev["gap_in"] <= 5].sort_values("N", ascending=False)
k = max(1, int(len(sub) * 0.03))
sert = sub.iloc[k:]
stat(sert, "sert genel")
med = sert["tarih"].median()
stat(sert[sert["tarih"] <= med], "sert 1.yari")
stat(sert[sert["tarih"] > med], "sert 2.yari")
for y in sorted(sert["yil"].unique()):
    stat(sert[sert["yil"] == y], f"sert {y}")
print("\nbitti")
