import time

import pandas as pd
import yfinance as yf

from tavan_aday import sembol_oku


def bilgi_al(sembol):
    for deneme in range(4):
        try:
            info = yf.Ticker(sembol).info
            return info.get("sector"), info.get("industry")
        except Exception as e:
            mesaj = str(e)
            bekle = 6 * (deneme + 1) if ("429" in mesaj or "Too Many" in mesaj) else 2
            time.sleep(bekle)
    return None, None


def main():
    semboller = sembol_oku()
    print("hisse:", len(semboller), flush=True)
    satirlar = []
    for i, s in enumerate(semboller, 1):
        sembol = s if s.endswith(".IS") else s + ".IS"
        sec, ind = bilgi_al(sembol)
        satirlar.append({"kod": sembol.replace(".IS", ""),
                         "sector": sec, "industry": ind})
        if i % 50 == 0:
            pd.DataFrame(satirlar).to_csv("sektor.csv", index=False)
            dolu = sum(1 for r in satirlar if r["sector"])
            print(f"{i}/{len(semboller)} islendi, dolu: {dolu}", flush=True)
        time.sleep(0.5)

    df = pd.DataFrame(satirlar)
    df.to_csv("sektor.csv", index=False)
    dolu = df["sector"].notna().sum()
    print("=" * 80)
    print(f"sektor dolu: {dolu}/{len(df)}")
    print("SEKTOR DAGILIMI:")
    print(df["sector"].value_counts(dropna=False).to_string())
    ind = df["industry"].value_counts()
    print(f"\nALT SEKTOR sayisi: {len(ind)}, en az 5 hisseli: {(ind >= 5).sum()}")
    print("EN BUYUK 15 ALT SEKTOR:")
    print(ind.head(15).to_string())
    print("\nbitti")


if __name__ == "__main__":
    main()
