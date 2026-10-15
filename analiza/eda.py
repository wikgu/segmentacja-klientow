"""Zestawienia do analizy eksploracyjnej (EDA) danych transakcyjnych i cech klientów."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analiza.cechy import CECHY_RFM

DNI_TYGODNIA = ["pon", "wt", "śr", "czw", "pt", "sob", "nd"]


def statystyki_opisowe(df: pd.DataFrame, kolumny) -> pd.DataFrame:
    """Liczebność, średnia, odchylenie standardowe, kwartyle i skośność wskazanych cech."""
    dane = df.loc[:, list(kolumny)]
    opis = pd.DataFrame(
        {
            "liczebnosc": dane.count(),
            "srednia": dane.mean(),
            "odch_std": dane.std(),
            "min": dane.min(),
            "Q1": dane.quantile(0.25),
            "mediana": dane.median(),
            "Q3": dane.quantile(0.75),
            "max": dane.max(),
            "skosnosc": dane.skew(),
        }
    )
    return opis


def podsumowanie_faktur(df: pd.DataFrame) -> pd.DataFrame:
    """Faktury sprzedażowe: liczba pozycji, liczba sztuk i wartość koszyka."""
    return (
        df[~df["Anulowanie"]]
        .groupby("InvoiceNo")
        .agg(pozycje=("StockCode", "size"), sztuki=("Quantity", "sum"), wartosc=("Wartosc", "sum"))
    )


def sprzedaz_miesieczna(df: pd.DataFrame) -> pd.DataFrame:
    """Wartość sprzedaży brutto, anulowań i netto oraz liczba faktur i klientów w miesiącach."""
    dane = df.assign(miesiac=df["InvoiceDate"].dt.to_period("M"))
    sprzedaz = dane[~dane["Anulowanie"]].groupby("miesiac")
    anulowania = dane[dane["Anulowanie"]].groupby("miesiac")
    wynik = pd.DataFrame(
        {
            "sprzedaz_brutto": sprzedaz["Wartosc"].sum(),
            "anulowania": -anulowania["Wartosc"].sum(),
            "faktury": sprzedaz["InvoiceNo"].nunique(),
            "aktywni_klienci": sprzedaz["CustomerID"].nunique(),
        }
    ).fillna(0)
    wynik["sprzedaz_netto"] = wynik["sprzedaz_brutto"] - wynik["anulowania"]
    wynik.index.name = "miesiac"
    return wynik


def sprzedaz_wg_kraju(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Kraje o największej sprzedaży netto: wartość, udział i liczba klientów."""
    grupy = df.groupby("Country")
    wynik = pd.DataFrame(
        {
            "sprzedaz_netto": grupy["Wartosc"].sum(),
            "klienci": grupy["CustomerID"].nunique(),
        }
    ).sort_values("sprzedaz_netto", ascending=False)
    wynik["udzial_proc"] = (100 * wynik["sprzedaz_netto"] / wynik["sprzedaz_netto"].sum()).round(2)
    return wynik.head(n)


def aktywnosc_tygodniowa(df: pd.DataFrame) -> pd.DataFrame:
    """Liczba faktur sprzedażowych w podziale na dzień tygodnia (wiersze) i godzinę (kolumny)."""
    faktury = df[~df["Anulowanie"]].drop_duplicates("InvoiceNo")
    tabela = pd.crosstab(faktury["InvoiceDate"].dt.dayofweek, faktury["InvoiceDate"].dt.hour)
    tabela = tabela.reindex(range(7), fill_value=0)
    tabela.index = DNI_TYGODNIA
    return tabela


def udzial_anulowan(df: pd.DataFrame) -> pd.Series:
    """Skala anulowań: odsetek faktur, pozycji i wartości sprzedaży, którą korygują."""
    anul = df["Anulowanie"]
    return pd.Series(
        {
            "faktury_proc": 100 * df.loc[anul, "InvoiceNo"].nunique() / df["InvoiceNo"].nunique(),
            "pozycje_proc": 100 * anul.mean(),
            "wartosc_proc": 100 * -df.loc[anul, "Wartosc"].sum() / df.loc[~anul, "Wartosc"].sum(),
            "klienci_z_anulowaniem_proc": 100 * df.loc[anul, "CustomerID"].nunique() / df["CustomerID"].nunique(),
        }
    ).round(2)


def koncentracja_wartosci(rfm: pd.DataFrame, kolumna: str = "Monetary") -> pd.DataFrame:
    """Krzywa koncentracji: jaki odsetek wartości generuje dany odsetek najlepszych klientów."""
    wartosci = np.sort(rfm[kolumna].clip(lower=0).to_numpy())[::-1]
    skumulowane = np.cumsum(wartosci) / wartosci.sum()
    return pd.DataFrame(
        {
            "odsetek_klientow": np.arange(1, len(wartosci) + 1) / len(wartosci),
            "odsetek_wartosci": skumulowane,
        }
    )


def udzial_najlepszych(rfm: pd.DataFrame, odsetek: float = 0.2, kolumna: str = "Monetary") -> float:
    """Udział w łącznej wartości ``odsetek`` klientów o najwyższej wartości cechy."""
    krzywa = koncentracja_wartosci(rfm, kolumna)
    return float(np.interp(odsetek, krzywa["odsetek_klientow"], krzywa["odsetek_wartosci"]))


def korelacje(rfm: pd.DataFrame, cechy=CECHY_RFM, metoda: str = "spearman") -> pd.DataFrame:
    """Macierz korelacji cech klientów (domyślnie Spearmana – odporna na skośność)."""
    return rfm.loc[:, list(cechy)].corr(method=metoda)
