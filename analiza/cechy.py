"""Budowa tabeli cech klientów (RFM), przekształcenia i skalowanie.

Definicje cech (jeden wiersz = jeden klient):

* **Recency** – liczba dni od dnia ostatniego zakupu do daty odniesienia
  (dzień po ostatniej transakcji w zbiorze),
* **Frequency** – liczba unikalnych faktur sprzedażowych (bez faktur anulujących),
  więc kilka pozycji na jednej fakturze to jedno zamówienie,
* **Monetary** – łączna wartość zakupów netto w GBP: sprzedaż pomniejszona o anulowania i zwroty.
"""

from __future__ import annotations

import pandas as pd

CECHY_RFM = ["Recency", "Frequency", "Monetary"]


def data_odniesienia(df: pd.DataFrame) -> pd.Timestamp:
    """Dzień następujący po ostatniej transakcji w zbiorze (bez części godzinowej)."""
    return df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)


def zbuduj_rfm(df: pd.DataFrame, data_ref: pd.Timestamp | None = None) -> pd.DataFrame:
    """Agreguje oczyszczone pozycje faktur do poziomu klienta.

    Wymaga kolumn ``Anulowanie`` i ``Wartosc`` (dodawanych w ``analiza.czyszczenie``).
    Oprócz R, F, M zwraca kolumny pomocnicze używane w profilowaniu i eksporcie.
    """
    data_ref = data_odniesienia(df) if data_ref is None else pd.Timestamp(data_ref)
    sprzedaz = df[~df["Anulowanie"]]
    anulowania = df[df["Anulowanie"]]

    zakupy = sprzedaz.groupby("CustomerID").agg(
        Ostatni_zakup=("InvoiceDate", "max"),
        Pierwszy_zakup=("InvoiceDate", "min"),
        Frequency=("InvoiceNo", "nunique"),
        Wartosc_brutto=("Wartosc", "sum"),
    )
    zwroty = anulowania.groupby("CustomerID")["Wartosc"].sum().abs().rename("Wartosc_zwrotow")
    netto = df.groupby("CustomerID")["Wartosc"].sum().rename("Monetary")
    # Kraj, z którego pochodzi najwięcej pozycji klienta (kilku klientów ma dwa kraje).
    kraj = (
        df.groupby(["CustomerID", "Country"]).size().rename("n").reset_index()
        .sort_values(["CustomerID", "n"], ascending=[True, False])
        .drop_duplicates("CustomerID")
        .set_index("CustomerID")["Country"]
        .rename("Kraj")
    )

    rfm = pd.concat([zakupy, zwroty, netto, kraj], axis=1)
    rfm["Frequency"] = rfm["Frequency"].fillna(0).astype(int)
    # Zaokrąglenie do pensów usuwa resztki z dodawania liczb zmiennoprzecinkowych
    # (np. zakup i jego pełne anulowanie dają dokładnie 0, a nie 1e-11).
    for kolumna in ["Wartosc_brutto", "Wartosc_zwrotow", "Monetary"]:
        rfm[kolumna] = rfm[kolumna].fillna(0.0).round(2)
    rfm["Recency"] = (data_ref - rfm["Ostatni_zakup"].dt.normalize()).dt.days
    rfm.index.name = "CustomerID"
    kolumny = CECHY_RFM + ["Wartosc_brutto", "Wartosc_zwrotow", "Pierwszy_zakup", "Ostatni_zakup", "Kraj"]
    return rfm[kolumny].sort_index()


def kontrola_agregacji(df: pd.DataFrame, rfm: pd.DataFrame) -> pd.Series:
    """Porównuje sumy w tabeli klientów z danymi transakcyjnymi (wykrywa podwójne liczenie)."""
    sprzedaz = df[~df["Anulowanie"]]
    return pd.Series(
        {
            "klienci w transakcjach": df["CustomerID"].nunique(),
            "wiersze tabeli cech": len(rfm),
            "faktury sprzedażowe": sprzedaz["InvoiceNo"].nunique(),
            "suma Frequency": int(rfm["Frequency"].sum()),
            "wartość netto transakcji": round(float(df["Wartosc"].sum()), 2),
            "suma Monetary": round(float(rfm["Monetary"].sum()), 2),
        },
        name="wartość",
    )
