"""Budowa tabeli cech klientów (RFM), przekształcenia i skalowanie.

Definicje cech (jeden wiersz = jeden klient):

* **Recency** – liczba dni od dnia ostatniego zakupu do daty odniesienia
  (dzień po ostatniej transakcji w zbiorze),
* **Frequency** – liczba unikalnych faktur sprzedażowych (bez faktur anulujących),
  więc kilka pozycji na jednej fakturze to jedno zamówienie,
* **Monetary** – łączna wartość zakupów netto w GBP: sprzedaż pomniejszona o anulowania i zwroty.

Przed segmentacją Frequency i Monetary są logarytmowane (``log1p``), bo mają bardzo silną
prawostronną skośność; Recency zostaje w dniach. Następnie wszystkie cechy są standaryzowane.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

CECHY_RFM = ["Recency", "Frequency", "Monetary"]
CECHY_LOG = ("Frequency", "Monetary")


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


def odfiltruj_klientow(rfm: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Wyłącza z segmentacji klientów bez zakupu netto (Frequency = 0 lub Monetary <= 0)."""
    maska = (rfm["Frequency"] > 0) & (rfm["Monetary"] > 0)
    wynik = rfm[maska].copy()
    wynik["Recency"] = wynik["Recency"].astype(int)
    wpis = {
        "krok": "klienci bez zakupu netto",
        "dzialanie": "wyłączenie z segmentacji",
        "uzasadnienie": "Klienci, których anulowania równoważą lub przewyższają zakupy (albo mają tylko "
                        "anulowania), nie mają dodatniej wartości zakupów do opisania.",
        "wiersze_przed": len(rfm),
        "wiersze_po": len(wynik),
        "liczba_obserwacji": int((~maska).sum()),
    }
    return wynik, wpis


def przeksztalc_cechy(
    rfm: pd.DataFrame,
    cechy=CECHY_RFM,
    log: bool | tuple | list = CECHY_LOG,
    winsoryzacja: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Zwraca kopię wybranych cech po opcjonalnym obcięciu kwantylowym i transformacji log1p.

    ``log`` to lista cech do zlogarytmowania (``True`` – wszystkie, ``False`` – żadna).
    ``winsoryzacja`` to para kwantyli, np. ``(0.01, 0.99)``; wartości spoza przedziału
    są przycinane do granic, a nie usuwane. Żadna obserwacja nie jest usuwana.
    """
    X = rfm.loc[:, list(cechy)].astype(float).copy()
    if winsoryzacja is not None:
        dolny, gorny = winsoryzacja
        X = X.clip(lower=X.quantile(dolny), upper=X.quantile(gorny), axis=1)
    do_log = list(cechy) if log is True else (list(log) if log else [])
    for cecha in do_log:
        if cecha in X.columns:
            X[cecha] = np.log1p(X[cecha])
    return X


def skaluj_cechy(X: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Standaryzuje cechy (średnia 0, odchylenie 1); skaler jest dopasowany tylko do ``X``.

    K-means liczy odległości euklidesowe, więc bez skalowania o podziale decydowałaby cecha
    o największym zakresie liczbowym (np. Recency w dniach), a nie wszystkie cechy po równo.
    """
    skaler = StandardScaler()
    wartosci = skaler.fit_transform(X)
    return pd.DataFrame(wartosci, index=X.index, columns=X.columns), skaler


def przygotuj_macierz_cech(
    rfm: pd.DataFrame,
    cechy=CECHY_RFM,
    log: bool | tuple | list = CECHY_LOG,
    winsoryzacja: tuple[float, float] | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Przekształca i skaluje cechy klientów przeznaczonych do segmentacji."""
    return skaluj_cechy(przeksztalc_cechy(rfm, cechy=cechy, log=log, winsoryzacja=winsoryzacja))
