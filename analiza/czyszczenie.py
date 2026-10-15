"""Kontrola jakości i czyszczenie danych transakcyjnych z dziennikiem decyzji.

Każda reguła czyszcząca jest osobną funkcją, która nie modyfikuje przekazanej ramki
i zwraca parę ``(nowa_ramka, wpis_dziennika)``. Funkcja :func:`wyczysc_dane` składa
reguły w potok i zwraca dziennik decyzji z liczbą obserwacji, których dotyczyła
każda operacja.
"""

from __future__ import annotations

import pandas as pd

WZORZEC_FAKTURY = r"^C?\d{6}$"

# Kody pozycji, które nie są sprzedażą towaru: koszty wysyłki, opłaty, rabaty, korekty ręczne.
KODY_NIETOWAROWE = {
    "POST": "opłata pocztowa",
    "DOT": "opłata pocztowa (dotcom)",
    "C2": "koszt przewozu",
    "M": "korekta ręczna",
    "D": "rabat",
    "S": "próbki",
    "B": "korekta złego długu",
    "BANK CHARGES": "opłata bankowa",
    "CRUK": "prowizja CRUK",
    "AMAZONFEE": "opłata Amazon",
}


def czy_anulowanie(df: pd.DataFrame) -> pd.Series:
    """Maska wierszy z faktur anulujących (numer faktury zaczyna się od 'C')."""
    return df["InvoiceNo"].str.startswith("C", na=False)


def czy_pozycja_nietowarowa(df: pd.DataFrame, kody: dict | set | None = None) -> pd.Series:
    """Maska wierszy z kodami pozycji nietowarowych (wielkość liter bez znaczenia)."""
    kody = KODY_NIETOWAROWE if kody is None else kody
    return df["StockCode"].str.upper().isin({k.upper() for k in kody})


# --------------------------------------------------------------------------- kontrola jakości


def raport_jakosci(df: pd.DataFrame) -> pd.DataFrame:
    """Zestawienie dla każdej kolumny: typ, braki, liczba unikalnych wartości, minimum i maksimum."""
    wiersze = []
    for kolumna in df.columns:
        seria = df[kolumna]
        liczbowa = pd.api.types.is_numeric_dtype(seria) or pd.api.types.is_datetime64_any_dtype(seria)
        wiersze.append(
            {
                "kolumna": kolumna,
                "typ": str(seria.dtype),
                "braki": int(seria.isna().sum()),
                "braki_proc": round(100 * seria.isna().mean(), 2),
                "unikalne": int(seria.nunique()),
                "min": seria.min() if liczbowa else None,
                "max": seria.max() if liczbowa else None,
            }
        )
    return pd.DataFrame(wiersze).set_index("kolumna")


def podsumowanie_problemow(df: pd.DataFrame) -> pd.Series:
    """Liczba wierszy z każdym typem problemu jakościowego (wartości niedozwolone i nietypowe)."""
    anulowanie = czy_anulowanie(df)
    return pd.Series(
        {
            "wiersze": len(df),
            "pelne duplikaty": int(df.duplicated().sum()),
            "brak CustomerID": int(df["CustomerID"].isna().sum()),
            "brak Description": int(df["Description"].isna().sum()),
            "wiersze faktur anulujących (C)": int(anulowanie.sum()),
            "Quantity <= 0": int((df["Quantity"] <= 0).sum()),
            "Quantity < 0 poza fakturami C": int(((df["Quantity"] < 0) & ~anulowanie).sum()),
            "UnitPrice = 0": int((df["UnitPrice"] == 0).sum()),
            "UnitPrice < 0": int((df["UnitPrice"] < 0).sum()),
            "nietypowy format InvoiceNo": int((~df["InvoiceNo"].str.match(WZORZEC_FAKTURY, na=False)).sum()),
            "pozycje nietowarowe": int(czy_pozycja_nietowarowa(df).sum()),
            "kraj 'Unspecified'": int((df["Country"] == "Unspecified").sum()),
            "faktury": int(df["InvoiceNo"].nunique()),
            "klienci": int(df["CustomerID"].nunique()),
        },
        name="liczba",
    )


def porownanie_przed_po(przed: pd.DataFrame, po: pd.DataFrame) -> pd.DataFrame:
    """Zestawienie problemów jakościowych przed i po czyszczeniu."""
    return pd.concat(
        [podsumowanie_problemow(przed).rename("przed"), podsumowanie_problemow(po).rename("po")],
        axis=1,
    )


def wartosci_skrajne(df: pd.DataFrame, kolumny=("Quantity", "UnitPrice"), mnoznik: float = 1.5) -> pd.DataFrame:
    """Liczba obserwacji poza przedziałem [Q1 - m*IQR, Q3 + m*IQR] dla wskazanych kolumn.

    Funkcja tylko raportuje wartości odstające, niczego nie usuwa.
    """
    wiersze = []
    for kolumna in kolumny:
        seria = df[kolumna].dropna()
        q1, q3 = seria.quantile([0.25, 0.75])
        iqr = q3 - q1
        dolna, gorna = q1 - mnoznik * iqr, q3 + mnoznik * iqr
        wiersze.append(
            {
                "cecha": kolumna,
                "Q1": q1,
                "Q3": q3,
                "dolna_granica": dolna,
                "gorna_granica": gorna,
                "ponizej": int((seria < dolna).sum()),
                "powyzej": int((seria > gorna).sum()),
                "odsetek_proc": round(100 * ((seria < dolna) | (seria > gorna)).mean(), 2),
                "max": seria.max(),
            }
        )
    return pd.DataFrame(wiersze).set_index("cecha")
