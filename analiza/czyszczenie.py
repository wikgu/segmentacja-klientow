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


# --------------------------------------------------------------------------- reguły czyszczenia


def _wpis(krok: str, dzialanie: str, uzasadnienie: str, przed: pd.DataFrame, po: pd.DataFrame,
          liczba_obserwacji: int | None = None) -> dict:
    """Pojedynczy wpis dziennika decyzji; domyślnie liczba obserwacji = liczba usuniętych wierszy."""
    if liczba_obserwacji is None:
        liczba_obserwacji = len(przed) - len(po)
    return {
        "krok": krok,
        "dzialanie": dzialanie,
        "uzasadnienie": uzasadnienie,
        "wiersze_przed": len(przed),
        "wiersze_po": len(po),
        "liczba_obserwacji": int(liczba_obserwacji),
    }


def usun_duplikaty(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Usuwa pełne duplikaty wierszy (zostawia pierwsze wystąpienie)."""
    wynik = df.drop_duplicates(keep="first").copy()
    return wynik, _wpis(
        "pełne duplikaty",
        "usunięcie",
        "Identyczna faktura, produkt, ilość, cena i czas co do minuty – traktujemy jako podwójny zapis tej samej pozycji.",
        df,
        wynik,
    )


def usun_brak_klienta(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Usuwa wiersze bez identyfikatora klienta."""
    wynik = df[df["CustomerID"].notna()].copy()
    return wynik, _wpis(
        "brak CustomerID",
        "usunięcie",
        "Bez identyfikatora nie da się przypisać transakcji do klienta, a jednostką analizy jest klient.",
        df,
        wynik,
    )


def usun_ujemne_ilosci_poza_anulowaniami(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Usuwa pozycje z ilością <= 0, które nie pochodzą z faktur anulujących (korekty magazynowe)."""
    maska = (df["Quantity"] <= 0) & ~czy_anulowanie(df)
    wynik = df[~maska].copy()
    return wynik, _wpis(
        "ilość <= 0 poza fakturami C",
        "usunięcie",
        "Ujemne ilości bez faktury anulującej to korekty stanów magazynowych (np. uszkodzenia), a nie zachowanie klienta.",
        df,
        wynik,
    )


def usun_niedodatnie_ceny(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Usuwa pozycje z ceną jednostkową <= 0."""
    wynik = df[df["UnitPrice"] > 0].copy()
    return wynik, _wpis(
        "cena <= 0",
        "usunięcie",
        "Pozycje bez dodatniej ceny (gratisy, korekty długu) nie wnoszą wartości zakupu i zawyżałyby liczbę zamówień.",
        df,
        wynik,
    )


def usun_pozycje_nietowarowe(df: pd.DataFrame, kody: dict | set | None = None) -> tuple[pd.DataFrame, dict]:
    """Usuwa pozycje o kodach nietowarowych (opłaty, wysyłka, rabaty, korekty ręczne)."""
    maska = czy_pozycja_nietowarowa(df, kody)
    wynik = df[~maska].copy()
    return wynik, _wpis(
        "pozycje nietowarowe",
        "usunięcie",
        "Opłaty pocztowe, bankowe, rabaty i korekty ręczne nie opisują zakupów produktów; korekty ręczne mają skrajne kwoty.",
        df,
        wynik,
    )


def oznacz_anulowania(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Dodaje kolumnę ``Anulowanie`` (faktura z prefiksem 'C'); niczego nie usuwa."""
    wynik = df.copy()
    wynik["Anulowanie"] = czy_anulowanie(wynik)
    return wynik, _wpis(
        "oznaczenie anulowań",
        "oznaczenie",
        "Anulowania i zwroty zostają w danych: pomniejszają wartość zakupów klienta (Monetary netto), ale nie są liczone jako zamówienia.",
        df,
        wynik,
        liczba_obserwacji=int(wynik["Anulowanie"].sum()),
    )


def dodaj_wartosc(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Dodaje kolumnę ``Wartosc`` = Quantity * UnitPrice (ujemna dla anulowań)."""
    wynik = df.copy()
    wynik["Wartosc"] = wynik["Quantity"] * wynik["UnitPrice"]
    return wynik, _wpis(
        "wartość pozycji",
        "nowa kolumna",
        "Wartość pozycji w GBP potrzebna do wyznaczenia Monetary.",
        df,
        wynik,
        liczba_obserwacji=len(wynik),
    )


# --------------------------------------------------------------------------- potok czyszczenia


KROKI_CZYSZCZENIA = (
    usun_duplikaty,
    usun_brak_klienta,
    usun_ujemne_ilosci_poza_anulowaniami,
    usun_niedodatnie_ceny,
    usun_pozycje_nietowarowe,
    oznacz_anulowania,
    dodaj_wartosc,
)


def wyczysc_dane(df: pd.DataFrame, kroki=KROKI_CZYSZCZENIA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Uruchamia kolejne reguły czyszczenia i zwraca dane po czyszczeniu oraz dziennik decyzji."""
    wynik = df
    dziennik = []
    for krok in kroki:
        wynik, wpis = krok(wynik)
        dziennik.append(wpis)
    return wynik.reset_index(drop=True), pd.DataFrame(dziennik)
