"""Wczytanie i opis zbioru UCI Online Retail.

Zbiór zawiera wszystkie transakcje brytyjskiego sklepu internetowego z okresu
01.12.2010–09.12.2011. Jeden wiersz to jedna pozycja faktury (produkt na fakturze).
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL_DANYCH = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
URL_OPISU = "https://archive.ics.uci.edu/dataset/352/online+retail"

KATALOG_PROJEKTU = Path(__file__).resolve().parent.parent
KATALOG_SUROWYCH = KATALOG_PROJEKTU / "data" / "raw"
# Kopia zbioru w formacie Parquet trzymana w repozytorium (licencja CC BY 4.0 pozwala ją udostępniać);
# dzięki niej aplikacja nie musi przy starcie pobierać i parsować pliku XLSX.
SCIEZKA_PARQUET = KATALOG_PROJEKTU / "data" / "online_retail.parquet"
NAZWA_PLIKU_XLSX = "Online Retail.xlsx"

KOLUMNY = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

OPIS_KOLUMN = {
    "InvoiceNo": "Numer faktury (6 cyfr); prefiks 'C' oznacza fakturę anulującą (korektę)",
    "StockCode": "Kod produktu (5 cyfr, czasem z literą); kody specjalne, np. POST, M, D, oznaczają pozycje nietowarowe",
    "Description": "Nazwa produktu",
    "Quantity": "Liczba sztuk w pozycji faktury; wartości ujemne to anulowania, zwroty i korekty",
    "InvoiceDate": "Data i godzina wystawienia faktury",
    "UnitPrice": "Cena jednostkowa w funtach szterlingach (GBP)",
    "CustomerID": "Pięciocyfrowy identyfikator klienta",
    "Country": "Kraj zamieszkania klienta",
}


def pobierz_dane(katalog: Path = KATALOG_SUROWYCH, url: str = URL_DANYCH) -> Path:
    """Pobiera archiwum z UCI i rozpakowuje plik XLSX, jeżeli nie ma go jeszcze na dysku.

    Zwraca ścieżkę do pliku ``Online Retail.xlsx``.
    """
    katalog = Path(katalog)
    sciezka_xlsx = katalog / NAZWA_PLIKU_XLSX
    if sciezka_xlsx.exists():
        return sciezka_xlsx

    katalog.mkdir(parents=True, exist_ok=True)
    sciezka_zip = katalog / "online_retail.zip"
    with urllib.request.urlopen(url, timeout=120) as odpowiedz, open(sciezka_zip, "wb") as plik:
        shutil.copyfileobj(odpowiedz, plik)
    with zipfile.ZipFile(sciezka_zip) as archiwum:
        archiwum.extract(NAZWA_PLIKU_XLSX, katalog)
    sciezka_zip.unlink()
    return sciezka_xlsx


def ujednolic_typy(df: pd.DataFrame) -> pd.DataFrame:
    """Nadaje kolumnom docelowe typy; nie modyfikuje przekazanej ramki.

    Numery faktur i kody produktów są tekstami (część z nich zawiera litery),
    identyfikator klienta jest liczbą całkowitą z dopuszczalnym brakiem (``Int64``).
    """
    wynik = df.loc[:, KOLUMNY].copy()
    for kolumna in ["InvoiceNo", "StockCode", "Description", "Country"]:
        wynik[kolumna] = wynik[kolumna].astype("str").str.strip()
    wynik["Quantity"] = wynik["Quantity"].astype("int64")
    wynik["InvoiceDate"] = pd.to_datetime(wynik["InvoiceDate"])
    wynik["UnitPrice"] = wynik["UnitPrice"].astype("float64")
    wynik["CustomerID"] = pd.to_numeric(wynik["CustomerID"]).round().astype("Int64")
    return wynik


def wczytaj_xlsx(katalog: Path = KATALOG_SUROWYCH, pobierz: bool = True) -> pd.DataFrame:
    """Wczytuje oryginalny plik XLSX z UCI (w razie potrzeby najpierw go pobiera)."""
    sciezka_xlsx = Path(katalog) / NAZWA_PLIKU_XLSX
    if not sciezka_xlsx.exists():
        if not pobierz:
            raise FileNotFoundError(f"Brak pliku {sciezka_xlsx}; uruchom pobierz_dane().")
        sciezka_xlsx = pobierz_dane(katalog)
    surowe = pd.read_excel(sciezka_xlsx, engine="calamine", dtype={"InvoiceNo": str, "StockCode": str})
    return ujednolic_typy(surowe)


def wczytaj_dane(sciezka_parquet: Path = SCIEZKA_PARQUET, katalog_surowych: Path = KATALOG_SUROWYCH,
                 pobierz: bool = True) -> pd.DataFrame:
    """Wczytuje surowe transakcje Online Retail.

    Czyta kopię Parquet z katalogu ``data/``; jeżeli jej nie ma, odtwarza ją z oryginalnego
    pliku XLSX (pobieranego z UCI), więc wynik da się w pełni odtworzyć ze źródła.
    """
    sciezka_parquet = Path(sciezka_parquet)
    if sciezka_parquet.exists():
        return ujednolic_typy(pd.read_parquet(sciezka_parquet))
    df = wczytaj_xlsx(katalog_surowych, pobierz=pobierz)
    sciezka_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(sciezka_parquet, index=False)
    return df


def opis_zbioru(df: pd.DataFrame) -> dict:
    """Zwraca podstawowe wskaźniki opisujące zbiór transakcji."""
    return {
        "liczba_wierszy": len(df),
        "liczba_kolumn": df.shape[1],
        "data_od": df["InvoiceDate"].min(),
        "data_do": df["InvoiceDate"].max(),
        "liczba_faktur": df["InvoiceNo"].nunique(),
        "liczba_klientow": df["CustomerID"].nunique(),
        "liczba_produktow": df["StockCode"].nunique(),
        "liczba_krajow": df["Country"].nunique(),
    }


def slownik_kolumn(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela z typem, przykładową wartością i znaczeniem każdej kolumny."""
    wiersze = []
    for kolumna in df.columns:
        niepuste = df[kolumna].dropna()
        wiersze.append(
            {
                "kolumna": kolumna,
                "typ": str(df[kolumna].dtype),
                "przyklad": str(niepuste.iloc[0]) if len(niepuste) else "",
                "znaczenie": OPIS_KOLUMN.get(kolumna, ""),
            }
        )
    return pd.DataFrame(wiersze)
