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
NAZWA_PLIKU_XLSX = "Online Retail.xlsx"
NAZWA_PLIKU_CACHE = "online_retail.parquet"

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


def wczytaj_dane(katalog: Path = KATALOG_SUROWYCH, pobierz: bool = True) -> pd.DataFrame:
    """Wczytuje surowe transakcje Online Retail.

    Przy pierwszym wywołaniu czyta plik XLSX (w razie potrzeby pobierając go z UCI)
    i zapisuje kopię w formacie Parquet; kolejne wywołania czytają już tylko tę kopię.
    """
    katalog = Path(katalog)
    sciezka_cache = katalog / NAZWA_PLIKU_CACHE
    if sciezka_cache.exists():
        return ujednolic_typy(pd.read_parquet(sciezka_cache))

    sciezka_xlsx = katalog / NAZWA_PLIKU_XLSX
    if not sciezka_xlsx.exists():
        if not pobierz:
            raise FileNotFoundError(f"Brak pliku {sciezka_xlsx}; uruchom pobierz_dane().")
        sciezka_xlsx = pobierz_dane(katalog)

    surowe = pd.read_excel(
        sciezka_xlsx,
        engine="calamine",
        dtype={"InvoiceNo": str, "StockCode": str},
    )
    df = ujednolic_typy(surowe)
    df.to_parquet(sciezka_cache, index=False)
    return df
