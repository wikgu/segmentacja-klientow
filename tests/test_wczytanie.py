"""Testy ujednolicania typów i opisu zbioru z modułu ``analiza.wczytanie``."""

import pandas as pd
import pytest

from analiza.wczytanie import KOLUMNY, opis_zbioru, slownik_kolumn, ujednolic_typy


@pytest.fixture
def surowe():
    """Kilka wierszy w postaci, w jakiej przychodzą z pliku XLSX (mieszane typy)."""
    return pd.DataFrame(
        {
            "InvoiceNo": [536365, 536365, "C536379"],
            "StockCode": [85123, "85123A", "D"],
            "Description": [" WHITE HANGING HEART ", None, "Discount"],
            "Quantity": [6, 2, -1],
            "InvoiceDate": ["2010-12-01 08:26", "2010-12-01 08:26", "2010-12-01 09:41"],
            "UnitPrice": [2.55, 3.39, 27.5],
            "CustomerID": [17850.0, None, 14527.0],
            "Country": ["United Kingdom", "United Kingdom", "United Kingdom"],
        }
    )


def test_ujednolic_typy_nadaje_docelowe_typy(surowe):
    wynik = ujednolic_typy(surowe)

    assert list(wynik.columns) == KOLUMNY
    assert wynik["InvoiceNo"].tolist() == ["536365", "536365", "C536379"]
    assert wynik["StockCode"].tolist() == ["85123", "85123A", "D"]
    assert str(wynik["CustomerID"].dtype) == "Int64"
    assert wynik["CustomerID"].isna().sum() == 1
    assert pd.api.types.is_datetime64_any_dtype(wynik["InvoiceDate"])
    assert wynik.loc[0, "Description"] == "WHITE HANGING HEART"
    assert wynik["Description"].isna().sum() == 1


def test_ujednolic_typy_nie_modyfikuje_wejscia(surowe):
    kopia = surowe.copy()
    ujednolic_typy(surowe)
    pd.testing.assert_frame_equal(surowe, kopia)


def test_opis_zbioru_liczy_faktury_i_klientow(surowe):
    opis = opis_zbioru(ujednolic_typy(surowe))

    assert opis["liczba_wierszy"] == 3
    assert opis["liczba_kolumn"] == 8
    assert opis["liczba_faktur"] == 2
    assert opis["liczba_klientow"] == 2
    assert opis["data_od"] == pd.Timestamp("2010-12-01 08:26")
    assert opis["data_do"] == pd.Timestamp("2010-12-01 09:41")


def test_slownik_kolumn_opisuje_kazda_kolumne(surowe):
    slownik = slownik_kolumn(ujednolic_typy(surowe))
    assert slownik["kolumna"].tolist() == KOLUMNY
    assert (slownik["znaczenie"] != "").all()
