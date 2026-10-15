"""Testy zestawień do analizy eksploracyjnej z modułu ``analiza.eda``."""

import pandas as pd
import pytest

from analiza.eda import (
    aktywnosc_tygodniowa,
    podsumowanie_faktur,
    sprzedaz_miesieczna,
    sprzedaz_wg_kraju,
    statystyki_opisowe,
    udzial_anulowan,
)


@pytest.fixture
def czyste():
    """Pozycje faktur po czyszczeniu (z kolumnami Anulowanie i Wartosc)."""
    df = pd.DataFrame(
        {
            "InvoiceNo": ["1001", "1001", "1002", "C1003", "1004"],
            "StockCode": ["A", "B", "A", "A", "C"],
            "Quantity": [2, 1, 4, -1, 10],
            "InvoiceDate": pd.to_datetime(
                ["2011-01-03 10:00", "2011-01-03 10:00", "2011-01-20 12:00", "2011-02-01 09:00", "2011-02-06 15:00"]
            ),
            "UnitPrice": [5.0, 10.0, 5.0, 5.0, 1.0],
            "CustomerID": pd.array([1, 1, 2, 1, 3], dtype="Int64"),
            "Country": ["United Kingdom", "United Kingdom", "France", "United Kingdom", "France"],
        }
    )
    df["Anulowanie"] = df["InvoiceNo"].str.startswith("C")
    df["Wartosc"] = df["Quantity"] * df["UnitPrice"]
    return df


def test_statystyki_opisowe_zgodne_z_pandas(czyste):
    opis = statystyki_opisowe(czyste, ["Quantity", "Wartosc"])
    assert opis.loc["Quantity", "liczebnosc"] == 5
    assert opis.loc["Quantity", "mediana"] == czyste["Quantity"].median()
    assert opis.loc["Wartosc", "Q3"] == czyste["Wartosc"].quantile(0.75)
    assert opis.loc["Wartosc", "odch_std"] == pytest.approx(czyste["Wartosc"].std())


def test_podsumowanie_faktur_pomija_anulowania(czyste):
    faktury = podsumowanie_faktur(czyste)
    assert faktury.index.tolist() == ["1001", "1002", "1004"]
    assert faktury.loc["1001", "pozycje"] == 2
    assert faktury.loc["1001", "wartosc"] == 20.0


def test_sprzedaz_miesieczna_netto_to_brutto_minus_anulowania(czyste):
    miesiace = sprzedaz_miesieczna(czyste)
    assert miesiace.loc[pd.Period("2011-01", "M"), "sprzedaz_brutto"] == 40.0
    assert miesiace.loc[pd.Period("2011-02", "M"), "anulowania"] == 5.0
    assert miesiace.loc[pd.Period("2011-02", "M"), "sprzedaz_netto"] == 5.0
    assert miesiace["sprzedaz_netto"].sum() == pytest.approx(czyste["Wartosc"].sum())
    assert miesiace.loc[pd.Period("2011-01", "M"), "faktury"] == 2


def test_sprzedaz_wg_kraju_sumuje_udzialy(czyste):
    kraje = sprzedaz_wg_kraju(czyste)
    assert kraje.index[0] == "France"
    assert kraje["udzial_proc"].sum() == pytest.approx(100, abs=0.01)
    assert kraje.loc["United Kingdom", "klienci"] == 1


def test_aktywnosc_tygodniowa_liczy_kazda_fakture_raz(czyste):
    tabela = aktywnosc_tygodniowa(czyste)
    assert tabela.to_numpy().sum() == 3  # trzy faktury sprzedażowe
    assert tabela.loc["pon", 10] == 1  # 2011-01-03 to poniedziałek
    assert tabela.loc["nd", 15] == 1  # 2011-02-06 to niedziela


def test_udzial_anulowan(czyste):
    udzial = udzial_anulowan(czyste)
    assert udzial["faktury_proc"] == 25.0
    assert udzial["wartosc_proc"] == pytest.approx(100 * 5 / 50)  # 5 GBP anulowań wobec 50 GBP sprzedaży
