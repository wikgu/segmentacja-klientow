"""Testy funkcji czyszczących z modułu ``analiza.czyszczenie``."""

import pandas as pd
import pytest

from analiza.czyszczenie import (
    KROKI_CZYSZCZENIA,
    podsumowanie_problemow,
    porownanie_przed_po,
    raport_jakosci,
    usun_brak_klienta,
    usun_duplikaty,
    usun_pozycje_nietowarowe,
    wartosci_skrajne,
    wyczysc_dane,
)


@pytest.fixture
def transakcje():
    """Dziesięć pozycji faktur, z których każda ilustruje inny przypadek do czyszczenia."""
    wiersze = [
        # InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
        ("536365", "85123A", "WHITE HANGING HEART", 6, "2010-12-01 08:26", 2.55, 17850, "United Kingdom"),
        ("536365", "85123A", "WHITE HANGING HEART", 6, "2010-12-01 08:26", 2.55, 17850, "United Kingdom"),  # duplikat
        ("536366", "22633", "HAND WARMER", 6, "2010-12-01 08:28", 1.85, 17850, "United Kingdom"),
        ("C536379", "85123A", "WHITE HANGING HEART", -2, "2010-12-02 09:41", 2.55, 17850, "United Kingdom"),  # anulowanie
        ("536370", "22728", "ALARM CLOCK", 24, "2010-12-01 08:45", 3.75, None, "France"),  # brak klienta
        ("536380", "22961", "JAM MAKING SET", -10, "2010-12-03 10:00", 1.45, 12583, "France"),  # korekta bez C
        ("536381", "22139", "RETROSPOT TEA SET", 3, "2010-12-03 11:00", 0.0, 12583, "France"),  # cena 0
        ("536382", "POST", "POSTAGE", 1, "2010-12-03 11:05", 18.0, 12583, "France"),  # opłata pocztowa
        ("536383", "m", "Manual", 1, "2010-12-03 11:10", 5.0, 12583, "France"),  # korekta ręczna
        ("536384", "21730", "GLASS STAR LANTERN", 4, "2010-12-04 12:00", 4.25, 12583, "France"),
    ]
    df = pd.DataFrame(
        wiersze,
        columns=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice",
                 "CustomerID", "Country"],
    )
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = pd.array(df["CustomerID"], dtype="Int64")
    return df


@pytest.mark.parametrize("krok", KROKI_CZYSZCZENIA, ids=lambda f: f.__name__)
def test_reguly_nie_modyfikuja_wejscia(transakcje, krok):
    kopia = transakcje.copy()
    krok(transakcje)
    pd.testing.assert_frame_equal(transakcje, kopia)


def test_potok_nie_modyfikuje_wejscia(transakcje):
    kopia = transakcje.copy()
    wyczysc_dane(transakcje)
    pd.testing.assert_frame_equal(transakcje, kopia)


def test_usun_duplikaty_zostawia_pierwsze_wystapienie(transakcje):
    wynik, wpis = usun_duplikaty(transakcje)
    assert len(wynik) == 9
    assert wpis["liczba_obserwacji"] == 1
    assert not wynik.duplicated().any()


def test_usun_brak_klienta_usuwa_tylko_braki(transakcje):
    wynik, wpis = usun_brak_klienta(transakcje)
    assert wynik["CustomerID"].notna().all()
    assert wpis["liczba_obserwacji"] == transakcje["CustomerID"].isna().sum() == 1


def test_pozycje_nietowarowe_bez_wzgledu_na_wielkosc_liter(transakcje):
    wynik, wpis = usun_pozycje_nietowarowe(transakcje)
    assert wpis["liczba_obserwacji"] == 2
    assert not wynik["StockCode"].str.upper().isin(["POST", "M"]).any()


def test_potok_zostawia_poprawne_pozycje_i_anulowania(transakcje):
    czyste, _ = wyczysc_dane(transakcje)

    assert czyste["InvoiceNo"].tolist() == ["536365", "536366", "C536379", "536384"]
    assert czyste["Anulowanie"].tolist() == [False, False, True, False]
    assert czyste.loc[czyste["Anulowanie"], "Wartosc"].item() == pytest.approx(-5.10)
    assert (czyste["UnitPrice"] > 0).all()


def test_dziennik_decyzji_zgadza_sie_z_liczbami_wierszy(transakcje):
    czyste, dziennik = wyczysc_dane(transakcje)

    assert dziennik["wiersze_przed"].iloc[0] == len(transakcje)
    assert dziennik["wiersze_po"].iloc[-1] == len(czyste)
    # każdy krok zaczyna od liczby wierszy, na której skończył poprzedni
    assert (dziennik["wiersze_przed"].iloc[1:].to_numpy() == dziennik["wiersze_po"].iloc[:-1].to_numpy()).all()
    usuniecia = dziennik[dziennik["dzialanie"] == "usunięcie"]
    assert (usuniecia["liczba_obserwacji"] == usuniecia["wiersze_przed"] - usuniecia["wiersze_po"]).all()
    assert dziennik.set_index("krok")["liczba_obserwacji"].to_dict() == {
        "pełne duplikaty": 1,
        "brak CustomerID": 1,
        "ilość <= 0 poza fakturami C": 1,
        "cena <= 0": 1,
        "pozycje nietowarowe": 2,
        "oznaczenie anulowań": 1,
        "wartość pozycji": 4,
    }


def test_podsumowanie_problemow_przed_i_po(transakcje):
    czyste, _ = wyczysc_dane(transakcje)
    przed = podsumowanie_problemow(transakcje)
    porownanie = porownanie_przed_po(transakcje, czyste)

    assert przed["pelne duplikaty"] == 1
    assert przed["brak CustomerID"] == 1
    assert przed["Quantity < 0 poza fakturami C"] == 1
    assert przed["UnitPrice = 0"] == 1
    assert przed["pozycje nietowarowe"] == 2
    assert porownanie.loc["pelne duplikaty", "po"] == 0
    assert porownanie.loc["wiersze", "po"] == len(czyste)


def test_raport_jakosci_i_wartosci_skrajne(transakcje):
    raport = raport_jakosci(transakcje)
    assert raport.loc["CustomerID", "braki"] == 1
    assert raport.loc["Quantity", "min"] == -10

    skrajne = wartosci_skrajne(transakcje, ["Quantity"])
    assert skrajne.loc["Quantity", "powyzej"] == 1  # 24 sztuki
    assert skrajne.loc["Quantity", "ponizej"] == 1  # -10 sztuk
