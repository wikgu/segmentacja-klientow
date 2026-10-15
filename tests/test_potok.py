"""Test całego potoku: od surowych transakcji do tabeli klientów z segmentem (tak jak w aplikacji)."""

import numpy as np
import pandas as pd
import pytest

from analiza.cechy import odfiltruj_klientow, zbuduj_rfm
from analiza.czyszczenie import wyczysc_dane
from analiza.segmentacja import segmentuj, tabela_do_eksportu
from analiza.wczytanie import ujednolic_typy


@pytest.fixture
def surowe_transakcje():
    """Syntetyczne transakcje 60 klientów z trzech grup oraz typowymi problemami jakości."""
    rng = np.random.default_rng(1)
    koniec = pd.Timestamp("2011-12-09 12:00")
    wiersze = []
    numer = 500000
    # (liczba zamówień, ile dni temu ostatni zakup, cena jednostkowa)
    for grupa, (zamowienia, recency, cena) in enumerate([(15, 5, 6.0), (4, 60, 3.0), (1, 300, 2.0)]):
        for klient in range(20):
            customer_id = 12000 + 100 * grupa + klient
            for z in range(zamowienia):
                numer += 1
                data = koniec - pd.Timedelta(days=recency + 7 * z + int(rng.integers(0, 3)))
                for produkt in ("22423", "85123A"):
                    wiersze.append((str(numer), produkt, "PRODUKT", int(rng.integers(2, 12)), data, cena,
                                    customer_id, "United Kingdom"))
    df = pd.DataFrame(wiersze, columns=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
                                        "UnitPrice", "CustomerID", "Country"])
    problemy = pd.DataFrame(
        [
            ("C599001", "22423", "PRODUKT", -2, koniec, 6.0, 12000, "United Kingdom"),  # zwrot
            ("599002", "POST", "POSTAGE", 1, koniec, 18.0, 12001, "United Kingdom"),  # opłata
            ("599003", "22423", "PRODUKT", 5, koniec, 6.0, None, "United Kingdom"),  # brak klienta
            ("599004", "22423", "PRODUKT", 5, koniec, 0.0, 12002, "United Kingdom"),  # cena 0
        ],
        columns=df.columns,
    )
    surowe = pd.concat([df, df.head(3), problemy], ignore_index=True)  # 3 duplikaty
    return ujednolic_typy(surowe)


def test_potok_od_surowych_danych_do_segmentow(surowe_transakcje):
    dane, dziennik = wyczysc_dane(surowe_transakcje)
    klienci, _ = odfiltruj_klientow(zbuduj_rfm(dane))
    wynik = segmentuj(klienci, k=3)
    eksport = tabela_do_eksportu(wynik.klienci, {0: "A", 1: "B", 2: "C"})

    assert dziennik.set_index("krok").loc["pełne duplikaty", "liczba_obserwacji"] == 3
    assert len(klienci) == 60
    assert len(eksport) == len(klienci) == eksport["CustomerID"].nunique()
    assert set(eksport["CustomerID"]) <= set(dane["CustomerID"])
    # trzy grupy o różnej aktywności powinny trafić do trzech różnych segmentów
    grupa = (eksport["CustomerID"] - 12000) // 100
    assert pd.crosstab(grupa, eksport["Segment"]).gt(0).sum(axis=1).eq(1).all()
    # najaktywniejsza grupa ma segment 0 (najwyższa mediana Monetary)
    assert (eksport.loc[grupa == 0, "Segment"] == 0).all()
