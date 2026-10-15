"""Testy budowy cech klientów z modułu ``analiza.cechy``."""

import numpy as np
import pandas as pd
import pytest

from analiza.cechy import (
    CECHY_RFM,
    kontrola_agregacji,
    odfiltruj_klientow,
    przeksztalc_cechy,
    przygotuj_macierz_cech,
    skaluj_cechy,
    zbuduj_rfm,
)


@pytest.fixture
def czyste():
    """Pozycje faktur po czyszczeniu dla czterech klientów.

    * klient 1 – dwie faktury (pierwsza z dwiema pozycjami) i częściowy zwrot,
    * klient 2 – jedna faktura z trzema pozycjami, ostatni dzień w danych,
    * klient 3 – wyłącznie faktura anulująca,
    * klient 4 – zamówienie anulowane w całości.
    """
    wiersze = [
        ("1001", 1, 2, "2011-01-03 10:00", 5.00, "United Kingdom"),
        ("1001", 1, 1, "2011-01-03 10:00", 10.00, "United Kingdom"),
        ("1002", 1, 4, "2011-03-10 12:00", 2.50, "United Kingdom"),
        ("C1003", 1, -1, "2011-03-15 09:00", 5.00, "United Kingdom"),
        ("1004", 2, 1, "2011-12-08 15:00", 3.00, "France"),
        ("1004", 2, 2, "2011-12-08 15:00", 4.00, "France"),
        ("1004", 2, 3, "2011-12-08 15:00", 1.00, "France"),
        ("C1005", 3, -2, "2011-06-01 11:00", 7.50, "Germany"),
        ("1006", 4, 10, "2011-07-01 10:00", 0.10, "United Kingdom"),
        ("C1007", 4, -10, "2011-07-01 10:20", 0.10, "United Kingdom"),
    ]
    df = pd.DataFrame(wiersze, columns=["InvoiceNo", "CustomerID", "Quantity", "InvoiceDate", "UnitPrice", "Country"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = pd.array(df["CustomerID"], dtype="Int64")
    df["Anulowanie"] = df["InvoiceNo"].str.startswith("C")
    df["Wartosc"] = df["Quantity"] * df["UnitPrice"]
    return df


def test_jeden_wiersz_na_klienta(czyste):
    rfm = zbuduj_rfm(czyste)
    assert rfm.index.is_unique
    assert len(rfm) == czyste["CustomerID"].nunique() == 4


def test_rfm_zgodne_z_obliczeniami_recznymi(czyste):
    rfm = zbuduj_rfm(czyste)
    # data odniesienia: dzień po ostatniej transakcji, czyli 2011-12-09
    assert rfm.loc[1, "Frequency"] == 2
    assert rfm.loc[1, "Monetary"] == pytest.approx(10 + 10 + 10 - 5)
    assert rfm.loc[1, "Recency"] == (pd.Timestamp("2011-12-09") - pd.Timestamp("2011-03-10")).days
    assert rfm.loc[2, "Recency"] == 1
    assert rfm.loc[2, "Monetary"] == pytest.approx(3 + 8 + 3)


def test_frequency_liczy_faktury_a_nie_pozycje(czyste):
    rfm = zbuduj_rfm(czyste)
    assert rfm.loc[2, "Frequency"] == 1  # trzy pozycje na jednej fakturze


def test_monetary_jest_wartoscia_netto_po_zwrotach(czyste):
    rfm = zbuduj_rfm(czyste)
    assert rfm.loc[1, "Wartosc_brutto"] == pytest.approx(30.0)
    assert rfm.loc[1, "Wartosc_zwrotow"] == pytest.approx(5.0)
    assert rfm.loc[1, "Monetary"] == pytest.approx(rfm.loc[1, "Wartosc_brutto"] - rfm.loc[1, "Wartosc_zwrotow"])
    assert rfm.loc[4, "Monetary"] == 0.0  # zakup i pełne anulowanie znoszą się dokładnie


def test_agregacja_nie_liczy_podwojnie(czyste):
    rfm = zbuduj_rfm(czyste)
    kontrola = kontrola_agregacji(czyste, rfm)
    assert kontrola["suma Monetary"] == pytest.approx(kontrola["wartość netto transakcji"])
    assert kontrola["suma Frequency"] == kontrola["faktury sprzedażowe"]
    assert kontrola["wiersze tabeli cech"] == kontrola["klienci w transakcjach"]


def test_zbuduj_rfm_nie_modyfikuje_wejscia(czyste):
    kopia = czyste.copy()
    zbuduj_rfm(czyste)
    pd.testing.assert_frame_equal(czyste, kopia)


def test_odfiltruj_klientow_bez_zakupu_netto(czyste):
    rfm = zbuduj_rfm(czyste)
    wynik, wpis = odfiltruj_klientow(rfm)
    assert wynik.index.tolist() == [1, 2]
    assert wpis["liczba_obserwacji"] == 2
    assert wpis["wiersze_przed"] - wpis["wiersze_po"] == wpis["liczba_obserwacji"]
    assert (wynik["Monetary"] > 0).all()


def test_przeksztalc_cechy_logarytmuje_tylko_wybrane(czyste):
    rfm, _ = odfiltruj_klientow(zbuduj_rfm(czyste))
    X = przeksztalc_cechy(rfm, log=["Frequency", "Monetary"])
    assert X["Recency"].tolist() == rfm["Recency"].astype(float).tolist()
    assert X["Monetary"].tolist() == pytest.approx(np.log1p(rfm["Monetary"]).tolist())
    assert przeksztalc_cechy(rfm, log=False).equals(rfm[CECHY_RFM].astype(float))


def test_winsoryzacja_przycina_zamiast_usuwac():
    rfm = pd.DataFrame({"Recency": range(1, 101), "Frequency": [1] * 99 + [500], "Monetary": [10.0] * 100})
    X = przeksztalc_cechy(rfm, log=False, winsoryzacja=(0.0, 0.95))
    assert len(X) == len(rfm)
    assert X["Frequency"].max() < 500
    assert X["Recency"].max() == pytest.approx(rfm["Recency"].quantile(0.95))


def test_skalowanie_standaryzuje_i_dopasowuje_sie_tylko_do_danych(czyste):
    rfm, _ = odfiltruj_klientow(zbuduj_rfm(czyste))
    X = przeksztalc_cechy(rfm)
    X_skal, skaler = skaluj_cechy(X)
    assert np.allclose(X_skal.mean(), 0)
    assert np.allclose(X_skal.std(ddof=0), 1)
    assert np.allclose(skaler.mean_, X.mean())
    assert X_skal.index.equals(rfm.index)


def test_przygotuj_macierz_cech_nie_modyfikuje_wejscia(czyste):
    rfm, _ = odfiltruj_klientow(zbuduj_rfm(czyste))
    kopia = rfm.copy()
    przygotuj_macierz_cech(rfm)
    pd.testing.assert_frame_equal(rfm, kopia)
