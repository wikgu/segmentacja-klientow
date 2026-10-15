"""Testy segmentacji K-means z modułu ``analiza.segmentacja``."""

import numpy as np
import pandas as pd
import pytest

from analiza.cechy import przygotuj_macierz_cech
from analiza.segmentacja import (
    dodatkowy_profil,
    opisz_segmenty,
    porownaj_k,
    porownaj_warianty,
    profiluj_segmenty,
    rzut_pca,
    segmentuj,
    stabilnosc_k,
    tabela_do_eksportu,
    uporzadkuj_etykiety,
)


@pytest.fixture
def klienci():
    """Trzy wyraźne grupy klientów (aktywni i wartościowi, przeciętni, uśpieni) – 90 klientów."""
    rng = np.random.default_rng(0)
    grupy = [
        # (Recency, Frequency, Monetary) – środki grup
        (10, 20, 8000.0),
        (60, 3, 600.0),
        (300, 1, 150.0),
    ]
    wiersze = []
    for r, f, m in grupy:
        for _ in range(30):
            wiersze.append(
                {
                    "Recency": int(max(1, rng.normal(r, r * 0.1))),
                    "Frequency": int(max(1, rng.normal(f, f * 0.1))),
                    "Monetary": float(max(1.0, rng.normal(m, m * 0.1))),
                }
            )
    df = pd.DataFrame(wiersze)
    df.index = pd.Index(range(10001, 10001 + len(df)), name="CustomerID")
    df["Wartosc_brutto"] = df["Monetary"]
    df["Wartosc_zwrotow"] = 0.0
    df["Pierwszy_zakup"] = pd.Timestamp("2011-01-01")
    df["Ostatni_zakup"] = pd.Timestamp("2011-12-10") - pd.to_timedelta(df["Recency"], unit="D")
    df["Kraj"] = "United Kingdom"
    return df


def test_liczba_przypisan_rowna_liczbie_klientow(klienci):
    wynik = segmentuj(klienci, k=3)
    assert len(wynik.klienci) == len(klienci)
    assert wynik.klienci["Segment"].notna().all()
    assert wynik.klienci.index.equals(klienci.index)


def test_etykiety_od_zera_do_k_minus_1(klienci):
    for k in (2, 3, 4):
        segmenty = segmentuj(klienci, k=k).klienci["Segment"]
        assert set(segmenty.unique()) == set(range(k))


def test_segmenty_numerowane_od_najwyzszej_mediany_monetary(klienci):
    wynik = segmentuj(klienci, k=3)
    mediany = wynik.klienci.groupby("Segment")["Monetary"].median()
    assert mediany.is_monotonic_decreasing


def test_wyraznie_rozdzielone_grupy_sa_odnajdywane(klienci):
    wynik = segmentuj(klienci, k=3)
    rzeczywiste = np.repeat([0, 1, 2], 30)  # kolejność grup zgodna z malejącym Monetary
    assert (wynik.klienci["Segment"].to_numpy() == rzeczywiste).all()


def test_wynik_jest_powtarzalny(klienci):
    pierwszy = segmentuj(klienci, k=4, random_state=7).klienci["Segment"]
    drugi = segmentuj(klienci, k=4, random_state=7).klienci["Segment"]
    pd.testing.assert_series_equal(pierwszy, drugi)


def test_segmentuj_nie_modyfikuje_wejscia(klienci):
    kopia = klienci.copy()
    segmentuj(klienci, k=3)
    pd.testing.assert_frame_equal(klienci, kopia)


def test_porownaj_k_zawiera_wszystkie_wartosci_k(klienci):
    X, _ = przygotuj_macierz_cech(klienci)
    porownanie = porownaj_k(X, k_values=range(2, 7))
    assert porownanie.index.tolist() == [2, 3, 4, 5, 6]
    assert porownanie["inertia"].is_monotonic_decreasing
    assert porownanie["silhouette"].between(-1, 1).all()
    assert porownanie.loc[3, "silhouette"] == porownanie["silhouette"].max()


def test_silhouette_na_probie_jest_odtwarzalny(klienci):
    X, _ = przygotuj_macierz_cech(klienci)
    pierwsze = porownaj_k(X, k_values=[3], proba_silhouette=50)
    drugie = porownaj_k(X, k_values=[3], proba_silhouette=50)
    assert pierwsze.loc[3, "silhouette"] == drugie.loc[3, "silhouette"]


def test_stabilnosc_w_przedziale_od_0_do_1(klienci):
    X, _ = przygotuj_macierz_cech(klienci)
    stabilnosc = stabilnosc_k(X, k_values=[3], liczba_prob=3)
    assert 0 <= stabilnosc.loc[3, "ari_min"] <= stabilnosc.loc[3, "ari_srednie"] <= 1


def test_porownaj_warianty_dla_kazdego_wariantu(klienci):
    warianty = porownaj_warianty(klienci, {"log F i M": {}, "bez log": {"log": False}}, k_values=[2, 3])
    assert warianty.index.get_level_values("wariant").unique().tolist() == ["log F i M", "bez log"]
    assert len(warianty) == 4


def test_uporzadkuj_etykiety():
    etykiety = np.array([5, 5, 7, 7, 9])
    wartosci = np.array([1.0, 2.0, 100.0, 90.0, 10.0])
    assert uporzadkuj_etykiety(etykiety, wartosci).tolist() == [2, 2, 0, 0, 1]


def test_profil_segmentow_w_oryginalnych_jednostkach(klienci):
    wynik = segmentuj(klienci, k=3)
    profil = profiluj_segmenty(wynik.klienci)
    assert profil["liczba_klientow"].sum() == len(klienci)
    assert profil["udzial_klientow_proc"].sum() == pytest.approx(100, abs=0.2)
    assert profil.loc[0, "Monetary_mediana"] == wynik.klienci.loc[wynik.klienci["Segment"] == 0, "Monetary"].median()


def test_opisy_i_dodatkowy_profil(klienci):
    wynik = segmentuj(klienci, k=3)
    opisy = opisz_segmenty(wynik.klienci)
    assert opisy.index.tolist() == [0, 1, 2]
    assert "wysoka wartość" in opisy[0]
    assert "dawno bez zakupu" in opisy[2]
    dodatkowe = dodatkowy_profil(wynik.klienci, pd.Timestamp("2011-12-10"))
    assert dodatkowe["wielka_brytania_proc"].eq(100).all()


def test_rzut_pca_i_eksport(klienci):
    wynik = segmentuj(klienci, k=3)
    rzut, wariancja = rzut_pca(wynik.X, wynik.klienci["Segment"])
    assert rzut.shape == (len(klienci), 3)
    assert 0 < wariancja.sum() <= 1
    eksport = tabela_do_eksportu(wynik.klienci, {0: "A", 1: "B", 2: "C"})
    assert len(eksport) == len(klienci)
    assert eksport.columns[:3].tolist() == ["CustomerID", "Segment", "Nazwa_segmentu"]
