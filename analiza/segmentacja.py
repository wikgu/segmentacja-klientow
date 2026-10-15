"""Segmentacja klientów algorytmem K-means, porównanie liczby klastrów i profilowanie."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn import config_context
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from analiza.cechy import CECHY_LOG, CECHY_RFM, przygotuj_macierz_cech

RANDOM_STATE = 42
N_INIT = 10
ZAKRES_K = range(2, 9)
PAMIEC_ROBOCZA_MB = 64

WYBRANE_K = 4
# Nazwy nadane po analizie profili dla k = 4 przy domyślnym przygotowaniu cech
# (segmenty są numerowane od najwyższej mediany Monetary).
NAZWY_SEGMENTOW = {
    0: "Najlepsi klienci",
    1: "Stali klienci",
    2: "Nowi i okazjonalni",
    3: "Uśpieni",
}


@dataclass
class WynikSegmentacji:
    """Wynik pełnego potoku segmentacji dla jednej wartości k."""

    klienci: pd.DataFrame
    X: pd.DataFrame
    model: KMeans
    skaler: StandardScaler
    k: int


def dopasuj_kmeans(X, k: int, n_init: int = N_INIT, random_state: int = RANDOM_STATE) -> KMeans:
    """Dopasowuje K-means z jawnie ustawionymi ``n_init`` i ``random_state`` (wyniki odtwarzalne)."""
    return KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(X)


def porownaj_k(
    X,
    k_values=ZAKRES_K,
    proba_silhouette: int | None = None,
    n_init: int = N_INIT,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Inertia, silhouette i indeks Daviesa–Bouldina dla kolejnych wartości k.

    Przy dużych zbiorach silhouette można liczyć na odtwarzalnej próbie
    o liczności ``proba_silhouette`` (losowanej z ``random_state``).
    """
    wiersze = []
    for k in k_values:
        model = dopasuj_kmeans(X, k, n_init=n_init, random_state=random_state)
        etykiety = model.labels_
        licznosci = np.bincount(etykiety, minlength=k)
        # Odległości między wszystkimi parami klientów liczone są w porcjach ograniczonych
        # do PAMIEC_ROBOCZA_MB (domyślnie scikit-learn rezerwuje na to do 1 GB).
        with config_context(working_memory=PAMIEC_ROBOCZA_MB):
            silhouette = silhouette_score(X, etykiety, sample_size=proba_silhouette, random_state=random_state)
        wiersze.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette,
                "davies_bouldin": davies_bouldin_score(X, etykiety),
                "najmniejszy_klaster": int(licznosci.min()),
                "najmniejszy_klaster_proc": 100 * licznosci.min() / len(etykiety),
            }
        )
    return pd.DataFrame(wiersze).set_index("k")


def stabilnosc_k(
    X,
    k_values=ZAKRES_K,
    liczba_prob: int = 20,
    n_init: int = N_INIT,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Stabilność podziału: średni skorygowany indeks Randa (ARI) między podziałem
    z pełnych danych a podziałami z modeli dopasowanych do prób bootstrapowych."""
    X = np.asarray(X)
    rng = np.random.default_rng(random_state)
    proby = [rng.integers(0, len(X), size=len(X)) for _ in range(liczba_prob)]
    wiersze = []
    for k in k_values:
        bazowe = dopasuj_kmeans(X, k, n_init=n_init, random_state=random_state).labels_
        ari = [
            adjusted_rand_score(
                bazowe, dopasuj_kmeans(X[idx], k, n_init=n_init, random_state=random_state).predict(X)
            )
            for idx in proby
        ]
        wiersze.append({"k": k, "ari_srednie": float(np.mean(ari)), "ari_min": float(np.min(ari))})
    return pd.DataFrame(wiersze).set_index("k")


def porownaj_warianty(
    rfm: pd.DataFrame,
    warianty: dict,
    k_values=ZAKRES_K,
    n_init: int = N_INIT,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Porównanie k dla różnych sposobów przygotowania cech (wrażliwość wyniku na transformację).

    ``warianty`` to słownik ``nazwa -> argumenty przygotuj_macierz_cech`` (np. ``{"log": False}``).
    """
    wyniki = {}
    for nazwa, opcje in warianty.items():
        X, _ = przygotuj_macierz_cech(rfm, **opcje)
        wyniki[nazwa] = porownaj_k(X, k_values, n_init=n_init, random_state=random_state)
    return pd.concat(wyniki, names=["wariant"])


def uporzadkuj_etykiety(etykiety, wartosci) -> np.ndarray:
    """Numeruje klastry od 0 w kolejności malejącej mediany ``wartosci`` (np. Monetary).

    Numery z K-means są arbitralne; stała kolejność ułatwia porównywanie wyników.
    """
    mediany = pd.Series(np.asarray(wartosci)).groupby(np.asarray(etykiety)).median()
    mapa = {stara: nowa for nowa, stara in enumerate(mediany.sort_values(ascending=False).index)}
    return np.array([mapa[e] for e in np.asarray(etykiety)])


def segmentuj(
    rfm: pd.DataFrame,
    k: int,
    cechy=CECHY_RFM,
    log=CECHY_LOG,
    winsoryzacja=None,
    n_init: int = N_INIT,
    random_state: int = RANDOM_STATE,
) -> WynikSegmentacji:
    """Pełny potok: przekształcenie cech, skalowanie, K-means i przypisanie segmentu klientom."""
    X, skaler = przygotuj_macierz_cech(rfm, cechy=cechy, log=log, winsoryzacja=winsoryzacja)
    model = dopasuj_kmeans(X, k, n_init=n_init, random_state=random_state)
    klienci = rfm.copy()
    klienci["Segment"] = uporzadkuj_etykiety(model.labels_, rfm["Monetary"])
    return WynikSegmentacji(klienci=klienci, X=X, model=model, skaler=skaler, k=k)


def profiluj_segmenty(klienci: pd.DataFrame, cechy=CECHY_RFM, kolumna: str = "Segment") -> pd.DataFrame:
    """Liczebność, udział oraz średnie i mediany cech w oryginalnych jednostkach dla każdego segmentu."""
    grupy = klienci.groupby(kolumna)
    profil = pd.DataFrame(
        {
            "liczba_klientow": grupy.size(),
            "udzial_klientow_proc": (100 * grupy.size() / len(klienci)).round(1),
            "udzial_wartosci_proc": (100 * grupy["Monetary"].sum() / klienci["Monetary"].sum()).round(1),
        }
    )
    for cecha in cechy:
        profil[f"{cecha}_mediana"] = grupy[cecha].median()
        profil[f"{cecha}_srednia"] = grupy[cecha].mean()
    return profil


def opisz_segmenty(klienci: pd.DataFrame, cechy=CECHY_RFM, kolumna: str = "Segment") -> pd.Series:
    """Krótki opis segmentów: w którym tercylu całej populacji leży mediana każdej cechy."""
    slowa = {
        "Recency": ("niedawny zakup", "zakup średnio dawno", "dawno bez zakupu"),
        "Frequency": ("rzadkie zakupy", "średnia częstotliwość", "częste zakupy"),
        "Monetary": ("niska wartość", "średnia wartość", "wysoka wartość"),
    }
    mediany = klienci.groupby(kolumna)[list(cechy)].median()
    opisy = {}
    for segment, wiersz in mediany.iterrows():
        czesci = []
        for cecha in cechy:
            t1, t2 = klienci[cecha].quantile([1 / 3, 2 / 3])
            poziom = 0 if wiersz[cecha] <= t1 else (2 if wiersz[cecha] > t2 else 1)
            czesci.append(slowa.get(cecha, ("niskie", "średnie", "wysokie"))[poziom])
        opisy[segment] = ", ".join(czesci)
    return pd.Series(opisy, name="opis")


def dodatkowy_profil(klienci: pd.DataFrame, data_ref, kolumna: str = "Segment") -> pd.DataFrame:
    """Cechy pomocnicze segmentów, nieużywane w K-means: staż, koszyk, klienci jednorazowi, zwroty, kraj."""
    staz = (pd.Timestamp(data_ref) - klienci["Pierwszy_zakup"].dt.normalize()).dt.days
    dane = klienci.assign(
        staz_dni=staz,
        nowy=staz <= 90,
        jednorazowy=klienci["Frequency"] == 1,
        koszyk=klienci["Monetary"] / klienci["Frequency"],
        ze_zwrotem=klienci["Wartosc_zwrotow"] > 0,
        z_wielkiej_brytanii=klienci["Kraj"] == "United Kingdom",
    )
    grupy = dane.groupby(kolumna)
    return pd.DataFrame(
        {
            "staz_dni_mediana": grupy["staz_dni"].median(),
            "nowi_90_dni_proc": 100 * grupy["nowy"].mean(),
            "jednorazowi_proc": 100 * grupy["jednorazowy"].mean(),
            "koszyk_mediana": grupy["koszyk"].median(),
            "ze_zwrotem_proc": 100 * grupy["ze_zwrotem"].mean(),
            "wielka_brytania_proc": 100 * grupy["z_wielkiej_brytanii"].mean(),
        }
    ).round(1)


def tabela_do_eksportu(klienci: pd.DataFrame, nazwy: dict | None = None) -> pd.DataFrame:
    """Tabela klientów z przypisanym segmentem w postaci gotowej do zapisu w CSV."""
    kolumny = ["Segment", *CECHY_RFM, "Wartosc_brutto", "Wartosc_zwrotow", "Pierwszy_zakup", "Ostatni_zakup", "Kraj"]
    wynik = klienci.loc[:, [k for k in kolumny if k in klienci.columns]].copy()
    if nazwy:
        wynik.insert(1, "Nazwa_segmentu", wynik["Segment"].map(nazwy))
    return wynik.reset_index()


def rzut_pca(X, etykiety=None, random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, np.ndarray]:
    """Rzut cech na dwie pierwsze składowe główne (tylko do wizualizacji).

    Zwraca ramkę ze współrzędnymi PC1, PC2 (i opcjonalnie segmentem) oraz udział
    wariancji wyjaśnianej przez obie składowe.
    """
    pca = PCA(n_components=2, random_state=random_state)
    wspolrzedne = pca.fit_transform(X)
    indeks = X.index if isinstance(X, pd.DataFrame) else None
    wynik = pd.DataFrame(wspolrzedne, columns=["PC1", "PC2"], index=indeks)
    if etykiety is not None:
        wynik["Segment"] = np.asarray(etykiety)
    return wynik, pca.explained_variance_ratio_
