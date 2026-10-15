"""Segmentacja klientów algorytmem K-means, porównanie liczby klastrów i profilowanie."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from analiza.cechy import CECHY_LOG, CECHY_RFM, przygotuj_macierz_cech

RANDOM_STATE = 42
N_INIT = 10
ZAKRES_K = range(2, 9)


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
        wiersze.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(
                    X, etykiety, sample_size=proba_silhouette, random_state=random_state
                ),
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
