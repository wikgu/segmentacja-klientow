"""Segmentacja klientów algorytmem K-means, porównanie liczby klastrów i profilowanie."""

from __future__ import annotations


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score


RANDOM_STATE = 42
N_INIT = 10
ZAKRES_K = range(2, 9)


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
