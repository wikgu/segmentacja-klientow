# Dane

Surowy zbiór **Online Retail** (UCI Machine Learning Repository, licencja CC BY 4.0) nie jest
przechowywany w repozytorium. Funkcja `analiza.wczytanie.wczytaj_dane()` przy pierwszym
uruchomieniu pobiera archiwum ze strony
<https://archive.ics.uci.edu/dataset/352/online+retail> do katalogu `data/raw/`
(ignorowanego przez Git) i zapisuje tam szybszą do odczytu kopię w formacie Parquet.

Źródło: Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C5BW33>
