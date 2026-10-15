# Dane

`online_retail.parquet` to pełna kopia zbioru **Online Retail** z UCI Machine Learning Repository
(541 909 pozycji faktur, 8 kolumn) zapisana w formacie Parquet. Zbiór jest udostępniony na licencji
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), która pozwala go rozpowszechniać z podaniem
źródła. Plik zawiera dane w postaci oryginalnej – zmieniono jedynie typy kolumn i usunięto zbędne spacje
na początku i końcu tekstów (`analiza.wczytanie.ujednolic_typy`). Kopia pozwala uruchomić aplikację bez
pobierania i parsowania pliku XLSX (ok. 23 MB).

Odtworzenie pliku ze źródła: usuń `online_retail.parquet` i uruchom `analiza.wczytanie.wczytaj_dane()` –
funkcja pobierze archiwum z <https://archive.ics.uci.edu/dataset/352/online+retail> do katalogu
`data/raw/` (ignorowanego przez Git) i ponownie zapisze kopię Parquet.

Źródło: Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C5BW33>
