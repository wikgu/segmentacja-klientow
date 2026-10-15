# Segmentacja klientów metodą K-means

Projekt zaliczeniowy z przedmiotu *Eksploracja danych*: kompletna, powtarzalna eksploracja danych
transakcyjnych sklepu internetowego (UCI Online Retail) – od danych surowych, przez kontrolę jakości
i analizę eksploracyjną, po segmentację klientów algorytmem K-means i prezentację wyników w aplikacji Streamlit.

**Aplikacja:** _link zostanie uzupełniony po wdrożeniu_ (instrukcja wdrożenia i lokalnego uruchomienia poniżej).

## Spis treści

1. [Cel projektu](#cel-projektu)
2. [Dane](#dane)
3. [Uruchomienie](#uruchomienie)
4. [Struktura repozytorium](#struktura-repozytorium)
5. [Kontrola jakości i czyszczenie](#kontrola-jakości-i-czyszczenie)
6. [Analiza eksploracyjna](#analiza-eksploracyjna)
7. [Cechy klientów](#cechy-klientów)
8. [K-means: porównanie wartości k](#k-means-porównanie-wartości-k)
9. [Charakterystyka segmentów](#charakterystyka-segmentów)
10. [Wnioski](#wnioski)
11. [Ograniczenia](#ograniczenia)
12. [Najważniejsze decyzje analityczne](#najważniejsze-decyzje-analityczne)
13. [Testy](#testy)

## Cel projektu

**Główne pytanie:** czy na podstawie zachowań zakupowych można wyróżnić interpretowalne segmenty klientów?

Pytania pomocnicze:

1. Jak różnią się klienci pod względem częstotliwości i wartości zakupów?
2. Ile segmentów jest uzasadnione w świetle wybranych metryk i interpretacji?
3. Czym charakteryzują się uzyskane segmenty i jakie są ograniczenia wnioskowania?

**Krótka odpowiedź:** tak – klienci dzielą się na cztery czytelne segmenty (najlepsi, stali, nowi i okazjonalni,
uśpieni), ale ich granice są umowne: klienci tworzą kontinuum zaangażowania, a K-means dzieli je na części
(silhouette 0,38).

## Dane

| | |
|---|---|
| **Źródło** | [UCI Machine Learning Repository – Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) (Chen, 2015, [doi:10.24432/C5BW33](https://doi.org/10.24432/C5BW33)), licencja CC BY 4.0 |
| **Opis** | wszystkie transakcje brytyjskiego sklepu internetowego z upominkami; wielu klientów to hurtownicy |
| **Zakres czasowy** | 01.12.2010 – 09.12.2011 |
| **Jednostka obserwacji** | pozycja faktury (produkt na fakturze); w segmentacji – **klient** |
| **Rozmiar** | 541 909 wierszy, 8 kolumn; 25 900 faktur, 4 372 klientów, 4 070 kodów produktów, 38 krajów |

| Kolumna | Znaczenie |
|---|---|
| `InvoiceNo` | numer faktury (6 cyfr); prefiks `C` oznacza fakturę anulującą |
| `StockCode` | kod produktu; kody specjalne (np. `POST`, `M`, `D`) to pozycje nietowarowe |
| `Description` | nazwa produktu |
| `Quantity` | liczba sztuk; wartości ujemne to anulowania, zwroty i korekty |
| `InvoiceDate` | data i godzina wystawienia faktury |
| `UnitPrice` | cena jednostkowa w GBP |
| `CustomerID` | identyfikator klienta |
| `Country` | kraj klienta |

Kopia zbioru w formacie Parquet (3,6 MB) znajduje się w [`data/`](data/README.md) – licencja CC BY 4.0
pozwala ją udostępniać. Funkcja `wczytaj_dane()` potrafi odtworzyć ją z oryginalnego pliku XLSX pobieranego z UCI.

## Uruchomienie

Wymagany Python 3.13.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

| Zadanie | Polecenie |
|---|---|
| testy | `pytest` |
| aplikacja | `streamlit run app.py` (otwiera się pod adresem <http://localhost:8501>) |
| notebook z analizą | `pip install -r requirements-dev.txt`, a następnie `jupyter nbconvert --to notebook --execute --inplace notebooks/analiza.ipynb` |
| odtworzenie danych ze źródła | usuń `data/online_retail.parquet` – przy następnym uruchomieniu dane zostaną pobrane z UCI |

Wszystkie wyniki są odtwarzalne: K-means ma ustawione `random_state = 42` i `n_init = 10`, a próby
bootstrapowe używają tego samego ziarna.

### Wdrożenie

* **Render** – repozytorium zawiera plik [`render.yaml`](render.yaml) (usługa *Web Service*, plan darmowy).
  W panelu Render: *New → Blueprint*, wskazać repozytorium. Aplikacja startuje poleceniem
  `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`. Darmowy plan ma 512 MB pamięci;
  aplikacja zajmuje ok. 400–450 MB, dlatego po wczytaniu danych przechowuje tylko zestawienia, a nie pełne transakcje.
* **Streamlit Community Cloud** (alternatywa z większym limitem pamięci) – *Create app*, repozytorium,
  gałąź `main`, plik `app.py`, w ustawieniach zaawansowanych Python 3.13.

## Struktura repozytorium

```
segmentacja-klientow/
├── app.py                    # aplikacja Streamlit
├── analiza/                  # cała logika – ta sama w notebooku, aplikacji i testach
│   ├── wczytanie.py          # pobranie, wczytanie i opis zbioru
│   ├── czyszczenie.py        # kontrola jakości, reguły czyszczenia, dziennik decyzji
│   ├── cechy.py              # tabela cech RFM, przekształcenia, skalowanie
│   ├── segmentacja.py        # K-means, porównanie k, stabilność, profile segmentów
│   ├── eda.py                # zestawienia do analizy eksploracyjnej
│   └── wykresy.py            # wykresy (matplotlib) używane w notebooku i aplikacji
├── tests/                    # testy pytest (53 testy)
├── notebooks/analiza.ipynb   # pełna analiza z komentarzami
├── figures/                  # wykresy zapisane przez notebook
├── data/                     # kopia zbioru (CC BY 4.0) i opis pochodzenia
├── requirements.txt          # zależności aplikacji i testów
├── requirements-dev.txt      # dodatkowe pakiety do uruchamiania notebooka
├── render.yaml               # konfiguracja wdrożenia na Render
└── .github/workflows/        # uruchamianie testów w GitHub Actions
```
