"""Aplikacja Streamlit: segmentacja klientów sklepu internetowego metodą K-means.

Wszystkie obliczenia pochodzą z pakietu ``analiza`` – z tych samych funkcji korzystają
notebook z analizą i testy. Uruchomienie: ``streamlit run app.py``.
"""

import pandas as pd
import streamlit as st

from analiza import czyszczenie
from analiza.cechy import CECHY_RFM, data_odniesienia, kontrola_agregacji, odfiltruj_klientow, zbuduj_rfm
from analiza.wczytanie import URL_OPISU, opis_zbioru, slownik_kolumn, wczytaj_dane

st.set_page_config(page_title="Segmentacja klientów K-means", layout="wide")

SEKCJE = ["Opis danych", "Jakość danych"]


# --------------------------------------------------------------------------- obliczenia (z pamięcią podręczną)


@st.cache_resource(show_spinner="Wczytywanie i czyszczenie danych…")
def przygotuj_dane() -> dict:
    """Wczytanie, czyszczenie i agregacja do poziomu klienta – raz na proces aplikacji.

    Zwracane ramki są tylko do odczytu (funkcje z pakietu ``analiza`` nie modyfikują wejścia).
    Surowe dane nie są przechowywane – zostają z nich tylko zestawienia potrzebne w aplikacji.
    """
    surowe = wczytaj_dane()
    dane, dziennik = czyszczenie.wyczysc_dane(surowe)
    klienci_wszyscy = zbuduj_rfm(dane)
    klienci, wpis_klienci = odfiltruj_klientow(klienci_wszyscy)
    return {
        "opis": opis_zbioru(surowe),
        "slownik": slownik_kolumn(surowe),
        "podglad": surowe.head(10),
        "raport_przed": czyszczenie.raport_jakosci(surowe),
        "raport_po": czyszczenie.raport_jakosci(dane),
        "przed_po": czyszczenie.porownanie_przed_po(surowe, dane),
        "dziennik": pd.concat([dziennik, pd.DataFrame([wpis_klienci])], ignore_index=True),
        "dane": dane,
        "klienci_wszyscy": klienci_wszyscy,
        "klienci": klienci,
        "data_ref": data_odniesienia(dane),
    }


# --------------------------------------------------------------------------- pomocnicze elementy widoku


def tabela(df: pd.DataFrame, precyzja: int = 2, **kwargs) -> None:
    """Tabela z liczbami w zapisie polskim (spacja tysięcy, przecinek dziesiętny)."""
    st.dataframe(df.style.format(precision=precyzja, thousands=" ", decimal=","), **kwargs)


def liczba(x: float, cyfry: int = 0) -> str:
    return f"{x:,.{cyfry}f}".replace(",", " ").replace(".", ",")


def raport_do_wyswietlenia(raport: pd.DataFrame) -> pd.DataFrame:
    """Kolumny min/max raportu jakości mieszają liczby i daty – do tabeli zamieniamy je na tekst."""
    return raport.assign(**{
        kolumna: raport[kolumna].map(lambda v: "" if v is None or pd.isna(v) else str(v))
        for kolumna in ["min", "max"]
    })


# --------------------------------------------------------------------------- sekcje


def sekcja_opis(d: dict) -> None:
    st.title("Segmentacja klientów sklepu internetowego metodą K-means")
    st.markdown(
        f"""
**Pytanie badawcze:** czy na podstawie zachowań zakupowych można wyróżnić interpretowalne segmenty klientów?

**Dane:** [UCI Online Retail]({URL_OPISU}) (licencja CC BY 4.0) – wszystkie transakcje brytyjskiego
sklepu internetowego z upominkami, w którym wielu klientów to hurtownicy. Jeden wiersz danych to jedna
pozycja faktury; w segmentacji jednostką analizy jest **klient** (cechy RFM: Recency, Frequency, Monetary).
"""
    )
    opis, klienci, dane = d["opis"], d["klienci"], d["dane"]
    st.subheader("Dane surowe")
    kolumny = st.columns(5)
    kolumny[0].metric("Pozycje faktur", liczba(opis["liczba_wierszy"]))
    kolumny[1].metric("Faktury", liczba(opis["liczba_faktur"]))
    kolumny[2].metric("Klienci", liczba(opis["liczba_klientow"]))
    kolumny[3].metric("Produkty", liczba(opis["liczba_produktow"]))
    kolumny[4].metric("Kraje", opis["liczba_krajow"])
    st.caption(
        f"Okres: {opis['data_od']:%d.%m.%Y} – {opis['data_do']:%d.%m.%Y}, kolumny: {opis['liczba_kolumn']}."
    )
    st.subheader("Po czyszczeniu")
    kolumny = st.columns(4)
    kolumny[0].metric("Pozycje faktur", liczba(len(dane)))
    kolumny[1].metric("Sprzedaż netto [GBP]", liczba(dane["Wartosc"].sum()))
    kolumny[2].metric("Klienci w segmentacji", liczba(len(klienci)))
    kolumny[3].metric("Data odniesienia (Recency)", f"{d['data_ref']:%d.%m.%Y}")
    st.subheader("Kolumny zbioru")
    st.dataframe(d["slownik"], hide_index=True)
    st.subheader("Podgląd danych surowych")
    st.dataframe(d["podglad"], hide_index=True)


def sekcja_jakosc(d: dict) -> None:
    st.header("Jakość danych i czyszczenie")
    st.markdown(
        """
Każda reguła czyszczenia jest osobną funkcją w `analiza/czyszczenie.py`, nie modyfikuje danych wejściowych
i zapisuje wpis w dzienniku decyzji. **Anulowania i zwroty** zostają w danych – pomniejszają wartość
zakupów netto, ale nie są liczone jako zamówienia. **Wartości skrajne nie są usuwane** – to rzeczywiste
duże zamówienia; ich wpływ ogranicza transformacja logarytmiczna cech.
"""
    )
    st.subheader("Zestawienie problemów przed i po czyszczeniu")
    tabela(d["przed_po"], precyzja=0)
    st.subheader("Dziennik decyzji")
    st.dataframe(d["dziennik"], hide_index=True)
    with st.expander("Raport kolumn: typy, braki, liczba unikalnych wartości"):
        lewa, prawa = st.columns(2)
        lewa.markdown("**Przed czyszczeniem**")
        lewa.dataframe(raport_do_wyswietlenia(d["raport_przed"]))
        prawa.markdown("**Po czyszczeniu**")
        prawa.dataframe(raport_do_wyswietlenia(d["raport_po"]))
    st.subheader("Wartości skrajne (reguła 1,5 IQR – raportowane, nie usuwane)")
    lewa, prawa = st.columns(2)
    lewa.markdown("**Pozycje faktur sprzedażowych**")
    with lewa:
        tabela(czyszczenie.wartosci_skrajne(d["dane"][~d["dane"]["Anulowanie"]], ["Quantity", "UnitPrice", "Wartosc"]))
    prawa.markdown("**Cechy klientów**")
    with prawa:
        tabela(czyszczenie.wartosci_skrajne(d["klienci"], CECHY_RFM))
    st.subheader("Kontrola agregacji do poziomu klienta")
    tabela(kontrola_agregacji(d["dane"], d["klienci_wszyscy"]).to_frame())
    st.caption("Suma Monetary jest równa wartości netto transakcji, a suma Frequency – liczbie faktur "
               "sprzedażowych, więc żadna pozycja nie została policzona podwójnie.")


# --------------------------------------------------------------------------- układ aplikacji


with st.sidebar:
    st.title("Segmentacja klientów")
    sekcja = st.radio("Sekcja", SEKCJE)

dane_aplikacji = przygotuj_dane()
if sekcja == SEKCJE[0]:
    sekcja_opis(dane_aplikacji)
else:
    sekcja_jakosc(dane_aplikacji)
