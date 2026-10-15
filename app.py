"""Aplikacja Streamlit: segmentacja klientów sklepu internetowego metodą K-means.

Wszystkie obliczenia pochodzą z pakietu ``analiza`` – z tych samych funkcji korzystają
notebook z analizą i testy. Uruchomienie: ``streamlit run app.py``.
"""

import pandas as pd
import streamlit as st

from analiza import czyszczenie, eda, wykresy
from analiza.cechy import (
    CECHY_LOG,
    CECHY_RFM,
    data_odniesienia,
    kontrola_agregacji,
    odfiltruj_klientow,
    przeksztalc_cechy,
    zbuduj_rfm,
)
from analiza.wczytanie import URL_OPISU, opis_zbioru, slownik_kolumn, wczytaj_dane

st.set_page_config(page_title="Segmentacja klientów K-means", layout="wide")

SEKCJE = ["Opis danych", "Jakość danych", "Analiza eksploracyjna"]


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


@st.cache_resource(show_spinner="Przygotowanie wykresów EDA…")
def wykresy_eda() -> dict:
    """Wykresy analizy eksploracyjnej (nie zależą od ustawień w panelu bocznym)."""
    d = przygotuj_dane()
    dane, klienci = d["dane"], d["klienci"]
    return {
        "miesiace": wykresy.wykres_sprzedazy_miesiecznej(
            eda.sprzedaz_miesieczna(dane), uwaga="Grudzień 2011 obejmuje tylko 9 dni (dane kończą się 09.12.2011)."
        ),
        "kraje": wykresy.wykres_krajow(eda.sprzedaz_wg_kraju(dane, n=11)),
        "tydzien": wykresy.wykres_aktywnosci_tygodniowej(eda.aktywnosc_tygodniowa(dane)),
        "rozklady": wykresy.wykres_rozkladow(klienci, tytul="Rozkłady cech klientów w skali oryginalnej"),
        "rozklady_log": wykresy.wykres_rozkladow(
            przeksztalc_cechy(klienci), tytul="Rozkłady cech po przekształceniu (log1p dla F i M)", log=CECHY_LOG
        ),
        "pudelkowy": wykresy.wykres_pudelkowy(klienci),
        "zaleznosci": wykresy.wykres_zaleznosci(klienci),
        "korelacje": wykresy.wykres_korelacji(eda.korelacje(klienci)),
        "koncentracja": wykresy.wykres_koncentracji(eda.koncentracja_wartosci(klienci)),
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


def sekcja_eda(d: dict) -> None:
    st.header("Analiza eksploracyjna")
    fig = wykresy_eda()
    dane, klienci = d["dane"], d["klienci"]

    st.subheader("Transakcje")
    tabela(eda.statystyki_opisowe(dane[~dane["Anulowanie"]], ["Quantity", "UnitPrice", "Wartosc"]))
    st.caption("Statystyki pozycji faktur sprzedażowych. Średnie są wyraźnie wyższe od median – rozkłady są "
               "silnie prawostronnie skośne (obok wielu drobnych zakupów są pojedyncze duże zamówienia hurtowe).")
    st.pyplot(fig["miesiace"])
    st.caption("Sprzedaż i liczba aktywnych klientów rosną od września do listopada 2011 (sezon przedświąteczny). "
               "Dlatego Recency liczona na 10.12.2011 faworyzuje klientów kupujących przed świętami.")
    lewa, prawa = st.columns(2)
    with lewa:
        st.pyplot(fig["kraje"])
        st.caption("Wielka Brytania to 82% sprzedaży i 90% klientów – wnioski dotyczą głównie rynku brytyjskiego.")
    with prawa:
        st.pyplot(fig["tydzien"])
        st.caption("Zamówienia składane są w godzinach pracy, nie ma faktur z soboty – rytm typowy dla klientów "
                   "firmowych (hurtowników).")

    st.subheader("Cechy klientów (RFM)")
    tabela(eda.statystyki_opisowe(klienci, CECHY_RFM))
    skala = st.radio("Skala histogramów", ["oryginalna", "po przekształceniu"], horizontal=True)
    st.pyplot(fig["rozklady"] if skala == "oryginalna" else fig["rozklady_log"])
    st.caption("Frequency i Monetary mają skrajną skośność (mediany 2 zamówienia i 650 GBP, maksima 206 i 278 778 GBP). "
               "Po log1p rozkłady są zbliżone do symetrycznych; Recency zostaje w dniach.")
    st.pyplot(fig["pudelkowy"])
    st.caption("Wartości odstające to rzeczywiści, najcenniejsi klienci – nie są usuwane.")
    lewa, srodek, prawa = st.columns(3)
    with lewa:
        st.pyplot(fig["zaleznosci"])
        st.caption("Klienci z wieloma zamówieniami wydają więcej i kupowali niedawno.")
    with srodek:
        st.pyplot(fig["korelacje"])
        st.caption("Korelacja Spearmana F–M wynosi 0,81 – cechy w dużej mierze opisują jeden wymiar zaangażowania.")
    with prawa:
        st.pyplot(fig["koncentracja"])
        st.caption(f"20% najcenniejszych klientów odpowiada za {eda.udzial_najlepszych(klienci):.0%} wartości netto.")

    with st.expander("Wnioski z analizy eksploracyjnej", expanded=True):
        st.markdown(
            """
1. **Koncentracja wartości** – *obserwacja:* 20% klientów generuje 74% wartości netto; *interpretacja:*
   warto wyodrębnić małą grupę kluczowych klientów, a cechy wartości trzeba przekształcić przed K-means.
2. **Klienci okazjonalni** – *obserwacja:* 34,6% klientów złożyło jedno zamówienie; *interpretacja:*
   duża część bazy to klienci jednorazowi lub nowi, ale roczne okno nie pozwala odróżnić nowych od utraconych.
3. **Jeden wymiar zaangażowania** – *obserwacja:* F i M są silnie skorelowane i maleją z Recency;
   *interpretacja:* segmenty będą się różnić głównie ogólnym poziomem aktywności.
4. **Sezonowość** – *obserwacja:* wrzesień–listopad to szczyt sprzedaży; *interpretacja:* wynik zależy
   od daty odniesienia dla Recency.
"""
        )


# --------------------------------------------------------------------------- układ aplikacji


with st.sidebar:
    st.title("Segmentacja klientów")
    sekcja = st.radio("Sekcja", SEKCJE)

dane_aplikacji = przygotuj_dane()
if sekcja == SEKCJE[0]:
    sekcja_opis(dane_aplikacji)
elif sekcja == SEKCJE[1]:
    sekcja_jakosc(dane_aplikacji)
else:
    sekcja_eda(dane_aplikacji)
