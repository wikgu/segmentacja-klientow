"""Aplikacja Streamlit: segmentacja klientów sklepu internetowego metodą K-means.

Wszystkie obliczenia pochodzą z pakietu ``analiza`` – z tych samych funkcji korzystają
notebook z analizą i testy. Uruchomienie: ``streamlit run app.py``.
"""

import gc

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
    przygotuj_macierz_cech,
    zbuduj_rfm,
)
from analiza.segmentacja import (
    N_INIT,
    NAZWY_SEGMENTOW,
    RANDOM_STATE,
    WYBRANE_K,
    ZAKRES_K,
    dodatkowy_profil,
    opisz_segmenty,
    porownaj_k,
    profiluj_segmenty,
    rzut_pca,
    segmentuj,
    tabela_do_eksportu,
)
from analiza.wczytanie import URL_OPISU, opis_zbioru, slownik_kolumn, wczytaj_dane

st.set_page_config(page_title="Segmentacja klientów K-means", layout="wide")

WARIANTY_CECH = {
    "log1p dla Frequency i Monetary (przyjęty w analizie)": CECHY_LOG,
    "log1p dla Recency, Frequency i Monetary": True,
    "bez transformacji": False,
}
DOMYSLNY_WARIANT = next(iter(WARIANTY_CECH))
SEKCJE = ["Opis danych", "Jakość danych", "Analiza eksploracyjna", "Segmentacja K-means", "Eksport wyników"]


# --------------------------------------------------------------------------- obliczenia (z pamięcią podręczną)


@st.cache_resource(show_spinner="Wczytywanie i czyszczenie danych…")
def przygotuj_dane() -> dict:
    """Wczytanie, czyszczenie i agregacja do poziomu klienta – raz na proces aplikacji.

    Zwracane ramki są tylko do odczytu (funkcje z pakietu ``analiza`` nie modyfikują wejścia).
    Z danych transakcyjnych zostają tylko zestawienia potrzebne w aplikacji, a same transakcje
    są zwalniane, żeby aplikacja mieściła się w niewielkiej pamięci (np. darmowy plan Render: 512 MB).
    """
    surowe = wczytaj_dane()
    wynik = {
        "opis": opis_zbioru(surowe),
        "slownik": slownik_kolumn(surowe),
        "podglad": surowe.head(10),
        "raport_przed": czyszczenie.raport_jakosci(surowe),
    }
    problemy_przed = czyszczenie.podsumowanie_problemow(surowe)
    dane, dziennik = czyszczenie.wyczysc_dane(surowe)
    del surowe
    gc.collect()

    klienci_wszyscy = zbuduj_rfm(dane)
    klienci, wpis_klienci = odfiltruj_klientow(klienci_wszyscy)
    sprzedaz = dane[~dane["Anulowanie"]]
    wynik.update(
        {
            "raport_po": czyszczenie.raport_jakosci(dane),
            "przed_po": pd.concat(
                [problemy_przed.rename("przed"), czyszczenie.podsumowanie_problemow(dane).rename("po")], axis=1
            ),
            "dziennik": pd.concat([dziennik, pd.DataFrame([wpis_klienci])], ignore_index=True),
            "kontrola_agregacji": kontrola_agregacji(dane, klienci_wszyscy),
            "skrajne_transakcje": czyszczenie.wartosci_skrajne(sprzedaz, ["Quantity", "UnitPrice", "Wartosc"]),
            "statystyki_transakcji": eda.statystyki_opisowe(sprzedaz, ["Quantity", "UnitPrice", "Wartosc"]),
            "miesiace": eda.sprzedaz_miesieczna(dane),
            "kraje": eda.sprzedaz_wg_kraju(dane, n=11),
            "tydzien": eda.aktywnosc_tygodniowa(dane),
            "liczba_pozycji": len(dane),
            "sprzedaz_netto": float(dane["Wartosc"].sum()),
            "data_ref": data_odniesienia(dane),
            "klienci": klienci,
        }
    )
    return wynik


@st.cache_resource(show_spinner="Przygotowanie wykresów EDA…")
def wykresy_eda() -> dict:
    """Wykresy analizy eksploracyjnej (nie zależą od ustawień w panelu bocznym)."""
    d = przygotuj_dane()
    klienci = d["klienci"]
    return {
        "miesiace": wykresy.wykres_sprzedazy_miesiecznej(
            d["miesiace"], uwaga="Grudzień 2011 obejmuje tylko 9 dni (dane kończą się 09.12.2011)."
        ),
        "kraje": wykresy.wykres_krajow(d["kraje"]),
        "tydzien": wykresy.wykres_aktywnosci_tygodniowej(d["tydzien"]),
        "rozklady": wykresy.wykres_rozkladow(klienci, tytul="Rozkłady cech klientów w skali oryginalnej"),
        "rozklady_log": wykresy.wykres_rozkladow(
            przeksztalc_cechy(klienci), tytul="Rozkłady cech po przekształceniu (log1p dla F i M)", log=CECHY_LOG
        ),
        "pudelkowy": wykresy.wykres_pudelkowy(klienci),
        "zaleznosci": wykresy.wykres_zaleznosci(klienci),
        "korelacje": wykresy.wykres_korelacji(eda.korelacje(klienci)),
        "koncentracja": wykresy.wykres_koncentracji(eda.koncentracja_wartosci(klienci)),
    }


@st.cache_data(show_spinner="Porównywanie wartości k…")
def porownanie_k(wariant: str) -> pd.DataFrame:
    X, _ = przygotuj_macierz_cech(przygotuj_dane()["klienci"], log=WARIANTY_CECH[wariant])
    return porownaj_k(X, k_values=ZAKRES_K)


@st.cache_data(show_spinner="Dopasowywanie K-means…")
def segmentacja(k: int, wariant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    wynik = segmentuj(przygotuj_dane()["klienci"], k, log=WARIANTY_CECH[wariant])
    return wynik.klienci, wynik.X


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


def nazwy_segmentow(k: int, wariant: str) -> dict:
    """Nazwy z analizy dotyczą k = 4 i przyjętego przygotowania cech; w innych ustawieniach – numery."""
    if k == WYBRANE_K and wariant == DOMYSLNY_WARIANT:
        return NAZWY_SEGMENTOW
    return {s: f"Segment {s}" for s in range(k)}


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
    opis, klienci = d["opis"], d["klienci"]
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
    kolumny[0].metric("Pozycje faktur", liczba(d["liczba_pozycji"]))
    kolumny[1].metric("Sprzedaż netto [GBP]", liczba(d["sprzedaz_netto"]))
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
        tabela(d["skrajne_transakcje"])
    prawa.markdown("**Cechy klientów**")
    with prawa:
        tabela(czyszczenie.wartosci_skrajne(d["klienci"], CECHY_RFM))
    st.subheader("Kontrola agregacji do poziomu klienta")
    tabela(d["kontrola_agregacji"].to_frame())
    st.caption("Suma Monetary jest równa wartości netto transakcji, a suma Frequency – liczbie faktur "
               "sprzedażowych, więc żadna pozycja nie została policzona podwójnie.")


def sekcja_eda(d: dict) -> None:
    st.header("Analiza eksploracyjna")
    fig = wykresy_eda()
    klienci = d["klienci"]

    st.subheader("Transakcje")
    tabela(d["statystyki_transakcji"])
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


def sekcja_segmentacja(d: dict, k: int, wariant: str) -> None:
    st.header(f"Segmentacja K-means (k = {k})")
    st.caption(f"Przygotowanie cech: {wariant}; standaryzacja StandardScaler dopasowana do "
               f"{liczba(len(d['klienci']))} klientów; KMeans(n_init={N_INIT}, random_state={RANDOM_STATE}).")
    if k != WYBRANE_K or wariant != DOMYSLNY_WARIANT:
        st.info(f"W analizie przyjęto k = {WYBRANE_K} i {DOMYSLNY_WARIANT.split(' (')[0]}. "
                "Nazwy segmentów dotyczą tego wariantu – przy innych ustawieniach segmenty są numerowane "
                "od najwyższej mediany Monetary i opisane automatycznie.")

    porownanie = porownanie_k(wariant)
    klienci, X = segmentacja(k, wariant)
    nazwy = nazwy_segmentow(k, wariant)

    st.subheader("Porównanie liczby klastrów")
    lewa, prawa = st.columns([3, 2])
    with lewa:
        st.pyplot(wykresy.wykres_wyboru_k(porownanie, wybrane_k=k))
    with prawa:
        tabela(porownanie[["inertia", "silhouette", "davies_bouldin", "najmniejszy_klaster"]], precyzja=3)
        st.caption("Inercja zawsze maleje z k – liczy się miejsce „łokcia”. Silhouette (im więcej, tym lepiej) "
                   "i Davies–Bouldin (im mniej, tym lepiej) nie mierzą użyteczności biznesowej segmentów.")

    wiersz = porownanie.loc[k]
    kolumny = st.columns(3)
    kolumny[0].metric("Silhouette", liczba(wiersz["silhouette"], 3))
    kolumny[1].metric("Inercja", liczba(wiersz["inertia"], 0))
    kolumny[2].metric("Najmniejszy segment", f"{int(wiersz['najmniejszy_klaster'])} klientów")

    st.subheader("Profile segmentów (oryginalne jednostki)")
    profil = profiluj_segmenty(klienci)
    profil.insert(0, "nazwa", profil.index.map(nazwy))
    profil["opis"] = opisz_segmenty(klienci)
    tabela(profil)
    dodatkowe = dodatkowy_profil(klienci, d["data_ref"])
    dodatkowe.insert(0, "nazwa", dodatkowe.index.map(nazwy))
    with st.expander("Cechy pomocnicze segmentów (nieużywane w K-means)"):
        tabela(dodatkowe, precyzja=1)
        st.caption("Staż – dni od pierwszego zakupu do daty odniesienia; nowi – pierwszy zakup w ostatnich "
                   "90 dniach; koszyk – Monetary / Frequency.")

    lewa, prawa = st.columns(2)
    with lewa:
        st.pyplot(wykresy.wykres_profili(X, klienci["Segment"], nazwy))
        st.caption("Średnie standaryzowanych cech: ile odchyleń standardowych segment leży od średniej klientów.")
    with prawa:
        rzut, wariancja = rzut_pca(X, klienci["Segment"])
        st.pyplot(wykresy.wykres_pca(rzut, wariancja, nazwy))
        st.caption(f"Rzut PCA zachowuje {wariancja.sum():.0%} wariancji cech. To tylko wizualizacja – nie jest "
                   "dowodem separacji segmentów w pełnej przestrzeni cech.")
    st.pyplot(wykresy.wykres_segmentow(klienci, nazwy=nazwy))


def sekcja_eksport(k: int, wariant: str) -> None:
    st.header("Eksport tabeli klientów z segmentem")
    klienci, _ = segmentacja(k, wariant)
    nazwy = nazwy_segmentow(k, wariant)
    eksport = tabela_do_eksportu(klienci, nazwy)
    wybrane = st.multiselect("Segmenty", options=list(nazwy), default=list(nazwy), format_func=nazwy.get)
    widok = eksport[eksport["Segment"].isin(wybrane)]
    st.caption(f"{liczba(len(widok))} klientów; k = {k}, przygotowanie cech: {wariant}.")
    st.download_button(
        "Pobierz CSV",
        data=widok.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"klienci_segmenty_k{k}.csv",
        mime="text/csv",
        type="primary",
    )
    st.dataframe(widok.head(500), hide_index=True)
    if len(widok) > 500:
        st.caption("Podgląd pokazuje pierwszych 500 klientów; plik CSV zawiera wszystkich wybranych.")


# --------------------------------------------------------------------------- układ aplikacji


with st.sidebar:
    st.title("Segmentacja klientów")
    sekcja = st.radio("Sekcja", SEKCJE)
    st.divider()
    k = st.slider("Liczba segmentów k", min_value=min(ZAKRES_K), max_value=max(ZAKRES_K), value=WYBRANE_K)
    wariant = st.selectbox("Przygotowanie cech", list(WARIANTY_CECH),
                           help="Sposób ograniczenia wpływu wartości skrajnych przed standaryzacją.")
    st.caption(f"W analizie wybrano k = {WYBRANE_K} – uzasadnienie w README i w notebooku.")

dane_aplikacji = przygotuj_dane()
if sekcja == SEKCJE[0]:
    sekcja_opis(dane_aplikacji)
elif sekcja == SEKCJE[1]:
    sekcja_jakosc(dane_aplikacji)
elif sekcja == SEKCJE[2]:
    sekcja_eda(dane_aplikacji)
elif sekcja == SEKCJE[3]:
    sekcja_segmentacja(dane_aplikacji, k, wariant)
else:
    sekcja_eksport(k, wariant)
