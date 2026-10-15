"""Wykresy wykorzystywane w notebooku z analizą i w aplikacji Streamlit.

Funkcje tworzą obiekty ``matplotlib.figure.Figure`` bez globalnego stanu ``pyplot``,
dzięki czemu można je bezpiecznie wywoływać wielokrotnie w aplikacji.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter

from analiza.cechy import CECHY_RFM

KOLORY = ["#2a6f97", "#e07a5f", "#3d9970", "#f2a541", "#7b4b94", "#5c677d", "#c44536", "#1b998b"]
JEDNOSTKI = {"Recency": "dni", "Frequency": "liczba zamówień", "Monetary": "GBP"}


def _tysiace(x, _pos) -> str:
    return f"{x / 1000:,.0f} tys.".replace(",", " ")


def zapisz_wykres(fig: Figure, sciezka: str | Path, dpi: int = 150) -> Path:
    """Zapisuje wykres do pliku PNG (tworzy katalog, jeżeli nie istnieje)."""
    sciezka = Path(sciezka)
    sciezka.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(sciezka, dpi=dpi, bbox_inches="tight")
    return sciezka


def wykres_sprzedazy_miesiecznej(miesiace: pd.DataFrame, uwaga: str | None = None) -> Figure:
    """Sprzedaż netto i liczba aktywnych klientów w kolejnych miesiącach."""
    fig = Figure(figsize=(10, 4.5))
    ax = fig.subplots()
    etykiety = [str(m) for m in miesiace.index]
    ax.bar(etykiety, miesiace["sprzedaz_netto"], color=KOLORY[0], label="sprzedaż netto")
    ax.yaxis.set_major_formatter(FuncFormatter(_tysiace))
    ax.set_ylabel("sprzedaż netto [GBP]")
    ax.tick_params(axis="x", rotation=45)
    ax2 = ax.twinx()
    ax2.plot(etykiety, miesiace["aktywni_klienci"], color=KOLORY[1], marker="o", label="aktywni klienci")
    ax2.set_ylabel("liczba aktywnych klientów")
    ax2.set_ylim(bottom=0)
    ax.set_title("Sprzedaż netto i liczba aktywnych klientów w miesiącach")
    uchwyty = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax.legend(uchwyty, [u.get_label() for u in uchwyty], loc="upper left")
    if uwaga:
        fig.text(0.01, 0.01, uwaga, fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.03 if uwaga else 0, 1, 1))
    return fig


def wykres_krajow(kraje: pd.DataFrame, pomin: str = "United Kingdom") -> Figure:
    """Sprzedaż netto w krajach o największym obrocie poza dominującym rynkiem krajowym.

    Kraj ``pomin`` jest opisany w tytule, bo jego słupek zdominowałby skalę wykresu.
    """
    fig = Figure(figsize=(9, 4.8))
    ax = fig.subplots()
    dane = kraje.drop(index=pomin, errors="ignore").sort_values("sprzedaz_netto")
    ax.barh(dane.index, dane["sprzedaz_netto"], color=KOLORY[0])
    for y, (wartosc, udzial, klienci) in enumerate(zip(dane["sprzedaz_netto"], dane["udzial_proc"], dane["klienci"])):
        ax.text(wartosc, y, f" {udzial:.1f}% · {klienci} klientów", va="center", fontsize=9)
    ax.xaxis.set_major_formatter(FuncFormatter(_tysiace))
    ax.set_xlim(right=dane["sprzedaz_netto"].max() * 1.35)
    ax.set_xlabel("sprzedaż netto [GBP]")
    tytul = "Kraje o największej sprzedaży netto"
    if pomin in kraje.index:
        tytul += (f"\n(poza wykresem: {pomin} – {kraje.loc[pomin, 'udzial_proc']:.1f}% sprzedaży, "
                  f"{kraje.loc[pomin, 'klienci']} klientów)")
    ax.set_title(tytul)
    fig.tight_layout()
    return fig


def wykres_aktywnosci_tygodniowej(tabela: pd.DataFrame) -> Figure:
    """Mapa ciepła liczby faktur według dnia tygodnia i godziny."""
    fig = Figure(figsize=(10, 3.8))
    ax = fig.subplots()
    obraz = ax.imshow(tabela.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_xticks(range(tabela.shape[1]), [str(g) for g in tabela.columns])
    ax.set_yticks(range(tabela.shape[0]), tabela.index)
    ax.set_xlabel("godzina wystawienia faktury")
    ax.set_title("Liczba faktur sprzedażowych według dnia tygodnia i godziny")
    fig.colorbar(obraz, ax=ax, label="liczba faktur")
    fig.tight_layout()
    return fig


def wykres_rozkladow(dane: pd.DataFrame, cechy=CECHY_RFM, tytul: str = "Rozkłady cech klientów",
                     log=()) -> Figure:
    """Histogramy cech klientów z zaznaczoną medianą.

    ``dane`` mogą być już przekształcone – cechy wymienione w ``log`` są wtedy opisane jako log1p.
    """
    do_log = list(cechy) if log is True else list(log or [])
    fig = Figure(figsize=(12, 3.8))
    osie = np.atleast_1d(fig.subplots(1, len(cechy)))
    for ax, cecha, kolor in zip(osie, cechy, KOLORY):
        wartosci = dane[cecha]
        jednostka = JEDNOSTKI.get(cecha, "")
        ax.hist(wartosci, bins=50, color=kolor, edgecolor="white")
        ax.axvline(wartosci.median(), color="black", linestyle="--", linewidth=1, label="mediana")
        ax.set_title(cecha + (" (log1p)" if cecha in do_log else ""))
        ax.set_xlabel(f"log1p({jednostka})" if cecha in do_log else jednostka)
        ax.legend()
    osie[0].set_ylabel("liczba klientów")
    fig.suptitle(tytul)
    fig.tight_layout()
    return fig


def wykres_pudelkowy(rfm: pd.DataFrame, cechy=CECHY_RFM) -> Figure:
    """Wykresy pudełkowe cech klientów w skali logarytmicznej (widoczne wartości odstające)."""
    fig = Figure(figsize=(12, 3.5))
    osie = np.atleast_1d(fig.subplots(1, len(cechy)))
    for ax, cecha, kolor in zip(osie, cechy, KOLORY):
        ax.boxplot(rfm[cecha].clip(lower=1e-2), vert=False, widths=0.6,
                   patch_artist=True, boxprops={"facecolor": kolor, "alpha": 0.6},
                   flierprops={"markersize": 2, "alpha": 0.4})
        ax.set_xscale("log")
        ax.set_yticks([])
        ax.set_title(cecha)
        ax.set_xlabel(f"{JEDNOSTKI.get(cecha, '')}, skala log")
    fig.suptitle("Wartości odstające cech klientów (reguła 1,5 IQR)")
    fig.tight_layout()
    return fig


def wykres_zaleznosci(rfm: pd.DataFrame) -> Figure:
    """Zależność wartości zakupów od liczby zamówień; kolor oznacza Recency."""
    fig = Figure(figsize=(7.5, 5.5))
    ax = fig.subplots()
    punkty = ax.scatter(rfm["Frequency"], rfm["Monetary"], c=rfm["Recency"], cmap="viridis",
                        s=8, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency – liczba zamówień (skala log)")
    ax.set_ylabel("Monetary – wartość netto [GBP] (skala log)")
    ax.set_title("Wartość zakupów a liczba zamówień")
    fig.colorbar(punkty, ax=ax, label="Recency [dni]")
    fig.tight_layout()
    return fig


def wykres_korelacji(macierz: pd.DataFrame, tytul: str = "Korelacja Spearmana cech klientów") -> Figure:
    """Mapa ciepła macierzy korelacji z wartościami w komórkach."""
    fig = Figure(figsize=(5, 4.2))
    ax = fig.subplots()
    obraz = ax.imshow(macierz.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(macierz)), macierz.columns)
    ax.set_yticks(range(len(macierz)), macierz.index)
    for i in range(len(macierz)):
        for j in range(len(macierz)):
            wartosc = macierz.iat[i, j]
            ax.text(j, i, f"{wartosc:.2f}", ha="center", va="center",
                    color="white" if abs(wartosc) > 0.6 else "black")
    ax.set_title(tytul)
    fig.colorbar(obraz, ax=ax)
    fig.tight_layout()
    return fig


def wykres_koncentracji(krzywa: pd.DataFrame) -> Figure:
    """Krzywa koncentracji wartości zakupów (jaki odsetek klientów daje jaki odsetek wartości)."""
    fig = Figure(figsize=(6, 4.5))
    ax = fig.subplots()
    ax.plot(krzywa["odsetek_klientow"], krzywa["odsetek_wartosci"], color=KOLORY[0], label="klienci")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="równy podział")
    udzial20 = np.interp(0.2, krzywa["odsetek_klientow"], krzywa["odsetek_wartosci"])
    ax.scatter([0.2], [udzial20], color=KOLORY[1], zorder=3)
    ax.annotate(f"20% klientów = {udzial20:.0%} wartości", (0.2, udzial20), xytext=(0.3, 0.55),
                arrowprops={"arrowstyle": "->"})
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("odsetek klientów (od najcenniejszych)")
    ax.set_ylabel("skumulowany odsetek wartości netto")
    ax.set_title("Koncentracja wartości zakupów")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def wykres_wyboru_k(porownanie: pd.DataFrame, wybrane_k: int | None = None,
                    stabilnosc: pd.DataFrame | None = None) -> Figure:
    """Metoda łokcia (inertia), współczynnik silhouette i opcjonalnie stabilność podziału dla kolejnych k."""
    liczba_paneli = 2 if stabilnosc is None else 3
    fig = Figure(figsize=(5.5 * liczba_paneli, 4))
    osie = fig.subplots(1, liczba_paneli)
    ax1, ax2 = osie[0], osie[1]
    ax1.plot(porownanie.index, porownanie["inertia"], marker="o", color=KOLORY[0])
    ax1.set_title("Metoda łokcia")
    ax1.set_ylabel("inertia (suma kwadratów odległości)")
    ax2.plot(porownanie.index, porownanie["silhouette"], marker="o", color=KOLORY[1])
    ax2.set_title("Współczynnik silhouette (im więcej, tym lepiej)")
    ax2.set_ylabel("średni silhouette")
    if stabilnosc is not None:
        ax3 = osie[2]
        ax3.plot(stabilnosc.index, stabilnosc["ari_srednie"], marker="o", color=KOLORY[2], label="średnia")
        ax3.fill_between(stabilnosc.index, stabilnosc["ari_min"], stabilnosc["ari_srednie"],
                         color=KOLORY[2], alpha=0.2, label="minimum–średnia")
        ax3.set_title("Stabilność podziału (bootstrap)")
        ax3.set_ylabel("skorygowany indeks Randa")
        ax3.set_ylim(0, 1.02)
        ax3.legend(loc="lower left")
    for ax in osie:
        ax.set_xlabel("liczba klastrów k")
        ax.set_xticks(list(porownanie.index))
        ax.grid(alpha=0.3)
        if wybrane_k is not None:
            ax.axvline(wybrane_k, color="grey", linestyle="--", linewidth=1)
    fig.tight_layout()
    return fig


def wykres_wariantow(warianty: pd.DataFrame) -> Figure:
    """Silhouette i wielkość najmniejszego klastra dla różnych sposobów przygotowania cech."""
    fig = Figure(figsize=(11, 4))
    ax1, ax2 = fig.subplots(1, 2)
    for (nazwa, grupa), kolor in zip(warianty.groupby(level="wariant", sort=False), KOLORY):
        k = grupa.index.get_level_values("k")
        ax1.plot(k, grupa["silhouette"], marker="o", color=kolor, label=nazwa)
        ax2.plot(k, grupa["najmniejszy_klaster_proc"], marker="o", color=kolor, label=nazwa)
    ax1.set_title("Silhouette")
    ax1.set_ylabel("średni silhouette")
    ax2.set_title("Najmniejszy klaster")
    ax2.set_ylabel("odsetek klientów [%] (skala log)")
    ax2.set_yscale("log")
    for ax in (ax1, ax2):
        ax.set_xlabel("liczba klastrów k")
        ax.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    fig.suptitle("Wrażliwość wyniku na przygotowanie cech")
    fig.tight_layout()
    return fig


def wykres_pca(rzut: pd.DataFrame, wariancja, nazwy: dict | None = None) -> Figure:
    """Rzut klientów na dwie pierwsze składowe główne, kolory według segmentu."""
    fig = Figure(figsize=(7.5, 5.5))
    ax = fig.subplots()
    for segment, grupa in rzut.groupby("Segment"):
        etykieta = nazwy.get(segment, f"Segment {segment}") if nazwy else f"Segment {segment}"
        ax.scatter(grupa["PC1"], grupa["PC2"], s=8, alpha=0.6, color=KOLORY[int(segment) % len(KOLORY)],
                   label=etykieta)
    ax.set_xlabel(f"PC1 ({wariancja[0]:.0%} wariancji)")
    ax.set_ylabel(f"PC2 ({wariancja[1]:.0%} wariancji)")
    ax.set_title("Segmenty w rzucie PCA 2D (wizualizacja, nie dowód separacji)")
    ax.legend(markerscale=2)
    fig.tight_layout()
    return fig


def wykres_segmentow(klienci: pd.DataFrame, cechy=CECHY_RFM, nazwy: dict | None = None) -> Figure:
    """Rozkłady cech w segmentach (wykresy pudełkowe, skala logarytmiczna)."""
    fig = Figure(figsize=(12, 4))
    osie = np.atleast_1d(fig.subplots(1, len(cechy)))
    segmenty = sorted(klienci["Segment"].unique())
    etykiety = [nazwy.get(s, str(s)) if nazwy else str(s) for s in segmenty]
    for ax, cecha in zip(osie, cechy):
        dane = [klienci.loc[klienci["Segment"] == s, cecha].clip(lower=1e-2) for s in segmenty]
        pudelka = ax.boxplot(dane, patch_artist=True, widths=0.6, flierprops={"markersize": 2, "alpha": 0.3})
        for pudelko, s in zip(pudelka["boxes"], segmenty):
            pudelko.set_facecolor(KOLORY[int(s) % len(KOLORY)])
            pudelko.set_alpha(0.7)
        ax.set_xticks(range(1, len(segmenty) + 1), etykiety, rotation=20, ha="right")
        ax.set_yscale("log")
        ax.set_title(f"{cecha} [{JEDNOSTKI.get(cecha, '')}]")
    fig.suptitle("Rozkład cech w segmentach (skala logarytmiczna)")
    fig.tight_layout()
    return fig


def wykres_profili(X: pd.DataFrame, etykiety, nazwy: dict | None = None) -> Figure:
    """Średnie standaryzowanych cech w segmentach (odległość od średniej w odchyleniach std)."""
    srednie = X.groupby(np.asarray(etykiety)).mean()
    fig = Figure(figsize=(6.5, 0.6 * len(srednie) + 1.8))
    ax = fig.subplots()
    granica = max(1.0, float(np.abs(srednie.to_numpy()).max()))
    obraz = ax.imshow(srednie.to_numpy(), cmap="RdBu_r", vmin=-granica, vmax=granica, aspect="auto")
    ax.set_xticks(range(srednie.shape[1]), srednie.columns)
    etykiety_y = [nazwy.get(s, f"Segment {s}") if nazwy else f"Segment {s}" for s in srednie.index]
    ax.set_yticks(range(len(srednie)), etykiety_y)
    for i in range(srednie.shape[0]):
        for j in range(srednie.shape[1]):
            wartosc = srednie.iat[i, j]
            ax.text(j, i, f"{wartosc:+.2f}", ha="center", va="center",
                    color="white" if abs(wartosc) > 0.6 * granica else "black")
    ax.set_title("Średnie cech po standaryzacji w segmentach")
    fig.colorbar(obraz, ax=ax, label="odchylenia standardowe od średniej")
    fig.tight_layout()
    return fig
