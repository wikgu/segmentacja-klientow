"""Wykresy wykorzystywane w notebooku z analizą i w aplikacji Streamlit.

Funkcje tworzą obiekty ``matplotlib.figure.Figure`` bez globalnego stanu ``pyplot``,
dzięki czemu można je bezpiecznie wywoływać wielokrotnie w aplikacji.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter


KOLORY = ["#2a6f97", "#e07a5f", "#3d9970", "#f2a541", "#7b4b94", "#5c677d", "#c44536", "#1b998b"]


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
