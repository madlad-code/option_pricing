# Optionsprissättning i Python

Ett kraftfullt verktyg för prissättning av optioner med Black-Scholes, Binomialträd och Monte Carlo-simuleringar. Projektet inkluderar en interaktiv Jupyter-dashboard för realtidsanalys.

## Mål med projektet
Huvudmålet var att utveckla en teknisk plattform som kan hantera komplexa finansiella beräkningar och visualisera dem på ett sätt som är lättillgängligt. Genom att implementera flera olika matematiska modeller kan användaren jämföra precision och beräkningshastighet mellan analytiska och numeriska lösningar.

## Kompetenser som testas
- **Kvantitativ finans:** Djup förståelse för optionsvärdering och riskparametrar (Grekerna).
- **Python-programmering:** Avancerad användning av `NumPy` och `SciPy` för matematiska beräkningar.
- **Data-visualisering:** Skapande av interaktiva dashboards med `ipywidgets` och `matplotlib`.
- **Stokastisk kalkyl:** Implementering av Geometric Brownian Motion för Monte Carlo-simuleringar.

## Innehåll
- **`black_scholes.py`**: Analytisk prissättning av europeiska optioner samt beräkning av Grekerna (Delta, Gamma, Theta, Vega, Rho).
- **`binomial.py`**: Diskret prissättning för både europeiska och amerikanska optioner med visualisering av beslutsträd.
- **`monte_carlo.py`**: Stokastisk simulering av prisbanor för europeiska och binära optioner.
- **`OptionPricingDashboard.ipynb`**: Interaktiv dashboard byggd med `ipywidgets`.

## Installation
Se till att du har Python installerat. Installera sedan nödvändiga bibliotek:
```bash
pip install numpy scipy matplotlib networkx ipywidgets jupyterlab
```

## Användning
För den bästa upplevelsen, använd den inbyggda dashboarden:
1. Starta Jupyter Lab: `jupyter lab`
2. Öppna `OptionPricingDashboard.ipynb`.
3. Kör alla celler.

## Licens
MIT
