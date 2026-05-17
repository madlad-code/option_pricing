# Optionsprissättning i Python

Ett kraftfullt verktyg för prissättning av optioner med Black-Scholes, Binomialträd och Monte Carlo-simuleringar. Projektet inkluderar en interaktiv Jupyter-dashboard för realtidsanalys.

## Innehåll

- **`black_scholes.py`**: Analytisk prissättning av europeiska optioner samt beräkning av Grekerna (Delta, Gamma, Theta, Vega, Rho).
- **`binomial.py`**: Diskret prissättning för både europeiska och amerikanska optioner med visualisering av beslutsträd.
- **`monte_carlo.py`**: Stokastisk simulering av prisbanor för europeiska och binära optioner.
- **`OptionPricingDashboard.ipynb`**: Interaktiv dashboard byggd med `ipywidgets`.

## Installation

Först, se till att du har Python installerat. Installera sedan nödvändiga bibliotek:

```bash
pip install numpy scipy matplotlib networkx ipywidgets jupyterlab
```

## Interaktiv Dashboard (Rekommenderas)

För den bästa upplevelsen, använd den inbyggda dashboarden. Den låter dig justera parametrar som volatilitet och tid i realtid och se hur de påverkar priser och riskparametrar direkt.

1. Starta Jupyter Lab:
   ```bash
   jupyter lab
   ```
2. Öppna `OptionPricingDashboard.ipynb`.
3. Kör alla celler (`Run > Run All Cells`).

## Användning av skript

Du kan även köra varje modell som fristående skript för att se exempel och generera grafer:

```bash
python black_scholes.py
python binomial.py
python monte_carlo.py
```

## Matematiska Koncept

- **Geometric Brownian Motion (GBM)**: Grunden för Monte Carlo och Black-Scholes.
- **Risk-Neutral Valuation**: Teorin bakom arbitragefri prissättning.
- **Greeks**: Känslighetsanalys för riskhantering.
- **Backwards Induction**: Används i binomialträdet för att värdera amerikanska optioner.

## Licens

MIT
