# Optionsprissättning i Python

Detta projekt innehåller en samling Python-program för optionsprissättning med olika metoder. Koden implementerar flera prissättningsmodeller och inkluderar verktyg för visualisering av optionsbeteende och riskparametrar.

## Modeller

### Black-Scholes-modellen
En grundläggande modell för optionsprissättning som ger en stängd formel för europeiska optioner. Modellen beräknar:
- Optionspriser baserat på konstant volatilitet
- Grekerna (delta, gamma, theta, vega, rho) för riskhantering

### Binomialträdsmodell
En diskret modell som är särskilt användbar för:
- Amerikansk optionsprissättning med möjlighet till tidig inlösen
- Flexibel hantering av volatilitet vid olika noder
- Tydlig visualisering av möjliga prisbanor

### Monte Carlo-simulering
Metod som använder slumpmässig provtagning för att uppskatta optionspriser:
- Lämpar sig för komplexa stigberoende optioner
- Kan hantera flera underliggande tillgångar
- Genererar konfidensintervall för prisuppskattningar

### Stokastiska volatilitetsmodeller (Heston)
En avancerad modell som hanterar varierande volatilitet genom att:
- Modellera volatilitet som en egen stokastisk process
- Fånga marknadsegenskaper som volatilitetskluster
- Integrera parametrar för mean reversion och långsiktig varians

### SABR-modellen
En modell populär på räntederivatmarknader som:
- Direkt modellerar volatilitetskurvan (smile/skew)
- Inkluderar CEV-parameter och volatilitet av volatilitet
- Bättre fångar marknadsobserverade mönster

## Installation

```bash
# Klona repository
git clone https://github.com/användarnamn/option-pricing-python.git
cd option-pricing-python

# Skapa virtuell miljö
python -m venv venv
source venv/bin/activate  # På Windows: venv\Scripts\activate

# Installera paket
pip install numpy scipy matplotlib pandas networkx
```

## Användning

Kör modellerna individuellt:

```bash
python black_scholes_model.py
python binomial_tree_model.py
python monte_carlo_model.py
```

Eller importera i egna program:

```python
from black_scholes_model import BlackScholesModel

# Skapa modellinstans
bs_model = BlackScholesModel()

# Sätt parametrar
S = 100      # Aktiepris
K = 100      # Lösenpris
T = 1.0      # Tid till förfall
r = 0.05     # Riskfri ränta
sigma = 0.2  # Volatilitet

# Beräkna pris och greker
call_price, call_greeks = bs_model.price_option(S, K, T, r, sigma, 'call')
print(f"Köpoptionspris: ${call_price:.4f}")
```

## Visualiseringar

Projektet inkluderar verktyg för att visualisera:
- Optionspriser vid olika nivåer
- Grekernas beteende över tid
- Volatilitetsytor
- Binomialträd och Monte Carlo-prisbanor

## Avancerade funktioner

- Implicit volatilitetsberäkning
- Amerikansk optionsvärdering
- Exotiska optioner
- Stokastisk volatilitetsmodellering
- Känslighetsanalys för parametrar

## Bidrag

Bidrag välkomnas! För att bidra:
1. Forka repositoryt
2. Skapa en funktionsgren
3. Committa ändringar
4. Pusha till din gren
5. Skapa en Pull Request

## Licens

Projektet använder MIT-licens. Se LICENSE-filen för detaljer.
