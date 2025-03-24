# Option Prissättning med Python

Detta projekt syftar till att utveckla en modell för prissättning av aktiemarknadens optioner genom att implementera metoder som Black-Scholes och Monte Carlo-simuleringar. Genom att analysera historisk data, volatilitet och riskfri ränta beräknas teoretiska priser, vilka sedan jämförs med faktiska marknadspriser.

## Funktioner
- **Black-Scholes-modellen:** Beräknar teoretiska optionpriser baserat på volatilitet, riskfri ränta och tid till förfall.
- **Monte Carlo-simuleringar:** Simulerar prisutvecklingen för att uppskatta optionpriser.
- **Dataanalys:** Importerar och analyserar historisk aktiedata.
- **Jämförelse:** Utvärderar teoretiska priser mot aktuella marknadspriser.

## Installation
1. Klona repot:
   ```bash
   git clone https://github.com/your_username/option_pricing.git
2. navigera in i projektmappen:
   cd option_pricing
3. Installera nödvändiga bibliotek:
   pip install -r requirements.txt
(Projektet kräver Python 3.7 eller högre.)

## Användning
Kör huvudprogrammet med:
python main.py

Justera konfigurationsparametrar, såsom riskfri ränta, volatilitet och antal simuleringar, i filen config.py.

## Projektstruktur
- **main.py:** Huvudfil för att köra simuleringarna.
- **black_scholes.py:** Innehåller funktioner för Black-Scholes-modellen.
- **monte_carlo.py:** Innehåller funktioner för Monte Carlo-simuleringar.
- **data_analysis.py:** Verktyg för att analysera historisk aktiedata.
- **config.py:** Konfigurationsfil med inställningar och parametrar.

För frågor kontakta mig via mejl:
Oscarenghag@gmail.com
