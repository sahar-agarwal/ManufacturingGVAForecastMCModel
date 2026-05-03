# Manufacturing GVA Growth Rate Forecast MC Model

# India Future Trajectory

An interactive **Streamlit** dashboard for quarterly manufacturing-growth forecasting in India using **ARIMAX**, Monte Carlo simulation, scenario analysis, and residual diagnostics. 

## Overview

This project loads quarterly macroeconomic data from `IDAML_Compiled.csv`, fits a SARIMAX model to the manufacturing growth series (`MFG_GVA`), and generates short-term forecasts, 3-year Monte Carlo projections, and 10-year fan-chart uncertainty bands. 
It is designed as a compact analytics app for exploring how manufacturing growth may evolve under different macroeconomic conditions. 
The interface includes dataset previews, forecast charts, confidence intervals, simulation controls, and model diagnostics. 

## Features

- Quarterly data parsing and cleaning.
- SARIMAX model fitting with exogenous macro variables.
- 1-year forecast with confidence interval.
- 3-year Monte Carlo simulation.
- 10-year fan chart with percentile bands.
- Residual diagnostics using ACF, PACF, and histogram plots.
- Downloadable forecast tables in CSV format.
## Project Structure

```text
.
├── main.py
├── IDAML_Compiled.csv
└── README.md
```

## Requirements

Install the required packages:

```bash
pip install streamlit pandas numpy plotly statsmodels matplotlib
```

## Run the App

Start the Streamlit app with:

```bash
streamlit run main.py
```

## Data Format

The app expects a CSV file named `IDAML_Compiled.csv` in the same folder as `main.py`.  
The file must contain a `DATE` column with quarterly values and the following variables:

- `MFG_GVA`
- `MFG_IIP`
- `WPI`
- `BANK_CREDIT`
- `REER`
- `EXPORTS`
- `GOVT_CAPEX`

The `DATE` column should be in a quarterly format such as `2010Q1`, `2010Q2`, and so on.

## Model Logic

The target series is `MFG_GVA`, which the app uses as the manufacturing growth variable.  
The model is fitted as an ARIMAX specification with user-controlled AR, differencing, MA, and seasonal parameters.  
Forecast inputs for the exogenous variables are created using recent averages, Monte Carlo perturbations, and compound-growth scenario paths. 

## Outputs

The app provides:

- Historical data preview.
- Cleaned data preview.
- 1-year point forecast with 95% confidence bands.
- 3-year simulated forecast distribution.
- 10-year fan chart with percentile bands.
- Residual ACF/PACF and histogram diagnostics.
- CSV downloads for forecasts and fan-chart percentiles. 

