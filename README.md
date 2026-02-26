# Volatility & Risk Management (Peru ADRs) — GARCH(1,1)

Quantitative finance project that estimates conditional volatility using a GARCH(1,1) model and computes risk measures (Value at Risk and Expected Shortfall) for internationally traded Peruvian assets (BAP, BVN) using Python.

## Methodology
- Daily price download using yfinance (2015–2025)
- Log-return transformation
- Visual evidence of volatility clustering
- GARCH(1,1) estimation with:
  - Normal distribution
  - Student-t distribution (heavy tails)
- 95% VaR and Expected Shortfall estimation
- Backtesting through violation frequency
- Normal vs Student-t comparison (extreme risk)

## Key Results
- High volatility persistence (α+β ≈ 0.95)
- Heavy-tailed behavior (ν ≈ 4.3)
- 95% VaR: violation rate close to the theoretical 5%
- Student-t captures extreme losses better than Normal distribution

## Tools
Python, yfinance, pandas, numpy, matplotlib, arch, scipy

## Outputs
The following files are generated in /data:
- raw_prices.csv
- log_returns.csv
- clustering_volatilidad.png
- var_garch_95.png
- comparacion_modelos.csv
- comparacion_volatilidades.png

## How to Run
```bash
pip install -r requirements.txt
py main.py
```

## requirements.txt

```txt
yfinance
pandas
numpy
matplotlib
arch
scipy
```

