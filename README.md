# Volatility & Risk Management (Peru ADRs) — GARCH(1,1)

Proyecto de finanzas cuantitativas que estima volatilidad condicional mediante GARCH(1,1) y calcula medidas de riesgo (VaR y Expected Shortfall) para activos peruanos cotizados internacionalmente (BAP, BVN) usando Python.

## Metodología
- Descarga de precios diarios con yfinance (2015–2025)
- Transformación a rendimientos logarítmicos
- Evidencia visual de clustering de volatilidad
- Estimación GARCH(1,1) con:
  - Distribución Normal
  - Distribución t-Student (colas pesadas)
- Cálculo de VaR 95% y Expected Shortfall
- Backtesting por proporción de violaciones
- Comparación Normal vs t-Student (riesgo extremo)

## Resultados clave 
- Persistencia alta de volatilidad (α+β ≈ 0.95)
- Colas pesadas (ν ≈ 4.3)
- VaR 95%: proporción de violaciones cercana al 5% teórico
- t-Student captura mejor pérdidas extremas que Normal

## Herramientas
Python, yfinance, pandas, numpy, matplotlib, arch, scipy

## Outputs
Se generan archivos en /data:
- raw_prices.csv
- log_returns.csv
- clustering_volatilidad.png
- var_garch_95.png
- comparacion_modelos.csv
- comparacion_volatilidades.png

## Cómo ejecutar
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
