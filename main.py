import yfinance as yf
import pandas as pd
import os

# ------------------------------------------------------
# Proyecto: Predicción de volatilidad y gestión de riesgo
# Activos: Empresas peruanas con cotización internacional
# Periodo de análisis: 2015-2025
# ------------------------------------------------------

# Selección de activos (sector financiero y minería)
tickers = ["BAP", "BVN"]

fecha_inicio = "2015-01-01"
fecha_fin = "2025-01-01"

# Descarga de precios históricos diarios
datos = yf.download(tickers, start=fecha_inicio, end=fecha_fin)

# Se utilizan precios ajustados para incorporar dividendos
precios = datos["Close"]
precios = precios[["BAP", "BVN"]]

# Guardar base de datos para asegurar reproducibilidad
if not os.path.exists("data"):
    os.makedirs("data")

precios.to_csv("data/raw_prices.csv")

print("Descarga completada.")
print(precios.head())

# Depuración de la base de datos
# Se eliminan observaciones con valores faltantes para garantizar coherencia temporal entre activos.

precios = precios.dropna()

print("\nBase de datos después de eliminar valores faltantes:")
print(precios.head())


# Transformación a rendimientos logarítmicos
# Se calculan rendimientos logarítmicos:
# r_t = ln(P_t / P_{t-1})

import numpy as np

rendimientos = np.log(precios / precios.shift(1))
rendimientos = rendimientos.dropna()

print("\nPrimeros rendimientos logarítmicos:")
print(rendimientos.head())


# Guardar rendimientos para análisis posterior

rendimientos.to_csv("data/log_returns.csv")

print("\nRendimientos guardados correctamente.")

# Análisis exploratorio de volatilidad
# Se grafican los rendimientos para identificar episodios de clustering de volatilidad.
# En mercados financieros, la varianza suele cambiar en el tiempo.

import matplotlib.pyplot as plt
plt.close('all')

plt.figure(figsize=(12,5))
plt.plot(rendimientos["BAP"])
plt.title("Rendimientos Logarítmicos - BAP")
plt.xlabel("Fecha")
plt.ylabel("Rendimiento")
plt.axhline(0, linestyle="--")
plt.tight_layout()
plt.savefig("data/clustering_volatilidad.png", dpi=300)
plt.close()

#Estimación modelo GARCH(1,1)
# Se estima un modelo GARCH(1,1) para modelar la volatilidad condicional de los rendimientos de BAP.
# Este modelo captura persistencia en la varianza.

from arch import arch_model

# Se multiplican los rendimientos por 100 para estabilidad numérica
serie = rendimientos["BAP"] * 100

modelo = arch_model(serie, vol="Garch", p=1, q=1, dist="t")
resultado = modelo.fit(disp="off")

print(resultado.summary())



# Cálculo de Value at Risk (VaR)
# Se calcula el VaR al 95% utilizando la volatilidad condicional estimada por el modelo GARCH(1,1).
# Se asume distribución normal.

import numpy as np
import scipy.stats as stats

# Extraer volatilidad condicional
volatilidad = resultado.conditional_volatility

# Nivel de confianza
nivel_confianza = 0.95

# Cuantil normal
z = stats.norm.ppf(1 - nivel_confianza)

# VaR diario (en porcentaje)
VaR = resultado.params["mu"] + z * volatilidad

# Convertimos rendimientos a la misma escala
rend_scaled = serie

# Visualización VaR vs Rendimientos

plt.figure(figsize=(12,6))
plt.plot(rend_scaled, label="Rendimientos")
plt.plot(VaR, label="VaR 95%", linestyle="--")
plt.title("Value at Risk (95%) - Modelo GARCH")
plt.legend()
plt.tight_layout()
plt.savefig("data/var_garch_95.png", dpi=300)
plt.close()

# Backtesting del VaR - Test de Kupiec
# Se evalúa si la proporción de violaciones del VaR coincide con el nivel de confianza teórico (5%).

violaciones = rend_scaled < VaR
num_violaciones = violaciones.sum()
n = len(rend_scaled)

proporcion_violaciones = num_violaciones / n

print("Número de violaciones:", num_violaciones)
print("Proporción de violaciones:", round(proporcion_violaciones,4))
print("Proporción teórica esperada:", 1 - nivel_confianza)

# Expected Shortfall (ES)
# El ES mide la pérdida promedio en los días donde el VaR es superado.
# Captura el riesgo en la cola extrema de la distribución.

ES = rend_scaled[violaciones].mean()

print("Expected Shortfall estimado:", round(ES,4))

# Comparación con distribución Normal

# Estimación GARCH con distribución normal
modelo_normal = arch_model(serie, vol="Garch", p=1, q=1, dist="normal")
resultado_normal = modelo_normal.fit(disp="off")

# Comparación de volatilidad condicional (Normal vs t-Student)
plt.figure(figsize=(12,6))
plt.plot(resultado.conditional_volatility, label="Volatilidad t-Student")
plt.plot(resultado_normal.conditional_volatility, label="Volatilidad Normal", linestyle="--")
plt.title("Comparación de Volatilidad Condicional - GARCH(1,1)")
plt.legend()
plt.tight_layout()
plt.savefig("data/comparacion_volatilidades.png", dpi=300)
plt.close()

print("Gráfico guardado: data/comparacion_volatilidades.png")

# Volatilidad condicional normal
vol_normal = resultado_normal.conditional_volatility

# VaR normal
z_normal = stats.norm.ppf(1 - nivel_confianza)
VaR_normal = resultado_normal.params["mu"] + z_normal * vol_normal

# Violaciones normal
violaciones_normal = serie < VaR_normal
num_violaciones_normal = violaciones_normal.sum()
proporcion_normal = num_violaciones_normal / n

# Expected Shortfall normal
ES_normal = serie[violaciones_normal].mean()

print("\n--- COMPARACIÓN NORMAL vs t-STUDENT ---")
print("Violaciones Normal:", num_violaciones_normal)
print("Proporción Normal:", round(proporcion_normal,4))
print("ES Normal:", round(ES_normal,4))

print("\nViolaciones t-Student:", num_violaciones)
print("Proporción t-Student:", round(proporcion_violaciones,4))
print("ES t-Student:", round(ES,4))

# Guardar comparación
comparacion = pd.DataFrame({
    "Modelo": ["Normal", "t-Student"],
    "Violaciones": [num_violaciones_normal, num_violaciones],
    "Proporcion": [proporcion_normal, proporcion_violaciones],
    "Expected_Shortfall": [ES_normal, ES]
})

comparacion.to_csv("data/comparacion_modelos.csv", index=False)

# Conclusión automática del modelo

print("\n--- CONCLUSIÓN DEL ANÁLISIS ---")

if proporcion_normal > proporcion_violaciones:
    print("La distribución Normal subestima riesgo extremo.")
else:
    print("La distribución t-Student captura mejor colas pesadas.")

if ES < ES_normal:
    print("t-Student presenta pérdidas extremas más realistas.")
    
print("\nProyecto finalizado correctamente.")
print("Archivos generados en carpeta /data")