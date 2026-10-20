# Resultados Experimentales de la Selección de Características con Metaheurísticas para IDS en MAWILab 2021

> **Declaración Metodológica y Muestreo del Corpus**:
> - **Universo de flujos MAWILab 2021**: **19.089.344 flujos totales** distribuidos en tres capturas (`20210616`: 9.49M, `20210210`: 7.37M, `20211013`: 2.24M).
> - **Extracción de la captura `2021-02-10`**: Procesada y extraída parcialmente desde su archivo PCAP original debido a limitaciones de tiempo computacional en la disección de paquetes.
> - **Herramienta de extracción de atributos**: Script propio en Python ([`extract_fixed.py`](file:///c:/Users/Jhojan/Documents/ciclo_26_2/IA2/Parcial/extract_fixed.py)) implementado con la biblioteca `dpkt` para ensamblado de flujos bidireccionales con ventanas de timeout (inactividad 120 s, duración máxima 300 s), extrayendo 50 atributos estadísticos directos de cabeceras IP/TCP.
> - **Corpus muestreado**: **3.000.000 de flujos** (15.7% del universo) estratificado por clase (22.82% ataque / 77.18% benigno):
>   - `20210616`: 1.500.000 flujos (15.8% de su traza).
>   - `20210210`: 1.000.000 flujos (13.6% de su traza).
>   - `20211013`: 500.000 flujos (22.4% de su traza).
> - **Partición de datos**: 70% entrenamiento (2.1M) y 30% prueba (900k).
> - **Paridad de Entrenamiento**: Todos los brazos experimentales (**Baseline-500k**, **Control Aleatorio**, **PSO**, **BA**) se entrenan estrictamente sobre la misma submuestra estratificada de **500.000 flujos** (`seed=42`) para garantizar igualdad computacional y no mezclar el efecto del subconjunto de características con el volumen de entrenamiento.

---

## 1. Tabla Oficial de Métricas en Partición *i.i.d.* ($n=30$ corridas pareadas, train=500k, test=900k)

| Brazo | $F_1$ Macro | Recall Ataque | Accuracy | TFP (Falsas Alarmas) | TFD (Alertas Falsas) | N° Features | RD% | Costo Búsqueda ($T_{\text{search}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clasificador Trivial** *(Zero-R)* | $0.4356 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.7718 \pm 0.0000$ | **$0.0000 \pm 0.0000$** | — | $0.0$ | — | $0.0\text{ s}$ |
| **Baseline (500k)** *(Ref. Formal)* | $0.7280 \pm 0.0001$ | **$0.7117 \pm 0.0004$** | $0.7858 \pm 0.0002$ | $0.1924 \pm 0.0003$ | $0.4776 \pm 0.0003$ | $50.0 \pm 0.0$ | $0.0\%$ | $0.0\text{ s}$ |
| **Control Aleatorio** ($k=21$) | $0.7161 \pm 0.0373$ | $0.7071 \pm 0.1036$ | $0.7739 \pm 0.0553$ | $0.2063 \pm 0.1020$ | $0.4716 \pm 0.0873$ | $21.0 \pm 0.0$ | $58.0\%$ | $0.0\text{ s}$ |
| **PSO** | **$0.7342 \pm 0.0155$** | $0.6478 \pm 0.0764$ | **$0.8026 \pm 0.0281$** | **$0.1516 \pm 0.0588$** | **$0.4202 \pm 0.0923$** | **$18.5 \pm 2.6$** | **$63.1\%$** | Med: **$59.1\text{ s}$** (Media: $307.8\text{ s}$) |
| **BA** | $0.7244 \pm 0.0016$ | $0.6993 \pm 0.0175$ | $0.7840 \pm 0.0029$ | $0.1910 \pm 0.0086$ | $0.4801 \pm 0.0058$ | $24.2 \pm 3.0$ | $51.5\%$ | Med: **$122.6\text{ s}$** (Media: $130.2\text{ s}$) |

*Notas sobre la tabla*:
1. Para el **Clasificador Trivial** (*Zero-R*), la Tasa de Falsos Descubrimientos (TFD = $\frac{FP}{FP+TP} = \frac{0}{0}$) no está matemáticamente definida, y la Reducción de Dimensionalidad (RD%) no aplica, por lo que se indican como `—`.
2. *Línea Base original de 2.1M*: $F_1 = 0.7286 \pm 0.0002$, Recall = $0.7183 \pm 0.0004$. No se utiliza en la inferencia estadística comparativa para preservar la estricta paridad muestral.

---

## 2. Test Global de Friedman y Rangos Medios ($k=4$ brazos, $N=30$ bloques, $df=3$)

De acuerdo con la metodología de Demšar (2006), el rango 1 corresponde al mejor desempeño individual en cada pliegue/semilla.

| Métrica | Estadístico de Friedman $\chi_F^2$ | Valor $p$ | Significativo ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: |
| **$F_1$ Macro** | $\chi_F^2 = 25.32$ | $p < 0.0001$ ($1.32 \times 10^{-5}$) | Sí |
| **Recall Ataque** | $\chi_F^2 = 55.40$ | $p < 0.0001$ ($5.64 \times 10^{-12}$) | Sí |
| **Accuracy** | $\chi_F^2 = 20.28$ | $p = 0.0001$ ($1.48 \times 10^{-4}$) | Sí |
| **TFP (Falsas Alarmas)** | $\chi_F^2 = 10.67$ | $p = 0.0137$ | Sí |
| **TFD (Alertas Falsas)** | $\chi_F^2 = 21.40$ | $p = 0.0001$ ($8.69 \times 10^{-5}$) | Sí |

### Rangos Medios por Brazo Experimental:

| Brazo | Rango Medio $F_1$ Macro | Rango Medio Recall Ataque | Rango Medio Accuracy | Rango Medio TFP | Rango Medio TFD |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline-500k** | **1.53** *(1°)* | **1.13** *(1°)* | **1.77** *(1°)* | **2.10** *(1°)* | **1.77** *(1°)* |
| **PSO** | 2.50 *(2°)* | 3.57 *(4°)* | 2.27 *(2°)* | 2.20 *(2°)* | 2.23 *(2°)* |
| **BA** | 2.93 *(3°)* | 2.77 *(3°)* | 2.80 *(3°)* | 2.67 *(3°)* | 2.80 *(3°)* |
| **Control Aleatorio** ($k=21$) | 3.03 *(4°)* | 2.53 *(2°)* | 3.17 *(4°)* | 3.03 *(4°)* | 3.20 *(4°)* |

> **Interpretación Crítica de Friedman**:
> La prueba de Friedman demuestra la existencia de heterogeneidad global entre los cuatro tratamientos evaluados en las 5 métricas. Sin embargo, esta significancia global está impulsada predominantemente por el **liderazgo de la Línea Base** (rango medio 1.53 en $F_1$ y 1.13 en Recall). El test global no demuestra que los algoritmos bioinspirados sean superiores a la línea base ni cualitativamente distintos entre sí; dicha diferenciación requiere el análisis exhaustivo de los contrastes pareados post-hoc.

---

## 3. Matriz Exhaustiva de Contrastes Pareados ($30$ contrastes, Wilcoxon + Corrección de Holm, $N=30$)

Se evaluaron formalmente los 30 contrastes posibles ($6 \text{ pares de brazos} \times 5 \text{ métricas}$) aplicando el test de rangos con signo de Wilcoxon con ajuste de Holm ($\alpha = 0.05$) y tamaño del efecto rank-biserial ($rb\_r$):

| Métrica | Contraste (A vs B) | Media A | Mediana A | Media B | Mediana B | $\Delta$ abs | $\Delta\%$ | Wilcoxon $W$ | $p_{\text{raw}}$ | $rb\_r$ | $p_{\text{Holm}}$ | Significativo |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$F_1$ Macro** | PSO vs BL500k | 0.7342 | 0.7249 | 0.7280 | 0.7279 | +0.0063 | +0.86% | 228.0 | 0.9354 | +0.019 | 1.0000 | No |
| **$F_1$ Macro** | BA vs BL500k | 0.7244 | 0.7246 | 0.7280 | 0.7279 | -0.0036 | -0.49% | 0.0 | < 0.0001 | -1.000 | < 0.0001 | **Sí (BA inferior)** |
| **$F_1$ Macro** | RAND vs BL500k | 0.7161 | 0.7233 | 0.7280 | 0.7279 | -0.0119 | -1.63% | 121.0 | 0.0208 | -0.480 | 0.3342 | No |
| **$F_1$ Macro** | PSO vs BA | 0.7342 | 0.7249 | 0.7244 | 0.7246 | +0.0098 | +1.36% | 120.0 | 0.0197 | +0.484 | 0.3342 | No |
| **$F_1$ Macro** | PSO vs RAND | 0.7342 | 0.7249 | 0.7161 | 0.7233 | +0.0181 | +2.53% | 134.0 | 0.0427 | +0.424 | 0.5264 | No |
| **$F_1$ Macro** | BA vs RAND | 0.7244 | 0.7246 | 0.7161 | 0.7233 | +0.0083 | +1.16% | 170.0 | 0.2054 | +0.269 | 1.0000 | No |
| **Recall Ataque** | PSO vs BL500k | 0.6478 | 0.7008 | 0.7117 | 0.7117 | -0.0639 | -8.98% | 0.0 | < 0.0001 | -1.000 | < 0.0001 | **Sí (Deterioro)** |
| **Recall Ataque** | BA vs BL500k | 0.6993 | 0.7052 | 0.7117 | 0.7117 | -0.0124 | -1.74% | 0.0 | < 0.0001 | -1.000 | < 0.0001 | **Sí (Deterioro)** |
| **Recall Ataque** | RAND vs BL500k | 0.7071 | 0.7052 | 0.7117 | 0.7117 | -0.0046 | -0.65% | 114.0 | 0.0137 | -0.510 | 0.2952 | No |
| **Recall Ataque** | PSO vs BA | 0.6478 | 0.7008 | 0.6993 | 0.7052 | -0.0516 | -7.37% | 76.0 | 0.0008 | -0.673 | **0.0215** | **Sí (BA supera a PSO)** |
| **Recall Ataque** | PSO vs RAND | 0.6478 | 0.7008 | 0.7071 | 0.7052 | -0.0593 | -8.39% | 105.0 | 0.0076 | -0.548 | 0.1827 | No |
| **Recall Ataque** | BA vs RAND | 0.6993 | 0.7052 | 0.7071 | 0.7052 | -0.0077 | -1.09% | 222.0 | 0.8394 | -0.045 | 1.0000 | No |
| **Accuracy** | PSO vs BL500k | 0.8026 | 0.7844 | 0.7858 | 0.7858 | +0.0168 | +2.14% | 145.0 | 0.0732 | +0.376 | 0.6592 | No |
| **Accuracy** | BA vs BL500k | 0.7840 | 0.7835 | 0.7858 | 0.7858 | -0.0018 | -0.23% | 86.0 | 0.0019 | -0.630 | **0.0485** | **Sí (BA inferior)** |
| **Accuracy** | RAND vs BL500k | 0.7739 | 0.7818 | 0.7858 | 0.7858 | -0.0118 | -1.51% | 137.0 | 0.0497 | -0.411 | 0.5468 | No |
| **Accuracy** | PSO vs BA | 0.8026 | 0.7844 | 0.7840 | 0.7835 | +0.0186 | +2.38% | 113.0 | 0.0128 | +0.514 | 0.2952 | No |
| **Accuracy** | PSO vs RAND | 0.8026 | 0.7844 | 0.7739 | 0.7818 | +0.0287 | +3.70% | 118.0 | 0.0175 | +0.492 | 0.3142 | No |
| **Accuracy** | BA vs RAND | 0.7840 | 0.7835 | 0.7739 | 0.7818 | +0.0100 | +1.30% | 161.0 | 0.1460 | +0.308 | 1.0000 | No |
| **TFP (FPR)** | PSO vs BL500k | 0.1516 | 0.1917 | 0.1924 | 0.1924 | -0.0407 | -21.16% | 133.0 | 0.0405 | -0.428 | 0.5264 | No |
| **TFP (FPR)** | BA vs BL500k | 0.1910 | 0.1931 | 0.1924 | 0.1924 | -0.0013 | -0.69% | 129.0 | 0.0327 | +0.445 | 0.4648 | No |
| **TFP (FPR)** | RAND vs BL500k | 0.2063 | 0.1948 | 0.1924 | 0.1924 | +0.0140 | +7.26% | 159.0 | 0.1347 | +0.316 | 1.0000 | No |
| **TFP (FPR)** | PSO vs BA | 0.1516 | 0.1917 | 0.1910 | 0.1931 | -0.0394 | -20.62% | 113.0 | 0.0128 | -0.514 | 0.2952 | No |
| **TFP (FPR)** | PSO vs RAND | 0.1516 | 0.1917 | 0.2063 | 0.1948 | -0.0547 | -26.50% | 103.0 | 0.0133 | -0.526 | 0.2952 | No |
| **TFP (FPR)** | BA vs RAND | 0.1910 | 0.1931 | 0.2063 | 0.1948 | -0.0153 | -7.41% | 163.0 | 0.1579 | -0.299 | 1.0000 | No |
| **TFD (FDR)** | PSO vs BL500k | 0.4202 | 0.4797 | 0.4776 | 0.4776 | -0.0574 | -12.02% | 137.0 | 0.0497 | -0.411 | 0.5468 | No |
| **TFD (FDR)** | BA vs BL500k | 0.4801 | 0.4812 | 0.4776 | 0.4776 | +0.0025 | +0.52% | 87.0 | 0.0020 | +0.626 | **0.0505** | No *(en el borde)* |
| **TFD (FDR)** | RAND vs BL500k | 0.4716 | 0.4840 | 0.4776 | 0.4776 | -0.0060 | -1.25% | 158.0 | 0.1294 | +0.320 | 1.0000 | No |
| **TFD (FDR)** | PSO vs BA | 0.4202 | 0.4797 | 0.4801 | 0.4812 | -0.0599 | -12.47% | 113.0 | 0.0128 | -0.514 | 0.2952 | No |
| **TFD (FDR)** | PSO vs RAND | 0.4202 | 0.4797 | 0.4716 | 0.4840 | -0.0514 | -10.90% | 128.0 | 0.0310 | -0.449 | 0.4648 | No |
| **TFD (FDR)** | BA vs RAND | 0.4801 | 0.4812 | 0.4716 | 0.4840 | +0.0085 | +1.79% | 180.0 | 0.2894 | -0.226 | 1.0000 | No |

---

### Análisis de Hallazgos Estadísticos Críticos:

1. **Aparente Contradicción de Signo en BA vs Línea Base (TFP)**:
   En el contraste de TFP entre BA y la línea base, la diferencia de medias es negativa ($\Delta = -0.0013$, media $0.1910$ vs $0.1924$), pero el coeficiente rank-biserial es positivo ($rb\_r = +0.445$). Al inspeccionar las **medianas**, se observa que la mediana de BA ($0.1931$) es superior a la de la línea base ($0.1924$). La distribución está asimétricamente sesgada: en la mayoría de las corridas pareadas (20 de 30), BA presentó una TFP ligeramente superior al baseline, produciendo una suma de rangos positivos mayor en la prueba de Wilcoxon, mientras que unas pocas corridas con TFP marcadamente bajo redujeron la media aritmética global.
2. **Contraste BA vs Línea Base en TFD en el Límite de Significancia**:
   El contraste de TFD entre BA y Línea Base arrojó $p_{\text{raw}} = 0.0020$. Tras la corrección conservadora por comparaciones múltiples de Holm, el valor resultante es $p_{\text{Holm}} = 0.0505$, situándose exactamente en el límite marginal del umbral $\alpha=0.05$, indicando una tendencia hacia una mayor tasa de alertas falsas en BA ($0.4801$ vs $0.4776$).
3. **PSO vs BA: Solo el Recall difiere significativamente**:
   Al contrastar directamente ambas metaheurísticas, únicamente el Recall de ataque resulta estadísticamente significativo ($p_{\text{Holm}} = 0.0215$, $rb\_r = -0.673$), favoreciendo a BA frente a PSO ($0.6993$ vs $0.6478$). En todas las demás dimensiones ($F_1$ Macro, Accuracy, TFP, TFD), los valores $p_{\text{Holm}} \ge 0.2952$ confirman que **no existen diferencias estadísticamente detectables** entre ambos algoritmos en particiones *i.i.d.*.
4. **Comparación con el Control Aleatorio**:
   Para Bat Algorithm (BA), todos los contrastes frente al Control Aleatorio arrojaron $p_{\text{Holm}} = 1.0000$. En rigor estadístico, esto indica que **no se detectó diferencia estadísticamente significativa** frente a una selección azarosa de variables de igual cardinalidad, aunque un valor $p=1.0$ no constituye una demostración formal de equivalencia.

---

## 4. Hipótesis Explicativa sobre la TFP en H.E.2: Mecanismo de Sesgo y Matrices de Confusión

Para dilucidar por qué PSO exhibe una reducción nominal de falsas alarmas (TFP $-21.16\%$) que coincide con una caída en la detección de ataques (Recall $-8.98\%$), se formula como **hipótesis explicativa** un desplazamiento sistemático de las predicciones hacia la clase mayoritaria (tráfico benigno).

### A. Función de Aptitud (Fitness)
La función de optimización empleada en la búsqueda bioinspirada se definió como:
$$\text{Fitness} = 0.95 \cdot F_1\text{ Macro} + 0.05 \cdot \text{RD}$$
donde $F_1\text{ Macro} = \frac{1}{2}\left(F_1^{\text{ataque}} + F_1^{\text{benigno}}\right)$.
Dado que el conjunto de prueba se compone de 694.647 flujos benignos ($77.18\%$) y 205.353 flujos de ataque ($22.82\%$), el componente $F_1^{\text{benigno}} = \frac{2 \cdot TN}{2 \cdot TN + FP + FN}$ se beneficia fuertemente de cualquier incremento en Verdaderos Negativos ($TN$) derivado de predecir tráfico benigno.

### B. Evidencia Empírica: Matrices de Confusión Agregadas ($N=30$ corridas, 27M flujos evaluados)

| Brazo | TP Total | FN Total | TN Total | FP Total | TP Promedio / corrida | FN Promedio / corrida | TN Promedio / corrida | FP Promedio / corrida |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (500k)** | 4.384.603 | 1.775.987 | 16.830.930 | 4.008.480 | 146.153,4 | 59.199,6 | 561.031,0 | 133.616,0 |
| **Control Aleatorio** | 4.355.966 | 1.804.624 | 16.540.040 | 4.299.370 | 145.198,9 | 60.154,1 | 551.334,7 | 143.312,3 |
| **PSO** | 3.990.706 | 2.169.884 | 17.679.185 | 3.160.225 | 133.023,5 | 72.329,5 | 589.306,2 | 105.340,8 |
| **BA** | 4.308.301 | 1.852.289 | 16.858.425 | 3.980.985 | 143.610,0 | 61.743,0 | 561.947,5 | 132.699,5 |

### Interpretación del Mecanismo:
Al contrastar **PSO vs Baseline-500k**:
- PSO reduce los Falsos Positivos promedio de $133.616$ a $105.341$ (un ahorro neto de **$28.275$ falsas alarmas** por corrida).
- Simultáneamente, los Falsos Negativos promedio se incrementan de $59.200$ a $72.330$ (**$13.130$ ataques reales no detectados** adicionales por corrida).
- Los Verdaderos Negativos ($TN$) crecen exactamente en la misma magnitud en que disminuyen los $FP$ ($+28.275$), aumentando la exactitud global (Accuracy) en $+1.68$ puntos porcentuales ($\frac{28.275 - 13.130}{900.000} \approx +0.0168$).

Por consiguiente, la evidencia de las matrices de confusión respalda la **hipótesis de que la reducción de TFP no constituye una depuración selectiva de anomalías sintácticas**, sino un sesgo conservador inducido por la poda de variables que empuja el umbral de decisión a clasificar más muestras como benignas a expensas de la detección de intrusiones.

---

## 5. Validación Temporal Leave-One-Capture-Out (LOO) sin Fuga Detectada

> **Protocolo Experimental Estricto sin Fuga**:
> Para evaluar la capacidad de generalización temporal ante deriva de concepto (*concept drift*), la búsqueda metaheurística (PSO y BA) se ejecutó de forma independiente y aislada sobre una submuestra del conjunto de entrenamiento de cada pliegue, **sin exponer jamás los datos de la captura dejada fuera** durante la fase de selección.
> - **Pliegue 1**: Test = `20210616` (449k flujos) | Train = `20210210` + `20211013` (500k flujos).
> - **Pliegue 2**: Test = `20210210` (300k flujos) | Train = `20210616` + `20211013` (500k flujos).
> - **Pliegue 3**: Test = `20211013` (150k flujos) | Train = `20210616` + `20210210` (500k flujos).

| Brazo | N° Feat Prom. | $F_1$ Macro | Recall Ataque | Accuracy | TFP (FPR) | TFD (FDR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clasificador Trivial** *(Zero-R)* | $0.0$ | $0.4361 \pm 0.0159$ | $0.0000 \pm 0.0000$ | **$0.7743 \pm 0.0494$** | **$0.0000 \pm 0.0000$** | — |
| **Línea Base (50 feat)** | $50.0$ | $0.4859 \pm 0.0356$ | $0.1977 \pm 0.1859$ | $0.6783 \pm 0.1451$ | $0.1711 \pm 0.2490$ | $0.5851 \pm 0.2096$ |
| **Control Aleatorio** ($k=21$) | $21.0$ | $0.4795 \pm 0.0293$ | $0.3144 \pm 0.2148$ | $0.6130 \pm 0.1572$ | $0.3090 \pm 0.2514$ | $0.6829 \pm 0.1608$ |
| **PSO-Fold** | **$17.3 \pm 4.7$** | **$0.5222 \pm 0.0827$** | $0.2646 \pm 0.1642$ | $0.6815 \pm 0.1473$ | $0.1928 \pm 0.2332$ | **$0.5756 \pm 0.2195$** |
| **BA-Fold** | $17.7 \pm 3.5$ | $0.4715 \pm 0.0357$ | **$0.4070 \pm 0.3255$** | $0.5921 \pm 0.1826$ | $0.3737 \pm 0.3270$ | $0.6535 \pm 0.1941$ |

### Hallazgos Rigurosos de la Validación Temporal:
1. **Inconsistencia del Accuracy frente al Clasificador Trivial**:
   La exactitud (Accuracy) de todos los modelos supervisados ($59.2\%$ a $68.2\%$) queda por debajo del clasificador trivial Zero-R ($77.43\%$). Esto confirma que en presencia de desbalance severo de clases y deriva temporal, Accuracy es una métrica engañosa e inadecuada.
2. **Desempeño de PSO y Limitaciones Muestrales en LOO**:
   PSO-Fold promedia un $F_1$ Macro de $0.5222$, lo que representa nominalmente $+0.0363$ (+3.63 pp) sobre la línea base completa ($0.4859$). No obstante, **la desviación estándar inter-pliegue de PSO ($\pm 0.0827$) supera con creces la ganancia observada**, y al contar únicamente con $N=3$ capturas históricas, no es factible aplicar pruebas estadísticas de hipótesis con potencia suficiente.
3. **Colapso de Bat Algorithm (BA-Fold)**:
   BA-Fold obtiene un $F_1$ Macro medio de $0.4715$, ubicándose **por debajo tanto de la Línea Base ($0.4859$) como del Control Aleatorio ($0.4795$)**. Aunque eleva el Recall al $40.7\%$, sufre una TFP crítica ($37.4\%$) y una tasa de falsas alarmas sobre alertas emitidas (TFD) del $65.4\%$.
4. **Sobreestimación en Particiones *i.i.d.***:
   La disparidad entre los resultados en partición aleatoria ($F_1 \approx 0.73$) y en validación temporal out-of-domain ($F_1 \approx 0.47 - 0.52$) evidencia que la división i.i.d. sobreestima drásticamente la capacidad de generalización del IDS.

---

## 6. Veredicto Oficial y Corregido de las Hipótesis de Investigación

| Hipótesis | Enunciado Operacional | Veredicto Experimental | Justificación Empírica y Estadística Rigurosa |
| :--- | :--- | :---: | :--- |
| **H.E.1** | *PSO y BA mejoran la eficacia de detección ($F_1$ Macro)* | **NO RESPALDADA** | En partición *i.i.d.* pareada ($N=30$), PSO no difiere significativamente de la línea base ($p_{\text{Holm}} = 1.0000$, $\Delta = +0.86\%$) y sufre una caída crítica en Recall de $-8.98\%$ ($p < 0.0001$). BA resulta significativamente inferior a la línea base en $F_1$ ($p < 0.0001$), Recall ($p < 0.0001$) y Accuracy ($p = 0.0485$). En validación LOO, la ventaja nominal de PSO ($+3.63$ pp) no cuenta con prueba estadística ($N=3$) y su dispersión ($\pm 0.083$) supera la ganancia, mientras que BA-Fold ($0.4715$) rinde por debajo de la línea base y del azar. |
| **H.E.2** | *PSO y BA reducen la Tasa de Falsos Positivos (TFP)* | **NO RESPALDADA** | La reducción nominal de TFP observada en PSO ($-21.16\%$) no alcanza significancia estadística tras el ajuste por multiplicidad de Holm ($p_{\text{Holm}} = 0.5264$). En BA, la TFP no experimenta variación ($p_{\text{Holm}} = 0.4648$). La evidencia de las matrices de confusión indica como posible explicación que la menor tasa de alarmas en PSO proviene del sesgo hacia la clase mayoritaria a expensas de omitir 13.130 ataques por corrida. |
| **H.E.3** | *Mejora en la eficiencia computacional* | **PARCIALMENTE RESPALDADA / CONDICIONADA** | En el benchmark pareado formal ($N=30$), la reducción de dimensionalidad reduce significativamente el tiempo de entrenamiento frente a la Línea Base: $+36.07\%$ en PSO ($16.91\text{ s}$ vs $26.45\text{ s}$, $p = 0.00002$) y $+26.47\%$ en BA ($19.45\text{ s}$, $p = 0.0071$). La inferencia mejora de forma estadísticamente significativa pero modesta: $+10.75\%$ en PSO ($2.014$ vs $2.256\ \mu\text{s/flujo}$, $p = 0.00015$) y $+3.69\%$ en BA ($2.173\ \mu\text{s}$, $p = 0.0026$). No obstante: (1) **PSO no supera al Control Aleatorio en tiempo de entrenamiento** ($16.91\text{ s}$ vs $16.73\text{ s}$, $p = 0.0920$), evidenciando que el ahorro proviene de la reducción del espacio y no de la optimización; (2) La búsqueda inicial tiene un costo promedio de $307.8\text{ s}$ en PSO (mediana $59.1\text{ s}$) y $130.2\text{ s}$ en BA; con un ahorro por entrenamiento de $9.54\text{ s}$ en PSO y $7.00\text{ s}$ en BA, la búsqueda se amortiza recién tras **32 ciclos de reentrenamiento** (usando la media) o **6 ciclos** (usando la mediana) en PSO, y **18 a 19 ciclos** en BA; o bien tras clasificar más de **1.270 millones de flujos** en inferencia. Por ende, la ventaja está condicionada a entornos de alto reentrenamiento periódico. |
| **H.G. / H4** | *La selección con metaheurísticas mejora el desempeño frente a la línea base* | **NO RESPALDADA** | La hipótesis general de mejora queda rechazada al comprobarse que ningún algoritmo bioinspirado supera a la línea base en $F_1$ Macro y que ambos deterioran el Recall. El test de Friedman resulta significativo ($\chi_F^2 = 25.32$ en $F_1$, $p < 0.0001$; $\chi_F^2 = 10.67$ en TFP, $p = 0.0137$; $\chi_F^2 = 55.40$ en Recall; $\chi_F^2 = 20.28$ en Acc; $\chi_F^2 = 21.40$ en TFD) debido al predominio de la Línea Base. Al comparar PSO vs BA, no hay diferencias en $F_1$, Acc, TFP ni TFD ($p_{\text{Holm}} \ge 0.2952$), siendo el Recall la única métrica con divergencia significativa ($p_{\text{Holm}} = 0.0215$). |

---

## 7. Limitaciones Metodológicas Declaradas

1. **Sobreestimación por Partición *i.i.d.***:
   La evaluación mediante partición estratificada aleatoria convencional (*i.i.d.*) sobreestima la eficacia práctica del IDS ($F_1 \approx 0.73$ frente a $0.47 - 0.52$ en validación temporal fuera de dominio). La proximidad temporal de paquetes en trazas continuas facilita el aprendizaje de correlaciones espurias de sesión.
2. **Tamaño Muestral de Validación Temporal ($N=3$)**:
   La evaluación LOO se restringe a tres capturas diarias del año 2021 debido a la disponibilidad y costo computacional de extracción de flujos crudos desde PCAP, impidiendo la inferencia estadística no paramétrica en este escenario.
3. **Amortización del Costo de Búsqueda Metaheurística**:
   Aunque el benchmark de $N=30$ corridas demostró reducciones estadísticamente significativas en entrenamiento e inferencia respecto a la línea base completa, el costo computacional de búsqueda ($T_{\text{search}}$) demanda decenas de reentrenamientos para amortizarse, limitando su viabilidad en infraestructuras donde el clasificador deba reentrenarse con baja frecuencia.
