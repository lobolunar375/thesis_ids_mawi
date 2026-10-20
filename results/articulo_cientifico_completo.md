# Evaluación Crítica de Metaheurísticas de Inteligencia de Enjambre (PSO y Bat Algorithm) para la Selección de Características en Detección de Intrusiones sobre Tráfico MAWILab: Un Enfoque con Controles Empíricos y Validación Temporal

**Autores**: Berrocal Barrientos Jorge Luis, Neira Herrera Jhojan  
**Institución**: Facultad de Ingeniería Eléctrica y Electrónica, Universidad Nacional de Ingeniería (UNI)  
**Curso**: Inteligencia Artificial II — Sección M (2026)  
**Profesor**: Dr. Yury Oscar Tello Canchapoma  

---

## Resumen

La selección de características en Sistemas de Detección de Intrusiones (IDS) busca reducir la dimensionalidad del tráfico de red manteniendo o mejorando la eficacia de detección. Numerosos trabajos en la literatura afirman que los algoritmos de inteligencia de enjambre (*Swarm Intelligence*) superan sistemáticamente a los modelos entrenados con el espacio completo de variables. En este trabajo evaluamos de forma empíricamente rigurosa dos metaheurísticas bioinspiradas con enfoque wrapper: **Optimización por Enjambre de Partículas (PSO)** y el **Algoritmo del Murciélago (Bat Algorithm, BA)**, acoplados a un clasificador Random Forest sobre tráfico real del repositorio **MAWILab 2021**. A diferencia de estudios convencionales, establecemos cinco salvaguardas metodológicas fundamentales: (1) comparación en estricta igualdad de datos de entrenamiento (**Línea Base de 500.000 flujos**), (2) inclusión de un **Control Aleatorio** ($k=21$ características al azar), (3) contraste contra el umbral de un **Clasificador Trivial** (Zero-R), (4) evaluación de generalización temporal mediante validación cruzada **Leave-One-Capture-Out (LOO) sin fuga de información detectada** sobre 3 capturas independientes, y (5) control de la tasa de error por familia mediante la corrección de **Holm-Bonferroni** junto al cálculo del tamaño del efecto (**Rank-Biserial $r$**).

Los resultados experimentales refutan la hipótesis de superioridad en particiones aleatorias homogéneas: en evaluación *i.i.d.*, **PSO y BA no superan a la línea base en $F_1$ macro** de forma estadísticamente significativa tras la corrección de Holm ($p_{\text{Holm}} = 1.0000$ para PSO; BA resulta significativamente inferior en $F_1$, Recall y Accuracy, y en todas sus métricas no se detectan diferencias estadísticamente significativas frente al control aleatorio). Asimismo, la reducción aparente de falsas alarmas observada en PSO se asocia empíricamente a un desplazamiento hacia la clase mayoritaria que induce un deterioro crítico en el Recall de ataque ($-8.98\%$, $p < 0.0001$, perdiendo $13.130$ detecciones de ataque por corrida). En la validación temporal Leave-One-Capture-Out, la línea base completa sufre una severa degradación ($F_1 = 0.4859$), mientras que PSO promedia $F_1 = 0.5222$, aunque con una dispersión inter-captura ($\pm 0.0827$) que excede la ganancia y sin alcanzar significancia estadística en una muestra de $N=3$ trazas; por su parte, BA-Fold ($F_1 = 0.4715$) rinde por debajo de la línea base y del control aleatorio. Adicionalmente, todos los modelos supervisados obtienen una exactitud inferior a la regla trivial Zero-R ($77.43\%$), y la partición *i.i.d.* sobreestima drásticamente la capacidad predictiva ($F_1 \approx 0.73$ frente a $\sim 0.49$ en LOO). Se concluye que las hipótesis de mejora en eficacia, reducción probada de falsas alarmas, eficiencia computacional y superioridad global de las metaheurísticas no quedan respaldadas empíricamente.

**Palabras clave**: Detección de Intrusiones, Inteligencia de Enjambre, PSO, Bat Algorithm, Selección de Características, MAWILab, Control Aleatorio, Deriva Temporal, Leave-One-Capture-Out.

---

## 1. Introducción

Los sistemas de detección de intrusiones en red basados en anomalías analizan flujos de comunicación descritos por decenas o cientos de atributos estadísticos extraídos de las cabeceras TCP/IP. La alta dimensionalidad introduce redundancia, ruido y una carga computacional prohibitiva para despliegues en tiempo real. Para mitigar este problema, la selección de características (*Feature Selection*, FS) con enfoque wrapper emplea algoritmos de optimización combinatoria para aislar un subconjunto óptimo de variables.

Dentro de las metaheurísticas bioinspiradas, la **Optimización por Enjambre de Partículas (PSO)** (Kennedy & Eberhart, 1995) y el **Algoritmo del Murciélago (Bat Algorithm, BA)** (Yang, 2010) han ganado amplia difusión en ciberseguridad (Thakkar & Lohiya, 2020; Ghanem et al., 2022). No obstante, una revisión crítica de la literatura revela problemas metodológicos recurrentes:
1. **Asimetría en las condiciones de entrenamiento**: Comparar modelos reducidos frente a modelos de referencia entrenados con diferentes volúmenes de datos, mezclando el impacto de la reducción de atributos con el tamaño muestral.
2. **Ausencia de controles aleatorios**: Atribuir las propiedades de un subconjunto reducido a la "inteligencia" del algoritmo sin verificar si una selección aleatoria de idéntica cardinalidad alcanza un rendimiento comparable.
3. **Optimismo espurio por partición aleatoria (*i.i.d. leakage*)**: Evaluar mediante partición aleatoria sobre tráfico continuo que comparte sesiones y contextos temporales, lo cual encubre la deriva de concepto (*concept drift*).
4. **Omisión de clasificadores triviales**: No contrastar las tasas de acierto contra el umbral basal de predecir sistemáticamente la clase mayoritaria en datasets fuertemente desbalanceados.

El presente estudio aborda estas limitaciones mediante un diseño experimental riguroso que evalúa PSO y BA en estricta paridad muestral frente a una línea base completa de 500.000 flujos, un control aleatorio pareado, un clasificador trivial Zero-R y una validación temporal Leave-One-Capture-Out sin fuga de información detectada.

---

## 2. Metodología Experimental

### 2.1. Corpus de Tráfico MAWILab y Protocolo de Muestreo

Se emplearon las trazas de tráfico real del repositorio internacional **MAWILab** (Measurement and Analysis on the WIDE Internet) correspondientes al año 2021. El universo total procesado a partir de los archivos crudos PCAP totalizó **19.089.344 flujos de red**:
- Captura `20210616`: 9.486.699 flujos.
- Captura `20210210`: 7.365.596 flujos. *Declaración metodológica*: Esta captura fue procesada y extraída parcialmente desde su archivo PCAP original debido a restricciones computacionales de tiempo en la disección exhaustiva de paquetes.
- Captura `20211013`: 2.237.049 flujos.

**Herramienta de Extracción de Características**: Para procesar las capturas masivas se desarrolló una herramienta propia en Python ([`extract_fixed.py`](file:///c:/Users/Jhojan/Documents/ciclo_26_2/IA2/Parcial/extract_fixed.py)) utilizando la biblioteca `dpkt`. La herramienta ensambla flujos bidireccionales según la 5-tupla IP con ventanas de inactividad de 120 s (*idle timeout*) y expiración forzada de 300 s (*hard timeout*), extrayendo 50 atributos estadísticos directos de cabeceras IP y TCP (distribución de longitudes, intervalos de inter-arribo, banderas TCP, tasas de transmisión). Esta solución superó las fallas por agotamiento de memoria experimentadas por herramientas convencionales (como CICFlowMeter o SiLK) ante volúmenes de decenas de gigabytes.

Para permitir una evaluación estadísticamente representativa y reproducible dentro de los límites de hardware, se extrajo una muestra estratificada de **3.000.000 de flujos** (15.7% del universo total), conservando rigurosamente la tasa de ataque natural de cada traza:
- `20210616`: 1.500.000 flujos (15.8% de su volumen).
- `20210210`: 1.000.000 flujos (13.6% de su volumen).
- `20211013`: 500.000 flujos (22.4% de su volumen).

El corpus consolidado presenta una distribución desbalanceada de **22.82% de tráfico malicioso (ataques)** y **77.18% de tráfico benigno**. La partición global se dividió en 70% entrenamiento (**2.100.000 flujos**) y 30% prueba (**900.000 flujos**).

### 2.2. Paridad Muestral y Brazos Experimentales

A fin de evitar distorsiones por tamaño muestral, todos los modelos evaluados en la fase principal se entrenaron estrictamente sobre la misma submuestra estratificada de **500.000 flujos** (`seed=42`) y se validaron sobre el conjunto independiente de **900.000 flujos de prueba**:

1. **Clasificador Trivial (Zero-R)**: Asigna la clase mayoritaria (Benigno) a todas las observaciones. Establece el umbral mínimo que cualquier clasificador supervisado debe superar.
2. **Línea Base 500k (BL500k)**: Random Forest (100 árboles, profundidad máxima 15, `class_weight='balanced'`) entrenado con los **50 atributos completos**. Constituye la **referencia formal** para todas las comparaciones estadísticas.
3. **Control Aleatorio (RAND)**: Random Forest entrenado seleccionando $k=21$ características al azar en cada una de las 30 semillas independientes (cardinalidad equivalente al promedio de PSO y BA).
4. **PSO Wrapper**: Enjambre de 10 partículas, 15 iteraciones, pesos cognitivo/social $c_1=1.5$, $c_2=1.5$, inercia $w=0.7$, velocidad máxima $v_{\text{max}} = 6.0$, mapeo binario mediante función sigmoide $S(v) = 1/(1+e^{-v})$.
5. **Bat Algorithm (BA) Wrapper**: Población de 10 individuos, 15 iteraciones, frecuencias en $[0, 2]$, decaimiento de sonoridad $\alpha=0.9$, incremento de emisión $\gamma=0.9$, sonoridad inicial $A_0=1.0$, tasa de pulso $r_0=0.5$.

La función de aptitud (*fitness*) en ambos algoritmos wrapper se definió como:
$$ \text{Fitness} = 0.95 \cdot F_{1,\text{macro}} + 0.05 \cdot \left(1 - \frac{|S|}{D}\right) $$
donde $|S|$ es la cantidad de características seleccionadas y $D=50$. La optimización se efectuó sobre un proxy de entrenamiento de 10.000 flujos y la máscara óptima de cada una de las 30 repeticiones fue reentrenada sobre el corpus completo de 500k flujos.

---

## 3. Resultados Experimentales en Partición *i.i.d.*

### 3.1. Eficacia de Clasificación ($n=30$ corridas pareadas)

La Tabla 1 consolida las métricas de evaluación obtenidas sobre el conjunto de prueba independiente de 900.000 flujos a lo largo de las 30 repeticiones pareadas por semilla.

**Tabla 1**: Métricas de rendimiento experimental (*i.i.d.* 900k flujos test, train 500k, $n=30$ corridas independientes pareadas).

| Brazo | $F_1$ Macro | Recall Ataque | Accuracy | TFP (FPR) | TFD (FDR) | N° Feat | RD% | Costo Búsqueda ($T_{\text{search}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clasificador Trivial** *(Zero-R)* | $0.4356 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.7718 \pm 0.0000$ | **$0.0000 \pm 0.0000$** | — | $0.0$ | — | $0.0\text{ s}$ |
| **Línea Base (500k)** *(Ref.)* | $0.7280 \pm 0.0001$ | **$0.7117 \pm 0.0004$** | $0.7858 \pm 0.0002$ | $0.1924 \pm 0.0003$ | $0.4776 \pm 0.0003$ | $50.0 \pm 0.0$ | $0.0\%$ | $0.0\text{ s}$ |
| **Control Aleatorio** ($k=21$) | $0.7161 \pm 0.0373$ | $0.7071 \pm 0.1036$ | $0.7739 \pm 0.0553$ | $0.2063 \pm 0.1020$ | $0.4716 \pm 0.0873$ | $21.0 \pm 0.0$ | $58.0\%$ | $0.0\text{ s}$ |
| **PSO** | **$0.7342 \pm 0.0155$** | $0.6478 \pm 0.0764$ | **$0.8026 \pm 0.0281$** | **$0.1516 \pm 0.0588$** | **$0.4202 \pm 0.0923$** | **$18.5 \pm 2.6$** | **$63.1\%$** | Med: **$59.1\text{ s}$** (Media: $307.8\text{ s}$) |
| **BA** | $0.7244 \pm 0.0016$ | $0.6993 \pm 0.0175$ | $0.7840 \pm 0.0029$ | $0.1910 \pm 0.0086$ | $0.4801 \pm 0.0058$ | $24.2 \pm 3.0$ | $51.5\%$ | Med: **$122.6\text{ s}$** (Media: $130.2\text{ s}$) |

*Nota metodológica*: Para el Clasificador Trivial (Zero-R), la Tasa de Falsos Descubrimientos ($\text{TFD} = \frac{FP}{FP+TP} = \frac{0}{0}$) no está definida y la reducción de dimensionalidad no aplica, denotándose con `—`.

### 3.2. Test Global de Friedman y Rangos Medios ($k=4$ brazos, $N=30$ bloques, $df=3$)

El test no paramétrico de Friedman aplicado sobre los 4 tratamientos a lo largo de los 30 bloques de evaluación evidenció heterogeneidad global en las 5 dimensiones evaluadas:
- **$F_1$ Macro**: $\chi_F^2(3) = 25.32$, $p < 0.0001$ ($1.32 \times 10^{-5}$)
- **Recall Ataque**: $\chi_F^2(3) = 55.40$, $p < 0.0001$ ($5.64 \times 10^{-12}$)
- **Accuracy**: $\chi_F^2(3) = 20.28$, $p = 0.0001$ ($1.48 \times 10^{-4}$)
- **TFP (Falsas Alarmas)**: $\chi_F^2(3) = 10.67$, $p = 0.0137$
- **TFD (Alertas Falsas)**: $\chi_F^2(3) = 21.40$, $p = 0.0001$ ($8.69 \times 10^{-5}$)

**Tabla 2**: Rangos medios del test de Friedman ($k=4$ tratamientos, $N=30$ bloques; rango 1 = mejor desempeño).

| Brazo | Rango $F_1$ Macro | Rango Recall Ataque | Rango Accuracy | Rango TFP | Rango TFD |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline-500k** | **1.53** *(1°)* | **1.13** *(1°)* | **1.77** *(1°)* | **2.10** *(1°)* | **1.77** *(1°)* |
| **PSO** | 2.50 *(2°)* | 3.57 *(4°)* | 2.27 *(2°)* | 2.20 *(2°)* | 2.23 *(2°)* |
| **BA** | 2.93 *(3°)* | 2.77 *(3°)* | 2.80 *(3°)* | 2.67 *(3°)* | 2.80 *(3°)* |
| **Control Aleatorio** ($k=21$) | 3.03 *(4°)* | 2.53 *(2°)* | 3.17 *(4°)* | 3.03 *(4°)* | 3.20 *(4°)* |

La significancia del test de Friedman se explica predominantemente por el **dominio de la Línea Base de 50 características** (rango medio 1.53 en $F_1$ y 1.13 en Recall). Un Friedman significativo no prueba que PSO y BA superen la línea base ni que posean dinámicas cualitativamente distintas entre sí, lo cual requiere el análisis de contrastes pareados post-hoc.

### 3.3. Matriz Exhaustiva de Contrastes Pareados ($30$ contrastes, Wilcoxon + Corrección de Holm)

La Tabla 3 expone los 30 contrastes evaluados ($6 \text{ pares} \times 5 \text{ métricas}$, $N=30$) reportando medias, medianas, $\Delta$ absoluta, $\Delta\%$ relativa, estadístico de Wilcoxon $W$, valor $p_{\text{raw}}$, tamaño del efecto rank-biserial $rb\_r$, valor ajustado $p_{\text{Holm}}$ y veredicto formal de significancia ($\alpha=0.05$):

**Tabla 3**: Contrastes pareados de Wilcoxon con ajuste de Holm-Bonferroni y tamaño del efecto rank-biserial ($N=30$).

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
| **TFD (FDR)** | BA vs BL500k | 0.4801 | 0.4812 | 0.4776 | 0.4776 | +0.0025 | +0.52% | 87.0 | 0.0020 | +0.626 | **0.0505** | No *(al borde)* |
| **TFD (FDR)** | RAND vs BL500k | 0.4716 | 0.4840 | 0.4776 | 0.4776 | -0.0060 | -1.25% | 158.0 | 0.1294 | +0.320 | 1.0000 | No |
| **TFD (FDR)** | PSO vs BA | 0.4202 | 0.4797 | 0.4801 | 0.4812 | -0.0599 | -12.47% | 113.0 | 0.0128 | -0.514 | 0.2952 | No |
| **TFD (FDR)** | PSO vs RAND | 0.4202 | 0.4797 | 0.4716 | 0.4840 | -0.0514 | -10.90% | 128.0 | 0.0310 | -0.449 | 0.4648 | No |
| **TFD (FDR)** | BA vs RAND | 0.4801 | 0.4812 | 0.4716 | 0.4840 | +0.0085 | +1.79% | 180.0 | 0.2894 | -0.226 | 1.0000 | No |

**Aspectos Relevantes de los Contrastes**:
1. *Discrepancia de Signo en BA vs BL500k (TFP)*: Aunque la media de BA es ligeramente menor a la de la línea base ($0.1910$ vs $0.1924$, $\Delta = -0.0013$), la mediana de BA es mayor ($0.1931$ vs $0.1924$) y el tamaño de efecto rank-biserial es positivo ($rb\_r = +0.445$). Esto obedece a una distribución asimétrica: en 20 de las 30 semillas pareadas, BA tuvo una TFP superior al baseline, produciendo mayor suma de rangos positivos en Wilcoxon, mientras que un número reducido de corridas con TFP marcadamente baja arrastró la media aritmética.
2. *Contraste BA vs BL500k en TFD*: Arroja un $p_{\text{raw}} = 0.0020$ que tras la corrección de Holm se ajusta a $p_{\text{Holm}} = 0.0505$, situándose exactamente en el límite de significancia estadística e indicando una tendencia hacia una mayor proporción de alertas falsas en BA ($0.4801$ vs $0.4776$).
3. *Diferenciación PSO vs BA*: Solo el Recall de ataque resulta estadísticamente significativo ($p_{\text{Holm}} = 0.0215$, $rb\_r = -0.673$), favoreciendo a BA. En $F_1$, Accuracy, TFP y TFD no se registran diferencias significativas ($p_{\text{Holm}} \ge 0.2952$).
4. *BA frente al Control Aleatorio*: En todas las métricas, $p_{\text{Holm}} = 1.0000$. En rigor estadístico, no se detectó diferencia significativa frente a una selección aleatoria, sin que ello constituya demostración de equivalencia.

---

## 4. Validación Temporal Leave-One-Capture-Out (LOO) sin Fuga Detectada

Para contrastar la validez de los modelos ante el fenómeno de deriva de concepto (*concept drift*), se diseñó un esquema **Leave-One-Capture-Out (LOO)** donde la selección de variables (PSO y BA) se confinó estrictamente dentro de la partición de entrenamiento de cada pliegue, **sin exponer jamás la captura de prueba**:
- **Pliegue 1**: Test en `20210616` (449k flujos) | Train en `20210210` + `20211013` (500k).
- **Pliegue 2**: Test en `20210210` (300k flujos) | Train en `20210616` + `20211013` (500k).
- **Pliegue 3**: Test en `20211013` (150k flujos) | Train en `20210616` + `20210210` (500k).

**Tabla 4**: Resultados promediados en validación temporal Leave-One-Capture-Out ($N=3$ capturas independientes).

| Brazo | N° Feat Prom. | $F_1$ Macro | Recall Ataque | Accuracy | TFP (FPR) | TFD (FDR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clasificador Trivial** *(Zero-R)* | $0.0$ | $0.4361 \pm 0.0159$ | $0.0000 \pm 0.0000$ | **$0.7743 \pm 0.0494$** | **$0.0000 \pm 0.0000$** | — |
| **Línea Base (50 feat)** | $50.0$ | $0.4859 \pm 0.0356$ | $0.1977 \pm 0.1859$ | $0.6783 \pm 0.1451$ | $0.1711 \pm 0.2490$ | $0.5851 \pm 0.2096$ |
| **Control Aleatorio** ($k=21$) | $21.0$ | $0.4795 \pm 0.0293$ | $0.3144 \pm 0.2148$ | $0.6130 \pm 0.1572$ | $0.3090 \pm 0.2514$ | $0.6829 \pm 0.1608$ |
| **PSO-Fold** | **$17.3 \pm 4.7$** | **$0.5222 \pm 0.0827$** | $0.2646 \pm 0.1642$ | $0.6815 \pm 0.1473$ | $0.1928 \pm 0.2332$ | **$0.5756 \pm 0.2195$** |
| **BA-Fold** | $17.7 \pm 3.5$ | $0.4715 \pm 0.0357$ | **$0.4070 \pm 0.3255$** | $0.5921 \pm 0.1826$ | $0.3737 \pm 0.3270$ | $0.6535 \pm 0.1941$ |

### Hallazgos de la Validación Temporal:
1. **Inconsistencia del Accuracy frente a la Regla Trivial**: La exactitud de todos los modelos supervisados ($59.2\%$ a $68.2\%$) queda marcadamente por debajo de la regla trivial Zero-R ($77.43\%$), confirmando que Accuracy es una métrica distorsionada en tráfico desbalanceado con deriva.
2. **Desempeño y Varianza de PSO-Fold**: PSO alcanza un $F_1$ Macro medio de $0.5222$ (+3.63 pp sobre la línea base de 50 variables). No obstante, la desviación estándar inter-captura ($\pm 0.0827$) es mayor que la ganancia media, y al tratarse de únicamente 3 pliegues temporales, no es posible aplicar pruebas estadísticas no paramétricas con potencia inferencial suficiente.
3. **Colapso de Bat Algorithm (BA-Fold)**: BA-Fold ($F_1 = 0.4715$) se ubica por debajo tanto de la Línea Base ($0.4859$) como del Control Aleatorio ($0.4795$), sufriendo una TFP del $37.4\%$ y una tasa de falsas alarmas sobre alertas emitidas (TFD) del $65.4\%$.
4. **Sobreestimación por Partición *i.i.d.***: La brecha entre el rendimiento en partición aleatoria ($F_1 \approx 0.73$) y en validación out-of-domain ($F_1 \approx 0.47 - 0.52$) confirma que la división *i.i.d.* convencional sobreestima drásticamente la generalización operativa del IDS.

---

## 5. Discusión Crítica

### 5.1. Hipótesis Explicativa sobre la Tasa de Falsos Positivos y Matrices de Confusión
La aparente reducción de TFP en PSO ($-21.16\%$) coincide con un desplome del Recall de ataque ($-8.98\%$, $p < 0.0001$). Para examinar el mecanismo subyacente, se plantea como **hipótesis explicativa** que la función de aptitud empleada:
$$ \text{Fitness} = 0.95 \cdot F_{1,\text{macro}} + 0.05 \cdot \text{RD} $$
al promediar simétricamente el $F_1$ de ambas clases en un corpus con $77.18\%$ de flujos benignos, premia indirectamente a los clasificadores conservadores. El $F_1$ benigno ($2 \cdot TN / (2 \cdot TN + FP + FN)$) se incrementa de forma notable al predecir la clase mayoritaria.

La inspección de las **matrices de confusión agregadas** ($N=30$ corridas, 27 millones de flujos evaluados) aporta sustento cuantitativo a esta hipótesis:
- **Baseline-500k**: Promedia $146.153$ TP, $59.200$ FN, $561.031$ TN y $133.616$ FP por corrida.
- **PSO**: Promedia $133.024$ TP, $72.330$ FN, $589.306$ TN y $105.341$ FP por corrida.
- **Balance Operativo**: PSO reduce los falsos positivos en **$28.275$ flujos**, pero simultáneamente incrementa los falsos negativos en **$13.130$ ataques omitidos** por corrida. Los $28.275$ flujos adicionales clasificados como benignos incrementan los verdaderos negativos exactamente en esa cuantía, elevando la Accuracy neta en $+1.68$ pp ($\frac{28.275 - 13.130}{900.000} \approx +0.0168$).

Por consiguiente, la evidencia sugiere como posible explicación que la menor tasa de alarmas en PSO no obedece a un filtrado selectivo de firmas espurias, sino a un desplazamiento del umbral de decisión hacia la clase mayoritaria.

### 5.2. Estabilidad de Atributos y Sesgo del Etiquetador
Ningún atributo superó el 70% de recurrencia en PSO. Las características más persistentes correspondieron a banderas TCP (`fwd_syn_cnt`, `fwd_fin_cnt`, `bwd_rst_cnt`, `bwd_ack_cnt`). Dado que las etiquetas de MAWILab provienen del consenso heurístico de detectores basados en umbrales de banderas y tasas de flujo, se formula la hipótesis de que las metaheurísticas podrían estar aprendiendo a reproducir las heurísticas sintácticas del etiquetador en lugar de firmas intrínsecas de ataque.

---

## 6. Conclusiones y Veredicto Formal de Hipótesis

1. **H.E.1 (Eficacia en Detección) $\rightarrow$ NO RESPALDADA**: En partición *i.i.d.*, PSO no difiere significativamente de la línea base en $F_1$ Macro ($p_{\text{Holm}} = 1.0000$, $\Delta = +0.86\%$) y degrada críticamente el Recall de ataque ($-8.98\%$, $p < 0.0001$). BA es significativamente inferior a la línea base en $F_1$, Recall y Accuracy ($p < 0.05$), sin diferenciarse del control aleatorio. En validación LOO, la ventaja nominal de PSO ($+3.63$ pp) no cuenta con prueba estadística ($N=3$) y su desviación ($\pm 0.083$) supera la ganancia, mientras que BA-Fold rinde por debajo de la línea base y del azar.
2. **H.E.2 (Reducción de Falsas Alarmas) $\rightarrow$ NO RESPALDADA**: La reducción nominal de TFP en PSO ($-21.16\%$) no alcanza significancia estadística tras el ajuste de Holm ($p_{\text{Holm}} = 0.5264$). En BA, la TFP no muestra variación ($p_{\text{Holm}} = 0.4648$). La evidencia de las matrices de confusión plantea como posible explicación que el descenso de FP en PSO proviene del sesgo hacia la clase mayoritaria a expensas de omitir $13.130$ ataques por corrida.
3. **H.E.3 (Eficiencia Computacional) $\rightarrow$ PARCIALMENTE RESPALDADA / CONDICIONADA**: En el benchmark pareado formal ($N=30$ corridas pareadas), la poda de atributos produce un ahorro significativo en tiempo de reentrenamiento frente a la línea base: $+36.07\%$ en PSO ($16.91\text{ s}$ vs $26.45\text{ s}$, $p = 0.00002$) y $+26.47\%$ en BA ($19.45\text{ s}$, $p = 0.0071$). La latencia de inferencia mejora de forma estadísticamente significativa pero modesta: $+10.75\%$ en PSO ($2.014$ vs $2.256\ \mu\text{s/flujo}$, $p = 0.00015$) y $+3.69\%$ en BA ($2.173\ \mu\text{s}$, $p = 0.0026$). No obstante: (1) **PSO no supera al Control Aleatorio en tiempo de entrenamiento** ($16.91\text{ s}$ vs $16.73\text{ s}$, $p = 0.0920$), evidenciando que la aceleración es producto exclusivo de la reducción de dimensiones y no de la búsqueda guiada; (2) La optimización bioinspirada introduce un costo de búsqueda de $307.8\text{ s}$ en PSO (mediana $59.1\text{ s}$) y $130.2\text{ s}$ en BA; dado que el ahorro por reentrenamiento es de $9.54\text{ s}$ en PSO y $7.00\text{ s}$ en BA, la búsqueda se amortiza recién tras **32 ciclos de reentrenamiento** (usando la media) o **6 ciclos** (usando la mediana) en PSO, y **18 a 19 ciclos** en BA; o bien tras clasificar más de **1.270 millones de flujos** en inferencia. Por ende, la ganancia de eficiencia queda estrictamente condicionada a infraestructuras con reentrenamiento periódico frecuente.
4. **H.G. / H4 (Hipótesis General de Mejora) $\rightarrow$ NO RESPALDADA**: La hipótesis general de que la selección bioinspirada mejora el desempeño del IDS queda rechazada: ningún algoritmo supera a la línea base en $F_1$ en *i.i.d.* y ambos deterioran el Recall. La significancia del test global de Friedman ($\chi_F^2 = 25.32$ en $F_1$, $p < 0.0001$; $\chi_F^2 = 10.67$ en TFP, $p = 0.0137$; $\chi_F^2 = 55.40$ en Recall; $\chi_F^2 = 20.28$ en Acc; $\chi_F^2 = 21.40$ en TFD) está explicada por el predominio de la Línea Base. Al contrastar directamente PSO vs BA, solo el Recall difiere significativamente ($p_{\text{Holm}} = 0.0215$); $F_1$, Accuracy, TFP y TFD no muestran diferencias significativas ($p_{\text{Holm}} \ge 0.2952$).

---

## 7. Limitaciones de la Investigación

1. **Sobreestimación por Partición *i.i.d.***: La evaluación mediante particiones aleatorias sobrestima la eficacia operativa del IDS ($F_1 \approx 0.73$ vs $0.47 - 0.52$ en LOO) debido a la correlación serial y persistencia de sesiones en trazas continuas.
2. **Muestra Restringida de Capturas Históricas ($N=3$)**: La validación LOO se circunscribe a tres capturas del año 2021 por las altas demandas computacionales de procesamiento de paquetes crudos, limitando la aplicación de pruebas no paramétricas formales en este escenario.
3. **Amortización del Costo de Búsqueda Metaheurística**: Aunque el benchmark pareado de $N=30$ corridas demostró reducciones estadísticamente significativas en entrenamiento e inferencia respecto a la línea base completa, el costo computacional de búsqueda ($T_{\text{search}}$) demanda decenas de reentrenamientos para amortizarse, limitando su viabilidad en infraestructuras donde el clasificador deba reentrenarse con baja frecuencia.

---

## Referencias

- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1-30.
- Ghanem, W. A. H. M., Ghaleb, S. A. A., et al. (2022). Cyber intrusion detection system based on a multiobjective binary bat algorithm for feature selection. *IEEE Access*, 10, 31924-31945.
- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *Proceedings of ICNN'95*, 4, 1942-1948.
- Mirjalili, S. (2016). Dragonfly algorithm: a new meta-heuristic optimization technique. *Neural Computing and Applications*, 27(4), 1053-1073.
- Thakkar, A., & Lohiya, R. (2020). Role of swarm and evolutionary algorithms in intrusion detection system: A survey. *Swarm and Evolutionary Computation*, 53, 100631.
- Yang, X. S. (2010). A new metaheuristic bat-inspired algorithm. *NICSO 2010*, 65-74.
