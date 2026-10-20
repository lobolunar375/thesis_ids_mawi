# Selección de Características con Metaheurísticas Bio-Inspiradas (PSO y Bat Algorithm) para Detección de Intrusiones en Tráfico Real de Enlace Troncal (MAWI)

[![Python](https://img.shields.io/badge/Python-3.11.9-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-MAWI%20WIDE%202021-lightgrey.svg)](http://mawi.wide.ad.jp/mawi/)
[![Status](https://img.shields.io/badge/Reproducibility-Verified%20(N=30)-brightgreen.svg)]()

Repositorio oficial del estudio experimental sobre selección de características en esquema *wrapper* mediante **Optimización por Enjambre de Partículas (PSO)** y **Algoritmo del Murciélago (Bat Algorithm - BA)** con clasificador **Random Forest**, evaluado sobre flujos de tráfico crudo extraídos directamente del enlace troncal transpacífico **WIDE/MAWI (Punto F, 2021)** y anotados mediante **MAWILab v1.1**.

---

## 📌 Tabla de Contenidos

- [Motivación y Contexto](#-motivación-y-contexto)
- [Dataset y Datos Crudos (MAWI 2021)](#-dataset-y-datos-crudos-mawi-2021)
- [Estructura del Repositorio](#-estructura-del-repositorio)
- [Diseño Experimental y Metodología](#-diseño-experimental-y-metodología)
- [Resultados Experimentales Clave](#-resultados-experimentales-clave)
  - [1. Rendimiento de Clasificación](#1-rendimiento-de-clasificación-n--30-réplicas)
  - [2. Benchmark Formal de Tiempos y Amortización](#2-benchmark-formal-de-tiempos-y-amortización)
  - [3. Contrastes Estadísticos (Wilcoxon + Holm)](#3-contrastes-estadísticos-post-hoc-wilcoxon--holm)
  - [4. Validación Fuera de Muestra (Leave-One-Capture-Out)](#4-validación-leave-one-capture-out-loo)
- [Veredicto de Hipótesis](#-veredicto-de-hipótesis)
- [Guía de Reproducción Rápida](#-guía-de-reproducción-rápida)
- [Artículo Académico (APA 7.ª edición)](#-artículo-académico-apa-7ª-edición)
- [Créditos y Licencia](#-créditos-y-licencia)

---

## 🔬 Motivación y Contexto

La mayoría de la literatura en selección de características bio-inspiradas para IDS (Intrusion Detection Systems) reporta precisiones superiores al 98 % sobre bancos sintéticos o de laboratorio (NSL-KDD, CIC-IDS2017, UNSW-NB15). Sin embargo, dichos entornos no reflejan las condiciones de tráfico de alta velocidad, el desbalance extremo de clases ni el desplazamiento distributivo (*covariate shift*) de enlaces troncales operativos.

Este proyecto evalúa bajo condiciones estrictas de paridad ($N = 30$ corridas pareadas, semillas estocásticas idénticas, control aleatorio y pruebas no paramétricas con corrección de Holm) si PSO y BA realmente mejoran la eficacia de detección frente al uso de todas las características disponibles, o si su beneficio se limita al coste computacional.

---

## 🌐 Dataset y Datos Crudos (MAWI 2021)

El corpus utilizado proviene del **Measurement and Analysis on the WIDE Internet (MAWI Working Group)** en el punto de muestreo F (enlace troncal transpacífico Japón–EE.UU., 150 Mbps), cruzado con el etiquetado heurístico de **MAWILab v1.1**.

### Enlaces Oficiales de Descarga Directa

| Fecha Captura | Archivo PCAP Crudo (Gzip) | Tamaño Bruto | Enlace de Descarga Directa (HTTP) |
| :---: | :---: | :---: | :--- |
| **2021-02-10** | `202102101400.pcap.gz` | $2{,}65\text{ GB}$ | [Descargar PCAP 20210210](http://mawi.wide.ad.jp/mawi/samplepoint-F/2021/202102101400.pcap.gz) |
| **2021-06-16** | `202106161400.pcap.gz` | $1{,}69\text{ GB}$ | [Descargar PCAP 20210616](http://mawi.wide.ad.jp/mawi/samplepoint-F/2021/202106161400.pcap.gz) |
| **2021-10-13** | `202110131400.pcap.gz` | $1{,}81\text{ GB}$ | [Descargar PCAP 20211013](http://mawi.wide.ad.jp/mawi/samplepoint-F/2021/202110131400.pcap.gz) |
| **Repositorio MAWI** | Punto de Muestreo F (2021) | — | [Archivo General MAWI WIDE 2021](http://mawi.wide.ad.jp/mawi/samplepoint-F/2021/) |
| **Etiquetas MAWILab** | XML / CSV v1.1 | ~500 KB | Incluidas en el repositorio: `data/raw/mawi_labels/` |

> **Nota:** Debido a que las trazas PCAP crudas totalizan **6,15 GB** y superan el límite de 100 MB por archivo de GitHub, las trazas se descargan bajo demanda utilizando el script automatizado provisto en `src/download_mawi.py`:
> ```bash
> python src/download_mawi.py --all
> ```

---

## 📁 Estructura del Repositorio

```text
thesis_ids/
├── config.yaml                     # Configuración central (semillas, rutas, parámetros de RF y metaheurísticas)
├── requirements.txt                # Dependencias exactas del entorno Python
├── README.md                       # Documentación técnica completa
├── .gitignore                      # Exclusión de trazas PCAP masivas (>1 GB) y cachés
│
├── data/                           # Datos y metadatos
│   ├── raw/mawi_labels/            # Etiquetas oficiales de MAWILab v1.1 (anomalous, suspicious, notice en CSV y XML)
│   └── processed/                  # Metadatos del corpus procesado
│       ├── dataset_stats.json      # Estadísticas de los 3 millones de flujos
│       └── feature_names.json      # Lista de los 50 atributos extraídos
│
├── src/                            # Pipeline modular ejecutable
│   ├── download_mawi.py            # Descarga automatizada de las capturas PCAP oficiales
│   ├── extract_flows.py            # Extracción de flujos bidireccionales con dpkt desde PCAP
│   ├── label_flows.py              # Cruce vectorial con etiquetas XML de MAWILab v1.1
│   ├── preprocesamiento.py         # Muestreo estratificado, imputación y partición 70/30 + LOO
│   ├── wrapper.py                  # Función de aptitud wrapper (RF rápido + CV)
│   ├── pso.py                      # Implementación de Binary PSO (función de transferencia sigmoide)
│   ├── ba.py                       # Implementación de Binary Bat Algorithm (sonoridad y pulso dinámico)
│   ├── baseline.py                 # Evaluación del brazo Línea Base (50 atributos)
│   ├── run_all.py                  # Orquestador del pipeline completo de 30 corridas
│   ├── benchmark_tiempos_30runs.py # Benchmark formal de tiempos de entrenamiento e inferencia
│   ├── run_stats_final.py          # Pruebas de Friedman, Wilcoxon pareado y corrección de Holm
│   └── run_loo_leakfree.py         # Validación cruzada Leave-One-Capture-Out
│
├── results/                        # Resultados empíricos verificables
│   ├── tables/                     # 17 tablas CSV oficiales con resultados consolidados
│   │   ├── resumen_metricas_v2.csv          # Medias y desviaciones de las 30 corridas
│   │   ├── benchmark_tiempos_oficial.csv    # Benchmark de tiempos y reducciones relativas
│   │   ├── tabla_contrastes_30_redondeada.csv # Contrastes pareados con ajuste de Holm
│   │   ├── friedman_rangos_medios.csv       # Rangos de Friedman por métrica
│   │   └── loo_leakfree_resumen.csv         # Resultados inter-captura LOO
│   ├── figures/                    # Gráficos generados
│   │   ├── boxplots_4arms.png               # Diagramas de caja de F1 macro entre brazos
│   │   ├── convergencia.png                 # Curvas de convergencia de aptitud
│   │   ├── f1_vs_features_v2.png            # Dispersión F1 vs. número de atributos
│   │   └── feature_frequency_v2.png         # Frecuencia de selección por atributo
│   ├── articulo_latex/             # Manuscrito formal en LaTeX (APA 7)
│   │   ├── articulo_cientifico.tex          # Código LaTeX completo
│   │   └── referencias.bib                  # Bibliografía en formato BibLaTeX-APA
│   └── resumen_resultados_v2.md    # Síntesis académica detallada
```

---

## ⚙️ Diseño Experimental y Metodología

* **Corpus MAWI:** Tres capturas de 2021 (febrero, junio y octubre), totalizando **3.000.000 de flujos** (22,82 % tráfico de ataque).
* **Partición:** 70 % entrenamiento ($2{,}100{,}000$ flujos) y 30 % prueba ($900{,}000$ flujos; $205{,}380$ ataques / $694{,}620$ benignos).
* **Brazos Experimentales:**
  1. `Línea Base`: Random Forest entrenado con los 50 atributos de flujo.
  2. `Random-k21`: Control aleatorio con 21 atributos al azar (igual cardinalidad que PSO).
  3. `PSO`: Selección wrapper con Particle Swarm Optimization binario.
  4. `Bat Algorithm`: Selección wrapper con Bat Algorithm binario.
* **Protocolo de Inferencia:**
  * Prueba de **Friedman** para evaluar diferencias globales entre los 4 brazos ($\alpha = 0{,}05$).
  * Prueba post-hoc de **Wilcoxon pareada** con corrección secuencial de **Holm** para comparaciones múltiples.
  * Tamaño del efecto estandarizado: Correlación biserial de rangos ($r_b$).

---

## 📊 Resultados Experimentales Clave

### 1. Rendimiento de Clasificación ($N = 30$ réplicas)

| Brazo Experimental | $F_1$-macro | Recall (clase ataque) | Accuracy | Tasa Falsos Positivos (TFP) | Tasa Falsos Desc. (TFD) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clasificador Trivial** | $0{,}4356$ | $0{,}0000$ | $0{,}7718$ | $0{,}0000$ | — |
| **Línea Base (50 atrib.)** | $0{,}7280 \pm 0{,}0001$ | $0{,}7117 \pm 0{,}0004$ | $0{,}7858 \pm 0{,}0002$ | $0{,}1924 \pm 0{,}0003$ | $0{,}4776 \pm 0{,}0003$ |
| **Random-k21 (control)** | $0{,}7161 \pm 0{,}0373$ | $0{,}7071 \pm 0{,}1036$ | $0{,}7739 \pm 0{,}0553$ | $0{,}2063 \pm 0{,}1020$ | $0{,}4716 \pm 0{,}0873$ |
| **PSO + Random Forest** | **$0{,}7342 \pm 0{,}0155$** | $0{,}6478 \pm 0{,}0764$ | **$0{,}8026 \pm 0{,}0281$** | **$0{,}1516 \pm 0{,}0588$** | **$0{,}4202 \pm 0{,}0923$** |
| **Bat Algorithm + RF** | $0{,}7244 \pm 0{,}0016$ | $0{,}6993 \pm 0{,}0175$ | $0{,}7840 \pm 0{,}0029$ | $0{,}1910 \pm 0{,}0086$ | $0{,}4801 \pm 0{,}0058$ |

> **Nota sobre Recall:** El recall de ataque en PSO cae de 0,7117 a 0,6478 (**$-6{,}39\text{ puntos porcentuales}$**; **$-8{,}98\,\%\text{ relativo}$**, $p_{\text{Holm}} < 0{,}001$), canjeando sensibilidad por especificidad y dejando inadvertidos a $13{,}127$ ataques adicionales por cada $900{,}000$ flujos.

---

### 2. Benchmark Formal de Tiempos y Amortización

| Brazo Experimental | Atributos ($\bar{k}$) | Reducción Atributos (RD) | $t_{\text{entrena}}$ | Reducción RTE | Latencia Inferencia | Reducción RTI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Línea Base** | $50{,}0 \pm 0{,}0$ | $0{,}0\,\%$ | $26{,}45\text{ s}$ | — | $2{,}256\,\mu\text{s/flujo}$ | — |
| **Random-k21** | $21{,}0 \pm 0{,}0$ | $58{,}0\,\%$ | $16{,}73\text{ s}$ | $+36{,}75\,\%$ | $2{,}071\,\mu\text{s/flujo}$ | $+8{,}23\,\%$ |
| **PSO** | **$18{,}5 \pm 2{,}6$** | **$63{,}1\,\%$** | **$16{,}91\text{ s}$** | **$+36{,}07\,\%$** | **$2{,}014\,\mu\text{s/flujo}$** | **$+10{,}75\,\%$** |
| **Bat Algorithm** | $24{,}2 \pm 3{,}0$ | $51{,}5\,\%$ | $19{,}45\text{ s}$ | $+26{,}47\,\%$ | $2{,}173\,\mu\text{s/flujo}$ | $+3{,}69\,\%$ |

* **Amortización del Costo de Búsqueda Offline:**
  * **PSO:** Costo de búsqueda $= 307{,}8\text{ s}$. Ahorro por reentrenamiento $= 9{,}54\text{ s}$. Requiere $\lceil 307{,}8 / 9{,}54 \rceil = \mathbf{33\text{ ciclos}}$ de reentrenamiento para amortizarse.
  * **BA:** Costo de búsqueda $= 42{,}3\text{ s}$. Ahorro por reentrenamiento $= 7{,}00\text{ s}$. Requiere $\lceil 42{,}3 / 7{,}00 \rceil = \mathbf{7\text{ ciclos}}$ para amortizarse.

---

### 3. Contrastes Estadísticos Post-Hoc (Wilcoxon + Holm)

| Comparación | Métrica | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Tamaño Efecto ($r_b$) | Veredicto ($\alpha = 0{,}05$) |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **PSO vs. Línea Base** | $F_1$-macro | $0{,}3050$ | $1{,}0000$ | $+0{,}019$ | **No significativo** (Invariante) |
| **PSO vs. Línea Base** | Recall (ataque) | $< 0{,}0001$ | $< 0{,}0010$ | $-1{,}000$ | **Significativo** (Línea Base superior; $-6{,}39\text{ pp}$) |
| **PSO vs. Línea Base** | TFP | $0{,}0880$ | $0{,}5264$ | $-0{,}421$ | **No significativo** tras corrección |
| **PSO vs. Línea Base** | $t_{\text{entrena}}$ | $0{,}00002$ | $< 0{,}0010$ | $-0{,}884$ | **Significativo** (PSO más rápido) |
| **PSO vs. Línea Base** | $t_{\text{infer}}$ | $0{,}00015$ | $< 0{,}0010$ | $-0{,}742$ | **Significativo** (PSO menor latencia) |
| **PSO vs. Random-k21** | $t_{\text{entrena}}$ | $0{,}0919$ | $0{,}0919$ | $-0{,}355$ | **No significativo** (Equivalente al azar) |
| **PSO vs. Random-k21** | $t_{\text{infer}}$ | $0{,}0262$ | $0{,}0524$ | $-0{,}462$ | **No significativo** tras Holm |
| **BA vs. Línea Base** | $F_1$-macro | $< 0{,}0001$ | $< 0{,}0010$ | $-1{,}000$ | **Significativo** (Línea Base superior) |
| **PSO vs. BA** | Recall (ataque) | $0{,}0036$ | $0{,}0215$ | $-0{,}673$ | **Significativo** (BA superior en recall) |

---

### 4. Validación Leave-One-Capture-Out (LOO)

Evaluación de la transferibilidad temporal inter-capturas:

| Captura de Prueba | Línea Base (50 atrib.) | Random-k21 | PSO | Bat Algorithm |
| :--- | :---: | :---: | :---: | :---: |
| **2021-02-10** | $0{,}519$ | $0{,}451$ | **$0{,}548$** | $0{,}452$ |
| **2021-06-16** | $0{,}457$ | $0{,}487$ | **$0{,}575$** | $0{,}495$ |
| **2021-10-13** | **$0{,}481$** | $0{,}501$ | $0{,}443$ | $0{,}468$ |
| **Media $\pm$ SD** | $0{,}486 \pm 0{,}031$ | $0{,}480 \pm 0{,}025$ | **$0{,}522 \pm 0{,}068$** | $0{,}472 \pm 0{,}022$ |

---

## 🎯 Veredicto de Hipótesis

* **H1 (Eficacia $F_1$-macro): NO RESPALDADA.** PSO no difiere estadísticamente de la línea base ($p_{\text{Holm}} = 1{,}0000; r_b = +0{,}019$) y BA es inferior ($p_{\text{Holm}} < 0{,}001$). Además, PSO reduce el recall en $6{,}39\text{ pp}$ ($8{,}98\,\%$; $p_{\text{Holm}} < 0{,}001$).
* **H2 (Falsas Alarmas TFP): NO RESPALDADA (Formalmente).** Aunque PSO reduce la TFP en un $21{,}2\,\%$ relativo (evitando $28{,}275$ falsas alarmas por corrida), el test de Holm arroja $p_{\text{Holm}} = 0{,}5264$ debido a la variabilidad entre semillas ($SD = 0{,}0588$).
* **H3 (Eficiencia Computacional $\ge 20\,\%$): PARCIALMENTE RESPALDADA.** El tiempo de entrenamiento disminuye **$36{,}07\,\%$** ($p = 0{,}00002 \ge 20\,\%$), pero la latencia de inferencia solo mejora **$10{,}75\,\%$** ($p = 0{,}00015 < 20\,\%$). Además, la equivalencia con el control aleatorio ($p = 0{,}0919$) demuestra que el ahorro se debe a la reducción dimensional y no a la metaheurística.


---

## 🚀 Guía de Reproducción Rápida

### Requisitos Previos
* Python 3.11+
* Memoria RAM mínima: 16 GB

### Instalación
```bash
# 1. Clonar el repositorio
git clone https://github.com/lobolunar375/thesis_ids_mawi.git
cd thesis_ids_mawi

# 2. Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: .\venv\Scripts\Activate.ps1

# 3. Instalar dependencias exactas
pip install -r requirements.txt
```

### Descarga de Datos Crudos y Ejecución del Pipeline
```bash
# Descargar las capturas PCAP crudas oficiales de MAWI (6.15 GB total)
python src/download_mawi.py --all

# Extraer y etiquetar flujos bidireccionales
python src/extract_flows.py 20210616
python src/label_flows.py

# Preprocesar datos (muestreo estratificado y partición)
python src/preprocesamiento.py

# Ejecutar las 30 corridas completas (Línea Base, Random-k21, PSO y BA)
python src/run_all.py

# Ejecutar el benchmark formal de tiempos de entrenamiento e inferencia
python src/benchmark_tiempos_30runs.py

# Computar estadísticas finales (Friedman, Wilcoxon pareado y Holm)
python src/run_stats_final.py

# Ejecutar validación Leave-One-Capture-Out
python src/run_loo_leakfree.py
```

---

## 📄 Artículo Académico (APA 7.ª edición)

El artículo formal se encuentra en `results/articulo_latex/articulo_cientifico.tex`, formateado con base en las directrices de la American Psychological Association (APA 7.ª edición).

Para compilar en Overleaf o localmente:
```bash
pdflatex articulo_cientifico.tex
biber articulo_cientifico
pdflatex articulo_cientifico.tex
pdflatex articulo_cientifico.tex
```

---

## 📜 Licencia y Cita

Este proyecto se distribuye bajo la licencia MIT. Para propósitos de citación académica:

```bibtex
@article{medina2026pso_ids,
  author    = {Medina Fuentes, Jhojan Sebastian},
  title     = {Optimización basada en inteligencia de enjambre (PSO y Bat Algorithm) para la selección de características en la detección de intrusiones sobre tráfico real de un enlace troncal transpacífico},
  journal   = {Universidad Nacional de Ingeniería - Inteligencia Artificial II},
  year      = {2026}
}
```
