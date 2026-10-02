# IA Semestre — Sistema de Análisis Documental 📄🤖

Este repositorio contiene el desarrollo progresivo de un Sistema de Análisis Documental impulsado por Inteligencia Artificial. El proyecto se enfoca en automatizar la clasificación, búsqueda y análisis de documentos (contratos, facturas, reportes técnicos, etc.) utilizando técnicas de Machine Learning y sistemas expertos.

Desarrollado para la asignatura de Inteligencia Artificial del programa de Ingeniería de Sistemas en la Escuela Tecnológica Instituto Técnico Central (ETITC).

---

## 🛠️ Tecnologías y Herramientas
El proyecto está desarrollado principalmente en **Python** haciendo uso de las siguientes librerías y entornos:
* **Scikit-learn:** Para la vectorización (TF-IDF) y modelos de clasificación (Naive Bayes, Regresión Logística).
* **NumPy & Pandas:** Para el manejo, estructuración y análisis de los datos.
* **Matplotlib:** Para la visualización de métricas y resultados.
* **JupyterLab:** Para la experimentación y análisis exploratorio de datos.

---

## 📅 Avance del Proyecto (Semana a Semana)

El desarrollo del sistema se divide en módulos incrementales. Cada semana aborda un concepto fundamental de la IA aplicado al dominio de los documentos:

### Semana 02: Fundamentos (`semana02_fundamentos.py`)
* **Objetivo:** Establecer las bases del proyecto y la lectura inicial de los datos.
* **Descripción:** Implementación de las estructuras de datos iniciales y algoritmos de base para procesar la información contenida en el dataset de prueba (`casos_ia.csv`).

### Semana 03: Taxonomía (`semana03_taxonomia.py`)
* **Objetivo:** Organizar y clasificar el dominio de conocimiento.
* **Descripción:** Desarrollo de la jerarquía conceptual de los documentos y categorías manejadas por el sistema (ej. Legal, Financiero, Técnico, RRHH), sentando las bases para una clasificación estructurada.

### Semana 04: Búsqueda (`semana04_busqueda.py`)
* **Objetivo:** Implementar métodos de búsqueda inteligente.
* **Descripción:** Integración de algoritmos de búsqueda (como grafos de búsqueda A* o árboles de decisión Minimax) para recorrer el espacio de soluciones y encontrar rutas óptimas dentro del análisis de los datos.

### Semana 05: Sistema Híbrido y Recuperación de Información (`semana05_sistema_hibrido.py`)
* **Objetivo:** Integrar un sistema experto con recuperación de información (NLP).
* **Descripción:** 
  * Construcción de una base de conocimiento documental (`base_conocimiento.txt`).
  * Implementación de **TF-IDF** (Term Frequency - Inverse Document Frequency) y similitud del coseno para recuperación de información relevante frente a consultas en texto libre.
  * Diseño de un **Sistema Experto** mediante reglas lógicas condicionales para emitir alertas, prioridades y recomendaciones basadas en el texto analizado.

### Semana 07: Representaciones del Reconocimiento (`semana07_representaciones.py`)
* **Objetivo:** Comparar representaciones numéricas, simbólicas y autómatas para la clasificación documental.
* **Descripción:** Implementación de distancias vectoriales, lógica proposicional de hechos y autómatas finitos para validación estructural de documentos.

### Semana 08: Representaciones del Reconocimiento con RNA y Ontologías (`semana08_representaciones.py`)
* **Objetivo:** Implementar el flujo integral Entrada -> Red Neuronal -> Base de Datos -> Ontología.
* **Descripción:** 
  * Clasificación con Red Neuronal Artificial (Perceptrón Multicapa MLPClassifier) sobre 6 categorías documentales.
  * Persistencia estructurada de evidencias en base de datos SQLite (`evidencia_reconocimiento.db`).
  * Enriquecimiento semántico y asignación de acciones mediante Grafo de Conocimiento y Ontología documental en GraphML.

### Semana 09: Reconocimiento de Imágenes: Características, Contornos y Segmentación (`semana09_vision.py`)
* **Objetivo:** Procesamiento digital de imágenes documentales (facturas escaneadas) para extracción de características, contornos y regiones conectadas.
* **Descripción:** 
  * Extracción de características numéricas: dimensiones, intensidad, distribución de color (RGB) y entropía de Shannon.
  * Detección de contornos mediante el algoritmo de **Canny** evaluando el efecto del parámetro de escala Gaussiana ($\sigma$).
  * Segmentación automática mediante el método de **Otsu**, separando la tinta/texto del papel soporte.
  * Análisis de **regiones conectadas** (componentes conexas) para identificar glifos, cuadrículas y sellos institucionales.
  * Generación de panel visual comparativo en `artifacts/semana09_vision.png`.

---

## ⚙️ Instalación y Configuración

Sigue estos pasos para clonar y ejecutar el proyecto en un entorno local:

1. **Clonar el repositorio:**
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd ia_semestre


   ## 👥 Autores
* **Jenny Valentina Rojas Orjuela**
* **Laury Dayana Gama Sosa**