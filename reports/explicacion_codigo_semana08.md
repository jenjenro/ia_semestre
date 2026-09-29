# Explicación Detallada Parte por Parte del Código
## `src/semana08_representaciones.py`

**Proyecto:** Sistema de Análisis y Gestión Documental con Inteligencia Artificial  
**Módulo:** Representaciones del Reconocimiento (Semana 08)  
**Objetivo del archivo:** Integrar un flujo de procesamiento inteligente completo:  
$$\text{Entrada de Documentos} \longrightarrow \text{Red Neuronal Artificial (RNA)} \longrightarrow \text{Evidencia Persistente (SQLite)} \longrightarrow \text{Significado Semántico (Ontología GraphML)}$$

---

## Índice General

1. [Estructura y Visión Global del Flujo](#1-estructura-y-visión-global-del-flujo)
2. [Bloque 0: Importaciones y Rutas del Sistema (Líneas 1 - 40)](#2-bloque-0-importaciones-y-rutas-del-sistema-líneas-1---40)
3. [Bloque 1: Dataset del Dominio Documental (Líneas 42 - 119)](#3-bloque-1-dataset-del-dominio-documental-líneas-42---119)
4. [Bloque 2: Ingesta y Lectura Multiformato (Líneas 121 - 264)](#4-bloque-2-ingesta-y-lectura-multiformato-líneas-121---264)
5. [Bloque 3: Base de Datos Relacional SQLite (Evidencia) (Líneas 266 - 373)](#5-bloque-3-base-de-datos-relacional-sqlite-evidencia-líneas-266---373)
6. [Bloque 4: Ontología del Conocimiento en GraphML (Líneas 375 - 540)](#6-bloque-4-ontología-del-conocimiento-en-graphml-líneas-375---540)
7. [Bloque 5: Red Neuronal Artificial y Validación (Líneas 542 - 612)](#7-bloque-5-red-neuronal-artificial-y-validación-líneas-542---612)
8. [Bloque 6: Pipeline Integrado de Inferencia (Líneas 614 - 716)](#8-bloque-6-pipeline-integrado-de-inferencia-líneas-614---716)
9. [Bloque 7: Generador Automático del Reporte Markdown (Líneas 718 - 920)](#9-bloque-7-generador-automático-del-reporte-markdown-líneas-718---920)
10. [Bloque 8: Bloque Principal de Ejecución (`main`) (Líneas 922 - 1004)](#10-bloque-8-bloque-principal-de-ejecución-main-líneas-922---1004)
11. [Resumen de Artefactos Generados](#11-resumen-de-artefactos-generados)

---

## 1. Estructura y Visión Global del Flujo

El código no se limita a arrojar una predicción numérica aislada en consola, sino que implementa una arquitectura por capas donde cada componente cumple un rol específico:

```text
┌───────────────────────┐
│ Documento en Disco    │ (.txt, .md, .docx, .pdf o texto directo)
└──────────┬────────────┘
           │ (1. Extracción de texto plano)
           ▼
┌───────────────────────┐
│ TfidfVectorizer       │ (Matriz dispersa con unigramas y bigramas)
└──────────┬────────────┘
           │ (2. Vector numérico de patrones)
           ▼
┌───────────────────────┐
│ MLPClassifier (RNA)   │ (32 neuronas ReLU + Softmax de 6 clases)
└──────────┬────────────┘
           │ (3. Predicción categórica + Probabilidad de Confianza)
           ▼
┌───────────────────────┬───────────────────────┐
│ Base de Datos SQLite  │ Ontología GraphML     │
│ (Registra Evidencia)  │ (Deriva Significado)  │
│ - ID transaccional    │ - Dominio corporativo │
│ - Documento / Fecha   │ - Depto. responsable  │
│ - Clase y Confianza   │ - Acción operativa    │
└──────────┬────────────┴───────────┬───────────┘
           │                        │
           └───────────┬────────────┘
                       ▼
         Resultado Integral Accionable
```

---

## 2. Bloque 0: Importaciones y Rutas del Sistema (Líneas 1 - 40)

### ¿Qué hace esta sección?
Declara las dependencias necesarias y centraliza la configuración de rutas del sistema operativo utilizando programación orientada a rutas con `pathlib.Path`.

```python
from datetime import datetime
from pathlib import Path
import os
import sqlite3
import unicodedata
import re

import docx              # Extracción de documentos Word (.docx)
import joblib            # Serialización y guardado de modelos entrenados
import networkx as nx    # Modelado de grafos y ontologías en GraphML
import numpy as np       # Operaciones matriciales numéricas
import pypdf             # Extracción de texto desde archivos PDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
```

### Rutas Dinámicas y Robustas
```python
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"
SAMPLES_DIR = ARTIFACTS_DIR / "muestras_prueba"
```
- `BASE_DIR`: Detecta la carpeta raíz del repositorio sin importar si el script se ejecuta desde Windows, Linux o una subcarpeta.
- Centraliza en variables constantes los nombres de los artefactos (`MODEL_PATH`, `DB_PATH`, `ONTOLOGY_PATH`, `REPORT_PATH`).

---

## 3. Bloque 1: Dataset del Dominio Documental (Líneas 42 - 119)

### ¿Qué hace esta sección?
Define el corpus de entrenamiento del sistema: **60 documentos etiquetados** adaptados estrictamente al contexto de gestión y análisis documental del proyecto del semestre.

### Estructura de Clases Balanceadas:
El dataset cuenta con **10 ejemplos por categoría** distribuidos equitativamente para evitar sesgos:
1. **`contrato`** (Dominio Legal): Contratos laborales, convenios de cooperación técnica, acuerdos de confidencialidad (NDA), contratos mercantiles y acuerdos de nivel de servicio (SLA).
2. **`factura`** (Dominio Financiero): Facturas electrónicas de venta, cuentas de cobro, comprobantes de egreso con retención en la fuente, IVA y códigos fiscales DIAN.
3. **`reporte_financiero`** (Dominio Financiero): Balances generales consolidados, estados de resultados (EBITDA), informes de auditoría contable NIIF y flujos de efectivo.
4. **`manual_tecnico`** (Dominio Tecnológico): Guías de despliegue en microservicios, documentación de APIs RESTful con OAuth2, manuales de usuario y mantenimiento de bases de datos.
5. **`memorando`** (Dominio Administrativo): Circulares internas de recursos humanos, avisos sobre teletrabajo, llamados de atención disciplinarios y calendarios de capacitación.
6. **`acta_reunion`** (Dominio Administrativo): Actas solemnes de junta directiva, deliberación de asambleas de accionistas y minutas de comités técnicos.

Cada elemento es una tupla: `("Texto descriptivo con vocabulario técnico representativo", "etiqueta_clase")`.

---

## 4. Bloque 2: Ingesta y Lectura Multiformato (Líneas 121 - 264)

### Función `extraer_texto_documento(fuente)`
Permite que el sistema reciba tanto rutas a archivos físicos como cadenas de texto directo.

```python
def extraer_texto_documento(fuente) -> tuple[str, str, str]:
```
1. **Detección de Entrada:** Si `fuente` es una cadena sin archivo físico existente, la procesa directamente como `"texto_directo"`.
2. **Despacho Polimórfico por Extensión:**
   - **`.txt` y `.md`:** Lee el archivo decodificando en `utf-8` con `errors="ignore"` para evitar caídas por caracteres no estándar.
   - **`.pdf`:** Instancia `pypdf.PdfReader` e itera sobre cada página extrayendo los bloques de texto. Si el PDF es escaneado (solo imagen), retorna cadena vacía de forma segura.
   - **`.docx`:** Instancia `docx.Document` y extrae el texto de todos los párrafos omitiendo bloques vacíos.
3. **Retorno Normalizado:** Devuelve `(nombre_archivo, texto_limpio, tipo_fuente)` para alimentar el pipeline.

### Función `crear_muestras_prueba()`
Genera automáticamente en el directorio `artifacts/muestras_prueba/` archivos reales en disco:
- `contrato_servicios_2026.txt` (.txt)
- `factura_proveedor_servicios.txt` (.txt)
- `balance_general_ejercicio.docx` (.docx creado dinámicamente con encabezados y párrafos de Word)
- `manual_despliegue_contenedores.md` (.md)
- `circular_politica_laboral.txt` (.txt)
- `acta_comite_directivo.txt` (.txt)
- `receta_reposteria_atipica.txt` (Caso especial fuera de dominio para validar el comportamiento ante anomalías)

---

## 5. Bloque 3: Base de Datos Relacional SQLite (Evidencia) (Líneas 266 - 373)

Garantiza el requisito de **registro de evidencia persistente** para que los resultados no queden únicamente en memoria volátil.

### Esquema Relacional de la Tabla
```sql
CREATE TABLE IF NOT EXISTS evidencia_reconocimiento (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    fuente_documento TEXT NOT NULL,
    tipo_fuente TEXT NOT NULL,
    resumen_contenido TEXT NOT NULL,
    categoria_real TEXT NOT NULL,
    categoria_predicha TEXT NOT NULL,
    nivel_confianza REAL NOT NULL,
    estado_proceso TEXT NOT NULL,
    categoria_dominio TEXT NOT NULL,
    departamento_destino TEXT NOT NULL,
    accion_sugerida TEXT NOT NULL,
    significado_ontologico TEXT NOT NULL
);
```

### Funciones Implementadas
- **`inicializar_base_datos(db_path)`:** Crea la base de datos si no existe, genera la tabla y define índices sobre `categoria_predicha` y `estado_proceso` para acelerar consultas analíticas.
- **`registrar_evidencia(registro, db_path)`:** Ejecuta una inserción parametrizada con `?` (evitando inyecciones SQL), trunca el resumen a 250 caracteres y retorna el `id` autoincremental asignado.
- **`consultar_evidencias(db_path, limite)`:** Configura `conn.row_factory = sqlite3.Row` para recuperar los registros más recientes como diccionarios listos para inspección.
- **`consultar_estadisticas_bd(db_path)`:** Ejecuta agregaciones SQL (`COUNT(*)`, `AVG(nivel_confianza)`, `GROUP BY categoria_predicha`) para generar métricas de trazabilidad global.

---

## 6. Bloque 4: Ontología del Conocimiento en GraphML (Líneas 375 - 540)

Representa el conocimiento del dominio y asigna **significado** a la predicción numérica.

### Función `construir_ontologia()`
Crea un grafo dirigido `networkx.DiGraph` con dos niveles:

1. **Conceptos Núcleo (8 conceptos):**
   - `DocumentoDigital`: Entidad documental analizada.
   - `RedNeuronal`: Agente inteligente clasificador.
   - `Prediccion`: Inferencia probabilística generada.
   - `EvidenciaBD`: Registro verificable en SQLite.
   - `TipoDocumento`: Taxonomía de clases reconocidas.
   - `CategoriaDominio`: Área macro institucional.
   - `DepartamentoResponsable`: Dependencia administrativa asignada.
   - `AccionFlujo`: Procedimiento operativo obligatorio.

2. **Relaciones como Proposiciones con Sentido de Frase:**
   - `RedNeuronal` **analiza_patrones_de** `DocumentoDigital`
   - `RedNeuronal` **genera_prediccion** `Prediccion`
   - `Prediccion` **asigna_clase_a** `TipoDocumento`
   - `Prediccion` **registra_evidencia_en** `EvidenciaBD`
   - `EvidenciaBD` **garantiza_trazabilidad_de** `DocumentoDigital`
   - `TipoDocumento` **pertenece_a_dominio** `CategoriaDominio`
   - `TipoDocumento` **enruta_hacia** `DepartamentoResponsable`
   - `TipoDocumento` **determina_accion** `AccionFlujo`

3. **Mapeos de Inferencia Específicos:**
   Conecta cada clase documental con su área (`Dominio_Legal`, `Dominio_Financiero`, `Dominio_Tecnologico`, `Dominio_Administrativo`), su oficina (`Area_Juridica`, `Contabilidad_Tesoreria`, `Operaciones_TI`, `Gestion_Humana`) y su acción inmediata (`Custodia_Firmas_Legales`, `Auditoria_Tributaria`, etc.).

### Función `guardar_ontologia_graphml(grafo, ruta)`
Exporta el grafo con todos los atributos de nodos y aristas al estándar XML abierto **GraphML** (`artifacts/ontologia_documental.graphml`).

### Función `interpretar_con_ontologia(ontologia, clase_predicha)`
Navega dinámicamente las aristas salientes (`out_edges`) de la clase predicha por la RNA:
```python
for _, destino, datos in ontologia.out_edges(clase_predicha_norm, data=True):
    rel = datos.get("label", "")
    if rel == "pertenece_a_dominio": dominio = destino
    elif rel == "enruta_hacia": departamento = destino
    elif rel == "determina_accion": accion = destino
```
Genera una explicación completa: transforma la predicción matemática en una instrucción empresarial estructurada.

---

## 7. Bloque 5: Red Neuronal Artificial y Validación (Líneas 542 - 612)

### Función `entrenar_modelo_neuronal(random_state)`

1. **Partición de Datos:**
   ```python
   X_train_raw, X_test_raw, y_train, y_test = train_test_split(
       textos, etiquetas, test_size=0.20, random_state=random_state, stratify=etiquetas
   )
   ```
   - Reserva 80% (48 documentos) para ajustar pesos y 20% (12 documentos) como conjunto de prueba ciego.
   - El parámetro `stratify=etiquetas` asegura que cada clase tenga exactamente la misma proporción de muestras en entrenamiento y test.

2. **Extracción Numérica de Características (TF-IDF):**
   ```python
   vectorizador = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=1000)
   ```
   - Modela unigramas (palabras individuales) y bigramas (pares de palabras consecutivas como *"balance general"* o *"cláusula penal"*).
   - `sublinear_tf=True`: Escala la frecuencia $1 + \log(TF)$ para evitar que palabras hiper-frecuentes distorsionen la representación.

3. **Arquitectura del Perceptrón Multicapa (MLPClassifier):**
   ```python
   modelo = MLPClassifier(
       hidden_layer_sizes=(32,),
       activation="relu",
       solver="adam",
       alpha=0.001,
       learning_rate_init=0.005,
       max_iter=600,
       random_state=random_state
   )
   ```
   - **Capa Oculta:** 32 neuronas artificiales con activación no lineal **ReLU** ($f(x) = \max(0, x)$).
   - **Optimizador Adam:** Ajusta dinámicamente la tasa de aprendizaje por peso sináptico mediante estimación de momentos de primer y segundo orden.
   - **Regularización L2 (`alpha=0.001`):** Penaliza pesos excesivos previniendo el sobreajuste (*overfitting*).
   - **Capa de Salida:** Genera un vector $z$ que la función **Softmax** transforma en probabilidades normalizadas que suman 1.0.

4. **Validación y Persistencia:**
   - Calcula la exactitud (`accuracy_score`), precisión, recall, F1-Score y matriz de confusión sobre el 20% de prueba.
   - Serializa el modelo entrenado y el vectorizador en `artifacts/modelo_red_neuronal.joblib` y `artifacts/vectorizador_tfidf.joblib`.

---

## 8. Bloque 6: Pipeline Integrado de Inferencia (Líneas 614 - 716)

### Función `clasificar_y_registrar_documento(...)`
Coordina el ciclo integral:

$$\text{Documento} \to \text{Validaciones} \to \text{RNA} \to \text{Ontología} \to \text{Filtro Certeza} \to \text{Registro SQLite}$$

1. **Validación de Legibilidad:** Si el archivo no tiene texto (por ejemplo, PDF escaneado sin OCR), no ejecuta la RNA; genera registro con estado `Rechazado (Sin contenido textual legible o requiere OCR)`.
2. **Validación de Coincidencia de Vocabulario:** Si el vector resultante tiene suma 0 (ningún término coincide con el vocabulario conocido), se marca como `Rechazado (Vocabulario desconocido)`.
3. **Inferencia Neuronal:**
   ```python
   probabilidades = modelo.predict_proba(vector)[0]
   idx_max = int(probabilidades.argmax())
   confianza = float(probabilidades[idx_max])
   clase_predicha = str(modelo.classes_[idx_max])
   ```
4. **Consulta Ontológica:** Obtiene dominio, departamento responsable y acción de flujo llamando a `interpretar_con_ontologia()`.
5. **Control de Incertidumbre:**
   - Si `confianza < umbral_minimo (0.40)`: Asigna el estado `Requiere Revisión Manual (Confianza XX% < 40%)`.
   - Si `confianza >= umbral_minimo`: Asigna `Procesado Exitosamente`.
6. **Persistencia en Base de Datos:** Llama a `registrar_evidencia()` y vincula el `id_bd` al diccionario retornado.

---

## 9. Bloque 7: Generador Automático del Reporte Markdown (Líneas 718 - 920)

### Función `generar_reporte_md(evaluacion, muestras_procesadas, ontologia)`
Automatiza la documentación del laboratorio generando [`reports/semana08.md`](file:///c:/Users/jenny/Documents/U/ia_semestre/reports/semana08.md):
- Extrae dinámicamente las métricas de validación obtenidas en la ejecución actual.
- Formatea la matriz de confusión en tablas Markdown legibles.
- Lista todas las relaciones semánticas de la ontología.
- Imprime la tabla de evidencias con los IDs de base de datos asignados.
- Expone el análisis técnico de las limitaciones encontradas (documentos escaneados, términos híbridos y jerga regional).

---

## 10. Bloque 8: Bloque Principal de Ejecución (`main`) (Líneas 922 - 1004)

Punto de entrada cuando el script es invocado directamente (`if __name__ == "__main__":`):

1. **Paso 1:** Inicializa la base de datos SQLite en `artifacts/evidencia_reconocimiento.db`.
2. **Paso 2:** Construye la ontología en memoria y la guarda en `artifacts/ontologia_documental.graphml`.
3. **Paso 3:** Entrena la Red Neuronal, evalúa las métricas de test e imprime el porcentaje de Accuracy.
4. **Paso 4:** Genera las muestras de prueba en disco y ejecuta el pipeline sobre cada una de ellas:
   - Archivos válidos (.txt, .docx, .md) alcanzan entre 92% y 99% de confianza y estado `Procesado Exitosamente`.
   - El caso anómalo (`receta_reposteria_atipica.txt`) arroja baja certeza (20.99% < 40%) y se clasifica automáticamente como `Requiere Revisión Manual`.
5. **Paso 5:** Realiza consultas SQL de verificación mostrando los últimos registros y estadísticas agregadas.
6. **Paso 6:** Genera el informe final `reports/semana08.md`.

---

## 11. Resumen de Artefactos Generados

Al finalizar la ejecución de `src/semana08_representaciones.py`, se producen los siguientes artefactos en el repositorio:

| Artefacto | Tipo | Propósito |
| :--- | :--- | :--- |
| [`artifacts/modelo_red_neuronal.joblib`](file:///c:/Users/jenny/Documents/U/ia_semestre/artifacts/modelo_red_neuronal.joblib) | Binario Joblib | Red Neuronal MLP entrenada lista para inferencia en producción. |
| [`artifacts/vectorizador_tfidf.joblib`](file:///c:/Users/jenny/Documents/U/ia_semestre/artifacts/vectorizador_tfidf.joblib) | Binario Joblib | Vocabulario y ponderaciones IDF del espacio vectorial. |
| [`artifacts/evidencia_reconocimiento.db`](file:///c:/Users/jenny/Documents/U/ia_semestre/artifacts/evidencia_reconocimiento.db) | Base de Datos SQLite | Registro persistente e indexado de cada predicción y auditoría. |
| [`artifacts/ontologia_documental.graphml`](file:///c:/Users/jenny/Documents/U/ia_semestre/artifacts/ontologia_documental.graphml) | XML GraphML | Estructura formal de conceptos y relaciones del dominio del proyecto. |
| [`artifacts/muestras_prueba/`](file:///c:/Users/jenny/Documents/U/ia_semestre/artifacts/muestras_prueba/) | Directorio | Archivos reales (.txt, .docx, .md) procesados en la demostración. |
| [`reports/semana08.md`](file:///c:/Users/jenny/Documents/U/ia_semestre/reports/semana08.md) | Markdown | Informe técnico formal requerido para la entrega académica. |
