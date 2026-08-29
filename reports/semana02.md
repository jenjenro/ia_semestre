# Semana 02 - Fundamentos aplicados al Sistema de Análisis Documental

Informe generado automáticamente por `src/semana02_fundamentos.py`.

## 1. Del dataset de práctica al problema del proyecto

La práctica original entrenaba una regresión logística sobre el dataset Iris. Aquí se conserva exactamente la misma lógica supervisada, pero aplicada al sistema de análisis documental.

| Elemento | Práctica original | Aplicación al proyecto |
|---|---|---|
| Datos | `load_iris()` | `data/casos_ia.csv` (casos reales del proyecto) |
| Entradas (X) | 4 medidas numéricas | Descripción textual del caso |
| Preparación | `StandardScaler` | `TfidfVectorizer` (unigramas, stopwords ES) |
| Etiquetas (y) | 3 especies | 7 categorías de la taxonomía documental |
| Modelo | `LogisticRegression` | `LogisticRegression` (unigramas, pesos balanceados) |
| Evaluación | `train_test_split` 75/25 | Validación cruzada (LOO y StratifiedKFold) |
| Métricas | Accuracy, matriz de confusión | Accuracy, matriz de confusión, precisión/recall |

### Por qué cambió el método de evaluación

El proyecto tiene **20 casos** y **7 categorías**. Un conjunto de prueba del 25% tendría solo 5 muestras, menos que el número de clases, por lo que una división estratificada es imposible y scikit-learn la rechaza. Además, una única división de 5 casos daría una accuracy con enorme varianza: cambiar la semilla cambiaría el resultado por completo.

La validación cruzada resuelve ese problema sin inventar datos: cada caso se predice con un modelo que nunca lo vio, y la métrica se calcula sobre las 20 predicciones.

## 2. Dataset supervisado

- Casos etiquetados: **20**
- Categorías: **7**

| Categoría | Casos |
|---|---:|
| Aprendizaje automático predictivo | 2 |
| Automatización documental inteligente | 2 |
| Búsqueda y recuperación documental | 4 |
| Clasificación documental | 3 |
| Procesamiento de lenguaje natural | 4 |
| Sistemas expertos y reglas documentales | 3 |
| Visión por computador y OCR | 2 |

La categoría menos frecuente tiene 2 ejemplos: ese valor es el que limita el número de particiones posibles en la validación cruzada estratificada.

## 3. Resultados

| Método de evaluación | Accuracy |
|---|---:|
| Leave-One-Out (20 entrenamientos) | **40.00%** |
| StratifiedKFold (k=2) | 35.00% |
| Reglas de la Semana 03 (baseline) | 80.00% |

### Efecto de las decisiones de diseño

Accuracy Leave-One-Out de cada configuración probada:

| Configuración | Accuracy |
|---|---:|
| Logistic Regression, unigramas + bigramas, sin balanceo | 20.00% |
| Logistic Regression, solo unigramas, sin balanceo | 20.00% |
| Logistic Regression, unigramas + bigramas, balanceada | 35.00% |
| Logistic Regression, solo unigramas, balanceada (modelo final) | 40.00% |

La configuración inicial (bigramas y sin balanceo) es la peor: con 20 documentos cortos, los bigramas casi solo aportan rasgos que aparecen una única vez, y sin `class_weight="balanced"` el modelo se inclina hacia las categorías con más ejemplos. Limitar a unigramas y balancear los pesos es lo que hace que el modelo llegue a su mejor resultado.

### Matriz de confusión (Leave-One-Out)

Filas: categoría real. Columnas: categoría predicha.

| Real \ Predicho | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Aprendizaje automático predictivo | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| Automatización documental inteligente | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
| Búsqueda y recuperación documental | 0 | 0 | 2 | 1 | 0 | 1 | 0 |
| Clasificación documental | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| Procesamiento de lenguaje natural | 0 | 0 | 0 | 1 | 3 | 0 | 0 |
| Sistemas expertos y reglas documentales | 0 | 1 | 2 | 0 | 0 | 0 | 0 |
| Visión por computador y OCR | 1 | 0 | 0 | 0 | 1 | 0 | 0 |

Referencia de columnas: **C1** = Aprendizaje automático predictivo, **C2** = Automatización documental inteligente, **C3** = Búsqueda y recuperación documental, **C4** = Clasificación documental, **C5** = Procesamiento de lenguaje natural, **C6** = Sistemas expertos y reglas documentales, **C7** = Visión por computador y OCR.

### Precisión y recall por categoría (Leave-One-Out)

```
                                         precision    recall  f1-score   support

      Aprendizaje automático predictivo       0.00      0.00      0.00         2
  Automatización documental inteligente       0.00      0.00      0.00         2
     Búsqueda y recuperación documental       0.50      0.50      0.50         4
               Clasificación documental       0.50      1.00      0.67         3
      Procesamiento de lenguaje natural       0.75      0.75      0.75         4
Sistemas expertos y reglas documentales       0.00      0.00      0.00         3
            Visión por computador y OCR       0.00      0.00      0.00         2

                               accuracy                           0.40        20
                              macro avg       0.25      0.32      0.27        20
                           weighted avg       0.33      0.40      0.35        20
```

## 4. Caso por caso (Leave-One-Out)

| # | Descripción | Categoría real | Predicción del modelo | Estado |
|---:|---|---|---|---|
| 1 | Escanear un contrato físico y convertirlo en texto utilizando OCR. | Visión por computador y OCR | Procesamiento de lenguaje natural | Revisar |
| 2 | Identificar automáticamente si un documento corresponde a una factura, contrato o certificado. | Clasificación documental | Clasificación documental | Coincide |
| 3 | Extraer nombres completos y números de identificación de documentos administrativos. | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural | Coincide |
| 4 | Extraer fechas importantes presentes en contratos y certificados. | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural | Coincide |
| 5 | Buscar una palabra específica dentro de un conjunto de documentos digitales. | Búsqueda y recuperación documental | Búsqueda y recuperación documental | Coincide |
| 6 | Encontrar todos los contratos relacionados con una determinada empresa. | Búsqueda y recuperación documental | Sistemas expertos y reglas documentales | Revisar |
| 7 | Verificar si un documento contiene todos los campos obligatorios. | Sistemas expertos y reglas documentales | Búsqueda y recuperación documental | Revisar |
| 8 | Detectar si un documento escaneado tiene partes ilegibles para solicitar una nueva captura. | Visión por computador y OCR | Aprendizaje automático predictivo | Revisar |
| 9 | Clasificar automáticamente documentos según su contenido textual. | Clasificación documental | Clasificación documental | Coincide |
| 10 | Generar un resumen automático del contenido de un informe. | Procesamiento de lenguaje natural | Clasificación documental | Revisar |
| 11 | Extraer nombres de personas y organizaciones mencionadas en un contrato. | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural | Coincide |
| 12 | Detectar documentos que contienen información faltante. | Sistemas expertos y reglas documentales | Automatización documental inteligente | Revisar |
| 13 | Comparar dos versiones de un documento para identificar diferencias. | Búsqueda y recuperación documental | Clasificación documental | Revisar |
| 14 | Detectar documentos duplicados dentro de un repositorio documental. | Aprendizaje automático predictivo | Sistemas expertos y reglas documentales | Revisar |
| 15 | Buscar documentos relacionados con una solicitud específica. | Búsqueda y recuperación documental | Búsqueda y recuperación documental | Coincide |
| 16 | Identificar automáticamente el tipo de documento a partir de su contenido. | Clasificación documental | Clasificación documental | Coincide |
| 17 | Validar que una solicitud incluya todos los documentos requeridos. | Sistemas expertos y reglas documentales | Búsqueda y recuperación documental | Revisar |
| 18 | Procesar automáticamente documentos escaneados y extraer su información relevante. | Automatización documental inteligente | Clasificación documental | Revisar |
| 19 | Detectar posibles anomalías en documentos administrativos mediante patrones. | Aprendizaje automático predictivo | Sistemas expertos y reglas documentales | Revisar |
| 20 | Generar un reporte con la información extraída de múltiples documentos. | Automatización documental inteligente | Sistemas expertos y reglas documentales | Revisar |

## 5. Vocabulario aprendido por el modelo

Términos con mayor peso positivo en la regresión logística entrenada con todos los casos. Muestran qué vocabulario del dominio documental usa el modelo para decidir cada categoría.

| Categoría | Términos más influyentes |
|---|---|
| Aprendizaje automático predictivo | `detectar`, `repositorio`, `duplicados`, `documental`, `posibles` |
| Automatización documental inteligente | `información`, `reporte`, `múltiples`, `extraída`, `escaneados` |
| Búsqueda y recuperación documental | `relacionados`, `específica`, `buscar`, `versiones`, `comparar` |
| Clasificación documental | `automáticamente`, `contenido`, `identificar`, `según`, `clasificar` |
| Procesamiento de lenguaje natural | `extraer`, `nombres`, `resumen`, `informe`, `automático` |
| Sistemas expertos y reglas documentales | `todos`, `faltante`, `contienen`, `validar`, `requeridos` |
| Visión por computador y OCR | `utilizando`, `texto`, `físico`, `ocr`, `convertirlo` |

## 6. Conclusiones

- La lógica de la práctica de fundamentos se traslada sin problema al proyecto: el cambio de fondo es sustituir el escalado numérico por una vectorización TF-IDF del texto.
- El vocabulario aprendido es coherente con el dominio (`ocr` para visión por computador, `buscar` para recuperación, `extraer` y `nombres` para PLN), así que el modelo sí capta señal real y no ruido aleatorio.
- Con validación cruzada Leave-One-Out el modelo alcanza **40.00%** de accuracy sobre los 20 casos reales del proyecto.
- Comparado con el clasificador por reglas de la Semana 03 (80.00%), el sistema de reglas de la Semana 03 supera al modelo aprendido por 40.00 puntos porcentuales.
- La principal limitación es el tamaño del dataset: con 2 o 3 ejemplos en varias categorías, el modelo tiene muy poca evidencia para aprender vocabulario discriminante, y por eso las categorías minoritarias son las que concentran los errores en la matriz de confusión.
- Que las reglas manuales ganen no significa que el enfoque supervisado sea inferior: significa que 20 ejemplos no bastan para que el modelo aprenda lo que un experto ya codificó a mano. Las reglas incorporan conocimiento del dominio que el modelo tendría que deducir de los datos.
- La mejora más rentable no es cambiar de algoritmo, sino ampliar y equilibrar el corpus etiquetado del proyecto.
