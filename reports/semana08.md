# Reporte Semana 08 - Representaciones del Reconocimiento

**Proyecto:** Sistema de Análisis y Gestión Documental con Inteligencia Artificial  
**Fecha de corte:** Septiembre 2026  
**Modalidad:** Individual  
**Entrega:** Repositorio GitHub – Commit Semana 8  

---

## 1. Qué reconoce el sistema

El sistema inteligente implementado aborda el reconocimiento y clasificación automática de documentos organizacionales y administrativos dentro del flujo documental institucional. En concordancia con el proyecto desarrollado durante el semestre, el componente inteligente reconoce **seis categorías documentales fundamentales**:

1. **`contrato`**: Acuerdos de voluntades, contratos laborales, de prestación de servicios, convenios y acuerdos de confidencialidad (NDA).
2. **`factura`**: Facturas electrónicas de venta, cuentas de cobro, comprobantes fiscales y recibos de pago.
3. **`reporte_financiero`**: Balances generales, estados de resultados, informes de auditoría contable y ejecuciones presupuestales.
4. **`manual_tecnico`**: Guías de instalación, especificaciones de arquitectura de software, manuales de usuario y documentación de APIs.
5. **`memorando`**: Comunicaciones internas de talento humano, circulares informativas y avisos institucionales.
6. **`acta_reunion`**: Actas de junta directiva, asambleas de accionistas, comités técnicos y minutas de seguimiento de proyectos.

El sistema recibe documentos en texto plano o archivos digitales en formatos **`.txt`, `.md`, `.pdf` y `.docx`**, reconociendo su contenido textual y determinando la tipología documental correspondiente.

---

## 2. Cómo funciona el modelo de reconocimiento (RNA)

### Arquitectura de la Red Neuronal Artificial
El reconocimiento de patrones no se realiza mediante palabras clave fijas o reglas rígidas, sino a través de un **Perceptrón Multicapa (MLPClassifier)**:

- **Capa de Entrada (TF-IDF Vector Space):**
  - Extracción de características textuales mediante `TfidfVectorizer` utilizando unigramas y bigramas (`ngram_range=(1, 2)`).
  - Ponderación sublineal de frecuencias (`sublinear_tf=True`) para suavizar el efecto de términos de alta frecuencia.
  - Dimensión del espacio vectorial de entrada: hasta 1000 términos y pares de palabras clave.
- **Capa Oculta (Extracción y Abstracción de Patrones):**
  - 32 neuronas artificiales densamente conectadas con función de activación no lineal **ReLU** ($f(x) = \max(0, x)$).
  - Captura combinaciones no lineales de términos representativos (por ejemplo, cláusulas jurídicas, estructuras contables, términos de ingeniería de software o protocolos de gestión humana).
- **Capa de Salida:**
  - 6 neuronas de salida correspondientes a las categorías documentales del sistema, con función de activación **Softmax**, que genera una distribución probabilística normalizada:
    $$P(C_i | X) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$
- **Algoritmo de Optimización y Entrenamiento:**
  - Optimizador **Adam** (Adaptive Moment Estimation) con tasa de aprendizaje inicial $\eta = 0.005$.
  - Regularización de pesos L2 (penalty $\alpha = 0.001$) para prevenir el sobreajuste.
  - Convergencia alcanzada en 86 épocas con una función de pérdida final de 0.0106.

### Proceso de Entrenamiento y Validación
El dataset fue dividido de forma estratificada:
- **Datos de Entrenamiento:** 48 documentos (80% del corpus etiquetado).
- **Datos de Prueba (Test Set):** 12 documentos (20% del corpus independiente, nunca vistos por el modelo).

### Métricas de Validación Obtenidas
- **Exactitud Global (Accuracy):** **91.67%** en el conjunto de prueba independiente.

| Categoría | Precisión | Recall | F1-Score | Muestras Test |
| :--- | :---: | :---: | :---: | :---: |
| **acta_reunion** | 100.0% | 50.0% | 66.7% | 2 |
| **contrato** | 100.0% | 100.0% | 100.0% | 2 |
| **factura** | 100.0% | 100.0% | 100.0% | 2 |
| **manual_tecnico** | 100.0% | 100.0% | 100.0% | 2 |
| **memorando** | 66.7% | 100.0% | 80.0% | 2 |
| **reporte_financiero** | 100.0% | 100.0% | 100.0% | 2 |
| **Promedio Ponderado** | 94.4% | 91.7% | 91.1% | 12 |


### Matriz de Confusión
| Real \ Pred | **acta_r** | **contra** | **factur** | **manual** | **memora** | **report** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **acta_r** | 1 | 0 | 0 | 0 | 1 | 0 |
| **contra** | 0 | 2 | 0 | 0 | 0 | 0 |
| **factur** | 0 | 0 | 2 | 0 | 0 | 0 |
| **manual** | 0 | 0 | 0 | 2 | 0 | 0 |
| **memora** | 0 | 0 | 0 | 0 | 2 | 0 |
| **report** | 0 | 0 | 0 | 0 | 0 | 2 |


---

## 3. Qué información registra la base de datos (Evidencia)

Para garantizar que las predicciones no queden únicamente en memoria RAM, se diseñó e implementó una estructura relacional en **SQLite** almacenada en el artefacto:  
`artifacts/evidencia_reconocimiento.db`.

### Esquema de la Tabla `evidencia_reconocimiento`
| Campo | Tipo | Propósito y Utilidad |
| :--- | :--- | :--- |
| `id` | `INTEGER PRIMARY KEY` | Identificador único autoincremental de la transacción o evidencia. |
| `timestamp` | `TEXT` | Marca de tiempo ISO-8601 en la que se ejecutó el reconocimiento. |
| `fuente_documento` | `TEXT` | Nombre del archivo analizado o identificador del documento. |
| `tipo_fuente` | `TEXT` | Formato físico o digital de origen (`pdf`, `docx`, `txt`, `texto_directo`). |
| `resumen_contenido` | `TEXT` | Extracto o muestra de los primeros caracteres del texto analizado. |
| `categoria_real` | `TEXT` | Etiqueta real conocida (o indicación de documento no etiquetado en producción). |
| `categoria_predicha` | `TEXT` | Tipo documental predicho por la red neuronal. |
| `nivel_confianza` | `REAL` | Probabilidad máxima arrojada por la capa Softmax de la RNA (0.00 a 1.00). |
| `estado_proceso` | `TEXT` | Indicador de control: `Procesado Exitosamente`, `Requiere Revisión` o `Rechazado`. |
| `categoria_dominio` | `TEXT` | Macro-área de la organización derivada de la ontología (`Dominio_Legal`, etc.). |
| `departamento_destino` | `TEXT` | Unidad administrativa responsable asignada por la ontología. |
| `accion_sugerida` | `TEXT` | Política operativa o procedimiento a aplicar sobre el documento. |
| `significado_ontologico` | `TEXT` | Frase interpretativa completa generada mediante el grafo del conocimiento. |

La base de datos cuenta con índices sobre `categoria_predicha` y `estado_proceso` para soportar consultas de auditoría, monitoreo de desviaciones e informes de gestión.

---

## 4. Qué conceptos y relaciones representa la ontología

La ontología del proyecto modela el conocimiento del dominio documental mediante un grafo dirigido estructurado, serializado en el estándar abierto:  
`artifacts/ontologia_documental.graphml`.

### Conceptos del Dominio (Nodos del Grafo)
1. **`DocumentoDigital`**: Representa el artefacto de entrada (archivo o texto).
2. **`RedNeuronal`**: Representa el agente de inteligencia artificial que procesa los patrones numéricos.
3. **`Prediccion`**: Representa la inferencia generada con su probabilidad asociada.
4. **`EvidenciaBD`**: Representa el registro persistente de auditoría en SQLite.
5. **`TipoDocumento`**: Representa la tipología clasificada (`contrato`, `factura`, `reporte_financiero`, etc.).
6. **`CategoriaDominio`**: Representa las áreas estratégicas de la institución (`Dominio_Legal`, `Dominio_Financiero`, `Dominio_Tecnologico`, `Dominio_Administrativo`).
7. **`DepartamentoResponsable`**: Representa las dependencias operativas (`Area_Juridica`, `Contabilidad_Tesoreria`, `Operaciones_TI`, `Gestion_Humana`).
8. **`AccionFlujo`**: Representa los protocolos de acción (`Custodia_Firmas_Legales`, `Auditoria_Tributaria`, `Despliegue_Documentacion_Tecnica`, `Difusion_Interna`).

### Relaciones entre Conceptos (Proposiciones Semánticas con Sentido de Frase)
Cada arista del grafo constituye una proposición semántica válida (`Sujeto -> Predicado -> Objeto`):

- `RedNeuronal` **analiza_patrones_de** `DocumentoDigital`
- `RedNeuronal` **genera_prediccion** `Prediccion`
- `Prediccion` **asigna_clase_a** `TipoDocumento`
- `Prediccion` **registra_evidencia_en** `EvidenciaBD`
- `EvidenciaBD` **garantiza_trazabilidad_de** `DocumentoDigital`
- `TipoDocumento` **pertenece_a_dominio** `CategoriaDominio`
- `TipoDocumento` **enruta_hacia** `DepartamentoResponsable`
- `TipoDocumento` **determina_accion** `AccionFlujo`
- `TipoDocumento` **incluye_tipo** `contrato`
- `TipoDocumento` **incluye_tipo** `factura`
- `TipoDocumento` **incluye_tipo** `reporte_financiero`
- `TipoDocumento` **incluye_tipo** `manual_tecnico`
- `TipoDocumento` **incluye_tipo** `memorando`
- `TipoDocumento` **incluye_tipo** `acta_reunion`
- `contrato` **pertenece_a_dominio** `Dominio_Legal`
- `contrato` **enruta_hacia** `Area_Juridica`
- `contrato` **determina_accion** `Custodia_Firmas_Legales`
- `factura` **pertenece_a_dominio** `Dominio_Financiero`
- `factura` **enruta_hacia** `Contabilidad_Tesoreria`
- `factura` **determina_accion** `Auditoria_Tributaria`
- `reporte_financiero` **pertenece_a_dominio** `Dominio_Financiero`
- `reporte_financiero` **enruta_hacia** `Contabilidad_Tesoreria`
- `reporte_financiero` **determina_accion** `Auditoria_Tributaria`
- `manual_tecnico` **pertenece_a_dominio** `Dominio_Tecnologico`
- `manual_tecnico` **enruta_hacia** `Operaciones_TI`
- `manual_tecnico` **determina_accion** `Despliegue_Documentacion_Tecnica`
- `memorando` **pertenece_a_dominio** `Dominio_Administrativo`
- `memorando` **enruta_hacia** `Gestion_Humana`
- `memorando` **determina_accion** `Difusion_Interna`
- `acta_reunion` **pertenece_a_dominio** `Dominio_Administrativo`
- `acta_reunion` **enruta_hacia** `Gestion_Humana`
- `acta_reunion` **determina_accion** `Difusion_Interna`


### Interpretación del Significado
Cuando la red neuronal predice, por ejemplo, que un archivo corresponde a una `factura`, la ontología permite interpretar que dicho documento pertenece al `Dominio_Financiero`, debe enrutarse hacia `Contabilidad_Tesoreria` y exige la acción operativa de `Auditoria_Tributaria`. De este modo, la predicción matemática se transforma en **conocimiento accionable** para la organización.

---

## 5. Integración completa: Flujo Verificado

Se ejecutaron pruebas integradas con documentos de diversos tipos y procedencias en disco (`.txt`, `.docx`, `.md`) para comprobar el ciclo completo:
$$\text{Entrada} \longrightarrow \text{Red Neuronal (RNA)} \longrightarrow \text{Evidencia (SQLite)} \longrightarrow \text{Significado (Ontología)}$$

### Evidencia de Registros Procesados
| ID BD | Documento Analizado | Formato | Cat. Real | Predicción RNA | Confianza | Estado | Destino Ontológico |
| :---: | :--- | :---: | :--- | :--- | :---: | :--- | :--- |
| 8 | `contrato_servicios_2026.txt` | `txt` | contrato | **contrato** | 92.3% | Procesado | Area_Juridica (Dominio_Legal) |
| 9 | `factura_proveedor_servicios.txt` | `txt` | factura | **factura** | 96.1% | Procesado | Contabilidad_Tesoreria (Dominio_Financiero) |
| 10 | `balance_general_ejercicio.docx` | `docx` | reporte_financiero | **reporte_financiero** | 99.7% | Procesado | Contabilidad_Tesoreria (Dominio_Financiero) |
| 11 | `manual_despliegue_contenedores.md` | `md` | manual_tecnico | **manual_tecnico** | 98.8% | Procesado | Operaciones_TI (Dominio_Tecnologico) |
| 12 | `circular_politica_laboral.txt` | `txt` | memorando | **memorando** | 98.8% | Procesado | Gestion_Humana (Dominio_Administrativo) |
| 13 | `acta_comite_directivo.txt` | `txt` | acta_reunion | **acta_reunion** | 98.5% | Procesado | Gestion_Humana (Dominio_Administrativo) |
| 14 | `receta_reposteria_atipica.txt` | `txt` | desconocido | **factura** | 21.0% | Revisión | Contabilidad_Tesoreria (Dominio_Financiero) |


---

## 6. Limitaciones encontradas

Durante la concepción, entrenamiento y validación del sistema se identificaron las siguientes limitaciones técnicas y operativas:

1. **Documentos basados en imágenes sin capa OCR:**  
   Si un archivo PDF o imagen contiene texto escaneado sin una capa OCR previa, la extracción textual produce cadenas vacías. El sistema cuenta con un mecanismo de detección temprana que rechaza el documento y sugiere un reescaneo con OCR, pero el clasificador neuronal en sí mismo no procesa píxeles directamente.
2. **Ambigüedad de vocabulario en documentos híbridos:**  
   Documentos como actas de junta directiva donde se discuten presupuestos financieros pueden contener términos simultáneos de finanzas y de gobernanza corporativa. En tales circunstancias, la red neuronal puede disminuir su nivel de certidumbre. El sistema mitiga este problema mediante un **umbral de confianza (threshold = 0.40)**: si la certeza es inferior, se clasifica en estado `Requiere Revisión Manual`.
3. **Escala y diversidad del corpus:**  
   El modelo actual fue entrenado sobre un conjunto de documentos estructurados en español. Textos provenientes de otras jurisdicciones o con jerga regional específica requerirán un proceso de reentrenamiento continuo (fine-tuning) alimentado por la misma base de datos de evidencia.

---

## 7. Estructura de Entregables en el Repositorio

El proyecto cumple a cabalidad con la estructura de archivos requerida:
```text
ia_semestre/
├── src/
│   └── semana08_representaciones.py       <- Código del modelo, base de datos y ontología
├── artifacts/
│   ├── modelo_red_neuronal.joblib        <- Red Neuronal MLP entrenada
│   ├── vectorizador_tfidf.joblib         <- Vocabulario y ponderaciones TF-IDF
│   ├── evidencia_reconocimiento.db       <- Base de datos SQLite con evidencia verificable
│   ├── ontologia_documental.graphml      <- Grafo de ontología en formato GraphML
│   └── muestras_prueba/                  <- Archivos reales (.txt, .docx, .md) procesados
└── reports/
    └── semana08.md                       <- Este informe técnico detallado
```
