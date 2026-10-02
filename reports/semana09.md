# Reporte Semana 09 - Reconocimiento de Imágenes: Características, Contornos y Segmentación

**Proyecto:** Sistema de Análisis y Gestión Documental con Inteligencia Artificial  
**Asignatura:** Inteligencia Artificial  
**Institución:** Escuela Tecnológica Instituto Técnico Central (ETITC)  
**Estudiante:** Jenny Valentina Rojas Orjuela  
**Fecha de entrega:** Sábado 3 de octubre de 2026 – 12:00 m.  
**Modalidad:** Individual  
**Entrega:** Repositorio GitHub – Commit Semana 9 (`semana09-reconocimiento-imagenes`)  

---

## 1. Selección de Imagen y Relación con el Proyecto

### 1.1 Qué representa la imagen
La imagen seleccionada y procesada (`data/imagen_proyecto.png`) corresponde a una **Factura Electrónica de Venta Comercial (No. FE-44910)**, emitida por la empresa proveedora *Infraestructura Cloud Andina S.A.S.* a la *Corporación Universitaria*. El documento contiene todos los elementos típicos de un documento tributario y administrativo real:
- **Encabezado y logotipo corporativo:** Razón social del emisor, NIT, régimen tributario y datos de contacto.
- **Identificador de factura y código CUFE:** Numeración oficial, hash criptográfico de validación fiscal DIAN y fechas de emisión/vencimiento.
- **Cuadro de datos del adquirente / cliente:** Información institucional del receptor.
- **Tabla estructurada de ítems facturados:** Cuadrícula con columnas de ítem, descripción de servicios en la nube, cantidad, valor unitario y subtotal.
- **Bloque de liquidación financiera y totales:** Desglose de subtotal, IVA del 19%, retención en la fuente (-4%) y total neto a pagar en pesos colombianos ($ 5.175.000 COP).
- **Código QR fiscal:** Matriz bidimensional de validación ante la autoridad tributaria (DIAN).
- **Sello institucional de auditoría contable y firma:** Estampilla de control en tinta azul con el texto *- APROBADO PARA PAGO -* y rúbrica manual de verificación.

Esta factura digitalizada representa la contraparte visual exacta de las muestras analizadas en la Semana 08 (`artifacts/muestras_prueba/factura_proveedor_servicios.txt`), permitiendo evaluar la transición de documentos puramente textuales hacia el procesamiento de imágenes documentales escaneadas.

```
+-----------------------------------------------------------------------------------+
| [LOGOTIPO]  INFRAESTRUCTURA CLOUD ANDINA S.A.S.        FACTURA ELECTRÓNICA        |
|             NIT: 900.542.118-4                         No. FE-44910               |
+-----------------------------------------------------------------------------------+
| FECHA: 28/08/2026 | CLIENTE: Corporación Universitaria | MONEDA: COP ($)         |
+-----------------------------------------------------------------------------------+
| ÍTEM | DESCRIPCIÓN                             | CANT | VR. UNITARIO | TOTAL      |
| 01   | Servidor Dedicado Cloud Compute         |  1   | $ 2.800.000  | $ 2.800.000|
| 02   | Almacenamiento Distribuido S3 5TB       |  1   | $   950.000  | $   950.000|
| 03   | Red Privada Virtual VPC y Firewall      |  1   | $   750.000  | $   750.000|
+-----------------------------------------------------------------------------------+
| CONDICIONES Y OBSERVACIONES            | SUBTOTAL:               $ 4.500.000 COP  |
| Pago transferencia Bancolombia         | IVA (19%):              $   855.000 COP  |
|                                        | RETENCIÓN (-4%):       -$   180.000 COP  |
|                                        | TOTAL A PAGAR:          $ 5.175.000 COP  |
+-----------------------------------------------------------------------------------+
| [QR FISCAL DIAN]     RÉGIMEN DE FACTURACIÓN     (SELLO AUDITORÍA Y FIRMA AZUL)    |
|                      Ley 1231 de 2008           ★ APROBADO PARA PAGO ★            |
+-----------------------------------------------------------------------------------+
```

### 1.2 Por qué es útil para el proyecto
Durante las semanas 01 a 08, el sistema operó asumiendo que los documentos estaban disponibles en formatos digitales de texto nativo (`.txt`, `.docx`, `.pdf` con capa de texto, `.md`). Sin embargo, en el entorno corporativo real:
1. Gran parte de las facturas, contratos notariales y actas de comités ingresan al sistema como **imágenes escaneadas, fotografías de archivo o documentos digitalizados en formato ráster**.
2. Un computador **no interpreta directamente** el contenido de una imagen; solo percibe una matriz tridimensional de números enteros $[0, 255]$ correspondientes a los canales rojo, verde y azul (RGB).
3. Las técnicas de visión por computador (filtrado, detección de contornos, umbralización y análisis de regiones conectadas) constituyen la **etapa fundamental de preprocesamiento e ingeniería de características** antes de poder invocar un motor OCR (Optical Character Recognition) o alimentar un modelo de clasificación profunda o estructuración documental (*Document Layout Analysis*).

### 1.3 Qué información se espera analizar
A través del procesamiento numérico de la imagen se busca extraer y analizar:
- **Fronteras estructurales y de diseño (*Layout*):** Reconocimiento de los límites de tablas, marcos de encabezado, cajas de metadatos y separadores de sección mediante gradientes de intensidad.
- **Separación figura/fondo:** Aislar los trazos de tinta (texto, líneas de cuadrícula, patrones del código QR y sellos) del soporte de papel blanco/claro.
- **Entidades gráficas independientes:** Cuantificar y caracterizar las componentes conexas para distinguir entre caracteres tipográficos individuales, puntuación ortográfica y macro-estructuras geométricas (como cuadrículas o sellos de aprobación).
- **Validación de autenticidad:** Identificar la presencia cromática y morfométrica de sellos institucionales y firmas que acrediten la autorización de pago de la factura.

---

## 2. Extracción de Características

Una imagen digital no posee semántica intrínseca para un procesador: es una función bidimensional de intensidad $I(x, y)$ o un tensor tridimensional $I(x, y, c)$ donde cada celda almacena un valor discreto de radiancia. En el sistema, la imagen fue cargada y convertida a representaciones matriciales mediante `numpy`, `PIL` y `scikit-image`.

### 2.1 Descomposición Estadística de Características
A partir de la ejecución del módulo `src/semana09_vision.py`, se obtuvieron las siguientes características numéricas de la imagen:

| Característica | Dimensión / Valor Obtenido | Significado Físico / Matemático |
| :--- | :---: | :--- |
| **Dimensiones Espaciales** | $1000 \times 1350$ px | Matriz de 1,350 filas por 1,000 columnas (relación de aspecto documental 1:1.35). |
| **Canales de Color** | 3 canales (RGB, uint8) | Espacio tricolor donde cada píxel tiene valores enteros en $[0, 255]$. |
| **Total de Píxeles** | $1,350,000$ píxeles | Cardinalidad total del espacio discreto de muestreo. |
| **Rango de Intensidad (Gris)** | $[0.0000, 1.0000]$ | Rango normalizado (0.0 representa negro puro / tinta máxima; 1.0 blanco puro / papel). |
| **Intensidad Promedio** | **$0.9219$** ($\pm 0.1903$) | Refleja un documento predominantemente blanco/claro: más del $92\%$ de la energía lumínica corresponde al fondo de papel. |
| **Intensidad Mediana** | **$0.9787$** | Confirma que el valor modal/típico de un píxel es el fondo de papel no entintado. |
| **Medias RGB** | $\bar{R} = 233.5, \bar{G} = 235.1, \bar{B} = 239.9$ | El canal azul presenta una ligera sobre-representación debido al sello de auditoría y franjas azules del encabezado. |
| **Píxeles Cromáticos** | $63,288$ px ($4.69\%$) | Píxeles donde $|R - B| > 25$, evidenciando elementos con color estructurado (sello azul y total rojo). |
| **Entropía de Shannon** | **$3.7723$ bits/píxel** | Grado de dispersión de información sobre 256 niveles de gris. Indica una distribución altamente bimodal y no uniforme. |

### 2.2 Qué características son útiles para el problema documental y por qué
1. **Intensidad (Escala de Grises):** Es la característica más crítica para la digitalización documental. La información textual y geométrica no depende del color del papel sino del contraste de reflectancia entre la tinta oscura y el fondo claro. Permite proyectar el tensor 3D a una sola matriz $I(x, y) \in [0, 1]$, reduciendo la complejidad computacional en un $66.7\%$ sin perder legibilidad tipográfica.
2. **Color (Espacio RGB):** Es fundamental para la **verificación forense de autenticidad**. Mientras el texto impreso es negro/gris neutro ($R \approx G \approx B$), el sello de auditoría y la firma manual utilizan tinta azul ($B \gg R$), y los indicadores de cobro o mora utilizan rojo ($R \gg B$). La separación por color permite aislar el sello de aprobación sin interferencia del texto subyacente.
3. **Bordes y Gradientes Espaciales:** Representan las discontinuidades abruptas de primer orden $\nabla I = \left(\frac{\partial I}{\partial x}, \frac{\partial I}{\partial y}\right)$. En documentos, los bordes delimitan las celdas de las tablas, los renglones, los contornos de cada letra y los límites de la página, sirviendo como base para segmentación estructural.
4. **Forma y Geometría de Componentes:** Permite clasificar regiones según su relación de aspecto, área y compacidad para discernir si una mancha de tinta es una letra, una línea horizontal divisoria o una rúbrica de firma.

---

## 3. Detección de Contornos mediante Canny

El algoritmo de Canny es el estándar óptimo para detección de bordes porque minimiza la probabilidad de falsos bordes, maximiza la localización espacial y garantiza una única respuesta por borde físico.

### 3.1 Etapas del Algoritmo
1. **Suavizado Gaussiano:** Convolución con un núcleo bidimensional parametrizado por la desviación estándar $\sigma$:
   $$G(x, y, \sigma) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$
2. **Cálculo del Gradiente:** Determinación de la magnitud del gradiente $|\nabla I|$ y su dirección $\theta$ mediante operadores de Sobel.
3. **Supresión de No Máximos:** Adelgazamiento de bordes preservando únicamente los píxeles que son máximos locales en la dirección ortogonal al contorno.
4. **Umbralización con Histéresis:** Aplicación de un umbral alto ($T_{\text{alto}}$) para iniciar bordes fuertes y un umbral bajo ($T_{\text{bajo}}$) para rastrear la continuidad de bordes débiles conectados.

### 3.2 Evaluación Experimental de la Variación del Parámetro $\sigma$
Se evaluó el comportamiento de Canny sobre la imagen en escala de grises variando sistemáticamente el parámetro $\sigma \in \{1.0, 1.5, 2.0, 3.0\}$:

| Parámetro $\sigma$ | Píxeles de Borde Detectados | Densidad de Borde (%) | Tipo de Elementos Capturados | Nivel de Detalle vs. Generalización |
| :---: | :---: | :---: | :--- | :--- |
| **$\sigma = 1.0$** | **$87,317$ px** | **$6.47\%$** | Letras pequeñas, trazos tipográficos finos, tildes, números de NIT, módulos del código QR, líneas milimétricas de cuadrícula. | **Alta resolución.** Óptimo para OCR y lectura de caracteres. Sensible a micro-rugosidades de digitalización. |
| **$\sigma = 1.5$** | $69,217$ px | $5.13\%$ | Caracteres medianos, palabras completas, líneas de separación y contornos exteriores del sello. | **Compromiso intermedio.** Se atenúan los puntos y serifas tipográficas menores. |
| **$\sigma = 2.0$** | $61,362$ px | $4.55\%$ | Textos en negrita (encabezados), bordes de cajas, marcos rectangulares de tablas y elipse del sello. | **Estructura media.** Los caracteres pequeños se fusionan o desaparecen parcialmente. |
| **$\sigma = 3.0$** | **$43,124$ px** | **$3.19\%$** | Cuadrícula de la tabla de servicios, marco perimetral exterior, recuadros de totales, divisor del banner y contorno del sello. | **Macro-estructura.** Suprime el texto interior y preserva la arquitectura del documento (*Page Layout*). |

```
Comparativa de Densidad de Bordes según Escala Gaussiana (Sigma):
Sigma = 1.0:  [##################################################] 6.47% (87,317 px) - Micro-detalles
Sigma = 1.5:  [#######################################           ] 5.13% (69,217 px) - Caracteres estándar
Sigma = 2.0:  [###################################               ] 4.55% (61,362 px) - Encabezados y bloques
Sigma = 3.0:  [########################                          ] 3.19% (43,124 px) - Estructura macro/tablas
```

### 3.3 Interpretación Técnica del Efecto de $\sigma$
El parámetro $\sigma$ actúa como un **filtro pasa-bajas espacial**:
- Al incrementar $\sigma$, el radio de convolución del filtro Gaussiano aumenta proporcionalmente ($3\sigma$). Esto promedia las variaciones de intensidad en vecindarios más grandes, eliminando las frecuencias espaciales altas donde residen los trazos tipográficos delgados (letras de 8 a 12 puntos).
- Como consecuencia, $\sigma = 1.0$ es indispensable para la **extracción de caracteres y lectura de datos**, mientras que $\sigma = 3.0$ es ideal para la **detección de tablas y zonificación (*layout analysis*)**, pues aísla las líneas divisorias sin el ruido visual producido por los miles de contornos de letras interiores.

---

## 4. Segmentación Mediante Umbral Automático de Otsu

La segmentación es el proceso de particionar la imagen en regiones con significado semántico. En documentos impresos o escaneados, el objetivo es separar el **primer plano** (texto, líneas de tabla, sellos, códigos) del **fondo** (papel soporte).

### 4.1 Principio Matemático del Método de Otsu
El método de Otsu calcula el umbral óptimo $T^*$ seleccionando el valor que maximiza la **varianza inter-clases** $\sigma_B^2(T)$, lo cual equivale matemáticamente a minimizar la varianza intra-clase $\sigma_W^2(T)$:

$$\sigma_B^2(T) = \omega_0(T) \omega_1(T) \left[ \mu_0(T) - \mu_1(T) \right]^2$$

Donde:
- $\omega_0(T)$ y $\omega_1(T)$ son las probabilidades acumuladas de las clases fondo y primer plano.
- $\mu_0(T)$ y $\mu_1(T)$ son las intensidades medias de cada clase para el umbral candidato $T$.

### 4.2 Resultados Obtenidos
Al aplicar `threshold_otsu` sobre la imagen normalizada $[0.0, 1.0]$, el algoritmo convergió en los siguientes parámetros:

- **Valor del Umbral de Otsu:** **$T = 0.6035$** (en escala entera $[0, 255]$: **$153.90$**).
- **Varianza Inter-Clases Máxima ($\sigma_B^2$):** **$0.033729$**, evidenciando un pico pronunciado de separabilidad estadística.
- **Partición Binaria Resultante:**
  - **Píxeles de Primer Plano (Tinta / Objetos de Interés, $I < T$):** **$90,611$ píxeles ($6.71\%$)**.
  - **Píxeles de Fondo (Papel Soporte, $I \ge T$):** **$1,259,389$ píxeles ($93.29\%$)**.

```
Distribución Bimodal de Intensidades y Umbral Óptimo de Otsu:

Densidad
  ▲
35│                                                    | Fondo Papel Blanco
30│                                                    | (~0.98)
25│                                                    | [████████████]
20│                                                    | [████████████]
15│                                                    | [████████████]
10│                                                    | [████████████]
 5│  Tinta Texto (~0.22)          Umbral Otsu (T=0.604)| [████████████]
 0└──[███]─────────────────────────────┼───────────────┴──────────────► Intensidad
    0.0       0.2       0.4           0.604    0.8       1.0
     (Negro)                                              (Blanco)
```

### 4.3 Explicación del Resultado
1. El histograma de intensidades (Panel 2 de la evidencia visual) muestra una **distribución bimodal asimétrica muy pronunciada**:
   - Un pico menor concentrado en intensidades bajas ($[0.05, 0.35]$) correspondiente a las tintas oscuras de impresión, las líneas de la cuadrícula y el sello.
   - Un pico masivo dominante centrado en $[0.95, 1.00]$ correspondiente al papel blanco.
2. El umbral automático $T = 0.6035$ ($153.9$) se sitúa en el valle intermedio de mínima densidad probabilística, garantizando una separación limpia:
   - Se suprime completamente cualquier ligera sombra o gradiente de digitalización del escáner.
   - No se pierden los trazos delgados de letras ni los módulos del código QR.
   - La máscara binaria resultante ($90,611$ píxeles útiles) retiene el $100\%$ de la información relevante para las fases posteriores de lectura y auditoría.

---

## 5. Análisis y Etiquetado de Regiones Conectadas

Una vez binarizada la imagen, la matriz booleana contiene conjuntos conexos de píxeles encendidos ($1$). Mediante el algoritmo de etiquetado con vecindad de 8-conectividad (`skimage.measure.label`), cada conjunto aislado de píxeles recibe un identificador entero único $k \in \{1, 2, \dots, N\}$.

### 5.1 Cuántas Regiones Fueron Encontradas
El análisis determinó un total de **$1,918$ regiones conectadas independientes** en el documento.

### 5.2 Caracterización y Clasificación de las Regiones
Mediante `regionprops`, se computaron las propiedades morfológicas (área, cuadro delimitador, centroide y relación de aspecto) de cada región. Los resultados cuantitativos se agrupan en tres escalas físicas:

| Categoría Morfológica | Rango de Área | Cantidad de Regiones | Porcentaje | Entidades Físicas Representadas en el Documento |
| :--- | :---: | :---: | :---: | :--- |
| **Micro-Regiones** | $< 15$ px | **$1,014$** | **$52.87\%$** | Puntos de la 'i', tildes ortográficas, puntos decimales de cifras monetarias, dos puntos (`:`), módulos individuales aislados del código QR y micro-ruido de textura. |
| **Regiones Medianas** | $15$ a $1,499$ px | **$895$** | **$46.66\%$** | Caracteres alfanuméricos completos (letras 'A', 'B', '0', '9'), símbolos monetarios (`$`), dígitos de NIT, palabras cortas continuas y trazos de firma. |
| **Macro-Regiones** | $\ge 1,500$ px | **$9$** | **$0.47\%$** | Marcos de la tabla de servicios, recuadro de totales, banner superior del encabezado, marco perimetral exterior y elipse del sello de auditoría. |
| **Totales Globales** | **$1$ a $13,459$ px** | **$1,918$** | **$100.0\%$** | Mediana: **$14.0$ px** \| Media: **$47.2$ px** |

### 5.3 ¿Corresponden o no a Objetos Reales?
En el contexto de la visión artificial aplicada a documentos, la correspondencia entre regiones conectadas y "objetos del dominio" requiere un análisis técnico riguroso:

1. **Correspondencia a Nivel Tipográfico (Glifos):**
   - La gran mayoría de las regiones medianas ($895$) corresponden efectivamente a **caracteres individuales** impresos en la factura (ej. cada número de la fecha `28/08/2026`, las letras de `SUBTOTAL SERVICIOS`, etc.).
   - Sin embargo, un carácter tipográfico no siempre equivale a una sola región conexa. Por ejemplo:
     - La letra minúscula `i` genera **dos regiones desconectadas**: el tallo vertical (categoría mediana) y el punto superior (categoría micro).
     - Las letras con tilde (`Í`, `Ó`) o signos de interrogación/puntuación (`:`, `;`, `!`) se fragmentan en dos o tres regiones conectadas independientes.
     - Letras cerradas como la `O`, `D` o `B` poseen "agujeros" interiores (*Euler characteristic* $< 1$) que forman parte de la misma componente conexa exterior.

2. **Correspondencia a Nivel Semántico (Palabras y Secciones):**
   - Una palabra como `INFORMACIÓN` no es una sola región conectada, sino un conjunto de 11 componentes independientes adyacentes.
   - Las macro-regiones ($9$) sí corresponden a **objetos estructurales de alto nivel**: el marco que encierra la tabla de ítems, el divisor de subtotales y el óvalo exterior del sello de aprobación.
   - En el código QR, los cuadros localizadores de las esquinas (*finder patterns*) son reconocidos como componentes conexas concéntricas de alta relevancia geométrica.

3. **Conclusión del Análisis:**
   Las regiones conectadas **no corresponden directamente a entidades semánticas abstractas completas** (como "el cliente" o "el total"), sino a los **átomos visuales constitutivos** (glifos, trazos y líneas). Para transformar estas $1,918$ componentes en información analizable por el clasificador del proyecto, es indispensable una etapa intermedia de agrupamiento espacial (*clustering* de renglones y palabras mediante operadores morfológicos de dilatación o distancias euclidianas de cajas envolventes).

---

## 6. Evidencia Visual Generada

Como evidencia del procesamiento realizado, se generó el artefacto visual multipanel en:  
`artifacts/semana09_vision.png` (resolución de 200 DPI, cuadrícula de $2 \times 3$ paneles).

```
+----------------------------------------------------------------------------------------------------+
|                                    EVIDENCIA VISUAL SEMANA 09                                      |
+---------------------------------+---------------------------------+--------------------------------+
| Panel 1:                        | Panel 2:                        | Panel 3:                       |
| Imagen Original del Proyecto    | Histograma de Intensidades      | Contornos Canny (Sigma = 1.0)  |
| Factura Electrónica FE-44910    | Umbral Otsu T = 0.604 (153.9)   | 87,317 px borde (6.47%)        |
| Espacio RGB (1000 x 1350 px)    | Separación bimodal fondo/tinta  | Detección fina de caracteres   |
+---------------------------------+---------------------------------+--------------------------------+
| Panel 4:                        | Panel 5:                        | Panel 6:                       |
| Contornos Canny (Sigma = 3.0)   | Segmentación Binaria Otsu       | Regiones Conectadas            |
| 43,124 px borde (3.19%)         | 90,611 px primer plano (6.71%)  | 1,918 componentes conexas      |
| Macro-estructura y tablas       | Máscara binaria tinta vs papel  | Mapa cromático de componentes  |
+---------------------------------+---------------------------------+--------------------------------+
```

### Descripción Detallada de los Paneles:
1. **Panel 1 (Imagen Original):** Muestra el documento comercial en su formato de entrada, con todos sus elementos contextuales (emisor, receptor, tabla de ítems, código QR y sello de auditoría).
2. **Panel 2 (Histograma y Umbral Otsu):** Ilustra la distribución de densidades de probabilidad en escala de grises. Se aprecia la clara bimodalidad y la línea discontinua roja que marca el umbral óptimo $T = 0.6035$, el cual separa el valle intermedio.
3. **Panel 3 (Canny $\sigma = 1.0$):** Resalta los bordes finos de alta frecuencia. Cada carácter, dígito, línea de división y elemento del código QR queda claramente delineado.
4. **Panel 4 (Canny $\sigma = 3.0$):** Demuestra el efecto de filtro pasa-bajas del incremento de escala Gaussiana. Los caracteres se atenúan y se preservan las líneas perimetrales de la tabla, los divisores principales y el contorno del sello.
5. **Panel 5 (Máscara Binaria Otsu):** Muestra la imagen segmentada en blanco y negro, donde los $90,611$ píxeles de tinta se encuentran aislados del fondo de papel, listos para binarización de OCR.
6. **Panel 6 (Regiones Conectadas):** Cada una de las $1,918$ componentes conexas se presenta con un color pseudocromático individual sobre fondo blanco, permitiendo apreciar visualmente la independencia de cada glifo, carácter y línea estructural.

---

## 7. Limitaciones Encontradas

Durante la implementación y experimentación con el pipeline de visión, se identificaron las siguientes limitaciones técnicas:

1. **Sensibilidad de Otsu Global a Variaciones de Iluminación:**
   El método de Otsu asume un fondo con iluminación uniforme en toda la escena. Si un documento físico es fotografiado con teléfono móvil o escaneado con la tapa abierta, se producen sombras no uniformes o degradados luminosos. En tales condiciones, un umbral global único clasifica incorrectamente zonas sombreadas como tinta, requiriendo **umbrales adaptativos locales (como los algoritmos de Sauvola o Niblack)**.
2. **Desconexión y Fragmentación Tipográfica:**
   El etiquetado de regiones conexas es puramente topológico y no semántico. Elementos que conceptualmente forman una sola letra (como la `i` o caracteres con tilde) o una sola palabra quedan fragmentados en múltiples regiones, lo que impide interpretar palabras directamente sin un modelo de agrupamiento geométrico.
3. **Fusión de Trazos por Baja Resolución (*Touching Characters*):**
   Si la resolución de escaneo desciende por debajo de 150 DPI o si la tinta física se expande sobre papel de baja calidad, letras consecutivas (como `cl`, `rn`, `fi`) entran en contacto físico, siendo etiquetadas como una sola macro-región y degradando la precisión del reconocimiento posterior.
4. **Pérdida de Información Cromática en Grayscale:**
   Al convertir la imagen a escala de grises para Canny y Otsu, se pierde la diferenciación entre tinta negra de imprenta y tinta azul de sellos y firmas, dificultando la discriminación forense si no se mantiene una rama paralela de procesamiento cromático en el espacio HSV o Lab.

---

## 8. Aplicación Futura dentro del Proyecto

El desarrollo alcanzado en esta Semana 09 sienta las bases para evolucionar el **Sistema de Análisis y Gestión Documental** hacia una plataforma multimodal integral:

1. **Clasificación Documental Multimodal (Fusión Imagen + Texto):**
   Integrar las características visuales extraídas en esta semana (densidad de bordes a $\sigma = 3.0$, relación de aspecto de macro-regiones, presencia de sello por análisis de color) como variables de entrada adicionales para el modelo de Red Neuronal Artificial (MLP) de la Semana 08. Un documento con alta densidad de líneas ortogonales será clasificado visualmente como `factura` o `reporte_financiero` antes de procesar su texto.
2. **Extracción Automática de Tablas (*Table Extraction*):**
   Utilizar la máscara de bordes de Canny a $\sigma = 3.0$ combinada con la transformada de Hough o proyecciones horizontales/verticales para detectar automáticamente las celdas de las tablas de ítems, permitiendo tabular los valores facturados en estructuras de base de datos relacional (SQLite).
3. **Módulo de Detección y Verificación de Sellos/Firmas:**
   Utilizar la segmentación cromática desarrollada ($|R - B| > 25$) y el etiquetado de regiones elípticas para aislar automáticamente el sello de auditoría contable. El sistema podrá comprobar de forma autónoma si una factura entrante cuenta con la firma de aprobación requerida para el pago institucional.
4. **Preprocesamiento Avanzado para Motor OCR:**
   Emplear la máscara binaria limpia obtenida mediante Otsu para alimentar motores de reconocimiento óptico de caracteres (ej. Tesseract / EasyOCR), mejorando drásticamente la tasa de acierto en la extracción de texto frente a imágenes crudas no procesadas.

---

## 9. Instrucciones de Ejecución y Reproducibilidad

Para reproducir todos los resultados, métricas y evidencias de esta semana:

```bash
# 1. Asegurar el entorno virtual y dependencias
source .venv/bin/activate  # En Linux/macOS
# o en Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# 2. Instalar requerimientos actualizados (incluye scikit-image)
pip install -r requirements.txt

# 3. Ejecutar el pipeline de visión artificial
python src/semana09_vision.py
```

Al finalizar la ejecución, el script confirmará:
- La extracción de características estadísticas de la matriz.
- El cálculo del umbral de Otsu ($T = 0.6035$).
- Los conteos de bordes para $\sigma \in \{1.0, 1.5, 2.0, 3.0\}$.
- La identificación de las $1,918$ regiones conectadas.
- La generación de la figura científica en `artifacts/semana09_vision.png`.
