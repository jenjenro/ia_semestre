# Explicación Detallada del Código: Semana 09 - Visión por Computador

El archivo `src/semana09_vision.py` implementa un flujo completo de procesamiento digital de imágenes enfocado en el análisis documental. A diferencia de la semana 08 (que procesaba texto con IA), este módulo trata a los documentos puramente como matrices de píxeles (imágenes) para extraer características numéricas, geométricas y estructurales utilizando la biblioteca `scikit-image` (`skimage`).

A continuación, se detalla el funcionamiento de cada sección del código:

---

### 1. Generación de la Imagen de Prueba (`asegurar_imagen_proyecto`)
Dado que los sistemas de visión necesitan imágenes para trabajar, esta función verifica si existe `imagen_proyecto.png`. Si no existe, genera dinámicamente una factura electrónica muy realista usando la librería `Pillow (PIL)`.
- Crea un lienzo blanco y dibuja rectángulos, líneas y textos usando `ImageDraw`.
- Simula una tabla de cobros, logotipos y totales.
- Genera un código QR simulado dibujando píxeles aleatorios en una cuadrícula.
- Genera un **sello azul** circular de "Auditoría Contable" y una firma.
- Por último, aplica un degradado de fondo y ruido gaussiano para simular que el documento fue *escaneado* y no es un PDF perfecto.

### 2. Extracción de Características Numéricas (`extraer_caracteristicas_imagen`)
Esta función convierte la imagen en datos estadísticos brutos.
- Convierte la imagen a escala de grises (`rgb2gray`) y extrae la intensidad mínima, máxima, media y mediana. 
- **Análisis de Color:** Separa los canales RGB. Además, busca píxeles donde el canal Rojo y el Azul tengan una diferencia mayor a 25 (`diff_chroma > 25`). Esto sirve para **detectar sellos o firmas** en tinta azul o roja, ignorando el texto negro impreso (donde R, G y B son casi iguales).
- **Entropía de Shannon:** Calcula cuánta "información" o variación hay en la imagen usando el histograma de colores.

### 3. Detección de Contornos (`evaluar_contornos_canny`)
Utiliza el algoritmo **Canny** (`skimage.feature.canny`) para encontrar los bordes de las figuras geométricas (letras, cajas de la tabla).
- Evalúa múltiples escalas Gaussianas iterando sobre una lista de `sigmas` (1.0, 1.5, 2.0, 3.0).
- Un `sigma` bajo (1.0) es muy sensible y detecta letras pequeñas y detalles finos.
- Un `sigma` alto (3.0) aplica un fuerte difuminado previo, ignorando el texto y detectando solo la macro-estructura (los recuadros grandes de la tabla).

### 4. Segmentación Automática (`segmentar_otsu`)
Para separar el texto/tinta del papel/fondo, no se puede usar un valor de corte fijo (ya que algunos escáneres son más oscuros que otros).
- Se usa `threshold_otsu()`, que calcula matemáticamente el umbral perfecto (T) que separa los píxeles claros de los oscuros maximizando la varianza inter-clases.
- Todo píxel por debajo de ese umbral (más oscuro) se clasifica como "Primer Plano" (tinta). Todo lo que está por encima es "Fondo" (papel).
- Calcula qué porcentaje de la hoja tiene tinta (`fg_pct`).

### 5. Análisis de Regiones Conectadas (`analizar_regiones_conectadas`)
Una vez separado el texto del fondo, el sistema necesita saber *qué son* esas manchas de tinta.
- Utiliza la función `label()` para agrupar píxeles negros que se tocan entre sí, formando "Regiones" o componentes conexas.
- Usa `regionprops()` para calcular el área (en píxeles) de cada mancha y las divide en tres categorías:
  - **Micro-regiones (< 15 px):** Puntos, comas, tildes, puntitos del código QR o simple ruido del escáner.
  - **Regiones medias (15 - 1500 px):** Letras individuales, números y glifos legibles.
  - **Macro-regiones (>= 1500 px):** Los recuadros exteriores de la factura, celdas de la tabla o el círculo del sello.
- El número de "regiones medias" es crucial en la API (`src/app.py`) para distinguir si es un plano arquitectónico (pocas letras) o una factura (miles de letras).

### 6. Evidencia Visual Multipanel (`generar_evidencia_visual`)
Toma todos los análisis matemáticos previos y utiliza `matplotlib` para generar un panel científico de 6 gráficas (`semana09_vision.png`):
1. **Imagen Original:** La factura escaneada.
2. **Histograma:** La distribución bimodal de los colores, marcando la línea vertical donde Otsu cortó el fondo de la tinta.
3. **Canny (Sigma=1.0):** El mapa de todos los bordes finos.
4. **Canny (Sigma=3.0):** El mapa de los bordes gruesos.
5. **Máscara Otsu:** Imagen en blanco y negro puro.
6. **Regiones Conectadas:** Colorea cada mancha de tinta independiente con un color aleatorio distinto usando `label2rgb`, demostrando cómo la computadora ve "objetos separados" en lugar de palabras.

---

### Conclusión y Conexión con el Proyecto
Toda esta lógica de la semana 09 está encapsulada e importada dentro del backend (`src/app.py`). Cuando subes una imagen PNG o JPG en el Dashboard, la API reutiliza estas funciones de `skimage` para determinar si es un Comprobante de Pago, una Ecografía Médica, o una Factura, basándose enteramente en **geometría y morfología sin necesidad de leer o entender el texto (OCR)**.

