# Semana 03 - Taxonomía de Inteligencia Artificial

## Resultado automático frente a clasificación manual de referencia

| Caso | Categoría automática principal | Categorías detectadas | Manual | Estado |
|---:|---|---|---|---|
| 1 | Visión por computador y OCR | Visión por computador y OCR, Procesamiento de lenguaje natural, Clasificación documental | Visión por computador y OCR | Coincide |
| 2 | Clasificación documental | Clasificación documental | Clasificación documental | Coincide |
| 3 | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural | Coincide |
| 4 | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural | Coincide |
| 5 | Búsqueda y recuperación documental | Búsqueda y recuperación documental | Búsqueda y recuperación documental | Coincide |
| 6 | Búsqueda y recuperación documental | Búsqueda y recuperación documental | Búsqueda y recuperación documental | Coincide |
| 7 | Sistemas expertos y reglas documentales | Sistemas expertos y reglas documentales | Sistemas expertos y reglas documentales | Coincide |
| 8 | Visión por computador y OCR | Visión por computador y OCR | Visión por computador y OCR | Coincide |
| 9 | Aprendizaje automático predictivo | Aprendizaje automático predictivo | Clasificación documental | Revisar |
| 10 | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural | Coincide |
| 11 | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural, Clasificación documental | Procesamiento de lenguaje natural | Coincide |
| 12 | Sistemas expertos y reglas documentales | Sistemas expertos y reglas documentales | Sistemas expertos y reglas documentales | Coincide |
| 13 | Requiere análisis | Requiere análisis | Búsqueda y recuperación documental | Revisar |
| 14 | Requiere análisis | Requiere análisis | Aprendizaje automático predictivo | Revisar |
| 15 | Búsqueda y recuperación documental | Búsqueda y recuperación documental | Búsqueda y recuperación documental | Coincide |
| 16 | Clasificación documental | Clasificación documental | Clasificación documental | Coincide |
| 17 | Sistemas expertos y reglas documentales | Sistemas expertos y reglas documentales | Sistemas expertos y reglas documentales | Coincide |
| 18 | Automatización documental inteligente | Automatización documental inteligente | Automatización documental inteligente | Coincide |
| 19 | Aprendizaje automático predictivo | Aprendizaje automático predictivo | Aprendizaje automático predictivo | Coincide |
| 20 | Requiere análisis | Requiere análisis | Automatización documental inteligente | Revisar |

Coincidencia con la referencia: **80.00%** (16/20).

## Cinco reglas propias

Se implementaron cinco reglas propias en `CUSTOM_RULES`, orientadas a mejorar la clasificación de problemas relacionados con el análisis documental.

### 1. Visión por computador y OCR

Se agregaron expresiones específicas para identificar documentos escaneados, imágenes y fotografías de documentos, así como procesos de conversión de imágenes a texto.

- `documento escaneado`
- `imagen del documento`
- `fotografia del documento`
- `convertir imagen en texto`

Estas reglas son pertinentes porque permiten diferenciar problemas donde el documento se encuentra inicialmente como una imagen y requiere OCR para obtener su contenido textual.

### 2. Procesamiento de lenguaje natural

Se agregaron reglas relacionadas con la extracción y análisis de información textual.

- `extraer nombres`
- `extraer fechas`
- `extraer entidades`
- `analizar contenido textual`

Estas expresiones permiten identificar tareas de extracción de información y análisis de texto, propias del procesamiento de lenguaje natural.

### 3. Clasificación documental

Se incorporaron reglas orientadas a identificar cuándo el objetivo principal es determinar el tipo o categoría de un documento.

- `identificar tipo de documento`
- `detectar tipo de documento`
- `reconocer documento`
- `clasificar documentos`

Estas reglas son pertinentes porque permiten diferenciar la clasificación documental de otras tareas de procesamiento de texto.

### 4. Sistemas expertos y reglas documentales

Se agregaron reglas relacionadas con la verificación de requisitos, validación de documentos y detección de información faltante.

- `verificar requisitos`
- `validar documento`
- `documento completo`
- `informacion faltante`

Estas expresiones permiten identificar problemas donde la decisión depende del cumplimiento de reglas o requisitos establecidos.

### 5. Automatización documental inteligente

Se agregaron reglas relacionadas con procesos automáticos de procesamiento, extracción y generación de información.

- `procesar automaticamente`
- `extraer automaticamente`
- `analizar automaticamente`
- `generar reporte`

Estas reglas son pertinentes porque permiten identificar procesos donde varias tareas documentales son ejecutadas automáticamente mediante sistemas de inteligencia artificial.

## Ampliación de categorías

Además de las cinco reglas propias, se ampliaron las palabras clave de las categorías existentes para mejorar la cobertura del dominio de análisis documental.

En **Visión por computador y OCR** se incorporaron términos como imagen, fotografía, cámara, escaneo, documento escaneado, OCR, reconocimiento óptico, digitalizado y captura.

En **Procesamiento de lenguaje natural** se incorporaron términos como texto, lenguaje, frase, párrafo, resumen, analizar texto, extraer nombres, extraer fechas y extraer entidades.

En **Aprendizaje automático predictivo** se incorporaron términos relacionados con predicción, probabilidad, patrones, anomalías, riesgo y detección automática.

En **Clasificación documental** se incorporaron expresiones relacionadas con tipos de documentos y ejemplos concretos como factura, contrato, certificado, acta, formulario y hoja de vida.

En **Búsqueda y recuperación documental** se incorporaron términos como buscar, búsqueda, encontrar, localizar, consultar, palabra clave, documentos relacionados y recuperar información.

En **Sistemas expertos y reglas documentales** se incorporaron términos como regla, validar, verificar, requisito, obligatorio, información faltante, cumple e incumple.

En **Automatización documental inteligente** se incorporaron términos como automatizar, procesar documento, procesamiento documental, procesar automáticamente, extraer información, generar reporte y flujo documental.

## Discrepancias y análisis

### Caso 9 - Clasificación automática de documentos

**Descripción:** Clasificar automáticamente documentos según su contenido textual.

- **Resultado automático:** Aprendizaje automático predictivo.
- **Referencia manual:** Clasificación documental.
- **Activador:** `clasificar automaticamente`.
- **Análisis:** la expresión utilizada en el caso coincide directamente con la categoría Aprendizaje automático predictivo. Sin embargo, el objetivo del problema es determinar la categoría o tipo de documentos, por lo que la referencia manual lo ubica en Clasificación documental.
- **Modificación propuesta:** retirar o restringir la expresión `clasificar automaticamente` de Aprendizaje automático predictivo o darle mayor prioridad a Clasificación documental cuando la clasificación tenga como objeto principal un documento.

### Caso 13 - Comparación de documentos

**Descripción:** Comparar dos versiones de un documento para identificar diferencias.

- **Resultado automático:** Requiere análisis.
- **Referencia manual:** Búsqueda y recuperación documental.
- **Activador:** no existe una coincidencia directa suficiente con las palabras clave actuales.
- **Análisis:** el problema se enfoca en comparar dos versiones de un documento e identificar diferencias. Esta operación no está representada explícitamente dentro de las palabras clave actuales de Búsqueda y recuperación documental.
- **Modificación propuesta:** agregar términos como `comparar documentos`, `comparar versiones`, `diferencias` o `comparación documental` a una categoría adecuada. Si se mantiene la referencia manual, podrían incorporarse estas expresiones a Búsqueda y recuperación documental.

### Caso 14 - Detección de documentos duplicados

**Descripción:** Detectar documentos duplicados dentro de un repositorio documental.

- **Resultado automático:** Requiere análisis.
- **Referencia manual:** Aprendizaje automático predictivo.
- **Activador:** no existe una palabra clave específica para `documentos duplicados` dentro de la categoría de aprendizaje automático predictivo.
- **Análisis:** la detección de duplicados puede utilizar patrones, similitud de contenido o técnicas de aprendizaje automático. Sin embargo, estas expresiones no fueron incluidas explícitamente en las reglas actuales.
- **Modificación propuesta:** agregar términos como `duplicado`, `documentos duplicados`, `detectar duplicados` y `similitud documental` a la categoría de Aprendizaje automático predictivo, si se desea conservar la clasificación manual establecida.

### Caso 20 - Generación automática de reportes

**Descripción:** Generar un reporte con la información extraída de múltiples documentos.

- **Resultado automático:** Requiere análisis.
- **Referencia manual:** Automatización documental inteligente.
- **Activador:** `generar reporte`.
- **Análisis:** aunque `generar reporte` está incluido en `CUSTOM_RULES` de Automatización documental inteligente, el resultado muestra que la expresión no fue suficiente para establecer la categoría principal bajo la lógica actual de clasificación.
- **Modificación propuesta:** reforzar la regla para que `generar reporte` tenga mayor peso cuando aparezca junto con expresiones como `información extraída`, `múltiples documentos` o `generar reporte automáticamente`.

## Nota técnica

Un problema real puede pertenecer a varias áreas de IA. La columna 'principal' utiliza la categoría con mayor cantidad de coincidencias, mientras que las demás coincidencias se conservan como categorías secundarias.

## Conclusión

Después de ampliar las categorías e implementar cinco reglas propias orientadas al dominio documental, el sistema obtuvo una coincidencia del **80.00%** con la clasificación manual de referencia. Los resultados muestran que las modificaciones realizadas permiten clasificar correctamente la mayoría de los casos. Las discrepancias encontradas permiten identificar oportunidades de mejora en la selección de palabras clave, el peso de las reglas y la prioridad entre categorías.