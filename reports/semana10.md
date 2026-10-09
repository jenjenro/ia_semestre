# Reporte Semana 10 - Reconocimiento de Imágenes (Texturas LBP, Regiones e Histogramas)

## 1. Selección de Imágenes y Aplicación al Proyecto
Para este ejercicio, se han seleccionado dos imágenes sintéticas que representan partes críticas del análisis documental (nuestro proyecto central):
1. **Imagen de Texto (`img_texto.png`)**: Representa un recorte de un bloque de texto denso (por ejemplo, los ítems de una factura o las cláusulas de un contrato). Se busca analizar la textura repetitiva y de alta frecuencia que generan las letras impresas.
2. **Imagen de Firma (`img_firma.png`)**: Representa un recorte del área de validación de un documento donde se encuentra una firma manuscrita. Se busca analizar los trazos continuos, orgánicos y dispersos, que contrastan fuertemente con la estructura rígida del texto impreso.

Ambas imágenes fueron procesadas añadiendo ruido gausiano al fondo para simular escaneos reales de papel.

## 2. Segmentación mediante Histogramas
Al procesar las imágenes, se calculó el histograma de intensidad de los píxeles. Dado que los documentos suelen tener texto oscuro (valores cercanos a 0) sobre un fondo claro (valores cercanos a 255), el histograma muestra dos picos distintos (bimodal).

**Umbral automático de Otsu:**
El método de Otsu busca automáticamente el punto exacto en el valle entre estos dos picos para minimizar la varianza intra-clase. 
- Al aplicar este umbral (ej. 79.00 para texto y 60.00 para la firma), los píxeles menores al umbral se clasifican como el "objeto" (letras o tinta de la firma) y los mayores como "fondo" (papel). 
- Esto genera una **máscara binaria** donde podemos separar perfectamente los trazos de interés del papel ruidoso de fondo.

## 3. Etiquetado de Regiones
Al usar `measure.label()` y `regionprops()` sobre la máscara binaria, se identificaron múltiples regiones conectadas, filtrando aquellas con un área menor a 10 píxeles para ignorar pequeños puntos de ruido.

**Resultados obtenidos:**
- **Imagen de Texto:** Se encontraron 231 regiones. El área promedio es de 154 píxeles (con una desviación estándar moderada de 86 píxeles). Esto tiene sentido: cada letra o palabra pequeña forma una región separada, resultando en muchas regiones de tamaño uniforme.
- **Imagen de Firma:** Se encontraron 4 regiones. El área promedio es de 526 píxeles (con una gran desviación de 683 píxeles). Las firmas consisten en trazos largos y continuos, por lo que resultan en muy pocas regiones pero con un área significativamente mayor y variable.

## 4. Análisis de Texturas LBP (Local Binary Patterns)
LBP analiza la textura de la imagen comparando cada píxel con sus vecinos. El histograma LBP resultante captura los micro-patrones (bordes, esquinas, áreas planas).

**Comparación entre imágenes:**
- **Textura de Texto:** Presenta una distribución mucho más equilibrada a lo largo del histograma LBP. Esto se debe a la gran cantidad de bordes verticales, horizontales, y esquinas que forman los caracteres impresos de manera repetitiva y estructurada.
- **Textura de Firma:** Presenta un histograma mucho más concentrado en patrones "planos" (fondo de papel) y bordes muy específicos (curvas suaves). Hay menos variabilidad estructural porque gran parte de la imagen es fondo en blanco y los trazos son líneas gruesas y continuas, sin las esquinas afiladas de la tipografía.

## 5. Limitaciones del enfoque
1. **Sensibilidad a la iluminación y ruido:** Aunque Otsu funciona muy bien en este entorno bimodal simulado, en escaneos reales con sombras, arrugas o manchas de café, un umbral global fallará. Se requeriría un umbral adaptativo (local).
2. **Rotación y Escala:** LBP clásico no es 100% invariante a cambios extremos de escala. Si escaneamos el documento con el doble de resolución, las texturas capturadas por el mismo radio de LBP cambiarán drásticamente.
3. **Firmas Fragmentadas:** El etiquetado de regiones puede sub-contar o sobre-contar componentes si la pluma se levantó del papel o si la tinta está desgastada.

## 6. Vector de Características
Toda la información extraída (número de regiones, área, desviación, histograma de intensidad normalizado e histograma LBP) ha sido concatenada usando NumPy en un único vector por imagen. Estos vectores han sido guardados en el archivo `artifacts/semana10_features.npy`. 

Este paso es vital para el proyecto, ya que ahora podemos usar estos vectores numéricos para entrenar un modelo de Machine Learning que pueda clasificar automáticamente si un recorte de documento corresponde a "Texto normal" o a una "Zona de Firma".
