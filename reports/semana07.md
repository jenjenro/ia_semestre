# Reporte Semana 07 - Representaciones del Reconocimiento

## Resultados de los Ejemplos Prácticos

### 1. Análisis Numérico (Similitud)
*Evaluando un documento desconocido frente a dos plantillas ideales:*
- **Distancia al modelo Contrato:** 20.025
- **Distancia al modelo Factura:** 400.185
- **Conclusión:** El sistema clasificó el archivo como **CONTRATO** por tener menor distancia.

### 2. Análisis Simbólico (Reglas y Metadatos)
*Clasificando documentos extraídos basándose en reglas lógicas estables:*
- **Ejemplo 1** (hechos: `{'fecha_valida', 'contiene_clausulas', 'contiene_firma'}`): El resultado fue **Contrato Válido**.
- **Ejemplo 2** (hechos: `{'contiene_tabla_precios', 'fecha_valida', 'contiene_nit'}`): El resultado fue **Factura Válida**.

### 3. Autómatas (Validación Secuencial)
*Se evaluaron distintas estructuras para verificar que una Factura tenga un Encabezado (E), seguido de Ítems (I) y termine en Total (T):*
- Estructura `EIT` -> **Aceptada**
- Estructura `EIIIT` -> **Aceptada**
- Estructura `ET` -> **Rechazada**
- Estructura `IEIT` -> **Rechazada**

---

## Análisis Comparativo de las Representaciones

A continuación se analizan los enfoques implementados en este sistema de gestión documental:

| Representación | Ventajas | Limitaciones | Pérdida de Información |
| :--- | :--- | :--- | :--- |
| **Numérica (Vectores)** | Cálculos matemáticos precisos, fácil de escalar y comparar similitudes. | Difícil de interpretar para un humano, no explica el *por qué* de la decisión. | Se pierde la estructura del texto original, el orden y el contexto semántico. |
| **Simbólica (Hechos)** | Lógica clara, decisiones fáciles de explicar mediante reglas. | Es rígida; si falta un hecho exacto, la regla falla por completo (no hay "casi parecido"). | Se pierde la magnitud y la variabilidad de los datos (todo es Blanco/Negro, Verdadero/Falso). |
| **Autómata (Estados)** | Excelente para validar el orden secuencial y estructuras estrictas. | Muy sensible a cambios menores o excepciones en la estructura. | Se ignora por completo el contenido semántico de las secciones, solo evalúa la estructura. |

*Reporte generado automáticamente mediante Python en el proyecto IA Semestre.*
