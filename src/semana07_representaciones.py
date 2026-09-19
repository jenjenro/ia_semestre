import numpy as np
import os

# ==========================================
# 1) Representación numérica
# ==========================================
plantilla_contrato = np.array([500.0, 2.0, 2.0])
plantilla_factura = np.array([80.0, 15.0, 0.0])
doc_desconocido = np.array([480.0, 3.0, 2.0]) 

dist_contrato = np.linalg.norm(doc_desconocido - plantilla_contrato)
dist_factura = np.linalg.norm(doc_desconocido - plantilla_factura)

resultado_numerico = "CONTRATO" if dist_contrato < dist_factura else "FACTURA"

print("--- 1. ANÁLISIS NUMÉRICO ---")
print(f"Distancia al Contrato: {dist_contrato:.3f}")
print(f"Distancia a la Factura: {dist_factura:.3f}")
print(f"-> Resultado: {resultado_numerico}\n")


# ==========================================
# 2) Representación simbólica
# ==========================================
print("--- 2. ANÁLISIS SIMBÓLICO ---")
hechos_doc1 = {"contiene_clausulas", "contiene_firma", "fecha_valida"}
hechos_doc2 = {"contiene_tabla_precios", "contiene_nit", "fecha_valida"}

def clasificar_documento(hechos):
    if {"contiene_clausulas", "contiene_firma"}.issubset(hechos):
        return "Contrato Válido"
    elif {"contiene_tabla_precios", "contiene_nit"}.issubset(hechos):
        return "Factura Válida"
    return "Documento Desconocido"

resultado_simbolico_1 = clasificar_documento(hechos_doc1)
resultado_simbolico_2 = clasificar_documento(hechos_doc2)

print(f"Doc1 clasificado como: {resultado_simbolico_1}")
print(f"Doc2 clasificado como: {resultado_simbolico_2}\n")


# ==========================================
# 3) Autómata 
# ==========================================
print("--- 3. ANÁLISIS DE AUTÓMATA ---")
def acepta_estructura_factura(estructura_doc):
    state = "q0"
    transitions = {
        ("q0", "E"): "q1", 
        ("q1", "I"): "q2",
        ("q2", "I"): "q2",
        ("q2", "T"): "q3",
    }
    for seccion in estructura_doc:
        state = transitions.get((state, seccion), "q_error")
        if state == "q_error": return False
    return state == "q3"

ejemplos_automata = ["EIT", "EIIIT", "ET", "IEIT"]
resultados_automata = {}

for doc in ejemplos_automata:
    es_valida = acepta_estructura_factura(doc)
    resultados_automata[doc] = es_valida # Guardamos el resultado para el reporte
    print(f"Estructura {doc} ¿Es factura válida?: {es_valida}")
print()


# ==========================================
# 4) Generación del Reporte Markdown Detallado
# ==========================================
print("--- 4. GENERANDO REPORTE ---")

os.makedirs("reports", exist_ok=True)
ruta_reporte = "reports/semana07.md"

# Formateamos los resultados del autómata como una lista de viñetas para Markdown
lista_automata_md = "\n".join([f"- Estructura `{k}` -> **{'Aceptada' if v else 'Rechazada'}**" for k, v in resultados_automata.items()])

contenido_md = f"""# Reporte Semana 07 - Representaciones del Reconocimiento

## Resultados de los Ejemplos Prácticos

### 1. Análisis Numérico (Similitud)
*Evaluando un documento desconocido frente a dos plantillas ideales:*
- **Distancia al modelo Contrato:** {dist_contrato:.3f}
- **Distancia al modelo Factura:** {dist_factura:.3f}
- **Conclusión:** El sistema clasificó el archivo como **{resultado_numerico}** por tener menor distancia.

### 2. Análisis Simbólico (Reglas y Metadatos)
*Clasificando documentos extraídos basándose en reglas lógicas estables:*
- **Ejemplo 1** (hechos: `{hechos_doc1}`): El resultado fue **{resultado_simbolico_1}**.
- **Ejemplo 2** (hechos: `{hechos_doc2}`): El resultado fue **{resultado_simbolico_2}**.

### 3. Autómatas (Validación Secuencial)
*Se evaluaron distintas estructuras para verificar que una Factura tenga un Encabezado (E), seguido de Ítems (I) y termine en Total (T):*
{lista_automata_md}

---

## Análisis Comparativo de las Representaciones

A continuación se analizan los enfoques implementados en este sistema de gestión documental:

| Representación | Ventajas | Limitaciones | Pérdida de Información |
| :--- | :--- | :--- | :--- |
| **Numérica (Vectores)** | Cálculos matemáticos precisos, fácil de escalar y comparar similitudes. | Difícil de interpretar para un humano, no explica el *por qué* de la decisión. | Se pierde la estructura del texto original, el orden y el contexto semántico. |
| **Simbólica (Hechos)** | Lógica clara, decisiones fáciles de explicar mediante reglas. | Es rígida; si falta un hecho exacto, la regla falla por completo (no hay "casi parecido"). | Se pierde la magnitud y la variabilidad de los datos (todo es Blanco/Negro, Verdadero/Falso). |
| **Autómata (Estados)** | Excelente para validar el orden secuencial y estructuras estrictas. | Muy sensible a cambios menores o excepciones en la estructura. | Se ignora por completo el contenido semántico de las secciones, solo evalúa la estructura. |

*Reporte generado automáticamente mediante Python en el proyecto IA Semestre.*
"""

with open(ruta_reporte, "w", encoding="utf-8") as archivo:
    archivo.write(contenido_md)

print(f"¡Reporte generado con éxito en '{ruta_reporte}'!")