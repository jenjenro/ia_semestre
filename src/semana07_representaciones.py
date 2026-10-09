import numpy as np
import os
import re

# ==========================================
# 1) Funciones de Extracción de Características
# ==========================================

def extraer_vector_numerico(texto):
    """Vector: [Cant. Palabras, Menciones 'firma' o 'notario', Menciones 'total' o '$']"""
    texto_min = texto.lower()
    palabras = len(texto_min.split())
    firmas = texto_min.count("firma") + texto_min.count("notario") + texto_min.count("acuerdan") + texto_min.count("suscrito")
    financiero = texto_min.count("total") + texto_min.count("$") + texto_min.count("nit") + texto_min.count("factura")
    return np.array([float(palabras), float(firmas), float(financiero)])

def extraer_hechos(texto):
    """Simbólico: Extrae un set de hechos lógicos del texto"""
    hechos = set()
    texto_min = texto.lower()
    if "cláusula" in texto_min or "contrato" in texto_min or "acuerdan" in texto_min:
        hechos.add("contiene_clausulas")
    if "firma" in texto_min or "suscrito" in texto_min or "notario" in texto_min:
        hechos.add("contiene_firma")
    if "nit" in texto_min or "factura" in texto_min:
        hechos.add("contiene_nit")
    if "precio" in texto_min or "total a pagar" in texto_min or "$" in texto_min:
        hechos.add("contiene_tabla_precios")
    return hechos

def extraer_secuencia(lineas):
    """Autómata: Transforma texto a secuencia (E, I, T)"""
    secuencia = ""
    for linea in lineas:
        linea_min = linea.lower().strip()
        if not linea_min: continue
        
        # Clasificar la línea en una sección estructural
        if "factura" in linea_min or "cliente" in linea_min or "nit" in linea_min:
            secuencia += "E" # Encabezado
        elif "$" in linea_min or "cantidad" in linea_min or "item" in linea_min:
            secuencia += "I" # Ítem / Renglón
        elif "total" in linea_min or "pagar" in linea_min:
            secuencia += "T" # Total
    return secuencia

# Plantillas / Modelos estáticos numéricos (Entrenamiento manual)
plantilla_contrato = np.array([40.0, 4.0, 0.0]) # Más palabras, varias firmas, nada financiero
plantilla_factura = np.array([40.0, 0.0, 6.0])  # Sin firmas, pero mucha terminología financiera

def clasificar_simbolico(hechos):
    if {"contiene_clausulas", "contiene_firma"}.issubset(hechos):
        return "Contrato Válido"
    elif {"contiene_tabla_precios", "contiene_nit"}.issubset(hechos):
        return "Factura Válida"
    return "Documento Desconocido"

def acepta_estructura_factura(secuencia):
    state = "q0"
    transitions = {
        ("q0", "E"): "q1", 
        ("q1", "I"): "q2",
        ("q2", "I"): "q2",
        ("q2", "T"): "q3",
    }
    for seccion in secuencia:
        state = transitions.get((state, seccion), "q_error")
        if state == "q_error": return False
    return state == "q3"

def procesar_archivo(ruta_archivo):
    print(f"\n==========================================")
    print(f" PROCESANDO: {ruta_archivo}")
    print(f"==========================================")
    
    try:
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            texto = f.read()
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            lineas = f.readlines()
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return None

    # 1. Numérico
    vector = extraer_vector_numerico(texto)
    dist_contrato = np.linalg.norm(vector - plantilla_contrato)
    dist_factura = np.linalg.norm(vector - plantilla_factura)
    res_numerico = "CONTRATO" if dist_contrato < dist_factura else "FACTURA"
    
    print("--- 1. ANÁLISIS NUMÉRICO ---")
    print(f"Vector extraído: {vector}")
    print(f"Distancia a Contrato: {dist_contrato:.2f} | Distancia a Factura: {dist_factura:.2f}")
    print(f"-> Clasificación Numérica: {res_numerico}")

    # 2. Simbólico
    hechos = extraer_hechos(texto)
    res_simbolico = clasificar_simbolico(hechos)
    print("\n--- 2. ANÁLISIS SIMBÓLICO ---")
    print(f"Hechos extraídos: {hechos}")
    print(f"-> Clasificación Simbólica: {res_simbolico}")

    # 3. Autómata
    secuencia = extraer_secuencia(lineas)
    es_valida = acepta_estructura_factura(secuencia)
    res_automata = "Estructura Correcta" if es_valida else "Estructura Inválida"
    print("\n--- 3. ANÁLISIS DE AUTÓMATA ---")
    print(f"Secuencia extraída: '{secuencia}'")
    print(f"-> Evaluación Autómata: {res_automata}")
    
    return {
        "archivo": os.path.basename(ruta_archivo),
        "vector": vector,
        "dist_contrato": dist_contrato,
        "dist_factura": dist_factura,
        "res_numerico": res_numerico,
        "hechos": hechos,
        "res_simbolico": res_simbolico,
        "secuencia": secuencia,
        "res_automata": res_automata
    }

# ==========================================
# Generación del Reporte Dinámico
# ==========================================
def generar_reporte(resultados):
    os.makedirs("reports", exist_ok=True)
    ruta_reporte = "reports/semana07.md"
    
    contenido = "# Reporte Semana 07 - Representaciones Adaptativas con Input Real\n\n"
    contenido += "En esta actualización, el sistema procesa archivos de texto reales extrayendo características y evaluándolas en tres modelos de representación (Numérico, Simbólico y Autómata).\n\n"
    
    for r in resultados:
        if r is None: continue
        contenido += f"## Documento Analizado: `{r['archivo']}`\n\n"
        
        contenido += "### 1. Análisis Numérico (Vectores)\n"
        contenido += f"- **Vector extraído:** `[{r['vector'][0]}, {r['vector'][1]}, {r['vector'][2]}]` *(Palabras, Menciones de Firmas, Menciones Financieras)*\n"
        contenido += f"- **Distancia a plantilla Contrato:** {r['dist_contrato']:.2f}\n"
        contenido += f"- **Distancia a plantilla Factura:** {r['dist_factura']:.2f}\n"
        contenido += f"- **Conclusión Numérica:** Clasificado como **{r['res_numerico']}**.\n\n"
        
        contenido += "### 2. Análisis Simbólico (Hechos Lógicos)\n"
        hechos_str = ", ".join(r['hechos']) if r['hechos'] else "Ninguno"
        contenido += f"- **Hechos detectados en el texto:** `{hechos_str}`\n"
        contenido += f"- **Conclusión Simbólica:** Evaluado como **{r['res_simbolico']}** por el motor de reglas deductivas.\n\n"
        
        contenido += "### 3. Autómatas (Validación Estructural Secuencial)\n"
        contenido += f"- **Secuencia extraída línea por línea:** `{r['secuencia'] if r['secuencia'] else 'Ninguna'}`\n"
        contenido += f"- **Conclusión del Autómata:** La estructura (Encabezado -> Ítems -> Total) fue evaluada como **{r['res_automata']}**.\n\n"
        contenido += "---\n"
    
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        f.write(contenido)
    print(f"\n¡Reporte dinámico generado con éxito en '{ruta_reporte}'!")


if __name__ == "__main__":
    resultados_totales = []
    
    print("Bienvenido al clasificador adaptativo de la Semana 07.")
    print("Puedes ingresar la ruta de los siguientes archivos de prueba generados:")
    print(" 1) x")
    print(" 2) data/pruebas_semana07/prueba_factura.txt")
    print(" 3) data/pruebas_semana07/prueba_invalida.txt")
    print("\nEscribe la ruta de un archivo para analizarlo, o presiona Enter para salir y generar el reporte final.")
    
    while True:
        try:
            ruta = input("\nRuta del archivo a analizar (Enter para terminar): ").strip()
            if not ruta:
                break
            if not os.path.exists(ruta):
                print("El archivo no existe. Intenta nuevamente.")
                continue
                
            res = procesar_archivo(ruta)
            if res:
                resultados_totales.append(res)
        except EOFError:
            # Manejar el caso de EOF si se corre de forma no interactiva
            break
            
    if resultados_totales:
        generar_reporte(resultados_totales)
    else:
        print("No se procesaron documentos. Saliendo...")