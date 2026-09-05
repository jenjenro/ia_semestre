import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# 1. Base de conocimiento
def cargar_base_conocimiento():
    # Obtenemos la ruta absoluta de la carpeta donde está este script ('src')
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    
    # Construimos la ruta apuntando a la carpeta 'data'
    ruta_absoluta = os.path.join(directorio_script, "..", "data", "base_conocimiento.txt")
    
    with open(ruta_absoluta, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# 2. Clasificación (15 ejemplos etiquetados)
# 10 adaptados al proyecto de Análisis Documental y 5 para completar la lógica
textos_entrenamiento = [
    "Contrato de prestación de servicios con cláusula de confidencialidad",
    "Factura de venta por servicios de desarrollo de software",
    "Balance general del tercer trimestre fiscal",
    "Acuerdo de no divulgación firmado por las partes",
    "Reporte de vulnerabilidades detectadas en el sistema",
    "Manual de usuario de la plataforma de análisis",
    "Acta de constitución de la sociedad anónima",
    "Cotización de servidores en la nube y arquitectura",
    "Guía de instalación de dependencias y despliegue",
    "Resolución de aprobación de presupuesto anual",
    "Solicitud de vacaciones del empleado de planta",
    "Certificado médico de incapacidad por tres días",
    "Llamado de atención por incumplimiento de horario",
    "Respuesta formal a queja interpuesta por un cliente",
    "Encuesta de satisfacción del producto entregado"
]

etiquetas_entrenamiento = [
    "Legal", "Financiero", "Financiero", "Legal", "Técnico",
    "Técnico", "Legal", "Financiero", "Técnico", "Financiero",
    "RRHH", "RRHH", "RRHH", "Atención", "Atención"
]

# Entrenamos el clasificador
clasificador = make_pipeline(TfidfVectorizer(), MultinomialNB())
clasificador.fit(textos_entrenamiento, etiquetas_entrenamiento)

# 3. Sistema Experto (5 Reglas)
def aplicar_reglas(texto, clase_predicha):
    texto_lower = texto.lower()
    
    # Regla 1 (Riesgo)
    if "confidencial" in texto_lower or "secreto" in texto_lower:
        return "R1: Riesgo Alto - Documento sensible detectado. Restringir acceso."
    # Regla 2 (Validación de firmas)
    elif clase_predicha == "Legal" and "firma" not in texto_lower:
        return "R2: Validación - Documento legal sin firma detectada. Requiere revisión."
    # Regla 3 (Auditoría)
    elif clase_predicha == "Financiero" and "balance" in texto_lower:
        return "R3: Auditoría - Balance detectado. Enviar a departamento de contabilidad."
    # Regla 4 (Seguridad)
    elif clase_predicha == "Técnico" and "vulnerabilidad" in texto_lower:
        return "R4: Seguridad - Reporte técnico crítico. Alertar al equipo IT."
    # Regla 5 (Prioridad)
    elif "urgente" in texto_lower:
        return "R5: Prioridad - Palabra clave 'urgente'. Procesamiento inmediato."
    
    return "R0: Documento estándar procesado correctamente sin alertas."

# 4. Motor de Recuperación de Información (TF-IDF y Similitud)
def procesar_consulta(consulta, base_conocimiento, vectorizer):
    # Unimos la consulta con la base de conocimiento para vectorizar
    textos_comparar = [consulta] + base_conocimiento
    tfidf_matrix = vectorizer.fit_transform(textos_comparar)
    
    # Calculamos similitud de la consulta (índice 0) contra el resto
    similitudes = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Obtenemos el mejor resultado
    indice_mejor_match = similitudes.argmax()
    mejor_match = base_conocimiento[indice_mejor_match]
    similitud_max = similitudes[indice_mejor_match]

    # Predecimos la categoría
    clase_predicha = clasificador.predict([consulta])[0]

    # Aplicamos reglas del sistema experto
    regla_activada = aplicar_reglas(consulta, clase_predicha)

    return regla_activada, mejor_match, similitud_max, clase_predicha

# 5. Ejecución de Pruebas y Reporte
if __name__ == "__main__":
    vectorizer_similitud = TfidfVectorizer()
    base_conocimiento = cargar_base_conocimiento()
    
    # 3 Consultas de prueba que activarán diferentes reglas
    consultas_prueba = [
        "Se adjunta un contrato confidencial para el nuevo proyecto, pero no tiene nada más.",
        "Reporte urgente de vulnerabilidades encontradas en la base de datos de producción.",
        "Solicito que se revise el balance general de este trimestre."
    ]

    reporte_md = "# Reporte Semana 05 - Pruebas Sistema Híbrido\n\n"

    for i, consulta in enumerate(consultas_prueba):
        regla, match, similitud, clase = procesar_consulta(consulta, base_conocimiento, vectorizer_similitud)
        
        resultado = f"### Prueba {i+1}\n"
        resultado += f"- **Consulta:** {consulta}\n"
        resultado += f"- **Regla activada:** {regla}\n"
        resultado += f"- **Evidencia (Información recuperada):** {match}\n"
        resultado += f"- **Similitud:** {similitud:.4f}\n"
        resultado += f"- **Clasificación (Categoría):** {clase}\n\n"
        
        print(resultado)
        reporte_md += resultado

    # Generar el archivo markdown
    # Construimos la ruta absoluta hacia la carpeta 'reports' en la raíz del proyecto
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    ruta_reports = os.path.join(directorio_script, "..", "reports")
    
    # Nos aseguramos de que exista (aunque ya la tienes creada)
    os.makedirs(ruta_reports, exist_ok=True)
    
    # Generamos la ruta final del archivo
    ruta_reporte = os.path.join(ruta_reports, "semana05.md")
    
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        f.write(reporte_md)
        
    print(f"✅ Reporte generado exitosamente en {ruta_reporte}")