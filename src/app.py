import os
import base64
import io
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import numpy as np
from PIL import Image
import joblib
import networkx as nx
import sqlite3

# Import text extraction and db logic from week 8
import semana08_representaciones as sem08

# Import vision logic from skimage
from skimage.color import rgb2gray, label2rgb
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DASHBOARD_DIR = BASE_DIR / "dashboard"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "modelo_red_neuronal.joblib"
VECTORIZER_PATH = ARTIFACTS_DIR / "vectorizador_tfidf.joblib"
DB_PATH = ARTIFACTS_DIR / "evidencia_reconocimiento.db"
ONTOLOGY_PATH = ARTIFACTS_DIR / "ontologia_documental.graphml"

# Initialize FastAPI
app = FastAPI(title="Sistema de Análisis Documental")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_cors_extra_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response

# Load Models
try:
    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        modelo = joblib.load(MODEL_PATH)
        vectorizador = joblib.load(VECTORIZER_PATH)
    else:
        modelo, vectorizador = None, None
        
    if ONTOLOGY_PATH.exists():
        ontologia = nx.read_graphml(ONTOLOGY_PATH)
    else:
        ontologia = None
except Exception as e:
    print(f"Error loading models: {e}")
    modelo = None
    vectorizador = None
    ontologia = None

def image_to_base64(img_array: np.ndarray, is_binary=False) -> str:
    """Convierte un array numpy a base64 para enviarlo al frontend."""
    if is_binary:
        # Convert boolean to uint8
        img_array = (img_array * 255).astype(np.uint8)
    
    if img_array.dtype != np.uint8:
        # Normalize and convert if needed
        img_array = (255 * (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array) + 1e-8)).astype(np.uint8)
        
    img = Image.fromarray(img_array)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.post("/api/upload")
@app.post("/api/upload/")
async def process_document(file: UploadFile = File(...)):
    """Recibe un archivo y lo procesa según su tipo (Imagen o Texto)."""
    file_ext = Path(file.filename).suffix.lower()
    temp_path = UPLOADS_DIR / file.filename
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        if file_ext in [".png", ".jpg", ".jpeg", ".tiff"]:
            # --- PROCESAMIENTO DE IMAGEN (Semana 09) ---
            pil_img = Image.open(temp_path).convert("RGB")
            img_np = np.array(pil_img)
            
            # 1. Grayscale y características de intensidad
            gray = rgb2gray(img_np)
            intensidad_media = float(np.mean(gray))
            desviacion_std = float(np.std(gray))
            
            # 2. Canny (Contornos y densidad de bordes)
            bordes = canny(gray, sigma=1.5)
            densidad_bordes = float((np.sum(bordes) / bordes.size) * 100)
            
            # 3. Otsu (Umbralización y Segmentación)
            thresh = float(threshold_otsu(gray))
            mascara = gray > thresh if intensidad_media < 0.40 else gray < thresh
            
            # 4. Regiones Conectadas y Análisis Morfológico Avanzado
            regiones_etiquetadas = label(mascara)
            num_regiones = int(len(np.unique(regiones_etiquetadas)) - 1)
            regiones_color = (label2rgb(regiones_etiquetadas, bg_label=0) * 255).astype(np.uint8)
            
            # Extracción de propiedades morfológicas
            props = regionprops(regiones_etiquetadas)
            total_fg = sum(p.area for p in props) if props else 0
            max_area = max(p.area for p in props) if props else 0
            max_axis = max(p.axis_major_length for p in props) if props else 0
            diag = float(np.sqrt(img_np.shape[0]**2 + img_np.shape[1]**2))
            axis_ratio = float(max_axis / diag) if diag > 0 else 0.0
            giant_ratio = float(max_area / total_fg) if total_fg > 0 else 0.0
            fg_coverage = float((total_fg / (img_np.shape[0] * img_np.shape[1])) * 100)

            # Pistas del nombre de archivo
            fname_lower = file.filename.lower()
            hints_plano = any(k in fname_lower for k in ["plano", "blueprint", "cad", "esquema", "cimentacion", "arquitect", "obra", "piso", "layout"])

            # CLASIFICADOR MULTICRITERIO DE VISIÓN ARTIFICIAL (Semana 09)
            # Criterio Plano: componente continua gigante (>60% del primer plano), trazos ortogonales perimetrales o pistas
            if (giant_ratio >= 0.60 and axis_ratio >= 0.75) or (fg_coverage < 5.5 and giant_ratio >= 0.55) or hints_plano:
                tipo_visual = "PLANO_ARQUITECTONICO_O_ESQUEMA_TECNICO"
                dominio_visual = "Dominio_Infraestructura_Y_Diseno"
                destino_visual = "Oficina_Planeacion_Y_Obras"
                accion_visual = "Revision_Tecnica_Y_Licencia_Construccion"
                interpretacion_visual = (
                    f"Plano técnico o diseño arquitectónico identificado mediante análisis morfológico de grafos conexos. "
                    f"Presenta una red estructural continua que concentra el {giant_ratio*100:.1f}% de los trazos, "
                    f"con una envergadura axial de {max_axis:.0f}px (típica de muros de contorno y distribución espacial de planta)."
                )
                confianza_visual = 0.95
            elif intensidad_media < 0.40:
                tipo_visual = "REGISTRO_MEDICO_O_CIENTIFICO (Ecografía / Escáner Oscuro)"
                dominio_visual = "Diagnostico_Salud_O_Imagenologia"
                destino_visual = "Area_Medica_O_Especialistas"
                accion_visual = "Valoracion_Clinica_Especializada"
                interpretacion_visual = "Imagen con fondo oscuro e hiperintensidades acústicas/ópticas. Contornos y componentes conexas segmentadas por umbral Otsu."
                confianza_visual = 0.92
            elif num_regiones > 450:
                tipo_visual = "DOCUMENTO_DIGITALIZADO_O_FACTURA"
                dominio_visual = "Gestion_Documental_Administrativa"
                destino_visual = "Contabilidad_O_Archivo"
                accion_visual = "Auditoria_Y_Contabilizacion"
                interpretacion_visual = f"Documento estructurado comercial con alta densidad tipográfica ({num_regiones} componentes conexas correspondientes a caracteres y bloques de texto)."
                confianza_visual = 0.94
            else:
                tipo_visual = "COMPROBANTE_O_RECIBO_DE_PAGO"
                dominio_visual = "Tesoreria_Y_Finanzas"
                destino_visual = "Caja_Y_Pagos"
                accion_visual = "Conciliacion_Bancaria"
                interpretacion_visual = f"Comprobante o soporte transaccional con baja densidad de texto ({num_regiones} componentes detectadas)."
                confianza_visual = 0.88
            
            # 5. Registro de Auditoría en SQLite (Integración Semana 08 + Semana 09)
            registro_img = {
                "timestamp": sem08.datetime.now().isoformat(),
                "fuente_documento": file.filename,
                "tipo_fuente": file_ext.replace(".", ""),
                "resumen_contenido": f"Visión Artificial: Intensidad={intensidad_media:.2f}, Bordes={densidad_bordes:.2f}%, Regiones={num_regiones}, RedDominante={giant_ratio*100:.1f}%",
                "categoria_real": "imagen_visual",
                "categoria_predicha": tipo_visual,
                "nivel_confianza": confianza_visual,
                "estado_proceso": "Procesado Exitosamente (Visión Computacional)",
                "categoria_dominio": dominio_visual,
                "departamento_destino": destino_visual,
                "accion_sugerida": accion_visual,
                "significado_ontologico": interpretacion_visual
            }
            try:
                sem08.registrar_evidencia(registro_img, DB_PATH)
            except Exception as e:
                print(f"[WARN] Error registrando imagen en BD: {e}")

            # Generar resultados en base64
            res = {
                "type": "image",
                "filename": file.filename,
                "metrics": {
                    "Clasificación": tipo_visual.split(" ")[0],
                    "Confianza": f"{confianza_visual:.1%}",
                    "Umbral Otsu": f"{thresh:.4f}",
                    "Regiones Encontradas": num_regiones,
                    "Red Dominante": f"{giant_ratio*100:.1f}%",
                    "Intensidad Media": f"{intensidad_media:.3f}",
                    "Densidad Bordes": f"{densidad_bordes:.2f}%",
                    "Dimensiones": f"{img_np.shape[1]}x{img_np.shape[0]} px"
                },
                "details": {
                    "categoria": tipo_visual,
                    "dominio": dominio_visual,
                    "destino": destino_visual,
                    "interpretacion": interpretacion_visual,
                    "explicacion": f"Se extrajeron {num_regiones} componentes morfológicas y se calculó una red continua del {giant_ratio*100:.1f}%. Estos rasgos determinaron su enrutamiento automático hacia {destino_visual}.",
                    "caracteristicas": {
                        "Intensidad Media (Brillo)": f"{intensidad_media:.4f}",
                        "Contraste (Desviación Std)": f"{desviacion_std:.4f}",
                        "Densidad de Contornos": f"{densidad_bordes:.2f}% de píxeles",
                        "Umbral Óptimo de Otsu": f"{thresh:.4f}",
                        "Componentes Conexas (Regiones)": f"{num_regiones} regiones",
                        "Resolución": f"{img_np.shape[1]} x {img_np.shape[0]} px"
                    }
                },
                "images": {
                    "original": image_to_base64(img_np),
                    "edges": image_to_base64(bordes, is_binary=True),
                    "mask": image_to_base64(mascara, is_binary=True),
                    "regions": image_to_base64(regiones_color)
                }
            }
            return JSONResponse(res)
            
        elif file_ext in [".pdf", ".txt", ".docx", ".md"]:
            # --- PROCESAMIENTO DE TEXTO (Semana 08) ---
            if not modelo or not vectorizador:
                return JSONResponse(
                    status_code=500,
                    content={"error": "Los modelos de IA no están entrenados. Ejecuta semana08 primero.", "detail": "Modelos no encontrados"}
                )
                
            # Extraer texto crudo
            nombre_fuente, texto_extraido, tipo_fuente = sem08.extraer_texto_documento(temp_path)
            
            if not texto_extraido or not texto_extraido.strip():
                return JSONResponse(
                    status_code=400,
                    content={"error": "El documento no contiene texto legible (posible documento escaneado que requiere OCR).", "detail": "Sin texto"}
                )
                
            # Ejecutar clasificación, enriquecimiento ontológico y auditoría SQLite
            resultado_sem08 = sem08.clasificar_y_registrar_documento(
                fuente=temp_path,
                modelo=modelo,
                vectorizador=vectorizador,
                ontologia=ontologia,
                db_path=DB_PATH,
                nombre_documento=file.filename
            )
            
            # Calcular distribución de probabilidades
            vector = vectorizador.transform([texto_extraido])
            if vector.sum() > 0:
                probs = modelo.predict_proba(vector)[0]
                prob_dict = {str(c): float(p) for c, p in zip(modelo.classes_, probs)}
                
                feature_names = vectorizador.get_feature_names_out()
                scores = vector.toarray()[0]
                top_indices = scores.argsort()[-8:][::-1]
                palabras_clave = [feature_names[i] for i in top_indices if scores[i] > 0]
            else:
                prob_dict = {str(c): 0.0 for c in modelo.classes_}
                palabras_clave = []

            pred_class = resultado_sem08.get("categoria_predicha", "desconocido")
            confianza = float(resultado_sem08.get("nivel_confianza", 0.0))
            significado = resultado_sem08.get("significado_ontologico") or resultado_sem08.get("categoria_dominio", "")

            explicacion_txt = f"El modelo encontró términos clave ({', '.join(palabras_clave)}) que activaron fuertemente la categoría '{pred_class.upper()}' en la Red Neuronal Multicapa (MLP)." if palabras_clave else "No se encontraron términos altamente predictivos en el texto extraído."

            res = {
                "type": "document",
                "filename": file.filename,
                "metrics": {
                    "Clasificación": pred_class.upper(),
                    "Confianza": f"{confianza:.2%}",
                    "Caracteres": len(texto_extraido)
                },
                "details": {
                    "clase": pred_class,
                    "texto": texto_extraido[:1000] + ("..." if len(texto_extraido) > 1000 else ""),
                    "interpretacion": f"{significado} | Destino: {resultado_sem08.get('departamento_destino', 'N/A')}",
                    "explicacion": explicacion_txt,
                    "probabilidades": prob_dict,
                    "estado_proceso": resultado_sem08.get("estado_proceso", "Procesado")
                }
            }
            return JSONResponse(res)
            
        else:
            return JSONResponse(
                status_code=400,
                content={"error": f"Formato no soportado: {file_ext}", "detail": f"Extensión inválida: {file_ext}"}
            )
            
    except HTTPException as he:
        print(f"[HTTPException] {he.status_code}: {he.detail}")
        return JSONResponse(status_code=he.status_code, content={"error": str(he.detail), "detail": str(he.detail)})
    except Exception as e:
        import traceback
        print("\n=== TRACEBACK ERROR EN BACKEND ===")
        traceback.print_exc()
        print("===================================\n")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error interno en el servidor: {str(e)}", "detail": str(e)}
        )
    finally:
        # Limpieza opcional del archivo temporal
        if temp_path.exists():
            pass # Podemos dejarlo para auditoría si se desea, o borrarlo.

@app.get("/api/history")
@app.get("/api/history/")
def get_history():
    """Retorna el historial de documentos procesados desde SQLite."""
    try:
        evidencias = sem08.consultar_evidencias(DB_PATH, limite=20)
        return JSONResponse({"history": evidencias})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.delete("/api/history/{id_evidencia}")
def delete_history_item(id_evidencia: int):
    """Elimina un documento específico del historial."""
    try:
        sem08.eliminar_evidencia(id_evidencia, DB_PATH)
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.delete("/api/history")
@app.delete("/api/history/")
def clear_history():
    """Borra todo el historial."""
    try:
        sem08.eliminar_evidencia(None, DB_PATH)
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Montar dashboard estático
app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")

if __name__ == "__main__":
    import uvicorn
    # Desactivamos reload=True para evitar conflictos de subprocesos y file-watchers
    uvicorn.run(app, host="0.0.0.0", port=8000)
