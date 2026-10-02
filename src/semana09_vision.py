"""
Sistema de Análisis y Gestión Documental con Inteligencia Artificial
Semana 09 - Reconocimiento de Imágenes: Características, Contornos y Segmentación

Flujo de Visión Artificial para Procesamiento de Documentos:
1. Carga y Selección de Imagen del Proyecto (Factura Electrónica FE-44910)
2. Extracción y Análisis Numérico de Características (Intensidad, Color, Textura, Bordes)
3. Detección de Contornos mediante Canny (Evaluación multiescala del parámetro Sigma)
4. Segmentación Automática mediante Umbral de Otsu (Separación Texto/Fondo)
5. Análisis y Etiquetado de Regiones Conectadas (Componentes Conexas)
6. Generación de Evidencia Visual Multipanel (artifacts/semana09_vision.png)
"""

from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.color import rgb2gray, label2rgb
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

# ==============================================================================
# 0. CONFIGURACIÓN DE RUTAS Y CONSTANTES DEL PROYECTO
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"

IMAGE_PATH = DATA_DIR / "imagen_proyecto.png"
OUTPUT_VISUAL_PATH = ARTIFACTS_DIR / "semana09_vision.png"
REPORT_PATH = REPORTS_DIR / "semana09.md"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. GENERADOR / VERIFICADOR DE LA IMAGEN DEL DOMINIO DEL PROYECTO
# ==============================================================================

def asegurar_imagen_proyecto(target_path: Path) -> Path:
    """
    Verifica la existencia de la imagen del proyecto. Si no existe, genera una
    factura electrónica comercial de alta fidelidad vinculada a las muestras
    previamente clasificadas en la Semana 08 (factura_proveedor_servicios.txt).
    """
    if target_path.exists():
        return target_path

    print(f"[INFO] Generando imagen representativa del dominio: {target_path}...")
    width, height = 1000, 1350
    img = Image.new("RGB", (width, height), color=(252, 252, 254))
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 24)
        font_subtitle = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
        font_bold = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 14)
        font_normal = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
        font_small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 11)
        font_tiny = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 10)
        font_stamp = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 12)
        font_stamp_sm = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 10)
    except Exception:
        font_title = font_subtitle = font_bold = font_normal = font_small = font_tiny = font_stamp = font_stamp_sm = ImageFont.load_default()

    # Marco estructural exterior
    draw.rectangle([25, 25, width - 26, height - 26], outline=(180, 190, 205), width=2)

    # Encabezado corporativo (Emisor)
    draw.rectangle([45, 45, 125, 125], fill=(30, 60, 115), outline=(20, 45, 90), width=2)
    draw.rectangle([65, 75, 105, 95], fill=(255, 255, 255))
    draw.ellipse([60, 70, 85, 95], fill=(255, 255, 255))
    draw.ellipse([80, 65, 105, 90], fill=(255, 255, 255))

    draw.text((140, 50), "INFRAESTRUCTURA CLOUD ANDINA S.A.S.", fill=(20, 35, 60), font=font_title)
    draw.text((140, 82), "NIT: 900.542.118-4  |  Régimen Común  |  Responsable de IVA", fill=(70, 80, 95), font=font_subtitle)
    draw.text((140, 102), "Calle 72 # 10-34, Piso 8, Bogotá D.C., Colombia  |  PBX: (601) 745-8900", fill=(70, 80, 95), font=font_subtitle)

    # Identificador de Factura Electrónica
    draw.rectangle([670, 45, width - 45, 125], fill=(240, 244, 250), outline=(30, 60, 115), width=2)
    draw.text((685, 52), "FACTURA ELECTRÓNICA DE VENTA", fill=(30, 60, 115), font=font_bold)
    draw.text((730, 72), "No. FE-44910", fill=(180, 20, 20), font=font_title)
    draw.text((685, 104), "CUFE: 8f4a-9b2c-e114-77a0-02bc-331e", fill=(90, 100, 115), font=font_tiny)

    # Separador horizontal
    draw.line([45, 145, width - 45, 145], fill=(30, 60, 115), width=2)

    # Metadatos del Documento
    draw.rectangle([45, 160, width - 45, 205], fill=(248, 249, 252), outline=(210, 215, 225), width=1)
    draw.text((60, 172), "Fecha Emisión: 28/08/2026", fill=(30, 30, 30), font=font_normal)
    draw.text((290, 172), "Fecha Vencimiento: 28/09/2026", fill=(30, 30, 30), font=font_normal)
    draw.text((540, 172), "Forma de Pago: Transferencia 30 Días", fill=(30, 30, 30), font=font_normal)
    draw.text((790, 172), "Moneda: COP ($)", fill=(30, 30, 30), font=font_bold)

    # Datos del Adquirente / Receptor
    draw.rectangle([45, 220, width - 45, 310], outline=(30, 60, 115), width=1)
    draw.rectangle([45, 220, width - 45, 245], fill=(230, 238, 250))
    draw.text((60, 225), "INFORMACIÓN DEL ADQUIRENTE / RECEPTOR", fill=(20, 45, 90), font=font_bold)

    draw.text((60, 255), "Razón Social: Corporación Universitaria para el Desarrollo Tecnológico", fill=(30, 30, 30), font=font_normal)
    draw.text((60, 275), "NIT / Identificación: 860.012.345-1", fill=(30, 30, 30), font=font_normal)
    draw.text((600, 255), "Dirección: Carrera 13 # 38-40, Sede Central", fill=(30, 30, 30), font=font_normal)
    draw.text((600, 275), "Ciudad: Bogotá D.C.  |  Email: facturas@etitc.edu.co", fill=(30, 30, 30), font=font_normal)

    # Tabla de Ítems / Conceptos Facturados
    table_top = 330
    table_bottom = 690
    draw.rectangle([45, table_top, width - 45, table_bottom], outline=(30, 60, 115), width=2)
    draw.rectangle([45, table_top, width - 45, table_top + 32], fill=(30, 60, 115))

    col_x = [45, 110, 580, 670, 810, width - 45]
    headers = ["ÍTEM", "DESCRIPCIÓN DEL SERVICIO / PRODUCTO", "CANT", "VR. UNITARIO", "VALOR TOTAL"]
    for i, h in enumerate(headers):
        draw.text((col_x[i] + 10, table_top + 8), h, fill=(255, 255, 255), font=font_bold)

    for x in col_x[1:-1]:
        draw.line([x, table_top, x, table_bottom], fill=(190, 200, 215), width=1)

    items_data = [
        ("01", "Servidor Dedicado Cloud Compute vCPU 16, RAM 64GB SSD NVMe 1TB", "1", "$ 2.800.000", "$ 2.800.000"),
        ("02", "Almacenamiento Distribuido Object Storage S3 Cloud - 5TB Backup", "1", "$   950.000", "$   950.000"),
        ("03", "Red Privada Virtual VPC, Firewall Gestionado y Gateway IPsec", "1", "$   750.000", "$   750.000"),
        ("04", "Monitoreo Proactivo de Cargas de Trabajo y Soporte 24/7 Nivel 3", "1", "$         0", "$         0 (BONIF.)"),
    ]

    row_y = table_top + 32
    for item in items_data:
        row_height = 55
        draw.line([45, row_y, width - 45, row_y], fill=(215, 225, 235), width=1)
        for col_idx, text in enumerate(item):
            draw.text((col_x[col_idx] + 12, row_y + 18), text, fill=(30, 30, 30), font=font_normal)
        row_y += row_height

    for y_fill in range(row_y, table_bottom, 45):
        draw.line([45, y_fill, width - 45, y_fill], fill=(235, 240, 248), width=1)

    # Bloque de Totales y Liquidación
    totals_top = 710
    totals_left = 550
    draw.rectangle([totals_left, totals_top, width - 45, totals_top + 160], outline=(30, 60, 115), width=2, fill=(250, 252, 255))

    totals_lines = [
        ("SUBTOTAL SERVICIOS:", "$ 4.500.000 COP", False),
        ("IVA RÉGIMEN GENERAL (19%):", "$   855.000 COP", False),
        ("RETENCIÓN EN LA FUENTE (-4%):", "- $   180.000 COP", False),
        ("TOTAL NETO A PAGAR:", "$ 5.175.000 COP", True),
    ]

    for idx, (label_txt, val, is_grand_total) in enumerate(totals_lines):
        y_pos = totals_top + 12 + idx * 36
        if is_grand_total:
            draw.rectangle([totals_left + 2, y_pos - 4, width - 47, y_pos + 32], fill=(230, 238, 250))
            draw.text((totals_left + 15, y_pos), label_txt, fill=(20, 45, 90), font=font_bold)
            draw.text((totals_left + 240, y_pos), val, fill=(180, 20, 20), font=font_bold)
        else:
            draw.line([totals_left, y_pos + 26, width - 45, y_pos + 26], fill=(225, 230, 240), width=1)
            draw.text((totals_left + 15, y_pos), label_txt, fill=(50, 60, 75), font=font_normal)
            draw.text((totals_left + 250, y_pos), val, fill=(30, 30, 30), font=font_normal)

    # Observaciones y condiciones comerciales
    draw.rectangle([45, totals_top, totals_left - 20, totals_top + 160], outline=(200, 210, 220), width=1, fill=(248, 249, 252))
    draw.text((60, totals_top + 15), "OBSERVACIONES Y CONDICIONES COMERCIALES:", fill=(30, 60, 115), font=font_bold)
    obs_text = [
        "1. Pago mediante transferencia a Bancolombia Cta Cte No. 031-99281-44.",
        "2. Servicio sujeto a contrato de prestación de servicios tecnológicos 2026.",
        "3. Genera retención en la fuente por servicios integrales TIC.",
        "4. Documento emitido según resolución DIAN No. 1876400000123.",
    ]
    for i, line in enumerate(obs_text):
        draw.text((60, totals_top + 45 + i * 24), line, fill=(70, 80, 90), font=font_small)

    # Sección Inferior: Código QR, Texto Legal y Sello de Aprobación
    bottom_top = 890

    # Código QR Fiscal DIAN
    qr_size = 130
    qr_x, qr_y = 55, bottom_top + 10
    draw.rectangle([qr_x - 5, qr_y - 5, qr_x + qr_size + 5, qr_y + qr_size + 5], fill=(255, 255, 255), outline=(150, 150, 150), width=1)

    def draw_finder(ox, oy):
        draw.rectangle([ox, oy, ox + 36, oy + 36], fill=(0, 0, 0))
        draw.rectangle([ox + 6, oy + 6, ox + 30, oy + 30], fill=(255, 255, 255))
        draw.rectangle([ox + 12, oy + 12, ox + 24, oy + 24], fill=(0, 0, 0))

    draw_finder(qr_x, qr_y)
    draw_finder(qr_x + qr_size - 36, qr_y)
    draw_finder(qr_x, qr_y + qr_size - 36)

    rng = np.random.default_rng(seed=42)
    grid_n = 21
    module_w = qr_size / grid_n
    for r in range(grid_n):
        for c in range(grid_n):
            if (r < 7 and c < 7) or (r < 7 and c >= grid_n - 7) or (r >= grid_n - 7 and c < 7):
                continue
            if rng.random() > 0.48:
                mx = qr_x + int(c * module_w)
                my = qr_y + int(r * module_w)
                draw.rectangle([mx, my, mx + int(module_w), my + int(module_w)], fill=(10, 10, 10))

    draw.text((qr_x, qr_y + qr_size + 12), "Validación Fiscal DIAN", fill=(60, 60, 60), font=font_tiny)

    # Texto Legal
    legal_x = 215
    draw.text((legal_x, bottom_top + 15), "RÉGIMEN DE FACTURACIÓN ELECTRÓNICA - REPÚBLICA DE COLOMBIA", fill=(40, 50, 65), font=font_bold)
    legal_lines = [
        "Esta factura electrónica de venta constituye título valor según Ley 1231 de 2008.",
        "El receptor declara haber recibido de conformidad los servicios descritos en este documento.",
        "Autorización DIAN No. 1876400021 de 2026-01-15, Vigencia 12 meses.",
        "Proveedor Tecnológico: Cloud Services & Software Colombia S.A.S. - NIT: 901.234.567-8",
    ]
    for i, line in enumerate(legal_lines):
        draw.text((legal_x, bottom_top + 40 + i * 20), line, fill=(80, 90, 105), font=font_small)

    # Sello de Auditoría Contable y Firma (Elemento clave de verificación documental)
    stamp_center_x, stamp_center_y = 820, bottom_top + 80
    stamp_box = [stamp_center_x - 125, stamp_center_y - 65, stamp_center_x + 125, stamp_center_y + 65]
    draw.ellipse(stamp_box, outline=(25, 65, 160), width=3)
    draw.ellipse([stamp_box[0] + 4, stamp_box[1] + 4, stamp_box[2] - 4, stamp_box[3] - 4], outline=(25, 65, 160), width=1)

    draw.text((stamp_center_x - 100, stamp_center_y - 48), "CORPORACIÓN UNIVERSITARIA", fill=(25, 65, 160), font=font_stamp_sm)
    draw.text((stamp_center_x - 75, stamp_center_y - 30), "AUDITORÍA CONTABLE", fill=(25, 65, 160), font=font_stamp)
    draw.text((stamp_center_x - 90, stamp_center_y - 8), "- APROBADO PARA PAGO -", fill=(180, 30, 30), font=font_stamp)
    draw.text((stamp_center_x - 75, stamp_center_y + 12), "REG: 2026-08-28 | SEC-09", fill=(25, 65, 160), font=font_tiny)

    sig_points = [
        (stamp_center_x - 70, stamp_center_y + 42),
        (stamp_center_x - 45, stamp_center_y + 26),
        (stamp_center_x - 15, stamp_center_y + 44),
        (stamp_center_x + 15, stamp_center_y + 22),
        (stamp_center_x + 40, stamp_center_y + 46),
        (stamp_center_x + 70, stamp_center_y + 36),
    ]
    for i in range(len(sig_points) - 1):
        draw.line([sig_points[i], sig_points[i + 1]], fill=(15, 45, 130), width=2)
    draw.line([stamp_center_x - 80, stamp_center_y + 48, stamp_center_x + 80, stamp_center_y + 48], fill=(25, 65, 160), width=1)

    # Simulación de textura/gradiente sutil de digitalización
    arr = np.array(img, dtype=np.float32)
    noise = rng.normal(0, 1.2, arr.shape)
    y_grad = np.linspace(0, 3, height).reshape(height, 1, 1)
    arr = np.clip(arr - y_grad + noise, 0, 255).astype(np.uint8)
    final_img = Image.fromarray(arr)

    final_img.save(target_path, dpi=(300, 300))
    print(f"[OK] Imagen del proyecto guardada exitosamente en: {target_path}")
    return target_path


# ==============================================================================
# 2. EXTRACCIÓN Y ANÁLISIS DE CARACTERÍSTICAS NUMÉRICAS
# ==============================================================================

def extraer_caracteristicas_imagen(img_rgb: np.ndarray, gray: np.ndarray) -> dict:
    """
    Realiza la descomposición matemática y estadística de la imagen:
    - Espacio de intensidad (escala de grises)
    - Separación y estadísticas de color (R, G, B)
    - Gradiente y energía de textura
    - Densidad espacial
    """
    height, width, channels = img_rgb.shape
    total_pixels = height * width

    # Estadísticas de intensidad (normalizada 0.0 - 1.0)
    int_min = float(np.min(gray))
    int_max = float(np.max(gray))
    int_mean = float(np.mean(gray))
    int_std = float(np.std(gray))
    int_median = float(np.median(gray))

    # Estadísticas de canales de color RGB (0 - 255)
    r_mean = float(np.mean(img_rgb[:, :, 0]))
    g_mean = float(np.mean(img_rgb[:, :, 1]))
    b_mean = float(np.mean(img_rgb[:, :, 2]))

    r_std = float(np.std(img_rgb[:, :, 0]))
    g_std = float(np.std(img_rgb[:, :, 1]))
    b_std = float(np.std(img_rgb[:, :, 2]))

    # Detección de píxeles cromáticos significativos (ej. sello azul o alerta roja)
    # La tinta monocromática de texto tiene R ~ G ~ B. Si |R - B| > 25, es color estructurado
    diff_chroma = np.abs(img_rgb[:, :, 0].astype(int) - img_rgb[:, :, 2].astype(int))
    chroma_pixels = int(np.count_nonzero(diff_chroma > 25))
    chroma_percentage = (chroma_pixels / total_pixels) * 100.0

    # Estimación de entropía de Shannon sobre intensidades cuantizadas (256 niveles)
    gray_uint8 = (gray * 255).astype(np.uint8)
    hist, _ = np.histogram(gray_uint8, bins=256, range=(0, 256), density=True)
    hist_nonzero = hist[hist > 0]
    shannon_entropy = float(-np.sum(hist_nonzero * np.log2(hist_nonzero)))

    stats = {
        "dimensiones": (height, width, channels),
        "total_pixeles": total_pixels,
        "intensidad_min": int_min,
        "intensidad_max": int_max,
        "intensidad_media": int_mean,
        "intensidad_std": int_std,
        "intensidad_mediana": int_median,
        "rgb_medias": (r_mean, g_mean, b_mean),
        "rgb_desviaciones": (r_std, g_std, b_std),
        "pixeles_cromaticos": chroma_pixels,
        "porcentaje_cromatico": chroma_percentage,
        "entropia_shannon": shannon_entropy,
    }
    return stats


# ==============================================================================
# 3. DETECCIÓN DE CONTORNOS MEDIANTE CANNY MULTIESCALA (SIGMA)
# ==============================================================================

def evaluar_contornos_canny(gray: np.ndarray, sigmas: list[float]) -> dict:
    """
    Aplica el algoritmo Canny variando el parámetro de escala Gaussiana (sigma)
    para evidenciar el compromiso entre detección de detalles finos y supresión de ruido.
    """
    resultados = {}
    for s in sigmas:
        # Canny de scikit-image aplica filtro Gaussiano con desviación estándar sigma
        edges = canny(gray, sigma=s)
        edge_pixels = int(np.count_nonzero(edges))
        edge_density = float((edge_pixels / edges.size) * 100.0)
        resultados[s] = {
            "mascara_bordes": edges,
            "total_pixeles_borde": edge_pixels,
            "densidad_borde_pct": edge_density,
        }
    return resultados


# ==============================================================================
# 4. SEGMENTACIÓN MEDIANTE UMBRAL AUTOMÁTICO DE OTSU
# ==============================================================================

def segmentar_otsu(gray: np.ndarray) -> dict:
    """
    Aplica el método de Otsu para calcular automáticamente el umbral óptimo
    que maximiza la varianza inter-clases.
    En documentos, el papel es claro (alta intensidad) y la tinta/líneas son oscuras (baja intensidad).
    Por tanto, la máscara binaria de interés (primer plano) es: gray < umbral.
    """
    thresh = float(threshold_otsu(gray))
    binary_mask = gray < thresh

    foreground_pixels = int(np.count_nonzero(binary_mask))
    background_pixels = int(np.count_nonzero(~binary_mask))
    total_pixels = gray.size

    fg_pct = float((foreground_pixels / total_pixels) * 100.0)
    bg_pct = float((background_pixels / total_pixels) * 100.0)

    # Cálculo de varianzas de clase
    fg_values = gray[binary_mask]
    bg_values = gray[~binary_mask]
    var_fg = float(np.var(fg_values)) if len(fg_values) > 0 else 0.0
    var_bg = float(np.var(bg_values)) if len(bg_values) > 0 else 0.0
    weight_fg = len(fg_values) / total_pixels
    weight_bg = len(bg_values) / total_pixels
    intra_class_var = weight_fg * var_fg + weight_bg * var_bg
    total_var = float(np.var(gray))
    inter_class_var = total_var - intra_class_var

    return {
        "umbral": thresh,
        "umbral_255": thresh * 255.0,
        "mascara_primer_plano": binary_mask,
        "pixeles_primer_plano": foreground_pixels,
        "porcentaje_primer_plano": fg_pct,
        "pixeles_fondo": background_pixels,
        "porcentaje_fondo": bg_pct,
        "varianza_intra_clase": intra_class_var,
        "varianza_inter_clase": inter_class_var,
    }


# ==============================================================================
# 5. ANÁLISIS Y ETIQUETADO DE REGIONES CONECTADAS
# ==============================================================================

def analizar_regiones_conectadas(binary_mask: np.ndarray) -> tuple[np.ndarray, list, dict]:
    """
    Etiqueta las regiones conexas (8-conectividad) sobre la máscara binaria.
    Calcula propiedades geométricas para interpretar la correspondencia con objetos reales:
    - Micro-regiones: puntos, tildes, signos de puntuación o ruido de fondo.
    - Regiones medias: caracteres alfanuméricos, glifos individuales, letras de tablas.
    - Macro-regiones: recuadros de tablas, bordes exteriores, logotipos y sellos.
    """
    labeled_matrix, num_features = label(binary_mask, connectivity=2, return_num=True)
    props = regionprops(labeled_matrix)

    areas = [p.area for p in props]

    # Clasificación por escala física de componentes
    micro_regions = [p for p in props if p.area < 15]
    medium_regions = [p for p in props if 15 <= p.area < 1500]
    macro_regions = [p for p in props if p.area >= 1500]

    resumen = {
        "total_regiones": num_features,
        "area_minima": int(min(areas)) if areas else 0,
        "area_maxima": int(max(areas)) if areas else 0,
        "area_mediana": float(np.median(areas)) if areas else 0.0,
        "area_media": float(np.mean(areas)) if areas else 0.0,
        "total_micro": len(micro_regions),
        "total_medias": len(medium_regions),
        "total_macro": len(macro_regions),
    }

    return labeled_matrix, props, resumen


# ==============================================================================
# 6. GENERACIÓN DE EVIDENCIA VISUAL MULTIPANEL (artifacts/semana09_vision.png)
# ==============================================================================

def generar_evidencia_visual(
    img_rgb: np.ndarray,
    gray: np.ndarray,
    canny_results: dict,
    otsu_res: dict,
    labeled_matrix: np.ndarray,
    output_path: Path
) -> None:
    """
    Construye una figura científica de 6 paneles en cuadrícula 2x3 con alta resolución:
    1. Imagen Original del Proyecto (RGB)
    2. Histograma Bimodal de Intensidades con Umbral Otsu Marcado
    3. Detección de Bordes Canny (Sigma = 1.0 - Alta Resolución / Caracteres)
    4. Efecto de Suavizado Canny (Sigma = 3.0 - Estructura Macro y Filtrado)
    5. Segmentación Binaria Otsu (Máscara de Tinta/Bordes vs Papel)
    6. Identificación de Regiones Conectadas (Mapa de Colores por Componente)
    """
    print(f"[INFO] Construyendo panel de evidencia visual en: {output_path}...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 14), dpi=200)

    # 1. Imagen Original
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("1. Imagen Original del Proyecto\n(Factura Electrónica FE-44910)", fontsize=11, fontweight="bold", pad=8)
    axes[0, 0].axis("off")

    # 2. Histograma de Intensidades y Umbral Otsu
    thresh = otsu_res["umbral"]
    axes[0, 1].hist(gray.ravel(), bins=60, color="#1e3c73", alpha=0.85, density=True, edgecolor="white", linewidth=0.5)
    axes[0, 1].axvline(thresh, color="#d9534f", linestyle="--", linewidth=2.5, label=f"Umbral Otsu (T = {thresh:.3f} / {thresh*255:.1f})")
    axes[0, 1].set_title("2. Histograma de Intensidades y Umbral Otsu\n(Separación Bimodal Fondo vs Tinta)", fontsize=11, fontweight="bold", pad=8)
    axes[0, 1].set_xlabel("Intensidad Normalizada (0.0 = Negro / Tinta, 1.0 = Blanco / Papel)", fontsize=10)
    axes[0, 1].set_ylabel("Densidad de Probabilidad", fontsize=10)
    axes[0, 1].legend(loc="upper left", frameon=True, facecolor="#f8f9fa", edgecolor="#ced4da", fontsize=9)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # 3. Contornos Canny con Sigma = 1.0
    edges_s1 = canny_results[1.0]["mascara_bordes"]
    px_s1 = canny_results[1.0]["total_pixeles_borde"]
    pct_s1 = canny_results[1.0]["densidad_borde_pct"]
    axes[0, 2].imshow(edges_s1, cmap="gray_r")
    axes[0, 2].set_title(f"3. Contornos Canny ($\\sigma = 1.0$)\n({px_s1:,} px borde - {pct_s1:.2f}% | Caracteres y Cuadrícula)", fontsize=11, fontweight="bold", pad=8)
    axes[0, 2].axis("off")

    # 4. Contornos Canny con Sigma = 3.0
    edges_s3 = canny_results[3.0]["mascara_bordes"]
    px_s3 = canny_results[3.0]["total_pixeles_borde"]
    pct_s3 = canny_results[3.0]["densidad_borde_pct"]
    axes[1, 0].imshow(edges_s3, cmap="gray_r")
    axes[1, 0].set_title(f"4. Contornos Canny ($\\sigma = 3.0$)\n({px_s3:,} px borde - {pct_s3:.2f}% | Macro-Estructura Filtrada)", fontsize=11, fontweight="bold", pad=8)
    axes[1, 0].axis("off")

    # 5. Segmentación Binaria Otsu
    binary_mask = otsu_res["mascara_primer_plano"]
    fg_px = otsu_res["pixeles_primer_plano"]
    fg_pct = otsu_res["porcentaje_primer_plano"]
    axes[1, 1].imshow(binary_mask, cmap="binary")
    axes[1, 1].set_title(f"5. Segmentación Binaria Otsu\n({fg_px:,} px tinta - {fg_pct:.2f}% | Región de Interés)", fontsize=11, fontweight="bold", pad=8)
    axes[1, 1].axis("off")

    # 6. Regiones Conectadas Etiquetadas (sobre fondo blanco)
    labeled_color = label2rgb(labeled_matrix, bg_label=0, bg_color=(1, 1, 1))
    total_regions = labeled_matrix.max()
    axes[1, 2].imshow(labeled_color)
    axes[1, 2].set_title(f"6. Identificación de Regiones Conectadas\n({total_regions:,} Componentes Conexas Etiquetadas)", fontsize=11, fontweight="bold", pad=8)
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Panel visual generado exitosamente en: {output_path}")


# ==============================================================================
# 7. FUNCIÓN PRINCIPAL DE EJECUCIÓN DEL PIPELINE
# ==============================================================================

def main():
    print("=" * 80)
    print(" SISTEMA DE ANÁLISIS DOCUMENTAL - SEMANA 09")
    print(" Visión por Computador: Características, Contornos y Segmentación")
    print("=" * 80)

    # 1. Asegurar la imagen del proyecto
    img_path = asegurar_imagen_proyecto(IMAGE_PATH)
    img_pil = Image.open(img_path)
    img_rgb = np.array(img_pil)

    # Conversión canónica a escala de grises normalizada [0.0, 1.0]
    gray = rgb2gray(img_rgb)

    # 2. Extracción de características
    print("\n--- PASO 1 Y 2: EXTRACCIÓN Y ANÁLISIS DE CARACTERÍSTICAS ---")
    stats = extraer_caracteristicas_imagen(img_rgb, gray)
    print(f"Dimensiones de la matriz:      {stats['dimensiones'][1]} px (ancho) x {stats['dimensiones'][0]} px (alto)")
    print(f"Canales de color:              {stats['dimensiones'][2]} (RGB, formato uint8)")
    print(f"Total de píxeles procesados:   {stats['total_pixeles']:,}")
    print(f"Rango de intensidad (gris):    [{stats['intensidad_min']:.4f}, {stats['intensidad_max']:.4f}]")
    print(f"Intensidad promedio:           {stats['intensidad_media']:.4f} (Desviación estándar: {stats['intensidad_std']:.4f})")
    print(f"Intensidad mediana:            {stats['intensidad_mediana']:.4f}")
    print(f"Medias por canal (R, G, B):    R={stats['rgb_medias'][0]:.1f}, G={stats['rgb_medias'][1]:.1f}, B={stats['rgb_medias'][2]:.1f}")
    print(f"Píxeles cromáticos (sellos):   {stats['pixeles_cromaticos']:,} ({stats['porcentaje_cromatico']:.2f}% de la superficie)")
    print(f"Entropía de información:       {stats['entropia_shannon']:.4f} bits/píxel")

    # 3. Detección de contornos con Canny
    print("\n--- PASO 3: DETECCIÓN DE CONTORNOS CON CANNY (VARIACIÓN DE SIGMA) ---")
    sigmas_evaluar = [1.0, 1.5, 2.0, 3.0]
    canny_res = evaluar_contornos_canny(gray, sigmas_evaluar)
    for s in sigmas_evaluar:
        px = canny_res[s]["total_pixeles_borde"]
        dens = canny_res[s]["densidad_borde_pct"]
        print(f"  • Canny (sigma = {s:.1f}): {px:>7,} píxeles de borde ({dens:.2f}% de la imagen)")

    # 4. Segmentación mediante umbral de Otsu
    print("\n--- PASO 4: SEGMENTACIÓN AUTOMÁTICA POR UMBRAL DE OTSU ---")
    otsu_res = segmentar_otsu(gray)
    print(f"Umbral Otsu calculado:         T = {otsu_res['umbral']:.4f} (Escala 0-255: {otsu_res['umbral_255']:.2f})")
    print(f"Píxeles de primer plano (tinta): {otsu_res['pixeles_primer_plano']:,} ({otsu_res['porcentaje_primer_plano']:.2f}%)")
    print(f"Píxeles de fondo (papel):      {otsu_res['pixeles_fondo']:,} ({otsu_res['porcentaje_fondo']:.2f}%)")
    print(f"Varianza inter-clase:          {otsu_res['varianza_inter_clase']:.6f} (Separabilidad óptima)")

    # 5. Análisis de regiones conectadas
    print("\n--- PASO 5: ANÁLISIS Y ETIQUETADO DE REGIONES CONECTADAS ---")
    labeled_matrix, props, res_regiones = analizar_regiones_conectadas(otsu_res["mascara_primer_plano"])
    print(f"Total de regiones encontradas: {res_regiones['total_regiones']:,}")
    print(f"  - Micro-regiones (< 15 px):    {res_regiones['total_micro']:,} (puntos, tildes, micro-ruido, QR)")
    print(f"  - Regiones medias (15-1500px): {res_regiones['total_medias']:,} (caracteres alfanuméricos, letras)")
    print(f"  - Macro-regiones (>= 1500 px): {res_regiones['total_macro']:,} (cuadrículas de tabla, marcos, sellos)")
    print(f"Área mínima: {res_regiones['area_minima']} px | Área máxima: {res_regiones['area_maxima']} px | Mediana: {res_regiones['area_mediana']:.1f} px")

    # 6. Generar evidencia visual
    print("\n--- PASO 6: GENERACIÓN DE EVIDENCIA VISUAL ---")
    generar_evidencia_visual(img_rgb, gray, canny_res, otsu_res, labeled_matrix, OUTPUT_VISUAL_PATH)

    print("\n" + "=" * 80)
    print(" PROCESAMIENTO COMPLETADO EXITOSAMENTE")
    print(f" Evidencia visual: {OUTPUT_VISUAL_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
