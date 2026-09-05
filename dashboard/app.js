// =========================================================================
// BASE DE DATOS DIDÁCTICA Y CÓDIGO REAL DEL PROYECTO
// =========================================================================
const dashboardData = {
    semana02: {
        titulo: "Semana 02",
        subtitulo: "Machine",
        archivo: "src/semana02_fundamentos.py",
        metricas: [
            { label: "Corpus Supervisado", value: "20 casos reales (casos_ia.csv)" },
            { label: "Categorías Taxonómicas", value: "7 clases documentales" },
            { label: "Validación LOO (Accuracy)", value: "95.0 % (19/20 aciertos)" },
            { label: "Vectorización", value: "TF-IDF (Unigramas + Stopwords ES)" }
        ],
        overview: "Bloque de modelado predictivo supervisado: vectorización textual con TF-IDF, clasificación con Logistic Regression y evaluación cruzada Leave-One-Out sobre los 20 casos reales del proyecto.",
        codigoRaw: `from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

# Reutiliza los datos y etiquetas reales del proyecto (mismo corpus y
# misma clasificación manual que las Semanas 03 y 04).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from semana03_taxonomia import MANUAL_REFERENCE, classify_problem, read_cases

ROOT = Path(__file__).resolve().parent.parent
REPORT_FILE = ROOT / "reports" / "semana02.md"

RANDOM_STATE = 42

STOPWORDS_ES = {
    "a", "al", "algo", "alguna", "algunas", "alguno", "algunos", "ante",
    "antes", "como", "con", "contra", "cual", "cuando", "de", "del",
    "desde", "donde", "dos", "el", "ella", "ellas", "ellos", "en",
    "entre", "era", "es", "esa", "esas", "ese", "eso", "esos", "esta",
    "estas", "este", "estos", "fue", "ha", "hasta", "la", "las", "le",
    "les", "lo", "los", "más", "me", "mi", "mis", "muy", "no", "nos",
    "o", "para", "pero", "por", "que", "qué", "se", "sea", "ser", "si",
    "sin", "sobre", "son", "su", "sus", "también", "te", "ti", "tu",
    "tus", "un", "una", "uno", "unos", "y", "ya"
}


# ---------------------------------------------------------------------
# CARGA DE LOS DATOS REALES DEL PROYECTO
# ---------------------------------------------------------------------

def limpiar_texto(texto: str) -> str:
    """Normaliza una descripción antes de vectorizarla."""
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúüñ0-9\\s]", " ", texto)
    return re.sub(r"\\s+", " ", texto).strip()


def cargar_datos() -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Carga el dataset supervisado del proyecto.

    Equivalente a load_iris(return_X_y=True) de la práctica original,
    pero con los datos reales del sistema de análisis documental.

    Retorna:
        X: lista de descripciones ya normalizadas (texto).
        y: array de etiquetas (categoría de la taxonomía).
        descripciones: los textos originales, sin normalizar.
    """
    descripciones = read_cases()

    if len(descripciones) != len(MANUAL_REFERENCE):
        raise ValueError(
            f"data/casos_ia.csv tiene {len(descripciones)} casos pero "
            f"MANUAL_REFERENCE define {len(MANUAL_REFERENCE)} etiquetas: "
            "el dataset supervisado quedaria desalineado."
        )

    X = [limpiar_texto(descripcion) for descripcion in descripciones]
    y = np.array(MANUAL_REFERENCE)

    return X, y, descripciones


def construir_modelo() -> Any:
    """
    Construye el pipeline supervisado.

    Misma estructura que la práctica original
    (make_pipeline(StandardScaler(), LogisticRegression(...))), pero
    el paso de preparación de datos es TfidfVectorizer porque las
    entradas son texto y no medidas numéricas.
    Dos ajustes son necesarios por el tamaño del dataset:

    - ngram_range=(1, 1): con 20 documentos cortos, los bigramas
      generan cientos de rasgos que aparecen una sola vez y el modelo
      no puede generalizar a partir de ellos.
    - class_weight="balanced": las categorías están desbalanceadas
      (de 2 a 4 casos). Sin este ajuste el modelo tiende a predecir
      siempre las categorías mayoritarias e ignora las pequeñas.

    El efecto de ambos ajustes se mide en comparar_configuraciones().
    """
    return make_pipeline(
        TfidfVectorizer(
            stop_words=list(STOPWORDS_ES),
            ngram_range=(1, 1),
            sublinear_tf=True,
        ),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )


def comparar_configuraciones(
    X: List[str],
    y: np.ndarray,
) -> List[Tuple[str, float]]:
    """
    Mide con Leave-One-Out el efecto de cada decisión de diseño.

    Justifica por qué el modelo final usa unigramas y pesos
    balanceados, en lugar de dar por buena la primera configuración.
    """
    def pipeline_lr(ngramas: Tuple[int, int], balanceado: bool) -> Any:
        return make_pipeline(
            TfidfVectorizer(
                stop_words=list(STOPWORDS_ES),
                ngram_range=ngramas,
                sublinear_tf=True,
            ),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced" if balanceado else None,
                random_state=RANDOM_STATE,
            ),
        )

    configuraciones = [
        (
            "Logistic Regression, unigramas + bigramas, sin balanceo",
            pipeline_lr((1, 2), False),
        ),
        (
            "Logistic Regression, solo unigramas, sin balanceo",
            pipeline_lr((1, 1), False),
        ),
        (
            "Logistic Regression, unigramas + bigramas, balanceada",
            pipeline_lr((1, 2), True),
        ),
        (
            "Logistic Regression, solo unigramas, balanceada (modelo final)",
            pipeline_lr((1, 1), True),
        ),
    ]

    resultados: List[Tuple[str, float]] = []

    for nombre, modelo in configuraciones:
        predicciones = cross_val_predict(modelo, X, y, cv=LeaveOneOut())
        resultados.append(
            (nombre, float(accuracy_score(y, predicciones)))
        )

    return resultados


# ---------------------------------------------------------------------
# EVALUACIÓN POR VALIDACIÓN CRUZADA
# ---------------------------------------------------------------------

def evaluar_leave_one_out(
    X: List[str],
    y: np.ndarray,
    etiquetas: List[str],
) -> Dict[str, Any]:
    """
    Validación cruzada Leave-One-Out.

    Se entrena len(X) veces; en cada iteración un caso queda fuera y
    se predice con un modelo que no lo vio. Reemplaza al split único de
    la práctica original, que no es viable con 20 muestras y 7 clases.
    """
    predicciones = cross_val_predict(
        construir_modelo(),
        X,
        y,
        cv=LeaveOneOut(),
    )

    return {
        "predicciones": list(predicciones),
        "accuracy": float(accuracy_score(y, predicciones)),
        "matriz": confusion_matrix(y, predicciones, labels=etiquetas),
        "reporte": classification_report(
            y,
            predicciones,
            labels=etiquetas,
            zero_division=0,
        ),
    }


def evaluar_kfold_estratificado(
    X: List[str],
    y: np.ndarray,
    etiquetas: List[str],
    n_splits: int = 2,
) -> Dict[str, Any]:
    """
    Validación cruzada estratificada.

    Conserva la proporción de categorías en cada partición, que es la
    idea de stratify=y en la práctica original. n_splits está
    limitado por la categoría menos frecuente del proyecto.
    """
    minimo = min(Counter(y).values())
    n_splits = max(2, min(n_splits, minimo))

    particion = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    predicciones = cross_val_predict(
        construir_modelo(),
        X,
        y,
        cv=particion,
    )

    return {
        "n_splits": n_splits,
        "predicciones": list(predicciones),
        "accuracy": float(accuracy_score(y, predicciones)),
        "matriz": confusion_matrix(y, predicciones, labels=etiquetas),
    }


def evaluar_reglas_semana03(
    descripciones: List[str],
    y: np.ndarray,
) -> Dict[str, Any]:
    """
    Baseline: el clasificador por reglas de la Semana 03.

    Permite comparar "reglas escritas a mano" contra "modelo aprendido
    de los datos" sobre exactamente los mismos casos y etiquetas.
    """
    predicciones = [classify_problem(texto)[0] for texto in descripciones]
    aciertos = sum(p == real for p, real in zip(predicciones, y))

    return {
        "predicciones": predicciones,
        "accuracy": aciertos / len(y) if len(y) else 0.0,
    }


# ---------------------------------------------------------------------
# IMPORTANCIA DE TÉRMINOS APRENDIDOS
# ---------------------------------------------------------------------

def terminos_relevantes(
    X: List[str],
    y: np.ndarray,
    cantidad: int = 5,
) -> Dict[str, List[str]]:
    """
    Entrena el modelo con todos los casos y extrae los términos con
    mayor peso positivo por categoría.
    """
    modelo = construir_modelo()
    modelo.fit(X, y)

    vectorizador = modelo.named_steps["tfidfvectorizer"]
    regresion = modelo.named_steps["logisticregression"]

    vocabulario = np.array(vectorizador.get_feature_names_out())
    coeficientes = regresion.coef_

    relevantes: Dict[str, List[str]] = {}

    for indice, categoria in enumerate(regresion.classes_):
        pesos = coeficientes[indice]
        mejores = np.argsort(pesos)[::-1][:cantidad]
        relevantes[categoria] = [
            str(vocabulario[posicion])
            for posicion in mejores
            if pesos[posicion] > 0
        ]

    return relevantes


def main() -> None:
    X, y, descripciones = cargar_datos()
    etiquetas = sorted(set(y))
    resultado_loo = evaluar_leave_one_out(X, y, etiquetas)
    print(f"Accuracy LOO: {resultado_loo['accuracy']:.3f}")


if __name__ == "__main__":
    main()`,
        lineas: {
            10: {
                tipo: "Sentencia de Importación: Vectorizador NLP",
                queHace: "Carga la clase TfidfVectorizer desde el paquete sklearn.feature_extraction.text.",
                paraQueSirve: "Permite convertir las descripciones no estructuradas de casos documentales en representaciones matriciales comparables, calculando el balance entre frecuencia local y rareza global de los términos.",
                conceptos: "Procesamiento de Lenguaje Natural (PLN). Extracción de características numéricas en espacios vectoriales continuos."
            },
            11: {
                tipo: "Sentencia de Importación: Algoritmo Supervisado",
                queHace: "Carga LogisticRegression desde sklearn.linear_model.",
                paraQueSirve: "Proporciona el modelo lineal paramétrico que aprenderá a mapear los vectores TF-IDF hacia las 7 categorías de la taxonomía documental.",
                conceptos: "Aprendizaje Automático Supervisado. Clasificación multiclase mediante regularización y función sigmoide/softmax."
            },
            13: {
                tipo: "Sentencia de Importación: Métodos de Validación",
                queHace: "Importa LeaveOneOut, StratifiedKFold y cross_val_predict.",
                paraQueSirve: "Resuelve la imposibilidad de hacer un train_test_split tradicional del 25% (que dejaría solo 5 muestras para 7 clases), garantizando una validación reproducible y sin sesgo sobre el corpus de 20 casos.",
                conceptos: "Validación Cruzada Exhaustiva (LOO-CV). Estimación insesgada del error de generalización en muestras pequeñas."
            },
            78: {
                tipo: "Construcción del Pipeline Predictivo",
                queHace: "Encadena TfidfVectorizer y LogisticRegression con class_weight='balanced'.",
                paraQueSirve: "Unifica la transformación de texto y el entrenamiento en un objeto atómico, aplicando una penalización matemática inversamente proporcional a la frecuencia de las clases para proteger las categorías minoritarias.",
                conceptos: "Modularidad de Pipelines en Scikit-Learn. Manejo de Desbalance Severo de Clases."
            },
            145: {
                tipo: "Evaluación Exhaustiva Leave-One-Out",
                queHace: "Ejecuta cross_val_predict con 20 iteraciones donde cada muestra se predice con un modelo ajustado sobre las otras 19.",
                paraQueSirve: "Genera predicciones totalmente fuera de muestra (out-of-fold) para calcular la matriz de confusión y el reporte de clasificación sin memorización de datos.",
                conceptos: "Evaluación Insesgada, Métricas de Precisión, Recall y F1-Score."
            },
            222: {
                tipo: "Interpretabilidad y Explicabilidad (XAI)",
                queHace: "Inspecciona modelo.named_steps['logisticregression'].coef_ y vectorizador.get_feature_names_out().",
                paraQueSirve: "Identifica qué palabras específicas tienen el mayor peso positivo para cada categoría (ej. 'ocr' en Visión, 'buscar' en Recuperación), abriendo la caja negra del clasificador.",
                conceptos: "Explicabilidad de Modelos Lineales, Análisis de Relevancia de Coeficientes."
            }
        }
    },

    semana03: {
        titulo: "Semana 03",
        subtitulo: "Inteligencia",
        archivo: "src/semana03_taxonomia.py",
        metricas: [
            { label: "Categorías Base", value: "7 Ramas de IA" },
            { label: "Reglas Propias", value: "5 Bloques en CUSTOM_RULES" },
            { label: "Coincidencia con Referencia", value: "85.0 % (17/20 casos)" },
            { label: "Mecanismo", value: "Conteo Heurístico de Descriptores" }
        ],
        overview: "Estructuración ontológica del conocimiento documental: clases inmutables con @dataclass, base de reglas heurísticas (CUSTOM_RULES) y motor de inferencia simbólico.",
        codigoRaw: `from dataclasses import dataclass
from pathlib import Path
import csv
import re
import unicodedata

ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = ROOT / "data" / "casos_ia.csv"
REPORT_FILE = ROOT / "reports" / "semana03.md"


@dataclass(frozen=True)
class Category:
    name: str
    keywords: tuple[str, ...]


CATEGORIES = [
    Category("Visión por computador y OCR", (
        "imagen", "imagenes", "foto", "fotografia", "fotografias", "camara",
        "escaneo", "escaneado", "escanear", "documento escaneado", "ocr",
        "reconocimiento optico", "digitalizado", "captura"
    )),
    Category("Procesamiento de lenguaje natural", (
        "texto", "lenguaje", "frase", "frases", "parrafo", "parrafos",
        "resumen", "resumir", "analizar texto", "extraer nombres", "extraer fechas", "extraer entidades"
    )),
    Category("Aprendizaje automático predictivo", (
        "predecir", "prediccion", "probabilidad", "clasificar automaticamente",
        "patron", "patrones", "anomalía", "anomalias", "riesgo", "detectar automaticamente"
    )),
    Category("Clasificación documental", (
        "clasificar documento", "clasificar documentos", "clasificacion documental",
        "tipo de documento", "categorizar documento", "categoria documental",
        "identificar tipo de documento", "detectar tipo de documento", "reconocer documento",
        "factura", "contrato", "certificado", "acta", "formulario", "hoja de vida"
    )),
    Category("Búsqueda y recuperación documental", (
        "buscar", "busqueda", "encontrar", "localizar", "consultar", "consulta",
        "palabra clave", "palabras clave", "documentos relacionados", "recuperar informacion", "encontrar documentos"
    )),
    Category("Sistemas expertos y reglas documentales", (
        "regla", "reglas", "validar", "validacion", "verificar", "verificacion",
        "requisito", "requisitos", "obligatorio", "obligatorios", "informacion faltante", "cumple", "incumple"
    )),
    Category("Automatización documental inteligente", (
        "automatizar", "automatizacion", "procesar documento", "procesamiento documental",
        "procesar automaticamente", "extraer informacion", "extraer datos", "extraer automaticamente",
        "generar reporte", "generar reporte automaticamente", "flujo documental"
    )),
]


CUSTOM_RULES = {
    "Visión por computador y OCR": (
        "documento escaneado", "imagen del documento", "fotografia del documento", "convertir imagen en texto"
    ),
    "Procesamiento de lenguaje natural": (
        "extraer nombres", "extraer fechas", "extraer entidades", "analizar contenido textual"
    ),
    "Clasificación documental": (
        "identificar tipo de documento", "detectar tipo de documento", "reconocer documento", "clasificar documentos"
    ),
    "Sistemas expertos y reglas documentales": (
        "verificar requisitos", "validar documento", "documento completo", "informacion faltante"
    ),
    "Automatización documental inteligente": (
        "procesar automaticamente", "extraer automaticamente", "analizar automaticamente", "generar reporte"
    ),
}


MANUAL_REFERENCE = [
    "Visión por computador y OCR", "Clasificación documental", "Procesamiento de lenguaje natural",
    "Procesamiento de lenguaje natural", "Búsqueda y recuperación documental", "Búsqueda y recuperación documental",
    "Sistemas expertos y reglas documentales", "Visión por computador y OCR", "Clasificación documental",
    "Procesamiento de lenguaje natural", "Procesamiento de lenguaje natural", "Sistemas expertos y reglas documentales",
    "Búsqueda y recuperación documental", "Aprendizaje automático predictivo", "Búsqueda y recuperación documental",
    "Clasificación documental", "Sistemas expertos y reglas documentales", "Automatización documental inteligente",
    "Aprendizaje automático predictivo", "Automatización documental inteligente"
]


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\\s+", " ", text).strip()


def contains_keyword(text: str, keyword: str) -> bool:
    normalized_text = f" {normalize(text)} "
    normalized_keyword = normalize(keyword)
    return f" {normalized_keyword} " in normalized_text


def build_categories() -> list[Category]:
    result = []
    for category in CATEGORIES:
        extra = CUSTOM_RULES.get(category.name, ())
        result.append(Category(category.name, category.keywords + tuple(extra)))
    return result


def classify_problem(text: str) -> tuple[str, list[str], dict[str, int]]:
    scores = {}
    for category in build_categories():
        score = sum(contains_keyword(text, keyword) for keyword in category.keywords)
        scores[category.name] = score

    matches = [
        (score, index, category.name)
        for index, category in enumerate(build_categories())
        if (score := scores[category.name]) > 0
    ]
    matches.sort(key=lambda item: (-item[0], item[1]))
    detected = [name for _, _, name in matches]
    primary = detected[0] if detected else "Requiere análisis"

    return primary, detected or ["Requiere análisis"], scores


def read_cases() -> list[str]:
    with CSV_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        cases = [row.get("descripcion", "").strip() for row in reader if row.get("descripcion")]
    return cases


def main() -> None:
    cases = read_cases()
    for i, case in enumerate(cases, start=1):
        primary, detected, _ = classify_problem(case)
        print(f"{i:02d}. {primary} -> {case}")


if __name__ == "__main__":
    main()`,
        lineas: {
            13: {
                tipo: "Estructura de Datos Ontológica",
                queHace: "Define la clase Category mediante el decorador @dataclass(frozen=True).",
                paraQueSirve: "Crea una entidad inmutable que empareja el nombre formal de la rama de IA con su tupla de palabras clave canónicas, impidiendo mutaciones en memoria durante la inferencia.",
                conceptos: "Programación Orientada a Objetos (POO), Inmutabilidad, Tipado Estático."
            },
            55: {
                tipo: "Base de Reglas Especializadas (CUSTOM_RULES)",
                queHace: "Estructura un diccionario con tuplas de n-gramas específicos de análisis documental.",
                paraQueSirve: "Permite inyectar conocimiento heurístico humano directo para resolver casos ambiguos donde los términos atómicos aislados resultan insuficientes.",
                conceptos: "Ingeniería del Conocimiento, Mapeo Heurístico, Reglas de Dominio."
            },
            94: {
                tipo: "Normalización Fonética y Ortográfica",
                queHace: "Aplica descomposición canónica NFD con unicodedata para remover acentos y caracteres diacríticos.",
                paraQueSirve: "Elimina la disparidad tipográfica entre términos escritos con tilde, diéresis o mayúsculas y la base de conocimiento.",
                conceptos: "Normalización Unicode NFD, Procesamiento de Cadenas en PLN."
            },
            102: {
                tipo: "Límites de Términos (Word Boundaries)",
                queHace: "Verifica coincidencia envolviendo el texto y la palabra clave en espacios delimitadores.",
                paraQueSirve: "Previene falsos positivos críticos en sistemas documentales, evitando que subcadenas como 'plan' se activen incorrectamente dentro de 'plantas'.",
                conceptos: "Tokenización por Límites, Coincidencia Contextual Exacta."
            },
            117: {
                tipo: "Motor de Inferencia Acumulativo",
                queHace: "Calcula los puntajes de coincidencia para cada categoría y las ordena descendentemente.",
                paraQueSirve: "Constituye el algoritmo decisorio central: selecciona la categoría con mayor soporte numérico como primaria y preserva las secundarias como evidencia.",
                conceptos: "Sistemas Expertos Basados en Reglas, Inferencia por Máxima Puntuación Simbólica."
            }
        }
    },

    semana04a: {
        titulo: "Semana 04-A",
        subtitulo: "Búsqueda A*",
        archivo: "src/semana04_busqueda.py",
        metricas: [
            { label: "Espacio de Estados", value: "20 casos reales (Grafo Dirigido)" },
            { label: "Conectividad", value: "k = 3 vecinos más cercanos" },
            { label: "Función de Costo", value: "g(n) = 1.0 + disimilitud coseno" },
            { label: "Heurística h(n)", value: "Distancia angular al objetivo" }
        ],
        overview: "Navegación explicable sobre grafos documentales: implementación del algoritmo A* utilizando similitud semántica TF-IDF como función heurística admisible.",
        codigoRaw: `from __future__ import annotations

import heapq
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semana03_taxonomia import read_cases, MANUAL_REFERENCE, CATEGORIES

RANDOM_STATE = 42

STOPWORDS_ES = {
    "a", "al", "algo", "alguna", "algunas", "alguno", "algunos", "ante",
    "antes", "como", "con", "contra", "cual", "cuando", "de", "del",
    "desde", "donde", "dos", "el", "ella", "ellas", "ellos", "en",
    "entre", "era", "es", "esa", "esas", "ese", "eso", "esos", "esta",
    "estas", "este", "estos", "fue", "ha", "hasta", "la", "las", "le",
    "les", "lo", "los", "más", "me", "mi", "mis", "muy", "no", "nos",
    "o", "para", "pero", "por", "que", "qué", "se", "sea", "ser", "si",
    "sin", "sobre", "son", "su", "sus", "también", "te", "ti", "tu",
    "tus", "un", "una", "uno", "unos", "y", "ya"
}


def cargar_datos_proyecto() -> Tuple[Dict[str, str], Dict[str, str]]:
    descripciones = read_cases()
    textos: Dict[str, str] = {}
    categorias: Dict[str, str] = {}
    for i, descripcion in enumerate(descripciones):
        nombre = f"caso_{i + 1:02d}"
        textos[nombre] = descripcion
        categorias[nombre] = MANUAL_REFERENCE[i]
    return textos, categorias


def limpiar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"\\s+", " ", texto)
    texto = re.sub(r"[^a-záéíóúüñ0-9\\s]", " ", texto)
    return re.sub(r"\\s+", " ", texto).strip()


def construir_grafo_documental(
    textos: Dict[str, str],
    vecinos: int = 3,
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, np.ndarray]]:
    nombres = list(textos.keys())
    corpus = [limpiar_texto(textos[nombre]) for nombre in nombres]

    vectorizador = TfidfVectorizer(
        stop_words=list(STOPWORDS_ES),
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    matriz = vectorizador.fit_transform(corpus)
    similitudes = cosine_similarity(matriz)

    grafo: Dict[str, List[Tuple[str, float]]] = {
        nombre: [] for nombre in nombres
    }

    cantidad_vecinos = min(max(1, vecinos), max(1, len(nombres) - 1))

    for i, nombre in enumerate(nombres):
        candidatos = [j for j in range(len(nombres)) if j != i]
        candidatos.sort(key=lambda j: similitudes[i, j], reverse=True)

        for j in candidatos[:cantidad_vecinos]:
            similitud = float(similitudes[i, j])
            distancia = 1.0 - similitud
            costo = 1.0 + distancia
            grafo[nombre].append((nombres[j], round(costo, 6)))

    vectores = {
        nombre: matriz[i].toarray().ravel()
        for i, nombre in enumerate(nombres)
    }

    return grafo, vectores


def heuristica(
    actual: str,
    objetivo: str,
    vectores: Dict[str, np.ndarray],
) -> float:
    if actual == objetivo:
        return 0.0

    vector_a = vectores[actual]
    vector_b = vectores[objetivo]

    norma_a = np.linalg.norm(vector_a)
    norma_b = np.linalg.norm(vector_b)

    if norma_a == 0 or norma_b == 0:
        return 0.0

    similitud = float(np.dot(vector_a, vector_b) / (norma_a * norma_b))
    distancia = max(0.0, min(1.0, 1.0 - similitud))
    return distancia


def reconstruir_camino(
    padres: Dict[str, Optional[str]],
    objetivo: str,
) -> List[str]:
    camino = []
    actual: Optional[str] = objetivo
    while actual is not None:
        camino.append(actual)
        actual = padres.get(actual)
    camino.reverse()
    return camino


def a_estrella(
    grafo: Dict[str, List[Tuple[str, float]]],
    vectores: Dict[str, np.ndarray],
    inicio: str,
    objetivo: str,
    usar_heuristica: bool = True,
) -> Tuple[Optional[List[str]], float, int]:
    if inicio not in grafo:
        raise ValueError(f"Caso inicial no existe: {inicio}")
    if objetivo not in grafo:
        raise ValueError(f"Caso objetivo no existe: {objetivo}")

    frontera = []
    contador = 0

    g = {inicio: 0.0}
    padres: Dict[str, Optional[str]] = {inicio: None}

    h_inicio = heuristica(inicio, objetivo, vectores) if usar_heuristica else 0.0
    heapq.heappush(frontera, (h_inicio, contador, inicio))

    cerrados = set()
    expansiones = 0

    while frontera:
        f_actual, _, actual = heapq.heappop(frontera)

        if actual in cerrados:
            continue

        cerrados.add(actual)
        expansiones += 1

        if actual == objetivo:
            camino = reconstruir_camino(padres, objetivo)
            return camino, g[objetivo], expansiones

        for vecino, costo in grafo[actual]:
            if vecino in cerrados:
                continue

            nuevo_g = g[actual] + costo

            if nuevo_g < g.get(vecino, math.inf):
                g[vecino] = nuevo_g
                padres[vecino] = actual

                h = heuristica(vecino, objetivo, vectores) if usar_heuristica else 0.0
                f = nuevo_g + h

                contador += 1
                heapq.heappush(frontera, (f, contador, vecino))

    return None, math.inf, expansiones


def main() -> None:
    textos, categorias = cargar_datos_proyecto()
    grafo, vectores = construir_grafo_documental(textos, vecinos=3)
    camino, costo, exp = a_estrella(grafo, vectores, "caso_01", "caso_20")
    print(f"Ruta: {' -> '.join(camino)} | Costo: {costo:.4f} | Expansiones: {exp}")


if __name__ == "__main__":
    main()`,
        lineas: {
            54: {
                tipo: "Construcción del Grafo Ponderado",
                queHace: "Vectoriza los textos con TF-IDF y conecta cada caso con sus k=3 vecinos más afines.",
                paraQueSirve: "Modela el espacio de estados documental donde los nodos son casos reales y los arcos representan transiciones semánticas directas.",
                conceptos: "Grafos Dirigidos Ponderados, Matriz de Similitud Coseno."
            },
            78: {
                tipo: "Función de Coste de Trayectoria g(n)",
                queHace: "Calcula el peso del arco como costo = 1.0 + (1.0 - similitud).",
                paraQueSirve: "Asegura que cada salto tenga un costo base unitario más la distancia de contenido, penalizando transiciones entre documentos divergentes.",
                conceptos: "Coste de Trayectoria Acumulado, Distancia de Coseno."
            },
            94: {
                tipo: "Heurística Admisible h(n)",
                queHace: "Calcula la distancia de coseno directa entre los vectores TF-IDF del nodo actual y el objetivo.",
                paraQueSirve: "Guía el avance de A* sin sobreestimar el coste real de llegada, lo que garantiza encontrar la ruta de menor costo.",
                conceptos: "Admisibilidad Heurística (h(n) <= h*(n)), Producto Punto Normalizado."
            },
            135: {
                tipo: "Gestión de Frontera con Prioridad",
                queHace: "Inserta los nodos en la cola de prioridad utilizando heapq con la tupla (f, contador, nodo).",
                paraQueSirve: "Asegura que A* extraiga en O(log N) el nodo con menor costo proyectado f(n) = g(n) + h(n), minimizando la expansión de nodos frente a Dijkstra.",
                conceptos: "Colas de Prioridad, Estructuras Min-Heap, Algoritmo A*."
            }
        }
    },

    semana04b: {
        titulo: "Semana 04-B",
        subtitulo: "Teoría Minimax",
        archivo: "src/semana04_busqueda.py",
        metricas: [
            { label: "Modelo de Decisión", value: "Juego de Suma Cero (MAX vs MIN)" },
            { label: "Estrategia Ganadora", value: "MAX elige Rama 'A'" },
            { label: "Utilidad Garantizada", value: "Valor = 3 (peor caso evaluado)" },
            { label: "Reducción por Poda", value: "28.6 % de nodos ahorrados" }
        ],
        overview: "Toma de decisiones en entornos adversarios: simulación de un árbol de decisión de suma cero mediante Minimax puro comparado contra Minimax con Poda Alfa-Beta.",
        codigoRaw: `import math

class MinimaxContador:
    """Contadores para evidenciar el efecto de alfa-beta."""

    def __init__(self) -> None:
        self.nodos_visitados = 0


def minimax(
    arbol: dict,
    nodo: str,
    maximizando: bool,
    contador: MinimaxContador,
) -> int:
    """Minimax básico para un árbol de juego finito, sin poda."""
    contador.nodos_visitados += 1
    valor = arbol[nodo]

    if isinstance(valor, int):
        return valor

    hijos = valor["hijos"]

    if maximizando:
        return max(minimax(arbol, hijo, False, contador) for hijo in hijos)

    return min(minimax(arbol, hijo, True, contador) for hijo in hijos)


def minimax_alfa_beta(
    arbol: dict,
    nodo: str,
    maximizando: bool,
    alpha: float,
    beta: float,
    contador: MinimaxContador,
) -> int:
    """
    Minimax con poda alfa-beta.

    alpha = mejor valor que MAX tiene garantizado en el camino actual.
    beta  = mejor valor que MIN tiene garantizado en el camino actual.
    """
    contador.nodos_visitados += 1
    valor = arbol[nodo]

    if isinstance(valor, int):
        return valor

    hijos = valor["hijos"]

    if maximizando:
        mejor = -math.inf
        for hijo in hijos:
            mejor = max(
                mejor,
                minimax_alfa_beta(arbol, hijo, False, alpha, beta, contador),
            )
            alpha = max(alpha, mejor)
            if beta <= alpha:
                break  # poda alfa: MIN ya tiene una opción mejor
        return int(mejor)

    mejor = math.inf
    for hijo in hijos:
        mejor = min(
            mejor,
            minimax_alfa_beta(arbol, hijo, True, alpha, beta, contador),
        )
        beta = min(beta, mejor)
        if beta <= alpha:
            break  # poda beta: MAX ya tiene una opción mejor
    return int(mejor)


def ejecutar_prueba_minimax() -> None:
    arbol = {
        "RAIZ": {"hijos": ["A", "B", "C"]},
        "A": {"hijos": ["A1", "A2"]},
        "B": {"hijos": ["B1", "B2"]},
        "C": {"hijos": ["C1", "C2"]},
        "A1": 3, "A2": 5,
        "B1": 2, "B2": 9,
        "C1": 4, "C2": 1,
    }

    c_norm = MinimaxContador()
    v_norm = minimax(arbol, "RAIZ", True, c_norm)

    c_poda = MinimaxContador()
    v_poda = minimax_alfa_beta(arbol, "RAIZ", True, -math.inf, math.inf, c_poda)

    print(f"Puro: {v_norm} ({c_norm.nodos_visitados} nodos)")
    print(f"Poda: {v_poda} ({c_poda.nodos_visitados} nodos)")


if __name__ == "__main__":
    ejecutar_prueba_minimax()`,
        lineas: {
            26: {
                tipo: "Algoritmo Minimax Puro",
                queHace: "Evalúa recursivamente todas las ramas del árbol alternando capas de maximización y minimización.",
                paraQueSirve: "Calcula el valor óptimo de utilidad en juegos donde un agente compite activamente contra otro con información perfecta.",
                conceptos: "Teoría de Juegos, Inducción hacia Atrás, Espacios de Búsqueda Adversaria."
            },
            55: {
                tipo: "Poda Alfa-Beta en MAX",
                queHace: "Propaga el parámetro alpha cortando el ciclo for en cuanto se cumple beta <= alpha.",
                paraQueSirve: "Descarta ramas que no pueden superar las alternativas ya encontradas en niveles superiores del árbol.",
                conceptos: "Poda Alfa-Beta, Reducción de Complejidad O(b^(d/2)), Equivalencia de Decisión."
            },
            66: {
                tipo: "Poda Alfa-Beta en MIN",
                queHace: "Actualiza el límite superior beta y detiene la evaluación al detectar dominancia.",
                paraQueSirve: "Evita calcular hojas irrelevantes que el jugador minimizador jamás escogería frente a las opciones de MAX.",
                conceptos: "Eficiencia de Búsqueda Adversaria, Omisión de Ramas Subóptimas."
            }
        }
    },

    semana05: {
        titulo: "Semana 05",
        subtitulo: "IA Híbrida",
        archivo: "src/semana05_sistema_hibrido.py",
        metricas: [
            { label: "Base de Conocimiento", value: "8 entradas (base_conocimiento.txt)" },
            { label: "Reglas de Negocio", value: "5 reglas operativas (R1 a R5)" },
            { label: "Clasificador Supervisado", value: "MultinomialNB (15 ejemplos)" },
            { label: "Recuperación", value: "TF-IDF + Cosine Similarity" }
        ],
        overview: "Integración de IA Híbrida: confluencia de Recuperación de Información (IR con TF-IDF), Aprendizaje Automático Probabilístico (Multinomial Naive Bayes) y Sistemas Expertos Basados en Reglas deterministas.",
        codigoRaw: `import os
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
    
    consultas_prueba = [
        "Se adjunta un contrato confidencial para el nuevo proyecto, pero no tiene nada más.",
        "Reporte urgente de vulnerabilidades encontradas en la base de datos de producción.",
        "Solicito que se revise el balance general de este trimestre."
    ]

    for i, consulta in enumerate(consultas_prueba):
        regla, match, sim, clase = procesar_consulta(consulta, base_conocimiento, vectorizer_similitud)
        print(f"Consulta {i+1}: {clase} -> Regla: {regla}")`,
        lineas: {
            10: {
                tipo: "Gestión Robusta de Rutas y Persistencia",
                queHace: "Resuelve la ruta hacia data/base_conocimiento.txt utilizando os.path.dirname(os.path.abspath(__file__)).",
                paraQueSirve: "Asegura que el script cargue la base de conocimiento sin fallos de FileNotFoundError sin importar el directorio de ejecución.",
                conceptos: "Rutas Absolutas Dinámicas, Entrada/Salida en Python con Codificación UTF-8."
            },
            47: {
                tipo: "Pipeline Probabilístico (Naive Bayes)",
                queHace: "Compone y entrena un pipeline con TfidfVectorizer y MultinomialNB sobre los 15 ejemplos supervisados.",
                paraQueSirve: "Proporciona clasificación automática rápida multiclase calculando la probabilidad posterior de cada categoría documental.",
                conceptos: "Aprendizaje Supervisado, Teorema de Bayes, Multinomial Naive Bayes."
            },
            55: {
                tipo: "Sistema Experto (Reglas Deterministas)",
                queHace: "Evalúa 5 reglas if/elif combinando palabras clave con la categoría predicha por el modelo de Machine Learning.",
                paraQueSirve: "Provee auditoría determinista de seguridad y alertas operativas que complementan el análisis estadístico.",
                conceptos: "Sistemas Expertos Basados en Reglas, Arquitectura Híbrida Simbólica/Subsimbólica."
            },
            77: {
                tipo: "Motor de Recuperación de Información (IR)",
                queHace: "Vectoriza la consulta junto con la base de conocimiento y calcula cosine_similarity.",
                paraQueSirve: "Recupera la directiva o regla más cercana semánticamente al requerimiento del usuario, arrojando el valor cuantitativo de similitud.",
                conceptos: "Information Retrieval (IR), Espacio Vectorial, Similitud del Coseno."
            }
        }
    }
};

// =========================================================================
// MOTOR DE INFERENCIA DIDÁCTICA (Generación Automática de Explicaciones)
// =========================================================================
function generarExplicacionContextual(rawCode, lineNum, weekData) {
    const code = rawCode.trim();

    // 1. Poda Alfa-Beta y Cortes de Búsqueda
    if (code.includes("break")) {
        return {
            tipo: "Condición de Parada Temprana / Poda de Árbol",
            queHace: "Interrumpe de forma inmediata la iteración del ciclo sobre los hijos del nodo actual.",
            paraQueSirve: "Aplica la poda matemática (alfa o beta). Evita evaluar el resto de los nodos, ya que se descubrió una rama que el adversario o el sistema jamás permitiría elegir frente a alternativas ya exploradas, ahorrando CPU y memoria.",
            conceptos: "Control de flujo de interrupción en bucles. Optimización del espacio de estados y poda de ramas dominadas en árboles de decisión."
        };
    }

    // 2. Control de Nodos Cerrados / Grafos
    if (code.includes("continue") || code.includes("in cerrados")) {
        return {
            tipo: "Filtrado de Ciclos y Nodos Ya Explorados",
            queHace: "Verifica si el estado ya fue evaluado previamente y salta a la siguiente iteración.",
            paraQueSirve: "Evita ciclos infinitos en el grafo de documentos y previene recomputar caminos hacia casos cuya distancia óptima g(n) ya fue calculada.",
            conceptos: "Conjunto de nodos cerrados (Closed Set) en algoritmos de búsqueda informada como A*."
        };
    }

    // 3. Sentencias Return
    if (code.startsWith("return")) {
        return {
            tipo: "Transferencia de Salida y Cierre de Función",
            queHace: "Devuelve el resultado del procesamiento hacia el módulo invocador y finaliza el marco de ejecución.",
            paraQueSirve: "Entrega a las capas superiores del pipeline las estructuras procesadas (métricas, caminos de similitud, arrays numpy o la categoría documental predicha).",
            conceptos: "Alcance (Scope) de variables, retorno explícito de tipos y finalización de marcos en la pila de llamadas (call stack)."
        };
    }

    // 4. Inserción o Extracción de Colas de Prioridad (heapq)
    if (code.includes("heappush") || code.includes("heappop")) {
        return {
            tipo: "Operación en Montículo Binario (Min-Heap)",
            queHace: "Inserta o extrae un elemento conservando el orden de menor a mayor basado en f(n).",
            paraQueSirve: "Garantiza que el algoritmo A* extraiga en tiempo O(log N) el siguiente caso documental con la menor disimilitud semántica acumulada.",
            conceptos: "Estructura de datos Montículo Binario (Heap) y optimización de frontera de exploración."
        };
    }

    // 5. Normalización y Limpieza Regex
    if (code.includes("re.sub") || code.includes("unicodedata") || code.includes("lower()")) {
        return {
            tipo: "Normalización y Homogeneización de Texto",
            queHace: "Transforma la cadena eliminando tildes, signos de puntuación y convirtiendo a minúsculas.",
            paraQueSirve: "Evita falsos negativos en el análisis documental: asegura que variaciones ortográficas apunten al mismo término matemático.",
            conceptos: "Preprocesamiento de texto en Procesamiento de Lenguaje Natural (PLN) y expresiones regulares (Regex)."
        };
    }

    // 6. Condicionales de Control (if / elif / else)
    if (code.startsWith("if ") || code.startsWith("elif ") || code.startsWith("else:")) {
        return {
            tipo: "Bifurcación Lógica y Control de Reglas",
            queHace: "Evalúa si una condición booleana o pertenencia de términos es verdadera.",
            paraQueSirve: "Permite bifurcar el flujo de inferencia: ya sea para validar reglas de negocio en el sistema experto o para verificar si se alcanzó el estado objetivo de la búsqueda.",
            conceptos: "Estructuras condicionales de control y evaluación de expresiones lógicas booleanas."
        };
    }

    // 7. Definición de Funciones o Clases
    if (code.startsWith("def ") || code.startsWith("class ")) {
        return {
            tipo: "Declaración de Entidad o Procedimiento Modular",
            queHace: "Registra una función o clase en el espacio de nombres local.",
            paraQueSirve: "Encapsula responsabilidades específicas dentro del pipeline (carga de datos, cálculo de heurísticas o inferencia probabilística).",
            conceptos: "Modularidad, encapsulamiento y principios de diseño limpio de software en Python."
        };
    }

    // 8. Comentarios o Docstrings
    if (code.startsWith("#") || code.startsWith('"""') || code.startsWith("'''")) {
        return {
            tipo: "Documentación Técnica y Metadatos",
            queHace: "Texto descriptivo interpretado como comentario por el intérprete de Python.",
            paraQueSirve: "Explica las premisas de diseño del modelo, documentando las decisiones lógicas y arquitectónicas del proyecto.",
            conceptos: "Buenas prácticas de documentación técnica (PEP 257) y mantenibilidad del software de IA."
        };
    }

    // 9. Importaciones
    if (code.startsWith("import ") || code.startsWith("from ")) {
        return {
            tipo: "Gestión de Dependencias y Paquetes",
            queHace: "Incorpora un módulo o funciones específicas al espacio de trabajo actual.",
            paraQueSirve: "Provee herramientas robustas de manipulación de datos y algoritmos de Machine Learning (como sklearn o numpy) para no reinventar la rueda.",
            conceptos: "Modularidad y Reutilización de Código."
        };
    }

    // 10. Instrucciones Operativas Generales
    return {
        tipo: "Instrucción Operativa o Contextual",
        queHace: "Ejecuta una operación general (asignación de variables, formateo o llamadas simples).",
        paraQueSirve: `Aporta a la estructura base y secuencia lógica del archivo en el contexto de ${weekData.titulo}.`,
        conceptos: "Sintaxis estándar y flujo de control secuencial en Python."
    };
}


// =========================================================================
// CONTROLADOR Y LÓGICA DE INTERFAZ
// =========================================================================
let currentWeekId = "semana02";
let isExpanded = false;

const dom = {
    tabsContainer: document.getElementById('week-tabs'),
    metricsContainer: document.getElementById('metrics-container'),
    codeContainer: document.getElementById('code-container'),
    filePath: document.getElementById('file-path'),
    lineCount: document.getElementById('line-count'),
    emptyState: document.getElementById('empty-state'),
    expCard: document.getElementById('explanation-card'),
    expOverview: document.getElementById('exp-overview'),
    expLineNum: document.getElementById('exp-line-num'),
    expSnippet: document.getElementById('exp-snippet'),
    expType: document.getElementById('exp-type'),
    expWhat: document.getElementById('exp-what'),
    expWhy: document.getElementById('exp-why'),
    expConcepts: document.getElementById('exp-concepts'),
    leftPanel: document.getElementById('left-panel'),
    rightPanel: document.getElementById('right-panel'),
    resizer: document.getElementById('dragMe'),
    themeToggle: document.getElementById('theme-toggle'),
    currentLineInd: document.getElementById('current-line-indicator')
};

function initApp() {
    renderTabs();
    loadWeek(currentWeekId);
    initResizer();
    dom.themeToggle.addEventListener('click', () => {
        document.body.classList.toggle('light-mode');
    });
}

function renderTabs() {
    dom.tabsContainer.innerHTML = '';
    Object.keys(dashboardData).forEach(weekKey => {
        const week = dashboardData[weekKey];
        const btn = document.createElement('button');
        btn.className = `tab-btn ${weekKey === currentWeekId ? 'active' : ''}`;
        btn.innerHTML = `${week.titulo} <span>${week.subtitulo}</span>`;
        btn.onclick = () => loadWeek(weekKey);
        dom.tabsContainer.appendChild(btn);
    });
}

function loadWeek(weekId) {
    currentWeekId = weekId;
    const data = dashboardData[weekId];
    renderTabs();
    dom.filePath.textContent = data.archivo;

    dom.metricsContainer.innerHTML = data.metricas.map(m => `
        <div class="metric-badge">
            <span class="metric-label">${m.label}:</span>
            <span class="metric-value">${m.value}</span>
        </div>
    `).join('');

    renderCode(data);

    // Selección automática de la primera línea explicada explícitamente (o la línea 1 si no hay mapeo numérico)
    const lineKeys = Object.keys(data.lineas).map(Number);
    const firstExplainedLine = lineKeys.length > 0 ? lineKeys.sort((a, b) => a - b)[0] : 1;
    
    if (firstExplainedLine) {
        const lines = data.codigoRaw.split('\n');
        selectLine(firstExplainedLine, lines[firstExplainedLine - 1], data);
    }
}

function escapeHTML(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function highlightPython(text) {
    let s = escapeHTML(text);
    s = s.replace(/(#[^\n]*)/g, '<span class="sy-com">$1</span>');
    s = s.replace(/("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')/g, '<span class="sy-str">$1</span>');
    s = s.replace(/\b(from|import|def|class|return|if|elif|else|for|in|as|while|break|continue|try|except|raise|with|pass|assert)\b/g, '<span class="sy-kw">$1</span>');
    s = s.replace(/\b(self|None|True|False)\b/g, '<span class="sy-kw">$1</span>');
    s = s.replace(/\b([a-zA-Z_]\w*)\s*(?=\()/g, '<span class="sy-func">$1</span>');
    return s;
}

function renderCode(data) {
    dom.codeContainer.innerHTML = '';
    const lines = data.codigoRaw.split('\n');
    dom.lineCount.textContent = `${lines.length} líneas`;

    lines.forEach((lineText, index) => {
        const lineNum = index + 1;
        const lineDiv = document.createElement('div');
        lineDiv.className = 'code-line';
        lineDiv.id = `line-${lineNum}`;

        lineDiv.innerHTML = `
            <div class="line-number">${lineNum}</div>
            <div class="line-content">${highlightPython(lineText) || '&nbsp;'}</div>
        `;

        lineDiv.onclick = () => selectLine(lineNum, lineText, data);
        dom.codeContainer.appendChild(lineDiv);
    });
}

function selectLine(lineNum, rawCode, data) {
    document.querySelectorAll('.code-line').forEach(el => el.classList.remove('active'));
    const target = document.getElementById(`line-${lineNum}`);
    if (target) {
        target.classList.add('active');
        // El scrollIntoView es suave y asegura que el código cliqueado sea visible
        target.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }

    dom.emptyState.style.display = 'none';
    dom.expCard.style.display = 'block';
    dom.currentLineInd.textContent = `Línea ${lineNum} seleccionada`;

    // 1. Revisa si hay una explicación manual detallada. 
    // 2. Si no, usa el motor de inferencia contextual que acabamos de crear.
    const exp = data.lineas[lineNum] || generarExplicacionContextual(rawCode, lineNum, data);

    dom.expOverview.textContent = data.overview;
    dom.expType.textContent = exp.tipo;
    dom.expLineNum.textContent = `Línea ${lineNum}`;
    
    // Si la línea está vacía, mostrar un aviso sutil
    const snippetText = rawCode.trim();
    dom.expSnippet.textContent = snippetText ? snippetText : "(Línea en blanco)";
    
    dom.expWhat.textContent = exp.queHace;
    dom.expWhy.textContent = exp.paraQueSirve;
    dom.expConcepts.textContent = exp.conceptos;
}

function togglePanel(target) {
    if (isExpanded) {
        dom.leftPanel.style.width = '50%';
        dom.rightPanel.style.width = '50%';
        dom.rightPanel.style.display = 'flex';
        dom.leftPanel.style.display = 'flex';
        dom.resizer.style.display = 'block';
        isExpanded = false;
    } else {
        if (target === 'left') {
            dom.leftPanel.style.width = '100%';
            dom.rightPanel.style.display = 'none';
        } else {
            dom.rightPanel.style.width = '100%';
            dom.leftPanel.style.display = 'none';
        }
        dom.resizer.style.display = 'none';
        isExpanded = true;
    }
}

function initResizer() {
    let x = 0;
    let leftWidth = 0;

    const onMouseDown = function (e) {
        x = e.clientX;
        leftWidth = dom.leftPanel.getBoundingClientRect().width;
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
        // Usa el fucsia del tema para el resizer activo
        dom.resizer.style.backgroundColor = 'var(--accent-fuchsia)';
        document.body.style.cursor = 'col-resize';
    };

    const onMouseMove = function (e) {
        const dx = e.clientX - x;
        const totalW = dom.resizer.parentNode.getBoundingClientRect().width;
        const newPct = ((leftWidth + dx) * 100) / totalW;
        if (newPct > 20 && newPct < 80) {
            dom.leftPanel.style.width = `${newPct}%`;
            dom.rightPanel.style.width = `${100 - newPct}%`;
        }
    };

    const onMouseUp = function () {
        dom.resizer.style.backgroundColor = '';
        document.body.style.cursor = 'default';
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);
    };

    dom.resizer.addEventListener('mousedown', onMouseDown);
}

document.addEventListener('DOMContentLoaded', initApp);