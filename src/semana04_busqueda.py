"""
Semana 04 - Marco Tecnológico de la Inteligencia Artificial
Sistema de Análisis Documental

Implementa:
1. A* para navegar el grafo de similitud entre los casos reales del
   proyecto (data/casos_ia.csv), usados también en la Semana 03.
2. Minimax + poda alfa-beta como práctica de referencia.

Integración con el proyecto
----------------------------
Este script NO usa un corpus sintético aparte: reutiliza directamente
`read_cases()` y `MANUAL_REFERENCE` de `semana03_taxonomia.py`, es
decir, los mismos 20 casos y las mismas categorías ya clasificadas en
la Semana 03. Cada "documento" del grafo es uno de esos 20 casos
reales del proyecto.

Aplicación de A* al proyecto
------------------------------
En un repositorio de casos de IA aplicados a documentos, es útil poder
explicar por qué dos casos aparentemente distintos están relacionados
(navegación semántica / recomendación explicable: "para llegar de este
caso a aquel, pasa por estos intermedios"), en lugar de solo calcular
una similitud directa entre dos puntos. Por eso el problema se modela
como búsqueda de la ruta de menor costo dentro de un grafo de
similitud, no como una simple comparación 1 a 1.

- Estado: caso actual (uno de los 20 casos de data/casos_ia.csv).
- Estados posibles: cualquier caso del proyecto.
- Acción: moverse del caso actual a uno de sus casos más similares
  (vecinos en el grafo de similitud TF-IDF).
- Transición: movimiento permitido por el grafo (dirigido: cada caso
  se conecta con sus propios vecinos más similares, no
  necesariamente de forma simétrica).
- Costo: 1 + distancia_TFIDF entre ambos casos (1 punto fijo por
  salto, más la disimilitud real entre ambos textos).
- Meta: llegar al caso objetivo.
- Heurística: distancia_TFIDF (coseno) del caso actual al objetivo.
- Criterio: minimizar f(n) = g(n) + h(n).

Por qué A* es pertinente: el costo de cada salto no es uniforme (dos
casos muy similares cuestan casi 1, dos casos muy distintos cuestan
casi 2), por lo que una búsqueda no informada (BFS) no bastaría para
garantizar la ruta más "semánticamente barata"; A* con una heurística
admisible garantiza encontrarla explorando menos nodos que Dijkstra.

Uso:
    python src/semana04_busqueda.py
"""

from __future__ import annotations

import heapq
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Reutiliza directamente los datos y categorías reales del proyecto
# (mismo corpus y misma clasificación manual que la Semana 03).
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

CATEGORY_NAMES = {c.name for c in CATEGORIES}
assert set(MANUAL_REFERENCE) <= CATEGORY_NAMES, (
    "MANUAL_REFERENCE contiene una categoria que no existe en CATEGORIES"
)


# ---------------------------------------------------------------------
# CARGA DE LOS DATOS REALES DEL PROYECTO
# ---------------------------------------------------------------------

def cargar_datos_proyecto() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Carga los 20 casos reales de data/casos_ia.csv (los mismos que usa
    semana03_taxonomia.py) y su categoría manual asociada.

    Retorna:
        textos:      {"caso_01": "Escanear un contrato...", ...}
        categorias:  {"caso_01": "Visión por computador y OCR", ...}
    """
    descripciones = read_cases()

    textos: Dict[str, str] = {}
    categorias: Dict[str, str] = {}

    for i, descripcion in enumerate(descripciones):
        nombre = f"caso_{i + 1:02d}"
        textos[nombre] = descripcion
        categorias[nombre] = MANUAL_REFERENCE[i]

    return textos, categorias


# ---------------------------------------------------------------------
# A* APLICADO AL SISTEMA DE ANÁLISIS DOCUMENTAL
# ---------------------------------------------------------------------

def limpiar_texto(texto: str) -> str:
    """Normaliza texto para la representación TF-IDF."""
    texto = texto.lower()
    texto = re.sub(r"\s+", " ", texto)
    texto = re.sub(r"[^a-záéíóúüñ0-9\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def construir_grafo_documental(
    textos: Dict[str, str],
    vecinos: int = 3,
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, np.ndarray]]:
    """
    Construye un grafo documental sobre los casos reales del proyecto.

    Cada caso es un estado. Se conectan los `vecinos` casos más
    similares a cada caso (grafo dirigido, no necesariamente
    simétrico). El costo de una transición es:

        costo = 1 + distancia_TFIDF

    donde:

        distancia_TFIDF = 1 - similitud_coseno
    """
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
    """
    Heurística admisible para el grafo usado:

        h(n) = distancia coseno(n, objetivo)

    Como cada transición cuesta 1 + distancia, y la distancia siempre
    está entre 0 y 1, esta heurística nunca supera el costo de una
    transición directa (o de ninguna ruta, ya que toda ruta con al
    menos un salto cuesta >= 1): por lo tanto nunca sobreestima.
    """
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
    """Reconstruye el camino encontrado por A*."""
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
    """
    Ejecuta A* (o Dijkstra si usar_heuristica=False, para comparar).

    f(n) = g(n) + h(n)

    Retorna:
        camino, costo total, cantidad de nodos expandidos.
    """
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


def imprimir_grafo(
    grafo: Dict[str, List[Tuple[str, float]]],
    categorias: Dict[str, str],
) -> None:
    print("\n--- GRAFO DOCUMENTAL (casos reales del proyecto) ---")
    for caso, vecinos in grafo.items():
        relaciones = ", ".join(
            f"{vecino} (costo={costo:.3f})" for vecino, costo in vecinos
        )
        print(f"{caso} [{categorias[caso]}] -> {relaciones}")


# ---------------------------------------------------------------------
# MINIMAX + PODA ALFA-BETA (práctica de referencia)
# ---------------------------------------------------------------------
#
# Por qué es una práctica de referencia y no del proyecto:
# el sistema de análisis documental (arriba) es un problema de
# búsqueda de un solo agente: no existe un segundo agente racional que
# compita activamente contra el sistema con un objetivo opuesto y
# turnos alternos. Minimax no aplica a ese dominio, así que se realiza
# aquí la práctica de referencia sobre un árbol de decisión abstracto,
# tal como exige la guía cuando Minimax no aplica al proyecto.
#
# Representación formal de esta práctica:
# - Estado: un nodo del árbol de decisión (RAIZ, A, B, C, A1, A2, ...).
# - Acciones disponibles: moverse a uno de los nodos hijos del estado
#   actual.
# - Jugador MAX: decide en la raíz (elige entre las estrategias A, B, C).
# - Jugador MIN: decide en el segundo nivel (responde dentro de cada
#   estrategia intentando minimizar el resultado de MAX).
# - Estados terminales: las hojas del árbol (A1, A2, B1, B2, C1, C2),
#   cada una con una utilidad ya conocida.
# - Función de utilidad: el valor entero asignado a cada hoja.
# - Decisión seleccionada: MAX elige la estrategia (A, B o C) cuyo peor
#   caso posible (el valor que MIN le dejaría) es el más alto de las
#   tres.

class MinimaxContador:
    """Contadores para evidenciar el efecto de alfa-beta."""

    def __init__(self) -> None:
        self.nodos_visitados = 0


def minimax(
    arbol: Dict,
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
    arbol: Dict,
    nodo: str,
    maximizando: bool,
    alpha: float,
    beta: float,
    contador: MinimaxContador,
) -> int:
    """
    Minimax con poda alfa-beta.

    alpha = mejor valor que MAX tiene garantizado hasta ahora en el
    camino actual del árbol.
    beta  = mejor valor que MIN tiene garantizado hasta ahora.

    Cuando beta <= alpha, el jugador de un nivel superior del árbol ya
    tiene una alternativa al menos igual de buena y jamás permitiría
    llegar a esta rama: se poda sin calcular su valor exacto. La poda
    nunca cambia el resultado final de Minimax, solo evita explorar
    ramas irrelevantes para la decisión.
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
                break  # poda: MIN ya tiene una opcion mejor mas arriba
        return int(mejor)

    mejor = math.inf
    for hijo in hijos:
        mejor = min(
            mejor,
            minimax_alfa_beta(arbol, hijo, True, alpha, beta, contador),
        )
        beta = min(beta, mejor)
        if beta <= alpha:
            break  # poda: MAX ya tiene una opcion mejor mas arriba
    return int(mejor)


def ejecutar_prueba_minimax() -> None:
    """
    Práctica de referencia de Minimax.

    El árbol representa un juego sencillo: MAX elige entre las
    estrategias A, B y C; MIN responde dentro de cada rama intentando
    minimizar el resultado de MAX. Las hojas ya tienen una utilidad
    conocida (no requieren evaluación adicional).
    """
    arbol = {
        "RAIZ": {"hijos": ["A", "B", "C"]},
        "A": {"hijos": ["A1", "A2"]},
        "B": {"hijos": ["B1", "B2"]},
        "C": {"hijos": ["C1", "C2"]},
        "A1": 3, "A2": 5,
        "B1": 2, "B2": 9,
        "C1": 4, "C2": 1,
    }

    contador_normal = MinimaxContador()
    valor_normal = minimax(arbol, "RAIZ", True, contador_normal)

    contador_poda = MinimaxContador()
    valor_poda = minimax_alfa_beta(arbol, "RAIZ", True, -math.inf, math.inf, contador_poda)

    # La estrategia ganadora es aquella cuyo peor caso (el minimo de
    # sus hojas) coincide con el valor minimax obtenido.
    peor_caso = {
        "A": min(arbol["A1"], arbol["A2"]),
        "B": min(arbol["B1"], arbol["B2"]),
        "C": min(arbol["C1"], arbol["C2"]),
    }
    estrategia_elegida = max(peor_caso, key=peor_caso.get)

    print("\n--- MINIMAX DE REFERENCIA (práctica, no aplica al proyecto) ---")
    print("Estados terminales y su utilidad:", 
          {k: v for k, v in arbol.items() if isinstance(v, int)})
    print(f"Utilidad con Minimax puro: {valor_normal}")
    print(f"Nodos visitados sin poda: {contador_normal.nodos_visitados}")
    print(f"Utilidad con poda alfa-beta: {valor_poda}")
    print(f"Nodos visitados con poda: {contador_poda.nodos_visitados}")
    reduccion = 100 * (1 - contador_poda.nodos_visitados / contador_normal.nodos_visitados)
    print(f"Reduccion de nodos por la poda: {reduccion:.1f}%")
    print(
        f"Decision de MAX: elegir la estrategia '{estrategia_elegida}', "
        f"cuyo peor caso garantiza utilidad {valor_normal} "
        "(MIN no puede llevarla a un valor menor)."
    )

    assert valor_normal == valor_poda, "La poda no deberia cambiar el resultado"


# ---------------------------------------------------------------------
# CASOS DE PRUEBA DE A* SOBRE LOS CASOS REALES DEL PROYECTO
# ---------------------------------------------------------------------

def ejecutar_casos_a_estrella(textos: Dict[str, str], categorias: Dict[str, str]) -> None:
    """Ejecuta 5 casos de prueba (minimo requerido: 3)."""

    # --- Grafo "normal": cada caso conectado a sus 3 vecinos mas similares ---
    grafo, vectores = construir_grafo_documental(textos, vecinos=3)
    imprimir_grafo(grafo, categorias)

    print("\n--- PRUEBAS A* (grafo con vecinos=3) ---")

    casos = [
        ("caso_01", "caso_20", "Caso 1: OCR (ingesta) -> Automatizacion (reporte final)"),
        ("caso_05", "caso_19", "Caso 2: Busqueda documental -> Aprendizaje predictivo (anomalias)"),
        ("caso_02", "caso_07", "Caso 3: Clasificacion documental -> Sistemas expertos (validacion)"),
    ]

    for inicio, objetivo, descripcion in casos:
        camino, costo, expansiones = a_estrella(grafo, vectores, inicio, objetivo)
        sin_heuristica = a_estrella(grafo, vectores, inicio, objetivo, usar_heuristica=False)

        print(f"\n{descripcion}")
        print(f"Inicio: {inicio} [{categorias[inicio]}] - \"{textos[inicio]}\"")
        print(f"Meta:   {objetivo} [{categorias[objetivo]}] - \"{textos[objetivo]}\"")

        if camino is None:
            print("Resultado: no se encontro una ruta.")
            continue

        print("Ruta encontrada:", " -> ".join(camino))
        print(f"Costo total: {costo:.4f}")
        print(f"Nodos expandidos con heuristica: {expansiones}")
        print(f"Nodos expandidos sin heuristica (h=0): {sin_heuristica[2]}")

    # --- Caso 4: mismas condiciones, grafo mas disperso (vecinos=1) ---
    # Demuestra que, al reducir las conexiones disponibles, A* puede
    # verse obligado a tomar una ruta distinta (o mas costosa).
    grafo_disperso, vectores_disperso = construir_grafo_documental(textos, vecinos=1)
    print("\n--- CASO 4: mismo par que el Caso 1, pero con grafo mas disperso (vecinos=1) ---")
    camino4, costo4, exp4 = a_estrella(grafo_disperso, vectores_disperso, "caso_01", "caso_20")
    print(f"Inicio: caso_01 [{categorias['caso_01']}]  Meta: caso_20 [{categorias['caso_20']}]")
    if camino4 is None:
        print("Resultado: no existe ruta con vecinos=1 (el grafo quedo demasiado disperso).")
    else:
        print("Ruta encontrada:", " -> ".join(camino4))
        print(f"Costo total: {costo4:.4f}  (vecinos=3 daba costo menor o igual; "
              "menos conexiones nunca puede bajar el costo optimo)")
        print(f"Nodos expandidos: {exp4}")

    # --- Caso 5: meta inalcanzable ---
    # Con vecinos=1 el grafo es lo bastante disperso como para que
    # algunos pares queden fuera de alcance. Se verifica cual par
    # concreto queda desconectado y se reporta correctamente.
    print("\n--- CASO 5: busqueda de un par sin ruta posible (grafo vecinos=1) ---")
    nombres = list(textos.keys())
    caso_sin_solucion = None
    for inicio in nombres:
        for objetivo in nombres:
            if inicio == objetivo:
                continue
            camino, _, _ = a_estrella(grafo_disperso, vectores_disperso, inicio, objetivo)
            if camino is None:
                caso_sin_solucion = (inicio, objetivo)
                break
        if caso_sin_solucion:
            break

    if caso_sin_solucion:
        inicio, objetivo = caso_sin_solucion
        print(f"Inicio: {inicio} [{categorias[inicio]}]  Meta: {objetivo} [{categorias[objetivo]}]")
        print("Resultado: A* agota la frontera y reporta correctamente que no hay ruta,")
        print("sin fallar ni entrar en bucle.")
    else:
        print("Con esta configuracion el grafo resulto conexo; no se encontro un par sin ruta.")


def generar_informe(
    ruta: Optional[Path] = None,
    textos: Optional[Dict[str, str]] = None,
    categorias: Optional[Dict[str, str]] = None,
) -> Path:
    """Genera automáticamente el informe Markdown de la Semana 04."""
    if ruta is None:
        ruta = Path(__file__).resolve().parent.parent / "reports" / "semana04.md"
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    if textos is None or categorias is None:
        textos, categorias = cargar_datos_proyecto()

    grafo, vectores = construir_grafo_documental(textos, vecinos=3)
    casos = [
        ("caso_01", "caso_20", "OCR (ingesta) -> automatización (reporte final)"),
        ("caso_05", "caso_19", "Búsqueda documental -> aprendizaje predictivo"),
        ("caso_02", "caso_07", "Clasificación documental -> sistemas expertos"),
    ]

    filas = []
    for inicio, objetivo, descripcion in casos:
        camino, costo, expansiones = a_estrella(grafo, vectores, inicio, objetivo)
        _, _, expansiones_dijkstra = a_estrella(
            grafo, vectores, inicio, objetivo, usar_heuristica=False
        )
        ruta_texto = " -> ".join(camino) if camino else "Sin ruta"
        filas.append(
            f"| {descripcion} | {ruta_texto} | "
            f"{costo:.4f} | {expansiones} | {expansiones_dijkstra} |"
        )

    arbol = {
        "RAIZ": {"hijos": ["A", "B", "C"]},
        "A": {"hijos": ["A1", "A2"]},
        "B": {"hijos": ["B1", "B2"]},
        "C": {"hijos": ["C1", "C2"]},
        "A1": 3, "A2": 5, "B1": 2, "B2": 9, "C1": 4, "C2": 1,
    }
    contador_normal = MinimaxContador()
    valor_normal = minimax(arbol, "RAIZ", True, contador_normal)
    contador_poda = MinimaxContador()
    valor_poda = minimax_alfa_beta(
        arbol, "RAIZ", True, -math.inf, math.inf, contador_poda
    )
    reduccion = 100 * (1 - contador_poda.nodos_visitados / contador_normal.nodos_visitados)

    contenido = f"""# Semana 04 — Búsqueda en el sistema de análisis documental

## Objetivo

Aplicar A* sobre un grafo de similitud TF-IDF construido con los casos reales de `data/casos_ia.csv`, y documentar Minimax con poda alfa-beta como práctica de referencia.

## Datos y modelo

- Casos analizados: **{len(textos)}**.
- Vecinos por caso: **3**.
- Costo de transición: `1 + distancia TF-IDF`.
- Heurística de A*: distancia coseno entre el caso actual y la meta.

## Pruebas de A*

| Caso | Ruta | Costo | Expansiones A* | Expansiones h=0 |
|---|---|---:|---:|---:|
{chr(10).join(filas)}

## Minimax y alfa-beta

- Valor con Minimax puro: **{valor_normal}**.
- Nodos visitados sin poda: **{contador_normal.nodos_visitados}**.
- Valor con poda alfa-beta: **{valor_poda}**.
- Nodos visitados con poda: **{contador_poda.nodos_visitados}**.
- Reducción de nodos: **{reduccion:.1f}%**.

La poda alfa-beta conserva la misma utilidad final y evita explorar ramas que no pueden mejorar la decisión de MAX.

## Ejecución

```bash
python src/semana04_busqueda.py
```

Este archivo se regenera automáticamente en `reports/semana04.md` cada vez que se ejecuta el script.
"""
    ruta.write_text(contenido, encoding="utf-8")
    return ruta

def main() -> None:
    print("=" * 70)
    print(" SEMANA 04 - A* EN EL SISTEMA DE ANÁLISIS DOCUMENTAL")
    print("=" * 70)

    textos, categorias = cargar_datos_proyecto()
    print(f"Casos cargados desde data/casos_ia.csv (Semana 03): {len(textos)}")

    ejecutar_casos_a_estrella(textos, categorias)
    ejecutar_prueba_minimax()
    ruta_informe = generar_informe(textos=textos, categorias=categorias)
    print(f"\nInforme generado: {ruta_informe}")

    print("\n" + "=" * 70)
    print(" PRÁCTICA FINALIZADA CORRECTAMENTE")
    print("=" * 70)


if __name__ == "__main__":
    main()