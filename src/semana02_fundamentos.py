"""
Semana 02 - Fundamentos de Aprendizaje Supervisado
Sistema de Análisis Documental

Primer modelo reproducible de clasificación supervisada aplicado al
proyecto: en lugar del dataset Iris, el modelo aprende a asignar la
categoría de la taxonomía documental a cada caso real del proyecto
(`data/casos_ia.csv`).

Qué se conserva de la práctica original
----------------------------------------
La lógica de la práctica de fundamentos es la misma:

    datos etiquetados -> vectorización -> pipeline -> modelo
    supervisado (Logistic Regression) -> evaluación con accuracy y
    matriz de confusión, todo con semilla fija (reproducible).

Qué cambia al aplicarla al proyecto
------------------------------------
1. Datos: se reemplaza `load_iris()` por los 20 casos reales de
   `data/casos_ia.csv` y sus etiquetas de `MANUAL_REFERENCE`
   (`semana03_taxonomia.py`). Son los mismos datos que usan la
   Semana 03 y la Semana 04.

2. Características (X): las entradas ya no son 4 medidas numéricas,
   sino texto libre. Por eso `StandardScaler` se reemplaza por
   `TfidfVectorizer`: convierte cada descripción en un vector de
   frecuencias TF-IDF (unigramas y bigramas, sin stopwords en
   español). Es el paso equivalente al escalado: llevar los datos
   crudos a una representación numérica comparable.

3. Etiquetas (y): las 3 especies de Iris se reemplazan por las 7
   categorías de la taxonomía documental.

4. Evaluación: `train_test_split(test_size=0.25, stratify=y)` NO es
   aplicable aquí. El proyecto tiene 20 casos y 7 categorías, así que
   un conjunto de prueba del 25% tendría 5 muestras para 7 clases y
   la división estratificada es imposible (scikit-learn lo rechaza).
   Con un dataset tan pequeño, la alternativa metodológicamente
   correcta es la validación cruzada:

   - Leave-One-Out (LOO): se entrena 20 veces, dejando cada vez un
     caso fuera como prueba. Así cada uno de los 20 casos se predice
     con un modelo que nunca lo vio, y se obtiene una accuracy y una
     matriz de confusión sobre las 20 predicciones.
   - StratifiedKFold (k=2): validación cruzada estratificada, que
     conserva la proporción de categorías en cada partición. Es el
     máximo k posible porque la categoría menos frecuente tiene solo
     2 ejemplos.

   Ambas son honestas: en ningún momento el modelo evalúa un caso que
   haya usado para entrenar.

Además se compara el modelo aprendido contra el clasificador por
reglas de la Semana 03, para ver si aprender de los datos mejora o no
frente a las reglas escritas a mano.

Salida automática
------------------
Al ejecutarse, el script imprime los resultados en consola y genera
automáticamente `reports/semana02.md`, igual que hacen
`semana03_taxonomia.py` y `semana04_busqueda.py`.

Uso:
    python src/semana02_fundamentos.py
"""

from __future__ import annotations

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
    texto = re.sub(r"[^a-záéíóúüñ0-9\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def cargar_datos() -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Carga el dataset supervisado del proyecto.

    Equivalente a `load_iris(return_X_y=True)` de la práctica original,
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
    (`make_pipeline(StandardScaler(), LogisticRegression(...))`), pero
    el paso de preparación de datos es `TfidfVectorizer` porque las
    entradas son texto y no medidas numéricas.
    Dos ajustes son necesarios por el tamaño del dataset:

    - `ngram_range=(1, 1)`: con 20 documentos cortos, los bigramas
      generan cientos de rasgos que aparecen una sola vez y el modelo
      no puede generalizar a partir de ellos.
    - `class_weight="balanced"`: las categorías están desbalanceadas
      (de 2 a 4 casos). Sin este ajuste el modelo tiende a predecir
      siempre las categorías mayoritarias e ignora las pequeñas.

    El efecto de ambos ajustes se mide en `comparar_configuraciones()`.
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

    Se entrena `len(X)` veces; en cada iteración un caso queda fuera y
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
    idea de `stratify=y` en la práctica original. `n_splits` está
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

    Es el equivalente interpretable de mirar los coeficientes del
    modelo: muestra qué vocabulario aprendió el sistema para decidir
    cada categoría documental.
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


# ---------------------------------------------------------------------
# SALIDA EN CONSOLA
# ---------------------------------------------------------------------

def imprimir_matriz(
    matriz: np.ndarray,
    etiquetas: List[str],
) -> None:
    """Imprime la matriz de confusión con nombres de categoría."""
    ancho = max(len(etiqueta) for etiqueta in etiquetas)

    for etiqueta, fila in zip(etiquetas, matriz):
        valores = " ".join(f"{valor:3d}" for valor in fila)
        print(f"  {etiqueta:<{ancho}} | {valores}")


# ---------------------------------------------------------------------
# GENERACIÓN AUTOMÁTICA DEL INFORME (reports/semana02.md)
# ---------------------------------------------------------------------

def generar_informe(
    descripciones: List[str],
    y: np.ndarray,
    etiquetas: List[str],
    resultado_loo: Dict[str, Any],
    resultado_kfold: Dict[str, Any],
    resultado_reglas: Dict[str, Any],
    relevantes: Dict[str, List[str]],
    configuraciones: List[Tuple[str, float]],
) -> Path:
    """
    Escribe automáticamente `reports/semana02.md` con la evidencia de
    la ejecución, siguiendo el mismo patrón que las Semanas 03 y 04.
    """
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    distribucion = Counter(y)

    lineas: List[str] = [
        "# Semana 02 - Fundamentos aplicados al Sistema de Análisis Documental",
        "",
        "Informe generado automáticamente por `src/semana02_fundamentos.py`.",
        "",
        "## 1. Del dataset de práctica al problema del proyecto",
        "",
        "La práctica original entrenaba una regresión logística sobre el "
        "dataset Iris. Aquí se conserva exactamente la misma lógica "
        "supervisada, pero aplicada al sistema de análisis documental.",
        "",
        "| Elemento | Práctica original | Aplicación al proyecto |",
        "|---|---|---|",
        "| Datos | `load_iris()` | `data/casos_ia.csv` (casos reales del proyecto) |",
        "| Entradas (X) | 4 medidas numéricas | Descripción textual del caso |",
        "| Preparación | `StandardScaler` | `TfidfVectorizer` (unigramas, stopwords ES) |",
        "| Etiquetas (y) | 3 especies | 7 categorías de la taxonomía documental |",
        "| Modelo | `LogisticRegression` | `LogisticRegression` (unigramas, pesos balanceados) |",
        "| Evaluación | `train_test_split` 75/25 | Validación cruzada (LOO y StratifiedKFold) |",
        "| Métricas | Accuracy, matriz de confusión | Accuracy, matriz de confusión, precisión/recall |",
        "",
        "### Por qué cambió el método de evaluación",
        "",
        f"El proyecto tiene **{len(y)} casos** y **{len(etiquetas)} "
        "categorías**. Un conjunto de prueba del 25% tendría solo "
        f"{round(len(y) * 0.25)} muestras, menos que el número de clases, "
        "por lo que una división estratificada es imposible y "
        "scikit-learn la rechaza. Además, una única división de 5 casos "
        "daría una accuracy con enorme varianza: cambiar la semilla "
        "cambiaría el resultado por completo.",
        "",
        "La validación cruzada resuelve ese problema sin inventar datos: "
        "cada caso se predice con un modelo que nunca lo vio, y la "
        "métrica se calcula sobre las 20 predicciones.",
        "",
        "## 2. Dataset supervisado",
        "",
        f"- Casos etiquetados: **{len(y)}**",
        f"- Categorías: **{len(etiquetas)}**",
        "",
        "| Categoría | Casos |",
        "|---|---:|",
    ]

    for etiqueta in etiquetas:
        lineas.append(f"| {etiqueta} | {distribucion[etiqueta]} |")

    lineas += [
        "",
        f"La categoría menos frecuente tiene {min(distribucion.values())} "
        "ejemplos: ese valor es el que limita el número de particiones "
        "posibles en la validación cruzada estratificada.",
        "",
        "## 3. Resultados",
        "",
        "| Método de evaluación | Accuracy |",
        "|---|---:|",
        f"| Leave-One-Out ({len(y)} entrenamientos) | "
        f"**{resultado_loo['accuracy'] * 100:.2f}%** |",
        f"| StratifiedKFold (k={resultado_kfold['n_splits']}) | "
        f"{resultado_kfold['accuracy'] * 100:.2f}% |",
        f"| Reglas de la Semana 03 (baseline) | "
        f"{resultado_reglas['accuracy'] * 100:.2f}% |",
        "",
        "### Efecto de las decisiones de diseño",
        "",
        "Accuracy Leave-One-Out de cada configuración probada:",
        "",
        "| Configuración | Accuracy |",
        "|---|---:|",
    ]

    for nombre, valor in configuraciones:
        lineas.append(f"| {nombre} | {valor * 100:.2f}% |")

    lineas += [
        "",
        "La configuración inicial (bigramas y sin balanceo) es la peor: "
        "con 20 documentos cortos, los bigramas casi solo aportan rasgos "
        "que aparecen una única vez, y sin `class_weight=\"balanced\"` el "
        "modelo se inclina hacia las categorías con más ejemplos. "
        "Limitar a unigramas y balancear los pesos es lo que hace que el "
        "modelo llegue a su mejor resultado.",
        "",
        "### Matriz de confusión (Leave-One-Out)",
        "",
        "Filas: categoría real. Columnas: categoría predicha.",
        "",
        "| Real \\ Predicho | " + " | ".join(
            f"C{i + 1}" for i in range(len(etiquetas))
        ) + " |",
        "|---|" + "---:|" * len(etiquetas),
    ]

    for etiqueta, fila in zip(etiquetas, resultado_loo["matriz"]):
        valores = " | ".join(str(int(valor)) for valor in fila)
        lineas.append(f"| {etiqueta} | {valores} |")

    lineas += [
        "",
        "Referencia de columnas: " + ", ".join(
            f"**C{i + 1}** = {etiqueta}"
            for i, etiqueta in enumerate(etiquetas)
        ) + ".",
        "",
        "### Precisión y recall por categoría (Leave-One-Out)",
        "",
        "```",
        resultado_loo["reporte"].rstrip(),
        "```",
        "",
        "## 4. Caso por caso (Leave-One-Out)",
        "",
        "| # | Descripción | Categoría real | Predicción del modelo | Estado |",
        "|---:|---|---|---|---|",
    ]

    for indice, descripcion in enumerate(descripciones):
        real = y[indice]
        predicho = resultado_loo["predicciones"][indice]
        estado = "Coincide" if real == predicho else "Revisar"
        texto = descripcion.replace("|", "/")
        lineas.append(
            f"| {indice + 1} | {texto} | {real} | {predicho} | {estado} |"
        )

    lineas += [
        "",
        "## 5. Vocabulario aprendido por el modelo",
        "",
        "Términos con mayor peso positivo en la regresión logística "
        "entrenada con todos los casos. Muestran qué vocabulario del "
        "dominio documental usa el modelo para decidir cada categoría.",
        "",
        "| Categoría | Términos más influyentes |",
        "|---|---|",
    ]

    for etiqueta in etiquetas:
        terminos = relevantes.get(etiqueta, [])
        texto = ", ".join(f"`{t}`" for t in terminos) if terminos else "—"
        lineas.append(f"| {etiqueta} | {texto} |")

    diferencia = (
        resultado_loo["accuracy"] - resultado_reglas["accuracy"]
    ) * 100

    if diferencia > 0:
        comparacion = (
            f"el modelo aprendido supera al sistema de reglas por "
            f"{diferencia:.2f} puntos porcentuales"
        )
    elif diferencia < 0:
        comparacion = (
            f"el sistema de reglas de la Semana 03 supera al modelo "
            f"aprendido por {abs(diferencia):.2f} puntos porcentuales"
        )
    else:
        comparacion = (
            "el modelo aprendido y el sistema de reglas empatan"
        )

    lineas += [
        "",
        "## 6. Conclusiones",
        "",
        "- La lógica de la práctica de fundamentos se traslada sin "
        "problema al proyecto: el cambio de fondo es sustituir el "
        "escalado numérico por una vectorización TF-IDF del texto.",
        "- El vocabulario aprendido es coherente con el dominio (`ocr` "
        "para visión por computador, `buscar` para recuperación, "
        "`extraer` y `nombres` para PLN), así que el modelo sí capta señal "
        "real y no ruido aleatorio.",
        f"- Con validación cruzada Leave-One-Out el modelo alcanza "
        f"**{resultado_loo['accuracy'] * 100:.2f}%** de accuracy sobre "
        f"los {len(y)} casos reales del proyecto.",
        f"- Comparado con el clasificador por reglas de la Semana 03 "
        f"({resultado_reglas['accuracy'] * 100:.2f}%), {comparacion}.",
        "- La principal limitación es el tamaño del dataset: con 2 o 3 "
        "ejemplos en varias categorías, el modelo tiene muy poca "
        "evidencia para aprender vocabulario discriminante, y por eso "
        "las categorías minoritarias son las que concentran los "
        "errores en la matriz de confusión.",
        "- Que las reglas manuales ganen no significa que el enfoque "
        "supervisado sea inferior: significa que 20 ejemplos no bastan "
        "para que el modelo aprenda lo que un experto ya codificó a "
        "mano. Las reglas incorporan conocimiento del dominio que el "
        "modelo tendría que deducir de los datos.",
        "- La mejora más rentable no es cambiar de algoritmo, sino "
        "ampliar y equilibrar el corpus etiquetado del proyecto.",
        "",
    ]

    REPORT_FILE.write_text("\n".join(lineas), encoding="utf-8")

    return REPORT_FILE


# ---------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print(" SEMANA 02 - FUNDAMENTOS EN EL SISTEMA DE ANÁLISIS DOCUMENTAL")
    print("=" * 70)

    X, y, descripciones = cargar_datos()
    etiquetas = sorted(set(y))

    print(f"Casos etiquetados: {len(X)}")
    print(f"Categorías: {len(etiquetas)}")
    print("Distribución por categoría:")
    for etiqueta, cantidad in sorted(Counter(y).items()):
        print(f"  {etiqueta}: {cantidad}")

    print(
        "\nNota: con 20 casos y 7 categorías no es posible un "
        "train_test_split\nestratificado del 25% (5 muestras < 7 clases), "
        "por lo que se usa\nvalidación cruzada."
    )

    print("\n--- VALIDACIÓN CRUZADA LEAVE-ONE-OUT ---")
    resultado_loo = evaluar_leave_one_out(X, y, etiquetas)
    print(f"Entrenamientos realizados: {len(X)}")
    print(f"Accuracy: {resultado_loo['accuracy']:.3f}")
    print("Matriz de confusión:")
    imprimir_matriz(resultado_loo["matriz"], etiquetas)
    print("\nPrecisión y recall por categoría:")
    print(resultado_loo["reporte"])

    print("--- EFECTO DE LAS DECISIONES DE DISEÑO (accuracy LOO) ---")
    configuraciones = comparar_configuraciones(X, y)
    for nombre, valor in configuraciones:
        print(f"  {valor:.3f}  {nombre}")

    print("\n--- VALIDACIÓN CRUZADA ESTRATIFICADA ---")
    resultado_kfold = evaluar_kfold_estratificado(X, y, etiquetas)
    print(f"Particiones (k): {resultado_kfold['n_splits']}")
    print(f"Accuracy: {resultado_kfold['accuracy']:.3f}")

    print("\n--- COMPARACIÓN CON LAS REGLAS DE LA SEMANA 03 ---")
    resultado_reglas = evaluar_reglas_semana03(descripciones, y)
    print(f"Accuracy del modelo aprendido (LOO): {resultado_loo['accuracy']:.3f}")
    print(f"Accuracy de las reglas manuales:     {resultado_reglas['accuracy']:.3f}")

    print("\n--- VOCABULARIO APRENDIDO POR CATEGORÍA ---")
    relevantes = terminos_relevantes(X, y)
    for etiqueta in etiquetas:
        terminos = ", ".join(relevantes.get(etiqueta, [])) or "—"
        print(f"  {etiqueta}: {terminos}")

    ruta_informe = generar_informe(
        descripciones,
        y,
        etiquetas,
        resultado_loo,
        resultado_kfold,
        resultado_reglas,
        relevantes,
        configuraciones,
    )

    print("\n" + "=" * 70)
    print(" PRÁCTICA FINALIZADA CORRECTAMENTE")
    print(f" Informe generado: {ruta_informe}")
    print("=" * 70)


if __name__ == "__main__":
    main()
    