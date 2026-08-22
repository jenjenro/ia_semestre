from dataclasses import dataclass
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
        "imagen",
        "imagenes",
        "foto",
        "fotografia",
        "fotografias",
        "camara",
        "escaneo",
        "escaneado",
        "escanear",
        "documento escaneado",
        "ocr",
        "reconocimiento optico",
        "digitalizado",
        "captura"
    )),

    Category("Procesamiento de lenguaje natural", (
        "texto",
        "lenguaje",
        "frase",
        "frases",
        "parrafo",
        "parrafos",
        "resumen",
        "resumir",
        "analizar texto",
        "extraer nombres",
        "extraer fechas",
        "extraer entidades",
    )),

    Category("Aprendizaje automático predictivo", (
        "predecir",
        "prediccion",
        "probabilidad",
        "clasificar automaticamente",
        "patron",
        "patrones",
        "anomalía",
        "anomalias",
        "riesgo",
        "detectar automaticamente"
    )),

    Category("Clasificación documental", (
        "clasificar documento",
        "clasificar documentos",
        "clasificacion documental",
        "tipo de documento",
        "categorizar documento",
        "categoria documental",
        "identificar tipo de documento",
        "detectar tipo de documento",
        "reconocer documento",
        "factura",
        "contrato",
        "certificado",
        "acta",
        "formulario",
        "hoja de vida",
    )),

    Category("Búsqueda y recuperación documental", (
        "buscar",
        "busqueda",
        "encontrar",
        "localizar",
        "consultar",
        "consulta",
        "palabra clave",
        "palabras clave",
        "documentos relacionados",
        "recuperar informacion",
        "encontrar documentos",
    )),

    Category("Sistemas expertos y reglas documentales", (
        "regla",
        "reglas",
        "validar",
        "validacion",
        "verificar",
        "verificacion",
        "requisito",
        "requisitos",
        "obligatorio",
        "obligatorios",
        "informacion faltante",
        "cumple",
        "incumple",
    )),

    Category("Automatización documental inteligente", (
        "automatizar",
        "automatizacion",
        "procesar documento",
        "procesamiento documental",
        "procesar automaticamente",
        "extraer informacion",
        "extraer datos",
        "extraer automaticamente",
        "generar reporte",
        "generar reporte automaticamente",
        "flujo documental",
    )),
]


CUSTOM_RULES = {
    "Visión por computador y OCR": (
        "documento escaneado",
        "imagen del documento",
        "fotografia del documento",
        "convertir imagen en texto",
    ),

    "Procesamiento de lenguaje natural": (
        "extraer nombres",
        "extraer fechas",
        "extraer entidades",
        "analizar contenido textual",
    ),

    "Clasificación documental": (
        "identificar tipo de documento",
        "detectar tipo de documento",
        "reconocer documento",
        "clasificar documentos",
    ),

    "Sistemas expertos y reglas documentales": (
        "verificar requisitos",
        "validar documento",
        "documento completo",
        "informacion faltante",
    ),

    "Automatización documental inteligente": (
        "procesar automaticamente",
        "extraer automaticamente",
        "analizar automaticamente",
        "generar reporte",
    ),
}


MANUAL_REFERENCE = [
    "Visión por computador y OCR",
    "Clasificación documental",
    "Procesamiento de lenguaje natural",
    "Procesamiento de lenguaje natural",
    "Búsqueda y recuperación documental",
    "Búsqueda y recuperación documental",
    "Sistemas expertos y reglas documentales",
    "Visión por computador y OCR",
    "Clasificación documental",
    "Procesamiento de lenguaje natural",
    "Procesamiento de lenguaje natural",
    "Sistemas expertos y reglas documentales",
    "Búsqueda y recuperación documental",
    "Aprendizaje automático predictivo",
    "Búsqueda y recuperación documental",
    "Clasificación documental",
    "Sistemas expertos y reglas documentales",
    "Automatización documental inteligente",
    "Aprendizaje automático predictivo",
    "Automatización documental inteligente",
]


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) != "Mn"
    )
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_header(text: str) -> str:
    return normalize(text).replace(" ", "")


def contains_keyword(text: str, keyword: str) -> bool:
    # Compara palabras/frases completas para evitar falsos positivos
    # como "plan" dentro de "plantas".
    normalized_text = f" {normalize(text)} "
    normalized_keyword = normalize(keyword)
    return f" {normalized_keyword} " in normalized_text


def build_categories() -> list[Category]:
    result = []

    for category in CATEGORIES:
        extra = CUSTOM_RULES.get(category.name, ())
        result.append(
            Category(
                category.name,
                category.keywords + tuple(extra)
            )
        )

    return result


def classify_problem(
    text: str
) -> tuple[str, list[str], dict[str, int]]:

    scores = {}

    for category in build_categories():
        score = sum(
            contains_keyword(text, keyword)
            for keyword in category.keywords
        )
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

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"No existe {CSV_FILE}. "
            "Crea data/casos_ia.csv antes de ejecutar la práctica."
        )

    # utf-8-sig elimina un BOM UTF-8 si el CSV fue guardado
    # por Excel u otro editor.
    with CSV_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(
                "El CSV está vacío o no contiene encabezados."
            )

        original_headers = list(reader.fieldnames)
        reader.fieldnames = [
            normalize_header(name)
            for name in reader.fieldnames
        ]

        if "descripcion" not in reader.fieldnames:
            raise ValueError(
                "No se encontró la columna 'descripcion'. "
                f"Encabezados encontrados: {original_headers}"
            )

        cases = []

        for row in reader:
            description = (row.get("descripcion") or "").strip()

            if description:
                cases.append(description)

        if len(cases) < 20:
            raise ValueError(
                f"La práctica requiere al menos 20 casos "
                f"y el archivo contiene {len(cases)}."
            )

    return cases


def write_report(results: list[dict]) -> None:

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    reference_count = min(
        len(results),
        len(MANUAL_REFERENCE)
    )

    matches = sum(
        results[i]["primary"] == MANUAL_REFERENCE[i]
        for i in range(reference_count)
    )

    accuracy = (
        100 * matches / reference_count
        if reference_count
        else 0.0
    )

    lines = [
        "# Semana 03 - Taxonomía de Inteligencia Artificial",
        "",
        "## Resultado automático frente a clasificación manual de referencia",
        "",
        "| Caso | Categoría automática principal | Categorías detectadas | Manual | Estado |",
        "|---:|---|---|---|---|",
    ]

    for i, result in enumerate(results, start=1):

        manual = (
            MANUAL_REFERENCE[i - 1]
            if i <= len(MANUAL_REFERENCE)
            else "Pendiente"
        )

        status = (
            "Coincide"
            if result["primary"] == manual
            else "Revisar"
        )

        detected = ", ".join(result["detected"])

        lines.append(
            f"| {i} | {result['primary']} | "
            f"{detected} | {manual} | {status} |"
        )

    lines += [
        "",
        f"Coincidencia con la referencia: "
        f"**{accuracy:.2f}%** ({matches}/{reference_count}).",
        "",

        "## Cinco reglas propias",
        "",
        "Se implementaron cinco reglas propias en `CUSTOM_RULES`, "
        "orientadas a mejorar la clasificación de problemas "
        "relacionados con el análisis documental.",
        "",

        "### 1. Visión por computador y OCR",
        "",
        "Se agregaron expresiones específicas para identificar "
        "documentos escaneados, imágenes y fotografías de documentos, "
        "así como procesos de conversión de imágenes a texto.",
        "",
        "- `documento escaneado`",
        "- `imagen del documento`",
        "- `fotografia del documento`",
        "- `convertir imagen en texto`",
        "",
        "Estas reglas son pertinentes porque permiten diferenciar "
        "problemas donde el documento se encuentra inicialmente "
        "como una imagen y requiere OCR para obtener su contenido textual.",
        "",

        "### 2. Procesamiento de lenguaje natural",
        "",
        "Se agregaron reglas relacionadas con la extracción y análisis "
        "de información textual.",
        "",
        "- `extraer nombres`",
        "- `extraer fechas`",
        "- `extraer entidades`",
        "- `analizar contenido textual`",
        "",
        "Estas expresiones permiten identificar tareas de extracción "
        "de información y análisis de texto, propias del procesamiento "
        "de lenguaje natural.",
        "",

        "### 3. Clasificación documental",
        "",
        "Se incorporaron reglas orientadas a identificar cuándo el "
        "objetivo principal es determinar el tipo o categoría de un documento.",
        "",
        "- `identificar tipo de documento`",
        "- `detectar tipo de documento`",
        "- `reconocer documento`",
        "- `clasificar documentos`",
        "",
        "Estas reglas son pertinentes porque permiten diferenciar la "
        "clasificación documental de otras tareas de procesamiento de texto.",
        "",

        "### 4. Sistemas expertos y reglas documentales",
        "",
        "Se agregaron reglas relacionadas con la verificación de "
        "requisitos, validación de documentos y detección de información faltante.",
        "",
        "- `verificar requisitos`",
        "- `validar documento`",
        "- `documento completo`",
        "- `informacion faltante`",
        "",
        "Estas expresiones permiten identificar problemas donde la "
        "decisión depende del cumplimiento de reglas o requisitos establecidos.",
        "",

        "### 5. Automatización documental inteligente",
        "",
        "Se agregaron reglas relacionadas con procesos automáticos "
        "de procesamiento, extracción y generación de información.",
        "",
        "- `procesar automaticamente`",
        "- `extraer automaticamente`",
        "- `analizar automaticamente`",
        "- `generar reporte`",
        "",
        "Estas reglas son pertinentes porque permiten identificar "
        "procesos donde varias tareas documentales son ejecutadas "
        "automáticamente mediante sistemas de inteligencia artificial.",
        "",

        "## Ampliación de categorías",
        "",
        "Además de las cinco reglas propias, se ampliaron las palabras "
        "clave de las categorías existentes para mejorar la cobertura "
        "del dominio de análisis documental.",
        "",
        "En **Visión por computador y OCR** se incorporaron términos "
        "como imagen, fotografía, cámara, escaneo, documento escaneado, "
        "OCR, reconocimiento óptico, digitalizado y captura.",
        "",
        "En **Procesamiento de lenguaje natural** se incorporaron "
        "términos como texto, lenguaje, frase, párrafo, resumen, "
        "analizar texto, extraer nombres, extraer fechas y extraer entidades.",
        "",
        "En **Aprendizaje automático predictivo** se incorporaron "
        "términos relacionados con predicción, probabilidad, patrones, "
        "anomalías, riesgo y detección automática.",
        "",
        "En **Clasificación documental** se incorporaron expresiones "
        "relacionadas con tipos de documentos y ejemplos concretos "
        "como factura, contrato, certificado, acta, formulario y hoja de vida.",
        "",
        "En **Búsqueda y recuperación documental** se incorporaron "
        "términos como buscar, búsqueda, encontrar, localizar, consultar, "
        "palabra clave, documentos relacionados y recuperar información.",
        "",
        "En **Sistemas expertos y reglas documentales** se incorporaron "
        "términos como regla, validar, verificar, requisito, obligatorio, "
        "información faltante, cumple e incumple.",
        "",
        "En **Automatización documental inteligente** se incorporaron "
        "términos como automatizar, procesar documento, procesamiento "
        "documental, procesar automáticamente, extraer información, "
        "generar reporte y flujo documental.",
        "",

        "## Discrepancias y análisis",
        "",

        "### Caso 9 - Clasificación automática de documentos",
        "",
        "**Descripción:** Clasificar automáticamente documentos según "
        "su contenido textual.",
        "",
        "- **Resultado automático:** Aprendizaje automático predictivo.",
        "- **Referencia manual:** Clasificación documental.",
        "- **Activador:** `clasificar automaticamente`.",
        "- **Análisis:** la expresión utilizada en el caso coincide "
        "directamente con la categoría Aprendizaje automático predictivo. "
        "Sin embargo, el objetivo del problema es determinar la categoría "
        "o tipo de documentos, por lo que la referencia manual lo ubica "
        "en Clasificación documental.",
        "- **Modificación propuesta:** retirar o restringir la expresión "
        "`clasificar automaticamente` de Aprendizaje automático predictivo "
        "o darle mayor prioridad a Clasificación documental cuando la "
        "clasificación tenga como objeto principal un documento.",
        "",

        "### Caso 13 - Comparación de documentos",
        "",
        "**Descripción:** Comparar dos versiones de un documento para "
        "identificar diferencias.",
        "",
        "- **Resultado automático:** Requiere análisis.",
        "- **Referencia manual:** Búsqueda y recuperación documental.",
        "- **Activador:** no existe una coincidencia directa suficiente "
        "con las palabras clave actuales.",
        "- **Análisis:** el problema se enfoca en comparar dos versiones "
        "de un documento e identificar diferencias. Esta operación no "
        "está representada explícitamente dentro de las palabras clave "
        "actuales de Búsqueda y recuperación documental.",
        "- **Modificación propuesta:** agregar términos como "
        "`comparar documentos`, `comparar versiones`, `diferencias` "
        "o `comparación documental` a una categoría adecuada. Si se "
        "mantiene la referencia manual, podrían incorporarse estas "
        "expresiones a Búsqueda y recuperación documental.",
        "",

        "### Caso 14 - Detección de documentos duplicados",
        "",
        "**Descripción:** Detectar documentos duplicados dentro de un "
        "repositorio documental.",
        "",
        "- **Resultado automático:** Requiere análisis.",
        "- **Referencia manual:** Aprendizaje automático predictivo.",
        "- **Activador:** no existe una palabra clave específica para "
        "`documentos duplicados` dentro de la categoría de aprendizaje "
        "automático predictivo.",
        "- **Análisis:** la detección de duplicados puede utilizar "
        "patrones, similitud de contenido o técnicas de aprendizaje "
        "automático. Sin embargo, estas expresiones no fueron incluidas "
        "explícitamente en las reglas actuales.",
        "- **Modificación propuesta:** agregar términos como "
        "`duplicado`, `documentos duplicados`, `detectar duplicados` "
        "y `similitud documental` a la categoría de Aprendizaje "
        "automático predictivo, si se desea conservar la clasificación "
        "manual establecida.",
        "",

        "### Caso 20 - Generación automática de reportes",
        "",
        "**Descripción:** Generar un reporte con la información "
        "extraída de múltiples documentos.",
        "",
        "- **Resultado automático:** Requiere análisis.",
        "- **Referencia manual:** Automatización documental inteligente.",
        "- **Activador:** `generar reporte`.",
        "- **Análisis:** aunque `generar reporte` está incluido en "
        "`CUSTOM_RULES` de Automatización documental inteligente, "
        "el resultado muestra que la expresión no fue suficiente para "
        "establecer la categoría principal bajo la lógica actual de "
        "clasificación.",
        "- **Modificación propuesta:** reforzar la regla para que "
        "`generar reporte` tenga mayor peso cuando aparezca junto con "
        "expresiones como `información extraída`, `múltiples documentos` "
        "o `generar reporte automáticamente`.",
        "",

        "## Nota técnica",
        "",
        "Un problema real puede pertenecer a varias áreas de IA. "
        "La columna 'principal' utiliza la categoría con mayor "
        "cantidad de coincidencias, mientras que las demás "
        "coincidencias se conservan como categorías secundarias.",
        "",
        
        "## Conclusión",
        "",
        f"Después de ampliar las categorías e implementar cinco reglas "
        f"propias orientadas al dominio documental, el sistema obtuvo "
        f"una coincidencia del **{accuracy:.2f}%** con la clasificación "
        f"manual de referencia. Los resultados muestran que las "
        f"modificaciones realizadas permiten clasificar correctamente "
        f"la mayoría de los casos. Las discrepancias encontradas "
        f"permiten identificar oportunidades de mejora en la selección "
        f"de palabras clave, el peso de las reglas y la prioridad entre "
        f"categorías."
    ]

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


def main() -> None:

    cases = read_cases()
    results = []

    print("=" * 80)
    print("SEMANA 03 - TAXONOMÍA DE INTELIGENCIA ARTIFICIAL")
    print("=" * 80)

    for i, case in enumerate(cases, start=1):

        primary, detected, scores = classify_problem(case)

        results.append({
            "description": case,
            "primary": primary,
            "detected": detected,
            "scores": scores,
        })

        print(f"{i:02d}. {case}")
        print(f" Principal: {primary}")
        print(f" Áreas detectadas: {', '.join(detected)}")

    write_report(results)

    print(f"\nCasos procesados: {len(results)}")
    print(f"Reporte generado: {REPORT_FILE}")


if __name__ == "__main__":
    main()