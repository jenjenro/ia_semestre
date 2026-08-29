# Semana 04 — Búsqueda en el sistema de análisis documental

## Objetivo

Aplicar A* sobre un grafo de similitud TF-IDF construido con los casos reales de `data/casos_ia.csv`, y documentar Minimax con poda alfa-beta como práctica de referencia.

## Datos y modelo

- Casos analizados: **20**.
- Vecinos por caso: **3**.
- Costo de transición: `1 + distancia TF-IDF`.
- Heurística de A*: distancia coseno entre el caso actual y la meta.

## Pruebas de A*

| Caso | Ruta | Costo | Expansiones A* | Expansiones h=0 |
|---|---|---:|---:|---:|
| OCR (ingesta) -> automatización (reporte final) | caso_01 -> caso_03 -> caso_18 -> caso_20 | 5.8429 | 9 | 13 |
| Búsqueda documental -> aprendizaje predictivo | caso_05 -> caso_14 -> caso_19 | 3.8273 | 5 | 7 |
| Clasificación documental -> sistemas expertos | caso_02 -> caso_13 -> caso_07 | 3.8353 | 5 | 7 |

## Minimax y alfa-beta

- Valor con Minimax puro: **3**.
- Nodos visitados sin poda: **10**.
- Valor con poda alfa-beta: **3**.
- Nodos visitados con poda: **9**.
- Reducción de nodos: **10.0%**.

La poda alfa-beta conserva la misma utilidad final y evita explorar ramas que no pueden mejorar la decisión de MAX.

## Ejecución

```bash
python src/semana04_busqueda.py
```

Este archivo se regenera automáticamente en `reports/semana04.md` cada vez que se ejecuta el script.
