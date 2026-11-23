# Proyecto Final – Hiper-heurística de Selección para TSP

Repositorio completo con la implementación de la **Hiper-heurística de Selección** para el Problema del Agente Viajero (TSP), evaluada sobre las 500 instancias (`train.csv`: 351, `test.csv`: 149) generadas con `SEED = 42`.

## Estructura del Proyecto

```text
proyecto_hiperheuristica_tsp/
├── data/
│   ├── instances/                      # 500 instancias inst_0001.csv .. inst_0500.csv
│   ├── train.csv                       # Conjunto de entrenamiento (351 filas)
│   └── test.csv                        # Conjunto de prueba (149 filas)
├── results/
│   ├── train_labeled.csv               # Entrenamiento con costos y etiqueta best_algorithm
│   ├── test_labeled.csv                # Prueba con costos y etiqueta best_algorithm
│   ├── metrics_summary.json            # Métricas completas (Accuracy, Baseline, Regret, etc.)
│   ├── confusion_matrix_and_features.png # Matriz de confusión e importancia de características
│   └── patterns_distribution_size.png  # Patrones de algoritmo ganador por distribución y tamaño n
├── src/
│   ├── tsp_algorithms.py               # 2-opt Hill Climbing, SA + 2-opt, y MST + 2-opt
│   ├── generate_labels.py              # Parte 1: Generación de costos y etiquetas
│   └── train_evaluate_ml.py            # Parte 2: Modelos ML, validación, Regret y análisis
├── requirements.txt
└── README.md
```

## Ejecución

```bash
pip install -r requirements.txt
python src/generate_labels.py
python src/train_evaluate_ml.py
```

## Parámetros Documentados de Recocido Simulado (SA + 2-opt)
- **Construcción inicial:** Vecino más cercano (`nearest_neighbor_tour`).
- **Temperatura inicial ($T_0$):** `15.0`
- **Factor de enfriamiento ($lpha$):** `0.995` (esquema geométrico $T_{k+1} = lpha \cdot T_k$).
- **Presupuesto de iteraciones:** $\max(500, 150 \cdot n)$ por instancia.
- **Refinamiento final:** Búsqueda local `2-opt` hasta óptimo local sobre la mejor solución hallada.
