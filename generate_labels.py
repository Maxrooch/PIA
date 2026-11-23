"""
Parte 1: Generación de etiquetas para la Hiper-heurística de Selección.
Lee data/train.csv, data/test.csv y data/instances/inst_NNNN.csv,
ejecuta los tres algoritmos de Tarea 2 y guarda results/train_labeled.csv y results/test_labeled.csv.
"""
import os
import csv
import pandas as pd
from tsp_algorithms import (
    solve_2opt_hill_climbing,
    solve_sa_2opt,
    solve_mst_2opt,
)


def load_points(filepath: str) -> list[tuple[float, float]]:
    points = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append((float(row["x"]), float(row["y"])))
    return points


def label_dataset(df: pd.DataFrame, instances_dir: str, split_name: str, seed: int = 42) -> pd.DataFrame:
    c_2opt_list, c_sa_list, c_mst_list, best_list = [], [], [], []
    total = len(df)
    print(f"Etiquetando '{split_name}' ({total} instancias)...")

    for idx, row in df.iterrows():
        inst_id = str(row["instance_id"])
        inst_path = os.path.join(instances_dir, f"{inst_id}.csv")
        points = load_points(inst_path)

        _, c_2opt = solve_2opt_hill_climbing(points)
        _, c_sa = solve_sa_2opt(points, seed=seed + idx, t0=15.0, alpha=0.995)
        _, c_mst = solve_mst_2opt(points)

        costs = {
            "2opt_hill_climbing": round(c_2opt, 8),
            "sa_2opt": round(c_sa, 8),
            "mst_2opt": round(c_mst, 8),
        }
        best_algo = min(costs, key=costs.get)

        c_2opt_list.append(costs["2opt_hill_climbing"])
        c_sa_list.append(costs["sa_2opt"])
        c_mst_list.append(costs["mst_2opt"])
        best_list.append(best_algo)

    out = df.copy()
    out["cost_2opt_hill_climbing"] = c_2opt_list
    out["cost_sa_2opt"] = c_sa_list
    out["cost_mst_2opt"] = c_mst_list
    out["best_algorithm"] = best_list
    return out


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    instances_dir = os.path.join(data_dir, "instances")

    train_labeled = label_dataset(train_df, instances_dir, "train", seed=42)
    test_labeled = label_dataset(test_df, instances_dir, "test", seed=42)

    train_labeled.to_csv(os.path.join(results_dir, "train_labeled.csv"), index=False)
    test_labeled.to_csv(os.path.join(results_dir, "test_labeled.csv"), index=False)

    full = pd.concat([train_labeled, test_labeled], ignore_index=True)
    print("\nBalance de clases obtenido (500 instancias):")
    for algo, cnt in full["best_algorithm"].value_counts().items():
        print(f"  - {algo}: {cnt} ({cnt / len(full) * 100:.2f}%)")


if __name__ == "__main__":
    main()
