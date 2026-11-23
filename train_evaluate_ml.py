"""
Parte 2: Entrenamiento, Evaluación y Análisis de la Hiper-heurística de Selección (ML).
Genera:
- Balance de clases y Baseline Trivial
- Comparación de modelos (Árbol de Decisión, KNN, Random Forest) con GridSearchCV
- Accuracy en prueba vs. Baseline
- Matriz de confusión y Regret (absoluto, relativo y tasa de regret=0)
- Importancia de características y análisis por distribución y tamaño n
- Gráficas listas para el reporte PDF en results/
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


COST_COL_MAP = {
    "2opt_hill_climbing": "cost_2opt_hill_climbing",
    "sa_2opt": "cost_sa_2opt",
    "mst_2opt": "cost_mst_2opt",
}


def compute_regret(df_test: pd.DataFrame, y_pred: np.ndarray) -> dict:
    regrets, rel_regrets = [], []
    for idx, (_, row) in enumerate(df_test.iterrows()):
        pred_cost = row[COST_COL_MAP[y_pred[idx]]]
        best_cost = min(
            row["cost_2opt_hill_climbing"],
            row["cost_sa_2opt"],
            row["cost_mst_2opt"],
        )
        reg = float(pred_cost - best_cost)
        rel = float((reg / best_cost) * 100.0) if best_cost > 0 else 0.0
        regrets.append(reg)
        rel_regrets.append(rel)
    return {
        "mean_regret": float(np.mean(regrets)),
        "median_regret": float(np.median(regrets)),
        "max_regret": float(np.max(regrets)),
        "mean_relative_regret_pct": float(np.mean(rel_regrets)),
        "zero_regret_rate_pct": float(np.mean(np.array(regrets) <= 1e-9) * 100.0),
    }


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["nn_to_pairwise_ratio"] = out["mean_nn_dist"] / (out["mean_pairwise_dist"] + 1e-9)
    out["centroid_to_pairwise_ratio"] = out["mean_centroid_dist"] / (out["mean_pairwise_dist"] + 1e-9)
    out["density_proxy"] = out["n"] / (out["std_coords"] ** 2 + 1e-9)
    return out


def save_plots(cm: np.ndarray, labels: list[str], feat_names: list[str], importances: np.ndarray, full_df: pd.DataFrame, results_dir: str):
    # 1. Matriz de confusión + Feature Importance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title("Matriz de Confusión (Conjunto de Prueba)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Algoritmo Predicho")
    axes[0].set_ylabel("Algoritmo Ganador Real")

    if importances is not None:
        idx = np.argsort(importances)
        axes[1].barh(np.array(feat_names)[idx], importances[idx], color="#2b5c8f")
        axes[1].set_title("Importancia de Características", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Importancia Relativa")
    else:
        axes[1].text(0.5, 0.5, "Importancia nativa no disponible", ha="center", va="center")
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix_and_features.png"), dpi=300)
    plt.close()

    # 2. Patrones por distribución y tamaño n
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    dist_ct = pd.crosstab(full_df["distribution"], full_df["best_algorithm"], normalize="index") * 100
    dist_ct.plot(kind="bar", stacked=True, ax=axes[0], colormap="Set2")
    axes[0].set_title("Algoritmo Ganador por Distribución (%)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Porcentaje de Instancias (%)")
    axes[0].set_xlabel("Distribución")
    axes[0].tick_params(axis="x", rotation=0)

    size_ct = pd.crosstab(full_df["n"], full_df["best_algorithm"], normalize="index") * 100
    size_ct.plot(kind="line", marker="o", ax=axes[1], linewidth=2)
    axes[1].set_title("Algoritmo Ganador según Tamaño n (%)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Porcentaje de Instancias (%)")
    axes[1].set_xlabel("Tamaño de Instancia (n)")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "patterns_distribution_size.png"), dpi=300)
    plt.close()


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")

    train_df = pd.read_csv(os.path.join(results_dir, "train_labeled.csv"))
    test_df = pd.read_csv(os.path.join(results_dir, "test_labeled.csv"))
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    print("=================================================================")
    print("1. BALANCE DE CLASES (CORRIDAS PROPIAS)")
    print("=================================================================")
    global_counts = full_df["best_algorithm"].value_counts()
    global_pcts = full_df["best_algorithm"].value_counts(normalize=True) * 100
    for algo in global_counts.index:
        print(f"  {algo:<22}: {global_counts[algo]:>3} ({global_pcts[algo]:>5.2f}%)")

    majority_class = train_df["best_algorithm"].value_counts().idxmax()
    y_test = test_df["best_algorithm"].values
    y_pred_base = np.full(len(test_df), majority_class)
    base_acc = accuracy_score(y_test, y_pred_base)
    base_reg = compute_regret(test_df, y_pred_base)

    print(f"\nBaseline Trivial (siempre predecir '{majority_class}'):")
    print(f"  - Accuracy en Test : {base_acc * 100:.2f}%")
    print(f"  - Regret Promedio  : {base_reg['mean_regret']:.6f} ({base_reg['mean_relative_regret_pct']:.3f}%)")

    train_feat = add_engineered_features(train_df)
    test_feat = add_engineered_features(test_df)

    num_cols = [
        "n", "mean_pairwise_dist", "std_coords", "mean_centroid_dist", "mean_nn_dist",
        "nn_to_pairwise_ratio", "centroid_to_pairwise_ratio", "density_proxy",
    ]
    cat_cols = ["distribution"]

    X_train = train_feat[num_cols + cat_cols]
    y_train = train_feat["best_algorithm"].values
    X_test = test_feat[num_cols + cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    candidates = {
        "DecisionTree": (
            DecisionTreeClassifier(random_state=42),
            {"clf__max_depth": [3, 4, 5, 6, None], "clf__min_samples_leaf": [2, 5, 10], "clf__class_weight": [None, "balanced"]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [3, 5, 7, 11, 15], "clf__weights": ["uniform", "distance"], "clf__metric": ["euclidean", "manhattan"]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [4, 6, 8, None], "clf__min_samples_leaf": [2, 5], "clf__class_weight": [None, "balanced"]},
        ),
    }

    print("\n=================================================================")
    print("2. ENTRENAMIENTO Y VALIDACIÓN CRUZADA DE MODELOS")
    print("=================================================================")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_model, best_name, best_score, best_params = None, "", -1.0, {}
    all_models_summary = {}

    for name, (est, grid_params) in candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", est)])
        grid = GridSearchCV(pipe, grid_params, cv=cv, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train, y_train)
        preds = grid.predict(X_test)
        acc = accuracy_score(y_test, preds)
        reg = compute_regret(test_df, preds)

        all_models_summary[name] = {
            "best_params": grid.best_params_,
            "cv_accuracy": float(grid.best_score_),
            "test_accuracy": float(acc),
            "test_regret": reg,
        }

        print(f"Modelo: {name}")
        print(f"  - Mejores Hiperparámetros : {grid.best_params_}")
        print(f"  - CV Accuracy (Train)     : {grid.best_score_ * 100:.2f}%")
        print(f"  - Test Accuracy           : {acc * 100:.2f}%")
        print(f"  - Test Regret Promedio    : {reg['mean_regret']:.6f} ({reg['mean_relative_regret_pct']:.3f}%)")

        if acc > best_score:
            best_score = acc
            best_model = grid.best_estimator_
            best_name = name
            best_params = grid.best_params_

    print("\n=================================================================")
    print(f"3. EVALUACIÓN FINAL DEL MEJOR MODELO: {best_name}")
    print("=================================================================")
    y_pred_final = best_model.predict(X_test)
    final_acc = accuracy_score(y_test, y_pred_final)
    final_reg = compute_regret(test_df, y_pred_final)
    labels = sorted(full_df["best_algorithm"].unique())
    cm = confusion_matrix(y_test, y_pred_final, labels=labels)

    print(f"Accuracy ({best_name}) : {final_acc * 100:.2f}% vs. Baseline: {base_acc * 100:.2f}%")
    print(f"Regret Promedio        : {final_reg['mean_regret']:.6f} (vs. Baseline: {base_reg['mean_regret']:.6f})")
    print(f"Regret Relativo        : {final_reg['mean_relative_regret_pct']:.3f}%")
    print(f"Tasa Regret = 0        : {final_reg['zero_regret_rate_pct']:.2f}%")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred_final, zero_division=0))
    print("Matriz de Confusión:")
    print(pd.DataFrame(cm, index=[f"Real_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels]))

    ohe_features = list(best_model.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_cols))
    all_feat_names = num_cols + ohe_features
    importances = getattr(best_model.named_steps["clf"], "feature_importances_", None)

    # Si el mejor modelo es KNN, extraer importancias de RandomForest para referencia analítica
    rf_importances = None
    if importances is None:
        rf_pipe = Pipeline([("prep", preprocessor), ("clf", RandomForestClassifier(n_estimators=200, random_state=42))])
        rf_pipe.fit(X_train, y_train)
        rf_importances = rf_pipe.named_steps["clf"].feature_importances_

    imp_to_plot = importances if importances is not None else rf_importances
    print("\nImportancia de Características:")
    for fn, iv in sorted(zip(all_feat_names, imp_to_plot), key=lambda x: x[1], reverse=True):
        print(f"  {fn:<30}: {iv:.4f}")

    print("\nPorcentaje de victoria por Distribución:")
    dist_ct = pd.crosstab(full_df["distribution"], full_df["best_algorithm"], normalize="index") * 100
    print(dist_ct.round(2))

    print("\nPorcentaje de victoria por Tamaño n:")
    size_ct = pd.crosstab(full_df["n"], full_df["best_algorithm"], normalize="index") * 100
    print(size_ct.round(2))

    save_plots(cm, labels, all_feat_names, imp_to_plot, full_df, results_dir)

    summary = {
        "best_model": best_name,
        "best_params": best_params,
        "majority_class_baseline": majority_class,
        "baseline_accuracy": float(base_acc),
        "model_accuracy": float(final_acc),
        "baseline_regret": base_reg,
        "model_regret": final_reg,
        "class_balance_counts": global_counts.to_dict(),
        "class_balance_pct": global_pcts.round(2).to_dict(),
        "all_models_comparison": all_models_summary,
        "feature_importances": dict(sorted(zip(all_feat_names, [round(float(x), 4) for x in imp_to_plot]), key=lambda x: x[1], reverse=True)),
        "by_distribution_pct": dist_ct.round(2).to_dict(),
        "by_size_n_pct": size_ct.round(2).to_dict(),
    }
    with open(os.path.join(results_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
