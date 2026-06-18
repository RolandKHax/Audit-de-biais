"""
Pipeline principal d'audit de biais.

Le script couvre les exigences du projet: EDA orientee biais, baselines
multiples, comparaison avec/sans attributs sensibles, métriques de fairness
par groupe, mitigation par reweighting/resampling et rapport reproductible.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bias_mitigation import PreprocessingDebias
from src.data_processing import DataProcessor, prepare_compas_data
from src.advanced_mitigation import (
    group_threshold_predictions,
    run_adversarial_debiasing_torch,
    run_fairlearn_reduction,
    tune_group_thresholds,
)
from src.metrics import (
    FairnessMetrics,
    PerformanceMetrics,
    bootstrap_confidence_interval,
    compute_multigroup_fairness,
    summarize_numeric_rows,
)
from src.visualization import FairnessVisualizer


class BiasAuditor:
    """Orchestre un audit complet et reproductible."""

    def __init__(self, config: dict):
        self.config = {
            "test_size": 0.15,
            "validation_size": 0.15,
            "seeds": [0, 1, 2, 3, 4],
            "models": ["logistic_regression", "random_forest", "mlp"],
            "feature_policies": ["without_sensitive", "with_sensitive"],
            "debiasing_methods": ["reweighting", "resampling"],
            "results_dir": "results",
            "bootstrap_iterations": 200,
            **config,
        }
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "config": self.config,
                "methodology": {
                    "split": "stratification label + attributs sensibles quand possible",
                    "feature_policies": {
                        "without_sensitive": "race/sex/gender exclus du modele mais conserves pour l'audit",
                        "with_sensitive": "attributs sensibles inclus pour comparaison diagnostique",
                    },
                    "fairness": "DP, disparate impact, equalized odds, TPR/FPR, precision et selection rate par groupe",
                },
            },
            "data_analysis": {},
            "baseline_evaluation": {},
            "debiasing_results": {},
            "recommendations": [],
        }
        self.fairness_viz = FairnessVisualizer(
            save_dir=self.config.get("output_dir", "reports/figures")
        )
        self.label_encoder = None
        self.results_dir = Path(self.config.get("results_dir", "results"))
        for subdir in ["figures", "tables", "metrics"]:
            (self.results_dir / subdir).mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        print("\n" + "=" * 72)
        print("ETAPE 1: CHARGEMENT ET ANALYSE DES DONNEES")
        print("=" * 72)

        data_path = self.config["data_path"]
        df = pd.read_csv(data_path)

        if self.config.get("preset") == "compas":
            df = prepare_compas_data(df)
            processed_path = Path("data/processed/compas_processed.csv")
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(processed_path, index=False)
            self.results["metadata"]["processed_data_path"] = str(processed_path)

        label_name = self.config["label_name"]
        missing_columns = [
            col for col in [label_name, *self.config["protected_attrs"]]
            if col not in df.columns
        ]
        if missing_columns:
            raise ValueError(f"Colonnes manquantes dans le dataset: {missing_columns}")

        df = df.dropna(subset=[label_name, *self.config["protected_attrs"]]).reset_index(drop=True)
        df[label_name] = self._coerce_binary_label(df[label_name])

        self.processor = DataProcessor(
            protected_attributes=self.config["protected_attrs"],
            label_name=label_name,
        )
        demographics = self.processor.explore_demographics(df)
        quality = self.processor.check_data_quality(df)
        proxies = self.processor.identify_proxies(df, threshold=0.3)
        base_rates = self._compute_base_rates(df)

        self.results["data_analysis"] = {
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "columns": list(df.columns),
            "demographics": demographics,
            "quality": quality,
            "proxies": proxies,
            "base_rates": base_rates,
        }

        print(f"Dataset: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        print(f"Label positif global: {df[label_name].mean():.3f}")
        for attr in self.config["protected_attrs"]:
            print(f"\nBase rates par {attr}:")
            for group, rate in base_rates[attr].items():
                print(f"  {group}: {rate:.3f}")

        try:
            self.fairness_viz.plot_demographic_distribution(
                df,
                self.config["protected_attrs"],
                label_name,
                save_name="eda_demographics.png",
            )
        except Exception as exc:
            print(f"Visualisation EDA ignoree: {exc}")

        return df

    def run_baseline_experiments(self, df: pd.DataFrame):
        print("\n" + "=" * 72)
        print("ETAPE 2: BASELINES MULTI-MODELES")
        print("=" * 72)

        experiments = {}
        primary = None
        primary_key = None

        for seed in self.config["seeds"]:
            split = self._make_split(df, seed)
            seed_key = f"seed_{seed}"
            experiments[seed_key] = {}

            for policy in self.config["feature_policies"]:
                policy_key = str(policy)
                experiments[seed_key][policy_key] = {}
                include_sensitive = policy == "with_sensitive"
                X_train, X_test = self._feature_frames(
                    split["train_df"], split["test_df"], include_sensitive
                )

                for model_name in self.config["models"]:
                    pipeline = self._build_pipeline(X_train, model_name, seed)
                    pipeline.fit(X_train, split["y_train"])
                    evaluation = self._evaluate_model(
                        pipeline,
                        X_test,
                        split["y_test"],
                        split["sensitive_test"],
                    )
                    experiments[seed_key][policy_key][model_name] = evaluation

                    print(
                        f"{seed_key} | {policy_key} | {model_name}: "
                        f"acc={evaluation['performance']['accuracy']:.3f}, "
                        f"f1={evaluation['performance']['f1_score']:.3f}"
                    )

                    if primary is None and policy == "without_sensitive":
                        primary = evaluation
                        primary_key = {
                            "seed": seed,
                            "policy": policy_key,
                            "model": model_name,
                        }
                        self.primary_split = split
                        self.primary_model = pipeline
                        self.primary_features = X_train.columns.tolist()

        summary = self._summarize_experiments(experiments)
        aggregate_summary = summarize_numeric_rows(
            summary,
            group_keys=["feature_policy", "model"],
        )
        self.results["baseline_evaluation"] = {
            "primary": primary_key,
            "performance": primary["performance"],
            "fairness": primary["fairness"],
            "classification_report": primary["classification_report"],
            "experiments": experiments,
            "summary": summary,
            "aggregate_summary": aggregate_summary,
        }

        self._plot_primary_baseline()

    def apply_debiasing(self):
        print("\n" + "=" * 72)
        print("ETAPE 3: MITIGATION DES BIAIS")
        print("=" * 72)

        split = self.primary_split
        X_train, X_test = self._feature_frames(
            split["train_df"],
            split["test_df"],
            include_sensitive=False,
        )
        X_val, _ = self._feature_frames(
            split["val_df"],
            split["test_df"],
            include_sensitive=False,
        )
        results = {}
        first_seed = int(self.config["seeds"][0])

        for attr in self.config["protected_attrs"]:
            sensitive_train = split["sensitive_train"][attr]
            sensitive_test = split["sensitive_test"][attr]
            counts = sensitive_train.value_counts()
            if len(counts) < 2:
                continue

            privileged = counts.idxmax()
            unprivileged = counts.idxmin()
            debias = PreprocessingDebias(attr, privileged, unprivileged)
            attr_results = {}

            if "reweighting" in self.config["debiasing_methods"]:
                weights = debias.reweighting(X_train, split["y_train"], sensitive_train)
                model = self._build_pipeline(X_train, "logistic_regression", first_seed)
                model.fit(X_train, split["y_train"], classifier__sample_weight=weights)
                attr_results["reweighting"] = self._evaluate_model(
                    model, X_test, split["y_test"], {attr: sensitive_test}
                )
                print(
                    f"{attr} | reweighting: "
                    f"acc={attr_results['reweighting']['performance']['accuracy']:.3f}, "
                    f"EO={attr_results['reweighting']['fairness'][attr]['equalized_odds_difference']:.3f}"
                )

            if "resampling" in self.config["debiasing_methods"]:
                X_res, y_res, s_res = debias.resampling_balance(
                    X_train,
                    split["y_train"],
                    sensitive_train,
                    strategy="oversample",
                )
                model = self._build_pipeline(X_res, "logistic_regression", first_seed)
                model.fit(X_res, y_res)
                attr_results["resampling"] = self._evaluate_model(
                    model, X_test, split["y_test"], {attr: sensitive_test}
                )
                attr_results["resampling"]["training_distribution"] = (
                    pd.DataFrame({"label": y_res, attr: s_res})
                    .value_counts()
                    .rename("count")
                    .reset_index()
                    .to_dict(orient="records")
                )
                print(
                    f"{attr} | resampling: "
                    f"acc={attr_results['resampling']['performance']['accuracy']:.3f}, "
                    f"EO={attr_results['resampling']['fairness'][attr]['equalized_odds_difference']:.3f}"
                )

            if "threshold" in self.config["debiasing_methods"]:
                base_model = self._build_pipeline(X_train, "logistic_regression", first_seed)
                base_model.fit(X_train, split["y_train"])
                val_scores = base_model.predict_proba(X_val)[:, 1]
                test_scores = base_model.predict_proba(X_test)[:, 1]
                thresholds = tune_group_thresholds(
                    split["y_val"],
                    val_scores,
                    split["sensitive_val"][attr],
                )
                y_pred = group_threshold_predictions(
                    test_scores,
                    sensitive_test,
                    thresholds,
                )
                evaluation = self._evaluate_predictions(
                    split["y_test"],
                    y_pred,
                    test_scores,
                    {attr: sensitive_test},
                )
                evaluation["thresholds"] = thresholds
                evaluation["ethical_note"] = (
                    "Seuils par groupe: amelioration technique possible, mais usage "
                    "sensible en contexte judiciaire et a encadrer juridiquement."
                )
                attr_results["threshold"] = evaluation
                print(
                    f"{attr} | threshold: "
                    f"acc={evaluation['performance']['accuracy']:.3f}, "
                    f"EO={evaluation['fairness'][attr]['equalized_odds_difference']:.3f}"
                )

            for constraint_name in ["demographic_parity", "equalized_odds"]:
                method_name = f"fairlearn_{constraint_name}"
                if method_name not in self.config["debiasing_methods"]:
                    continue
                estimator = self._build_pipeline(X_train, "logistic_regression", first_seed)
                optional = run_fairlearn_reduction(
                    estimator,
                    X_train,
                    split["y_train"],
                    sensitive_train,
                    X_test,
                    constraint_name=constraint_name,
                )
                if optional.available:
                    evaluation = self._evaluate_predictions(
                        split["y_test"],
                        optional.payload["y_pred"],
                        optional.payload["y_scores"],
                        {attr: sensitive_test},
                    )
                    attr_results[method_name] = evaluation
                    print(
                        f"{attr} | {method_name}: "
                        f"acc={evaluation['performance']['accuracy']:.3f}, "
                        f"EO={evaluation['fairness'][attr]['equalized_odds_difference']:.3f}"
                    )
                else:
                    attr_results[method_name] = {
                        "available": False,
                        "reason": optional.reason,
                    }

            if "adversarial_pytorch" in self.config["debiasing_methods"]:
                preprocessor = self._build_pipeline(
                    X_train, "logistic_regression", first_seed
                ).named_steps["preprocess"]
                optional = run_adversarial_debiasing_torch(
                    preprocessor,
                    X_train,
                    split["y_train"],
                    sensitive_train,
                    X_test,
                    lambdas=[0.0, 0.1, 0.5, 1.0],
                    epochs=int(self.config.get("adversarial_epochs", 20)),
                    random_state=first_seed,
                )
                if optional.available:
                    adv_results = {}
                    for lambda_value, payload in optional.payload.items():
                        adv_results[lambda_value] = self._evaluate_predictions(
                            split["y_test"],
                            payload["y_pred"],
                            payload["y_scores"],
                            {attr: sensitive_test},
                        )
                    attr_results["adversarial_pytorch"] = adv_results
                    best_lambda = min(
                        adv_results,
                        key=lambda key: adv_results[key]["fairness"][attr]["equalized_odds_difference"],
                    )
                    print(
                        f"{attr} | adversarial_pytorch best_lambda={best_lambda}: "
                        f"EO={adv_results[best_lambda]['fairness'][attr]['equalized_odds_difference']:.3f}"
                    )
                else:
                    attr_results["adversarial_pytorch"] = {
                        "available": False,
                        "reason": optional.reason,
                    }

            results[attr] = attr_results

        self.results["debiasing_results"] = results
        self._plot_debiasing_comparison()

    def generate_recommendations(self):
        print("\n" + "=" * 72)
        print("ETAPE 4: RECOMMANDATIONS")
        print("=" * 72)

        recommendations = []
        fairness = self.results["baseline_evaluation"]["fairness"]

        for attr, metrics in fairness.items():
            dp = metrics["demographic_parity_difference"]
            eo = metrics["equalized_odds_difference"]
            fpr = metrics["fpr_difference"]
            recommendations.append({
                "priority": "INFO",
                "category": "Fairness",
                "issue": f"Audit par groupe realise pour {attr}: DP={dp:.3f}, EO={eo:.3f}, FPR diff={fpr:.3f}",
                "recommendation": "Presenter ces valeurs avec les tableaux TPR/FPR/selection rate par groupe dans le rapport final.",
                "expected_impact": "Montre que l'analyse ne se limite pas a l'accuracy globale.",
            })
            if eo > 0.10:
                recommendations.append({
                    "priority": "HIGH",
                    "category": "Fairness",
                    "issue": f"Equalized odds eleve pour {attr} ({eo:.3f})",
                    "recommendation": "Prioriser l'analyse TPR/FPR par groupe avant tout deploiement.",
                    "expected_impact": "Reduction du risque d'erreurs asymetriques entre groupes.",
                })
            if fpr > 0.10:
                recommendations.append({
                    "priority": "HIGH",
                    "category": "Judicial risk",
                    "issue": f"Ecart de faux positifs notable pour {attr} ({fpr:.3f})",
                    "recommendation": "Dans un cas type COMPAS, documenter explicitement les faux positifs par groupe.",
                    "expected_impact": "Meilleure prise en compte de l'impact humain des erreurs.",
                })
            if dp > 0.10:
                recommendations.append({
                    "priority": "MEDIUM",
                    "category": "Fairness",
                    "issue": f"Parite demographique imparfaite pour {attr} ({dp:.3f})",
                    "recommendation": "Comparer reweighting, resampling et seuils de decision sur validation.",
                    "expected_impact": "Choix plus transparent du compromis performance/fairness.",
                })

        for attr, proxy_list in self.results["data_analysis"].get("proxies", {}).items():
            if proxy_list:
                top = ", ".join(proxy["feature"] for proxy in proxy_list[:3])
                recommendations.append({
                    "priority": "MEDIUM",
                    "category": "Data",
                    "issue": f"Variables proxy detectees pour {attr}: {top}",
                    "recommendation": "Conserver ces variables dans l'audit, meme si les attributs sensibles sont exclus du modele.",
                    "expected_impact": "Evite la conclusion fausse selon laquelle supprimer race/sex suffit.",
                })

        recommendations.append({
            "priority": "HIGH",
            "category": "Governance",
            "issue": "Une mesure ponctuelle ne garantit pas une fairness durable.",
            "recommendation": "Mettre en place un monitoring periodique accuracy, F1, DP, EO, TPR et FPR par groupe.",
            "expected_impact": "Detection des derives de donnees et de performance.",
        })
        recommendations.append({
            "priority": "MEDIUM",
            "category": "Methodology",
            "issue": "Les metriques de fairness traduisent des definitions differentes de l'equite.",
            "recommendation": "Justifier le choix d'equalized odds comme critere principal quand les faux positifs/faux negatifs ont un impact humain fort.",
            "expected_impact": "Discussion critique plus solide pour un cas de type COMPAS.",
        })

        self.results["recommendations"] = recommendations
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
        recommendations.sort(key=lambda rec: priority_order.get(rec["priority"], 9))
        for rec in recommendations:
            print(f"[{rec['priority']}] {rec['issue']}")

    def generate_report(self):
        print("\n" + "=" * 72)
        print("ETAPE 5: RAPPORTS")
        print("=" * 72)

        self._save_result_artifacts()

        output_path = Path(self.config.get("output_path", "reports/audit_report.html"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self._html_report(), encoding="utf-8")

        json_path = output_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(
                self._strip_runtime_arrays(self.results),
                indent=2,
                ensure_ascii=False,
                default=self._json_default,
            ),
            encoding="utf-8",
        )

        print(f"Rapport HTML: {output_path}")
        print(f"Resultats JSON: {json_path}")
        print(f"Artefacts tabulaires: {self.results_dir}")

    def run(self):
        try:
            df = self.load_data()
            self.run_baseline_experiments(df)
            self.apply_debiasing()
            self.generate_recommendations()
            self.generate_report()
            print("\nAudit termine avec succes.")
        except Exception:
            import traceback

            traceback.print_exc()
            sys.exit(1)

    def _coerce_binary_label(self, y: pd.Series) -> np.ndarray:
        if y.nunique() != 2:
            raise ValueError("L'audit attend une cible binaire.")
        if set(pd.Series(y).dropna().unique()).issubset({0, 1, False, True}):
            return y.astype(int).to_numpy()
        self.label_encoder = LabelEncoder()
        return self.label_encoder.fit_transform(y.astype(str))

    def _compute_base_rates(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        label = self.config["label_name"]
        return {
            attr: {
                str(group): float(rate)
                for group, rate in df.groupby(attr)[label].mean().sort_index().items()
            }
            for attr in self.config["protected_attrs"]
        }

    def _make_split(self, df: pd.DataFrame, seed: int) -> Dict:
        label_name = self.config["label_name"]
        stratify_col = df[label_name].astype(str)
        for attr in self.config["protected_attrs"]:
            stratify_col = stratify_col + "_" + df[attr].astype(str)

        try:
            train_val_df, test_df = train_test_split(
                df,
                test_size=float(self.config["test_size"]),
                random_state=seed,
                stratify=stratify_col,
            )
        except ValueError:
            train_val_df, test_df = train_test_split(
                df,
                test_size=float(self.config["test_size"]),
                random_state=seed,
                stratify=df[label_name],
            )

        val_size = float(self.config.get("validation_size", 0.15))
        relative_val_size = val_size / max(1e-9, 1.0 - float(self.config["test_size"]))
        train_val_stratify = train_val_df[label_name].astype(str)
        for attr in self.config["protected_attrs"]:
            train_val_stratify = train_val_stratify + "_" + train_val_df[attr].astype(str)

        try:
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=relative_val_size,
                random_state=seed,
                stratify=train_val_stratify,
            )
        except ValueError:
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=relative_val_size,
                random_state=seed,
                stratify=train_val_df[label_name],
            )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        return {
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "y_train": train_df[label_name].astype(int),
            "y_val": val_df[label_name].astype(int),
            "y_test": test_df[label_name].astype(int),
            "sensitive_train": train_df[self.config["protected_attrs"]].copy(),
            "sensitive_val": val_df[self.config["protected_attrs"]].copy(),
            "sensitive_test": test_df[self.config["protected_attrs"]].copy(),
        }

    def _feature_frames(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        include_sensitive: bool,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        drop_columns = [self.config["label_name"]]
        if not include_sensitive:
            drop_columns.extend(self.config["protected_attrs"])
        X_train = train_df.drop(columns=[col for col in drop_columns if col in train_df.columns])
        X_test = test_df.drop(columns=[col for col in drop_columns if col in test_df.columns])
        return X_train, X_test

    def _build_pipeline(self, X: pd.DataFrame, model_name: str, seed: int) -> Pipeline:
        numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        categorical_cols = [col for col in X.columns if col not in numeric_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]),
                    numeric_cols,
                ),
                (
                    "cat",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]),
                    categorical_cols,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        models = {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
            "random_forest": RandomForestClassifier(
                n_estimators=80,
                max_depth=8,
                min_samples_leaf=5,
                random_state=seed,
                n_jobs=-1,
            ),
            "mlp": MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                alpha=1e-3,
                learning_rate_init=1e-3,
                max_iter=80,
                early_stopping=True,
                random_state=seed,
            ),
        }
        if model_name not in models:
            raise ValueError(f"Modele inconnu: {model_name}")
        return Pipeline([("preprocess", preprocessor), ("classifier", models[model_name])])

    def _evaluate_model(
        self,
        model: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sensitive_test: pd.DataFrame | Dict[str, pd.Series],
    ) -> Dict:
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)[:, 1]
        else:
            y_scores = y_pred

        return self._evaluate_predictions(y_test, y_pred, y_scores, sensitive_test)

    def _evaluate_predictions(
        self,
        y_test: pd.Series,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        sensitive_test: pd.DataFrame | Dict[str, pd.Series],
    ) -> Dict:

        performance = PerformanceMetrics.compute_metrics(y_test, y_pred, y_scores)
        performance = {k: v for k, v in performance.items() if k != "confusion_matrix"}
        confidence_intervals = {
            "accuracy": bootstrap_confidence_interval(
                y_test,
                y_pred,
                lambda yt, yp: PerformanceMetrics.compute_metrics(yt, yp)["accuracy"],
                n_iterations=int(self.config.get("bootstrap_iterations", 200)),
            ),
            "f1_score": bootstrap_confidence_interval(
                y_test,
                y_pred,
                lambda yt, yp: PerformanceMetrics.compute_metrics(yt, yp)["f1_score"],
                n_iterations=int(self.config.get("bootstrap_iterations", 200)),
            ),
        }
        fairness = {}
        for attr in sensitive_test:
            sensitive = np.asarray(sensitive_test[attr])
            multigroup = compute_multigroup_fairness(y_test, y_pred, sensitive, y_scores)
            counts = pd.Series(sensitive).value_counts()
            if len(counts) >= 2:
                privileged = counts.idxmax()
                unprivileged = counts.idxmin()
                pairwise = FairnessMetrics(attr, privileged, unprivileged).compute_all_metrics(
                    y_test, y_pred, sensitive, y_scores
                )
                multigroup["pairwise_privileged"] = str(privileged)
                multigroup["pairwise_unprivileged"] = str(unprivileged)
                multigroup["pairwise"] = {
                    key: value for key, value in pairwise.items()
                    if key not in {"interpretation", "group_metrics"}
                }
                multigroup["interpretation"] = pairwise.get("interpretation", {})
            fairness[attr] = multigroup

        return {
            "performance": self._clean_numbers(performance),
            "confidence_intervals": self._clean_numbers(confidence_intervals),
            "fairness": self._clean_numbers(fairness),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
            "predictions": {
                "y_pred": y_pred,
                "y_scores": y_scores,
            },
        }

    def _summarize_experiments(self, experiments: Dict) -> List[Dict]:
        rows = []
        for seed_key, policies in experiments.items():
            for policy, models in policies.items():
                for model_name, result in models.items():
                    row = {
                        "seed": seed_key,
                        "feature_policy": policy,
                        "model": model_name,
                        "accuracy": result["performance"]["accuracy"],
                        "f1_score": result["performance"]["f1_score"],
                        "roc_auc": result["performance"].get("roc_auc"),
                    }
                    for attr, fairness in result["fairness"].items():
                        row[f"{attr}_dp_diff"] = fairness["demographic_parity_difference"]
                        row[f"{attr}_eo_diff"] = fairness["equalized_odds_difference"]
                        row[f"{attr}_fpr_diff"] = fairness["fpr_difference"]
                    rows.append(row)
        return rows

    def _plot_primary_baseline(self):
        split = self.primary_split
        primary = self.results["baseline_evaluation"]
        pred = primary["experiments"][f"seed_{primary['primary']['seed']}"][
            primary["primary"]["policy"]
        ][primary["primary"]["model"]]["predictions"]
        first_attr = self.config["protected_attrs"][0]
        sensitive = split["sensitive_test"][first_attr].to_numpy()
        group_names = {value: f"{first_attr}={value}" for value in np.unique(sensitive)}
        try:
            self.fairness_viz.plot_confusion_matrices_by_group(
                split["y_test"].to_numpy(),
                pred["y_pred"],
                sensitive,
                group_names,
                save_name="baseline_confusion_matrices.png",
            )
            self.fairness_viz.plot_roc_curves_by_group(
                split["y_test"].to_numpy(),
                pred["y_scores"],
                sensitive,
                group_names,
                save_name="baseline_roc_curves.png",
            )
        except Exception as exc:
            print(f"Visualisations baseline ignorees: {exc}")

    def _plot_debiasing_comparison(self):
        if not self.results["debiasing_results"]:
            return
        first_attr = next(iter(self.results["debiasing_results"]))
        metrics_dict = {
            "baseline": self.results["baseline_evaluation"]["fairness"][first_attr]
        }
        for method, result in self.results["debiasing_results"][first_attr].items():
            if isinstance(result, dict) and "fairness" in result:
                metrics_dict[method] = result["fairness"][first_attr]
            elif method == "adversarial_pytorch" and isinstance(result, dict):
                for lambda_value, lambda_result in result.items():
                    if isinstance(lambda_result, dict) and "fairness" in lambda_result:
                        metrics_dict[f"adv_lambda_{lambda_value}"] = lambda_result["fairness"][first_attr]
        try:
            self.fairness_viz.plot_fairness_metrics_comparison(
                metrics_dict,
                save_name="debiasing_comparison.png",
            )
        except Exception as exc:
            print(f"Visualisation mitigation ignoree: {exc}")

    def _html_report(self) -> str:
        perf = self.results["baseline_evaluation"]["performance"]
        fairness_rows = self._fairness_rows_html()
        summary_rows = self._summary_rows_html()
        mitigation_rows = self._mitigation_rows_html()
        recommendation_rows = self._recommendation_rows_html()
        primary = self.results["baseline_evaluation"]["primary"]

        return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Rapport d'audit de biais</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; line-height: 1.5; }}
    h1, h2 {{ color: #1f4e79; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #eef4f8; }}
    .metric {{ display: inline-block; margin: 8px 18px 8px 0; padding: 10px 14px; background: #f6f8fa; }}
    .warn {{ color: #9a3412; font-weight: bold; }}
  </style>
</head>
<body>
  <h1>Audit de biais d'un modele de classification</h1>
  <p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
  <p><strong>Baseline primaire:</strong> seed={primary['seed']}, politique={primary['policy']}, modele={primary['model']}.</p>

  <h2>Resume executif</h2>
  <p>L'audit mesure la performance globale et les disparites par groupe protege.
  Il compare les modeles avec et sans attributs sensibles afin d'eviter la conclusion
  erronee selon laquelle supprimer ces colonnes suffit a supprimer le biais.</p>
  <div class="metric">Accuracy: {perf['accuracy']:.3f}</div>
  <div class="metric">Precision: {perf['precision']:.3f}</div>
  <div class="metric">Recall: {perf['recall']:.3f}</div>
  <div class="metric">F1: {perf['f1_score']:.3f}</div>
  <div class="metric">ROC-AUC: {perf.get('roc_auc', 0):.3f}</div>

  <h2>Comparaison des baselines</h2>
  {summary_rows}

  <h2>Fairness par attribut protege</h2>
  {fairness_rows}

  <h2>Mitigation</h2>
  {mitigation_rows}

  <h2>Recommandations</h2>
  {recommendation_rows}

  <h2>Methodologie</h2>
  <p>Split stratifie par label et attributs sensibles quand les effectifs le permettent.
  Les variables categorielles sont encodees par one-hot encoding, les variables numeriques
  sont imputees et standardisees. Les mesures principales sont demographic parity,
  disparate impact, equalized odds, TPR/FPR, precision et selection rate par groupe.</p>
</body>
</html>"""

    def _summary_rows_html(self) -> str:
        rows = self.results["baseline_evaluation"]["summary"]
        headers = ["seed", "feature_policy", "model", "accuracy", "f1_score", "roc_auc"]
        html = "<table><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
        for row in rows:
            html += "<tr>" + "".join(
                f"<td>{self._format_cell(row.get(h))}</td>" for h in headers
            ) + "</tr>"
        return html + "</table>"

    def _fairness_rows_html(self) -> str:
        html = ""
        for attr, metrics in self.results["baseline_evaluation"]["fairness"].items():
            html += f"<h3>{attr}</h3>"
            html += (
                "<table><tr><th>Groupe</th><th>n</th><th>Base rate</th>"
                "<th>Selection rate</th><th>TPR</th><th>FPR</th><th>Precision</th></tr>"
            )
            for group, group_metrics in metrics["group_metrics"].items():
                html += (
                    f"<tr><td>{group}</td><td>{group_metrics['sample_size']}</td>"
                    f"<td>{group_metrics['base_rate']:.3f}</td>"
                    f"<td>{group_metrics['selection_rate']:.3f}</td>"
                    f"<td>{group_metrics['tpr']:.3f}</td>"
                    f"<td>{group_metrics['fpr']:.3f}</td>"
                    f"<td>{group_metrics['precision']:.3f}</td></tr>"
                )
            html += "</table>"
            html += (
                f"<p>DP diff: <strong>{metrics['demographic_parity_difference']:.3f}</strong>, "
                f"DI: <strong>{metrics['disparate_impact']:.3f}</strong>, "
                f"EO diff: <strong>{metrics['equalized_odds_difference']:.3f}</strong>, "
                f"FPR diff: <strong>{metrics['fpr_difference']:.3f}</strong>.</p>"
            )
        return html

    def _mitigation_rows_html(self) -> str:
        if not self.results["debiasing_results"]:
            return "<p>Aucune mitigation appliquee.</p>"
        html = "<table><tr><th>Attribut</th><th>Methode</th><th>Accuracy</th><th>F1</th><th>DP diff</th><th>EO diff</th><th>FPR diff</th></tr>"
        for attr, methods in self.results["debiasing_results"].items():
            for method, result in methods.items():
                if method == "adversarial_pytorch" and isinstance(result, dict):
                    for lambda_value, lambda_result in result.items():
                        if not isinstance(lambda_result, dict) or "fairness" not in lambda_result:
                            continue
                        fairness = lambda_result["fairness"][attr]
                        perf = lambda_result["performance"]
                        html += (
                            f"<tr><td>{attr}</td><td>adversarial λ={lambda_value}</td>"
                            f"<td>{perf['accuracy']:.3f}</td>"
                            f"<td>{perf['f1_score']:.3f}</td>"
                            f"<td>{fairness['demographic_parity_difference']:.3f}</td>"
                            f"<td>{fairness['equalized_odds_difference']:.3f}</td>"
                            f"<td>{fairness['fpr_difference']:.3f}</td></tr>"
                        )
                    continue
                if not isinstance(result, dict) or "fairness" not in result:
                    reason = result.get("reason", "indisponible") if isinstance(result, dict) else "indisponible"
                    html += (
                        f"<tr><td>{attr}</td><td>{method}</td>"
                        f"<td colspan='5'>Indisponible: {reason}</td></tr>"
                    )
                    continue
                fairness = result["fairness"][attr]
                perf = result["performance"]
                html += (
                    f"<tr><td>{attr}</td><td>{method}</td>"
                    f"<td>{perf['accuracy']:.3f}</td>"
                    f"<td>{perf['f1_score']:.3f}</td>"
                    f"<td>{fairness['demographic_parity_difference']:.3f}</td>"
                    f"<td>{fairness['equalized_odds_difference']:.3f}</td>"
                    f"<td>{fairness['fpr_difference']:.3f}</td></tr>"
                )
        return html + "</table>"

    def _recommendation_rows_html(self) -> str:
        html = "<table><tr><th>Priorite</th><th>Categorie</th><th>Constat</th><th>Action</th></tr>"
        for rec in self.results["recommendations"]:
            html += (
                f"<tr><td>{rec['priority']}</td><td>{rec['category']}</td>"
                f"<td>{rec['issue']}</td><td>{rec['recommendation']}</td></tr>"
            )
        return html + "</table>"

    def _save_result_artifacts(self):
        summary = self.results["baseline_evaluation"].get("summary", [])
        aggregate = self.results["baseline_evaluation"].get("aggregate_summary", [])
        if summary:
            pd.DataFrame(summary).to_csv(self.results_dir / "tables" / "baseline_summary.csv", index=False)
        if aggregate:
            pd.DataFrame(aggregate).to_csv(self.results_dir / "tables" / "baseline_summary_aggregate.csv", index=False)

        group_rows = []
        for attr, fairness in self.results["baseline_evaluation"].get("fairness", {}).items():
            for group, metrics in fairness.get("group_metrics", {}).items():
                row = {"attribute": attr, "group": group}
                row.update(metrics)
                group_rows.append(row)
        if group_rows:
            pd.DataFrame(group_rows).to_csv(self.results_dir / "tables" / "fairness_by_group.csv", index=False)

        mitigation_rows = []
        for attr, methods in self.results.get("debiasing_results", {}).items():
            for method, result in methods.items():
                if method == "adversarial_pytorch":
                    for lambda_value, lambda_result in result.items():
                        if isinstance(lambda_result, dict) and "fairness" in lambda_result:
                            row = {
                                "attribute": attr,
                                "method": "adversarial_pytorch",
                                "lambda": lambda_value,
                            }
                            row.update(lambda_result["performance"])
                            row.update({
                                "dp_diff": lambda_result["fairness"][attr]["demographic_parity_difference"],
                                "eo_diff": lambda_result["fairness"][attr]["equalized_odds_difference"],
                                "fpr_diff": lambda_result["fairness"][attr]["fpr_difference"],
                            })
                            mitigation_rows.append(row)
                    continue
                if isinstance(result, dict) and "fairness" in result:
                    row = {"attribute": attr, "method": method}
                    row.update(result["performance"])
                    row.update({
                        "dp_diff": result["fairness"][attr]["demographic_parity_difference"],
                        "eo_diff": result["fairness"][attr]["equalized_odds_difference"],
                        "fpr_diff": result["fairness"][attr]["fpr_difference"],
                    })
                    mitigation_rows.append(row)
                else:
                    mitigation_rows.append({
                        "attribute": attr,
                        "method": method,
                        "available": False,
                        "reason": result.get("reason", "") if isinstance(result, dict) else "",
                    })
        if mitigation_rows:
            pd.DataFrame(mitigation_rows).to_csv(self.results_dir / "tables" / "mitigation_summary.csv", index=False)

        metrics_path = self.results_dir / "metrics" / "audit_metrics.json"
        metrics_path.write_text(
            json.dumps(self._strip_runtime_arrays(self.results), indent=2, ensure_ascii=False, default=self._json_default),
            encoding="utf-8",
        )

    def _format_cell(self, value):
        if isinstance(value, float):
            return f"{value:.3f}"
        if value is None:
            return ""
        return str(value)

    def _clean_numbers(self, obj):
        if isinstance(obj, dict):
            return {str(k): self._clean_numbers(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._clean_numbers(v) for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj
        return obj

    def _json_default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    def _strip_runtime_arrays(self, obj):
        """Retire les predictions brutes du JSON public pour garder un rapport lisible."""
        if isinstance(obj, dict):
            return {
                key: self._strip_runtime_arrays(value)
                for key, value in obj.items()
                if key != "predictions"
            }
        if isinstance(obj, list):
            return [self._strip_runtime_arrays(value) for value in obj]
        return obj


def _csv_list(values: Iterable[str]) -> List[str]:
    output = []
    for value in values:
        output.extend(part.strip() for part in value.split(",") if part.strip())
    return output


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Audit de biais pour modeles de classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Chemin du CSV")
    parser.add_argument("--model", help="Modele pickle externe pour compatibilite historique")
    parser.add_argument("--protected-attrs", nargs="+", required=True, help="Attributs proteges")
    parser.add_argument("--label", required=True, help="Colonne cible binaire")
    parser.add_argument(
        "--debiasing",
        nargs="*",
        choices=[
            "reweighting",
            "resampling",
            "threshold",
            "fairlearn_demographic_parity",
            "fairlearn_equalized_odds",
            "adversarial_pytorch",
        ],
        default=["reweighting", "resampling"],
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["logistic_regression", "random_forest", "mlp"],
        help="Modeles: logistic_regression random_forest mlp",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=[42])
    parser.add_argument(
        "--feature-policies",
        nargs="*",
        default=["without_sensitive", "with_sensitive"],
        choices=["without_sensitive", "with_sensitive"],
    )
    parser.add_argument("--preset", choices=["compas"], help="Preset de nettoyage dataset")
    parser.add_argument("--output", default="reports/audit_report.html")
    parser.add_argument("--output-dir", default="reports/figures")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--adversarial-epochs", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = {
        "data_path": args.data,
        "model_path": args.model,
        "protected_attrs": args.protected_attrs,
        "label_name": args.label,
        "debiasing_methods": args.debiasing,
        "models": _csv_list(args.models),
        "seeds": args.seeds,
        "feature_policies": _csv_list(args.feature_policies),
        "preset": args.preset,
        "output_path": args.output,
        "output_dir": args.output_dir,
        "results_dir": args.results_dir,
        "bootstrap_iterations": args.bootstrap_iterations,
        "adversarial_epochs": args.adversarial_epochs,
    }
    BiasAuditor(config).run()


if __name__ == "__main__":
    main()
