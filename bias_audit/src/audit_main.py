"""
Script principal d'audit de biais - Interface en ligne de commande.
Permet d'auditer un modèle de classification et générer un rapport complet.

Usage:
    python -m src.audit_main --data data/dataset.csv \\
                             --model models/model.pkl \\
                             --protected-attrs gender race \\
                             --label label \\
                             --output reports/audit_report.html
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Ajouter la racine du projet au path pour supporter:
#   python -m src.audit_main
#   python src/audit_main.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing import DataProcessor
from src.metrics import FairnessMetrics, PerformanceMetrics
from src.bias_mitigation import PreprocessingDebias
from src.visualization import FairnessVisualizer, PerformanceVisualizer


class BiasAuditor:
    """Classe principale pour orchestrer l'audit de biais"""
    
    def __init__(self, config: dict):
        self.config = config
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'config': config
            },
            'data_analysis': {},
            'baseline_evaluation': {},
            'debiasing_results': {},
            'recommendations': []
        }
        
        # Initialiser les visualiseurs
        self.fairness_viz = FairnessVisualizer(save_dir=config.get('output_dir', 'reports/figures'))
        self.perf_viz = PerformanceVisualizer(save_dir=config.get('output_dir', 'reports/figures'))
    
    def load_data(self):
        """Charge et prépare les données"""
        print("\n" + "="*60)
        print("ÉTAPE 1: CHARGEMENT DES DONNÉES")
        print("="*60)
        
        data_path = self.config['data_path']
        print(f"Chargement depuis: {data_path}")
        
        df = pd.read_csv(data_path)
        print(f"✓ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        
        # Initialiser le processor
        self.processor = DataProcessor(
            protected_attributes=self.config['protected_attrs'],
            label_name=self.config['label_name']
        )
        
        # Analyse exploratoire
        print("\nAnalyse démographique...")
        demographics = self.processor.explore_demographics(df)
        self.results['data_analysis']['demographics'] = demographics
        
        # Afficher les distributions
        for attr, stats in demographics.items():
            if attr != 'intersectional':
                print(f"\n{attr}:")
                if 'counts' in stats:
                    for value, count in stats['counts'].items():
                        prop = stats['proportions'][value]
                        print(f"  {value}: {count} ({prop*100:.1f}%)")
        
        # Vérifier la qualité
        print("\nVérification de la qualité des données...")
        quality = self.processor.check_data_quality(df)
        self.results['data_analysis']['quality'] = quality
        
        missing = sum(quality['missing_values'].values())
        print(f"  Valeurs manquantes: {missing}")
        print(f"  Doublons: {quality['duplicates']}")
        
        # Identifier les proxies
        print("\nIdentification des proxies...")
        proxies = self.processor.identify_proxies(df, threshold=0.3)
        self.results['data_analysis']['proxies'] = proxies
        
        for attr, proxy_list in proxies.items():
            print(f"\n  {attr}: {len(proxy_list)} proxies détectés")
            for proxy in proxy_list[:3]:
                print(f"    - {proxy['feature']}: corrélation={proxy['correlation']:.3f}")
        
        return df
    
    def load_or_train_model(self, df: pd.DataFrame):
        """Charge un modèle existant ou en entraîne un nouveau"""
        print("\n" + "="*60)
        print("ÉTAPE 2: MODÈLE")
        print("="*60)
        
        # Préparer les données
        df_encoded = self.processor.encode_categorical(df)
        X_train, X_test, y_train, y_test = self.processor.split_data(df_encoded)
        
        # Sauvegarder les données de test
        self.X_test = X_test
        self.y_test = y_test
        self.sensitive_test = {}
        for attr in self.config['protected_attrs']:
            self.sensitive_test[attr] = X_test[attr].values
        
        # Retirer les attributs protégés des features
        features_to_drop = self.config['protected_attrs'] + [self.config['label_name']]
        X_train_clean = X_train.drop(columns=[col for col in features_to_drop if col in X_train.columns])
        X_test_clean = X_test.drop(columns=[col for col in features_to_drop if col in X_test.columns])
        
        self.X_test_clean = X_test_clean
        
        # Charger ou entraîner le modèle
        model_path = self.config.get('model_path')
        
        if model_path and Path(model_path).exists():
            print(f"Chargement du modèle depuis: {model_path}")
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("✓ Modèle chargé")
        else:
            print("Entraînement d'un nouveau modèle (Logistic Regression)...")
            from sklearn.linear_model import LogisticRegression
            
            self.model = LogisticRegression(max_iter=1000, random_state=42)
            self.model.fit(X_train_clean, y_train)
            
            print("✓ Modèle entraîné")
            
            # Sauvegarder si chemin spécifié
            if model_path:
                Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                with open(model_path, 'wb') as f:
                    pickle.dump(self.model, f)
                print(f"✓ Modèle sauvegardé: {model_path}")
        
        return X_train_clean, X_test_clean, y_train, y_test
    
    def evaluate_baseline(self):
        """Évalue le modèle baseline"""
        print("\n" + "="*60)
        print("ÉTAPE 3: ÉVALUATION BASELINE")
        print("="*60)
        
        # Prédictions
        y_pred = self.model.predict(self.X_test_clean)
        y_scores = self.model.predict_proba(self.X_test_clean)[:, 1]
        
        # Performance globale
        print("\nPerformance globale:")
        perf = PerformanceMetrics.compute_metrics(self.y_test, y_pred, y_scores)
        
        for metric, value in perf.items():
            if metric not in ['confusion_matrix', 'true_negatives', 'false_positives', 
                             'false_negatives', 'true_positives']:
                print(f"  {metric}: {value:.4f}")
        
        self.results['baseline_evaluation']['performance'] = perf
        
        # Fairness par attribut protégé
        print("\nMétriques de fairness:")
        fairness_results = {}
        
        for attr in self.config['protected_attrs']:
            print(f"\n  Attribut: {attr}")
            
            sensitive = self.sensitive_test[attr]
            unique_values = np.unique(sensitive)
            
            if len(unique_values) < 2:
                print(f"    ⚠ Attribut {attr} n'a qu'une seule valeur unique, skip")
                continue
            
            # Choisir privilégié/défavorisé
            counts = pd.Series(sensitive).value_counts()
            privileged = counts.idxmax()
            unprivileged = counts.idxmin()
            
            fm = FairnessMetrics(attr, privileged, unprivileged)
            fairness = fm.compute_all_metrics(self.y_test, y_pred, sensitive, y_scores)
            
            fairness_results[attr] = fairness
            
            # Afficher les métriques clés
            print(f"    Demographic Parity Difference: {fairness['demographic_parity_difference']:.4f}")
            print(f"    Disparate Impact: {fairness['disparate_impact']:.4f}")
            print(f"    Equalized Odds (avg): {fairness['average_odds_difference']:.4f}")
            
            # Interprétation
            print(f"\n    Interprétation:")
            for metric, interp in fairness['interpretation'].items():
                print(f"      {metric}: {interp}")
        
        self.results['baseline_evaluation']['fairness'] = fairness_results
        
        # Visualisations
        print("\nGénération des visualisations baseline...")
        
        # Matrices de confusion par groupe (premier attribut)
        if self.config['protected_attrs']:
            first_attr = self.config['protected_attrs'][0]
            sensitive = self.sensitive_test[first_attr]
            unique_vals = np.unique(sensitive)
            group_names = {val: f"{first_attr}={val}" for val in unique_vals}
            
            self.fairness_viz.plot_confusion_matrices_by_group(
                self.y_test, y_pred, sensitive, group_names,
                save_name='baseline_confusion_matrices.png'
            )
            
            self.fairness_viz.plot_roc_curves_by_group(
                self.y_test, y_scores, sensitive, group_names,
                save_name='baseline_roc_curves.png'
            )
        
        print("✓ Évaluation baseline terminée")
    
    def apply_debiasing(self, X_train, y_train, sensitive_train):
        """Applique différentes techniques de débiaisage"""
        print("\n" + "="*60)
        print("ÉTAPE 4: DÉBIAISAGE")
        print("="*60)
        
        methods = self.config.get('debiasing_methods', ['reweighting', 'resampling'])
        results = {}
        
        for attr in self.config['protected_attrs']:
            print(f"\nDébiaisage pour l'attribut: {attr}")
            
            sensitive = sensitive_train[attr]
            unique_values = np.unique(sensitive)
            
            if len(unique_values) < 2:
                continue
            
            counts = pd.Series(sensitive).value_counts()
            privileged = counts.idxmax()
            unprivileged = counts.idxmin()
            
            attr_results = {}
            
            # Reweighting
            if 'reweighting' in methods:
                print("\n  Méthode: Reweighting")
                debias = PreprocessingDebias(attr, privileged, unprivileged)
                weights = debias.reweighting(X_train, y_train, sensitive)
                
                model_reweighted = LogisticRegression(max_iter=1000, random_state=42)
                model_reweighted.fit(X_train, y_train, sample_weight=weights)
                
                y_pred = model_reweighted.predict(self.X_test_clean)
                y_scores = model_reweighted.predict_proba(self.X_test_clean)[:, 1]
                
                # Évaluer
                fm = FairnessMetrics(attr, privileged, unprivileged)
                fairness = fm.compute_all_metrics(self.y_test, y_pred, 
                                                 self.sensitive_test[attr], y_scores)
                perf = PerformanceMetrics.compute_metrics(self.y_test, y_pred, y_scores)
                
                attr_results['reweighting'] = {
                    'fairness': fairness,
                    'performance': perf
                }
                
                print(f"    DI après reweighting: {fairness['disparate_impact']:.4f}")
                print(f"    Accuracy: {perf['accuracy']:.4f}")
            
            # Resampling
            if 'resampling' in methods:
                print("\n  Méthode: Resampling")
                debias = PreprocessingDebias(attr, privileged, unprivileged)
                X_res, y_res, s_res = debias.resampling_balance(
                    X_train, y_train, sensitive, strategy='oversample'
                )
                
                model_resampled = LogisticRegression(max_iter=1000, random_state=42)
                model_resampled.fit(X_res, y_res)
                
                y_pred = model_resampled.predict(self.X_test_clean)
                y_scores = model_resampled.predict_proba(self.X_test_clean)[:, 1]
                
                fm = FairnessMetrics(attr, privileged, unprivileged)
                fairness = fm.compute_all_metrics(self.y_test, y_pred,
                                                 self.sensitive_test[attr], y_scores)
                perf = PerformanceMetrics.compute_metrics(self.y_test, y_pred, y_scores)
                
                attr_results['resampling'] = {
                    'fairness': fairness,
                    'performance': perf
                }
                
                print(f"    DI après resampling: {fairness['disparate_impact']:.4f}")
                print(f"    Accuracy: {perf['accuracy']:.4f}")
            
            results[attr] = attr_results
        
        self.results['debiasing_results'] = results
        
        # Visualisation comparative
        if results:
            print("\nGénération des visualisations comparatives...")
            
            # Prendre le premier attribut pour la visualisation
            first_attr = list(results.keys())[0]
            
            # Préparer les métriques pour comparaison
            metrics_dict = {
                'baseline': self.results['baseline_evaluation']['fairness'][first_attr]
            }
            
            for method, result in results[first_attr].items():
                metrics_dict[method] = result['fairness']
            
            self.fairness_viz.plot_fairness_metrics_comparison(
                metrics_dict,
                save_name='debiasing_comparison.png'
            )
        
        print("✓ Débiaisage terminé")
    
    def generate_recommendations(self):
        """Génère des recommandations basées sur l'audit"""
        print("\n" + "="*60)
        print("ÉTAPE 5: RECOMMANDATIONS")
        print("="*60)
        
        recommendations = []
        
        # Analyser les résultats baseline
        for attr, fairness in self.results['baseline_evaluation']['fairness'].items():
            di = fairness.get('disparate_impact', 1.0)
            dpd = abs(fairness.get('demographic_parity_difference', 0))
            
            # Recommandations basées sur disparate impact
            if di < 0.8:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Fairness',
                    'issue': f'Disparate Impact critique pour {attr} ({di:.3f} < 0.8)',
                    'recommendation': f'Appliquer reweighting ou resampling pour équilibrer les taux de sélection entre groupes de {attr}',
                    'expected_impact': 'Amélioration significative du Disparate Impact'
                })
            
            elif di > 1.25:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'Fairness',
                    'issue': f'Disparate Impact inversé pour {attr} ({di:.3f} > 1.25)',
                    'recommendation': f'Vérifier si sur-compensation. Ajuster les méthodes de débiaisage.',
                    'expected_impact': 'Équilibrage du taux de sélection'
                })
            
            # Recommandations basées sur demographic parity
            if dpd > 0.2:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Fairness',
                    'issue': f'Grande disparité démographique pour {attr} (DPD={dpd:.3f})',
                    'recommendation': 'Considérer post-processing avec threshold optimization',
                    'expected_impact': 'Réduction de la disparité démographique'
                })
        
        # Recommandations sur les proxies
        proxies = self.results['data_analysis'].get('proxies', {})
        for attr, proxy_list in proxies.items():
            if len(proxy_list) > 0:
                top_proxies = [p['feature'] for p in proxy_list[:3]]
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'Data',
                    'issue': f'Proxies détectés pour {attr}: {", ".join(top_proxies)}',
                    'recommendation': f'Considérer la suppression ou transformation de ces features. Monitorer leur impact.',
                    'expected_impact': 'Réduction du biais indirect'
                })
        
        # Recommandations sur la qualité des données
        quality = self.results['data_analysis'].get('quality', {})
        missing = sum(quality.get('missing_values', {}).values())
        if missing > 0:
            recommendations.append({
                'priority': 'LOW',
                'category': 'Data Quality',
                'issue': f'{missing} valeurs manquantes détectées',
                'recommendation': 'Implémenter une stratégie robuste de gestion des valeurs manquantes',
                'expected_impact': 'Amélioration de la qualité des prédictions'
            })
        
        # Recommandations générales
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Governance',
            'issue': 'Absence de monitoring continu',
            'recommendation': 'Mettre en place un système de monitoring des métriques de fairness en production',
            'expected_impact': 'Détection précoce des dérives de fairness'
        })
        
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Documentation',
            'issue': 'Documentation du modèle incomplète',
            'recommendation': 'Créer une Model Card documentant les décisions de fairness et les limitations',
            'expected_impact': 'Meilleure transparence et gouvernance'
        })
        
        self.results['recommendations'] = recommendations
        
        # Afficher les recommandations
        print("\nRecommandations générées:")
        for rec in sorted(recommendations, key=lambda x: {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}[x['priority']]):
            print(f"\n[{rec['priority']}] {rec['category']}")
            print(f"  Issue: {rec['issue']}")
            print(f"  Recommandation: {rec['recommendation']}")
            print(f"  Impact attendu: {rec['expected_impact']}")
        
        print("\n✓ Recommandations générées")
    
    def generate_report(self):
        """Génère un rapport HTML complet"""
        print("\n" + "="*60)
        print("ÉTAPE 6: GÉNÉRATION DU RAPPORT")
        print("="*60)
        
        output_path = self.config.get('output_path', 'reports/audit_report.html')
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Générer le HTML
        html_content = self._generate_html_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ Rapport généré: {output_path}")
        
        # Sauvegarder aussi en JSON
        json_path = output_path.replace('.html', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            # Convertir les objets numpy en types Python natifs
            results_json = json.loads(json.dumps(self.results, default=str))
            json.dump(results_json, f, indent=2)
        
        print(f"✓ Résultats JSON: {json_path}")
    
    def _generate_html_report(self):
        """Génère le contenu HTML du rapport"""
        
        html = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport d'Audit de Biais - {datetime.now().strftime('%Y-%m-%d')}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .section {{
            background: white;
            padding: 30px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .metric {{
            display: inline-block;
            background: #f8f9fa;
            padding: 15px 25px;
            margin: 10px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }}
        .recommendation {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }}
        .recommendation.high {{
            background: #f8d7da;
            border-color: #dc3545;
        }}
        .recommendation.medium {{
            background: #fff3cd;
            border-color: #ffc107;
        }}
        .recommendation.low {{
            background: #d1ecf1;
            border-color: #17a2b8;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        .status-good {{ color: #28a745; font-weight: bold; }}
        .status-warning {{ color: #ffc107; font-weight: bold; }}
        .status-bad {{ color: #dc3545; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Rapport d'Audit de Biais</h1>
            <p>Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
            <p>Dataset: {self.config['data_path']}</p>
        </div>
        
        <div class="section">
            <h2>📈 Résumé Exécutif</h2>
            <p>Ce rapport présente les résultats de l'audit de biais effectué sur le modèle de classification.
            L'audit a évalué les métriques de fairness pour les attributs protégés: {', '.join(self.config['protected_attrs'])}.</p>
        </div>
        
        <div class="section">
            <h2>📊 Performance Baseline</h2>
            {self._format_performance_section()}
        </div>
        
        <div class="section">
            <h2>⚖️ Métriques de Fairness</h2>
            {self._format_fairness_section()}
        </div>
        
        <div class="section">
            <h2>🔧 Résultats du Débiaisage</h2>
            {self._format_debiasing_section()}
        </div>
        
        <div class="section">
            <h2>💡 Recommandations</h2>
            {self._format_recommendations_section()}
        </div>
        
        <div class="section">
            <h2>📝 Méthodologie</h2>
            <p>Cet audit a été réalisé en suivant les meilleures pratiques en matière d'évaluation de fairness:</p>
            <ul>
                <li>Calcul de métriques standards (Demographic Parity, Equalized Odds, Disparate Impact)</li>
                <li>Évaluation stratifiée par groupe</li>
                <li>Application et comparaison de techniques de débiaisage</li>
                <li>Génération de recommandations actionnables</li>
            </ul>
        </div>
    </div>
</body>
</html>
        """
        
        return html
    
    def _format_performance_section(self):
        perf = self.results['baseline_evaluation']['performance']
        
        html = '<div>'
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            if metric in perf:
                value = perf[metric]
                html += f'''
                <div class="metric">
                    <div class="metric-label">{metric.replace('_', ' ').title()}</div>
                    <div class="metric-value">{value:.4f}</div>
                </div>
                '''
        html += '</div>'
        
        return html
    
    def _format_fairness_section(self):
        fairness = self.results['baseline_evaluation']['fairness']
        
        html = ''
        for attr, metrics in fairness.items():
            html += f'<h3>Attribut: {attr}</h3>'
            html += '<table><tr><th>Métrique</th><th>Valeur</th><th>Statut</th></tr>'
            
            di = metrics.get('disparate_impact', 1.0)
            dpd = abs(metrics.get('demographic_parity_difference', 0))
            
            # Disparate Impact
            di_status = 'status-good' if 0.8 <= di <= 1.25 else 'status-bad'
            html += f'<tr><td>Disparate Impact</td><td>{di:.4f}</td><td class="{di_status}">{"✓ Acceptable" if 0.8 <= di <= 1.25 else "✗ Problématique"}</td></tr>'
            
            # Demographic Parity
            dpd_status = 'status-good' if dpd <= 0.1 else ('status-warning' if dpd <= 0.2 else 'status-bad')
            html += f'<tr><td>Demographic Parity Difference</td><td>{dpd:.4f}</td><td class="{dpd_status}">{"✓ Bon" if dpd <= 0.1 else ("⚠ Modéré" if dpd <= 0.2 else "✗ Mauvais")}</td></tr>'
            
            html += '</table>'
        
        return html
    
    def _format_debiasing_section(self):
        if not self.results['debiasing_results']:
            return '<p>Aucune technique de débiaisage appliquée.</p>'
        
        html = ''
        for attr, methods in self.results['debiasing_results'].items():
            html += f'<h3>Attribut: {attr}</h3>'
            html += '<table><tr><th>Méthode</th><th>DI Avant</th><th>DI Après</th><th>Accuracy Avant</th><th>Accuracy Après</th></tr>'
            
            baseline_di = self.results['baseline_evaluation']['fairness'][attr]['disparate_impact']
            baseline_acc = self.results['baseline_evaluation']['performance']['accuracy']
            
            for method, result in methods.items():
                di_after = result['fairness']['disparate_impact']
                acc_after = result['performance']['accuracy']
                
                html += f'''
                <tr>
                    <td>{method.title()}</td>
                    <td>{baseline_di:.4f}</td>
                    <td>{di_after:.4f}</td>
                    <td>{baseline_acc:.4f}</td>
                    <td>{acc_after:.4f}</td>
                </tr>
                '''
            
            html += '</table>'
        
        return html
    
    def _format_recommendations_section(self):
        recommendations = self.results['recommendations']
        
        html = ''
        for rec in sorted(recommendations, key=lambda x: {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}[x['priority']]):
            priority_class = rec['priority'].lower()
            html += f'''
            <div class="recommendation {priority_class}">
                <strong>[{rec['priority']}] {rec['category']}</strong><br>
                <em>Issue:</em> {rec['issue']}<br>
                <em>Recommandation:</em> {rec['recommendation']}<br>
                <em>Impact attendu:</em> {rec['expected_impact']}
            </div>
            '''
        
        return html
    
    def run(self):
        """Exécute l'audit complet"""
        print("\n" + "="*60)
        print("DÉMARRAGE DE L'AUDIT DE BIAIS")
        print("="*60)
        
        try:
            # 1. Charger les données
            df = self.load_data()
            
            # 2. Charger/entraîner le modèle
            X_train, X_test, y_train, y_test = self.load_or_train_model(df)
            
            # 3. Évaluation baseline
            self.evaluate_baseline()
            
            # 4. Débiaisage
            # Préparer les données d'entraînement pour le débiaisage
            df_encoded = self.processor.encode_categorical(df)
            X_train_full, _, y_train_full, _ = self.processor.split_data(df_encoded)
            
            sensitive_train = {}
            for attr in self.config['protected_attrs']:
                sensitive_train[attr] = X_train_full[attr]
            
            # Retirer les attributs protégés
            features_to_drop = self.config['protected_attrs'] + [self.config['label_name']]
            X_train_clean = X_train_full.drop(columns=[col for col in features_to_drop if col in X_train_full.columns])
            
            self.apply_debiasing(X_train_clean, y_train_full, sensitive_train)
            
            # 5. Recommandations
            self.generate_recommendations()
            
            # 6. Rapport
            self.generate_report()
            
            print("\n" + "="*60)
            print("✓ AUDIT TERMINÉ AVEC SUCCÈS")
            print("="*60)
            print(f"\nRapport disponible: {self.config.get('output_path', 'reports/audit_report.html')}")
            
        except Exception as e:
            print(f"\n✗ ERREUR: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def parse_arguments():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(
        description='Audit de biais pour modèles de classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  
  # Audit avec données et modèle existant
  python -m src.audit_main --data data/dataset.csv --model models/model.pkl \\
                           --protected-attrs gender race --label label
  
  # Audit avec entraînement de nouveau modèle
  python -m src.audit_main --data data/dataset.csv --protected-attrs gender \\
                           --label outcome --debiasing reweighting resampling
        """
    )
    
    parser.add_argument('--data', required=True, help='Chemin vers le fichier de données (CSV)')
    parser.add_argument('--model', help='Chemin vers le modèle (pickle). Si absent, entraîne un nouveau modèle')
    parser.add_argument('--protected-attrs', nargs='+', required=True, 
                       help='Noms des attributs protégés')
    parser.add_argument('--label', required=True, help='Nom de la colonne cible')
    parser.add_argument('--debiasing', nargs='*', 
                       choices=['reweighting', 'resampling'],
                       default=['reweighting', 'resampling'],
                       help='Techniques de débiaisage compatibles Python 3.14 à appliquer')
    parser.add_argument('--output', default='reports/audit_report.html',
                       help='Chemin du rapport de sortie')
    parser.add_argument('--output-dir', default='reports/figures',
                       help='Répertoire pour les figures')
    
    return parser.parse_args()


def main():
    """Point d'entrée principal"""
    args = parse_arguments()
    
    # Configuration
    config = {
        'data_path': args.data,
        'model_path': args.model,
        'protected_attrs': args.protected_attrs,
        'label_name': args.label,
        'debiasing_methods': args.debiasing,
        'output_path': args.output,
        'output_dir': args.output_dir
    }
    
    # Créer et exécuter l'auditeur
    auditor = BiasAuditor(config)
    auditor.run()


if __name__ == "__main__":
    main()
