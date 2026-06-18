"""
Module de visualisation pour l'audit de biais.
Génère des graphiques professionnels pour l'analyse et les rapports.
"""

import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configuration globale
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
COLORS = {'primary': '#2E86AB', 'secondary': '#A23B72', 'accent': '#F18F01', 
          'success': '#06A77D', 'warning': '#F77F00', 'danger': '#D62828'}


class FairnessVisualizer:
    """Visualisations pour les métriques de fairness"""
    
    def __init__(self, save_dir='reports/figures'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_demographic_distribution(self, df: pd.DataFrame, 
                                     protected_attrs: List[str],
                                     label_name: str,
                                     save_name: str = 'demographic_dist.png'):
        """Visualise la distribution démographique"""
        
        n_attrs = len(protected_attrs)
        fig, axes = plt.subplots(1, n_attrs, figsize=(6*n_attrs, 5))
        
        if n_attrs == 1:
            axes = [axes]
        
        for idx, attr in enumerate(protected_attrs):
            # Distribution de l'attribut protégé
            ax1 = axes[idx]
            
            # Créer un tableau croisé
            cross_tab = pd.crosstab(df[attr], df[label_name], normalize='index') * 100
            
            cross_tab.plot(kind='bar', ax=ax1, color=[COLORS['primary'], COLORS['secondary']])
            ax1.set_title(f'Distribution des labels par {attr}', fontsize=12, fontweight='bold')
            ax1.set_xlabel(attr, fontsize=11)
            ax1.set_ylabel('Pourcentage (%)', fontsize=11)
            ax1.legend(title=label_name, labels=['Négatif', 'Positif'])
            ax1.tick_params(axis='x', rotation=45)
            
            # Ajouter les valeurs sur les barres
            for container in ax1.containers:
                ax1.bar_label(container, fmt='%.1f%%', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return fig
    
    def plot_fairness_metrics_comparison(self, metrics_dict: Dict[str, Dict],
                                        save_name: str = 'fairness_comparison.png'):
        """
        Compare les métriques de fairness entre différents modèles/configurations.
        
        Args:
            metrics_dict: {'baseline': {...}, 'debiased': {...}}
        """
        
        # Sélectionner les métriques clés
        key_metrics = [
            'demographic_parity_difference',
            'disparate_impact',
            'average_odds_difference',
            'equal_opportunity_difference'
        ]
        
        # Préparer les données
        data = []
        for model_name, metrics in metrics_dict.items():
            for metric_name in key_metrics:
                if metric_name in metrics:
                    data.append({
                        'Model': model_name,
                        'Metric': metric_name.replace('_', ' ').title(),
                        'Value': metrics[metric_name]
                    })
        
        df_plot = pd.DataFrame(data)
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Grouper par métrique
        metrics_unique = df_plot['Metric'].unique()
        x = np.arange(len(metrics_unique))
        width = 0.8 / len(metrics_dict)
        
        for idx, model_name in enumerate(metrics_dict.keys()):
            model_data = df_plot[df_plot['Model'] == model_name]
            values = [model_data[model_data['Metric'] == m]['Value'].values[0] 
                     if len(model_data[model_data['Metric'] == m]) > 0 else 0
                     for m in metrics_unique]
            
            offset = (idx - len(metrics_dict)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model_name, alpha=0.8)
            
            # Ajouter les valeurs sur les barres
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        # Zone acceptable (approximative)
        ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.3, label='Seuil acceptable (±0.1)')
        ax.axhline(y=-0.1, color='green', linestyle='--', alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        ax.set_xlabel('Métrique de Fairness', fontsize=12, fontweight='bold')
        ax.set_ylabel('Valeur', fontsize=12, fontweight='bold')
        ax.set_title('Comparaison des Métriques de Fairness', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_unique, rotation=15, ha='right')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return fig
    
    def plot_confusion_matrices_by_group(self, y_true: np.ndarray,
                                         y_pred: np.ndarray,
                                         sensitive_features: np.ndarray,
                                         group_names: Dict,
                                         save_name: str = 'confusion_matrices.png'):
        """Affiche les matrices de confusion par groupe"""
        
        from sklearn.metrics import confusion_matrix
        
        groups = np.unique(sensitive_features)
        n_groups = len(groups)
        
        fig, axes = plt.subplots(1, n_groups, figsize=(6*n_groups, 5))
        
        if n_groups == 1:
            axes = [axes]
        
        for idx, group in enumerate(groups):
            mask = sensitive_features == group
            cm = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1])
            
            # Normaliser
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            # Plot
            sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues', 
                       ax=axes[idx], cbar_kws={'label': 'Pourcentage (%)'},
                       vmin=0, vmax=100)
            
            group_name = group_names.get(group, str(group))
            axes[idx].set_title(f'Groupe: {group_name}\n(n={mask.sum()})', 
                              fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Prédiction', fontsize=11)
            axes[idx].set_ylabel('Vérité', fontsize=11)
            axes[idx].set_xticklabels(['Négatif', 'Positif'])
            axes[idx].set_yticklabels(['Négatif', 'Positif'], rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return fig
    
    def plot_roc_curves_by_group(self, y_true: np.ndarray,
                                y_scores: np.ndarray,
                                sensitive_features: np.ndarray,
                                group_names: Dict,
                                save_name: str = 'roc_curves.png'):
        """Courbes ROC par groupe"""
        
        from sklearn.metrics import roc_curve, auc
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        groups = np.unique(sensitive_features)
        colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
        
        for idx, group in enumerate(groups):
            mask = sensitive_features == group
            
            fpr, tpr, _ = roc_curve(y_true[mask], y_scores[mask])
            roc_auc = auc(fpr, tpr)
            
            group_name = group_names.get(group, str(group))
            ax.plot(fpr, tpr, color=colors[idx], lw=2, 
                   label=f'{group_name} (AUC = {roc_auc:.3f}, n={mask.sum()})')
        
        # Ligne de référence
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Hasard (AUC = 0.5)')
        
        ax.set_xlabel('Taux de Faux Positifs', fontsize=12)
        ax.set_ylabel('Taux de Vrais Positifs', fontsize=12)
        ax.set_title('Courbes ROC par Groupe', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return fig
    
    def plot_disparate_impact_radar(self, metrics_dict: Dict[str, Dict],
                                    save_name: str = 'disparate_impact_radar.png'):
        """Graphique radar pour le disparate impact"""
        
        fig = go.Figure()
        
        metrics_to_plot = [
            'demographic_parity_ratio',
            'disparate_impact',
            'equal_opportunity_difference',
            'predictive_parity_difference'
        ]
        
        for model_name, metrics in metrics_dict.items():
            values = []
            labels = []
            
            for metric in metrics_to_plot:
                if metric in metrics:
                    # Transformer pour que 1.0 soit "parfait"
                    if 'ratio' in metric or 'impact' in metric:
                        # Ratio: 1.0 est parfait
                        val = abs(1.0 - abs(metrics[metric]))
                    else:
                        # Différence: 0 est parfait
                        val = 1.0 - abs(metrics[metric])
                    
                    values.append(max(0, min(1, val)))  # Clip entre 0 et 1
                    labels.append(metric.replace('_', ' ').title())
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=labels,
                fill='toself',
                name=model_name.title(),
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Radar de Fairness (1.0 = Parfait)",
            title_font_size=16
        )
        
        fig.write_html(self.save_dir / save_name.replace(".png", ".html"))
        fig.show()
        
        return fig
    
    def plot_intersectional_analysis(self, df: pd.DataFrame,
                                     attr1: str, attr2: str,
                                     label_name: str,
                                     save_name: str = 'intersectional.png'):
        """Analyse intersectionnelle (2 attributs protégés)"""
        
        # Créer une colonne combinée
        df['combined'] = df[attr1].astype(str) + ' & ' + df[attr2].astype(str)
        
        # Calculer les taux de labels positifs
        positive_rates = df.groupby('combined')[label_name].mean() * 100
        sample_sizes = df.groupby('combined').size()
        
        # Créer le DataFrame pour plotting
        plot_df = pd.DataFrame({
            'Group': positive_rates.index,
            'Positive_Rate': positive_rates.values,
            'Sample_Size': sample_sizes.values
        })
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(14, 6))
        
        bars = ax.bar(range(len(plot_df)), plot_df['Positive_Rate'], 
                     color=COLORS['primary'], alpha=0.7, edgecolor='black')
        
        # Colorer différemment selon le taux
        global_rate = df[label_name].mean() * 100
        for idx, (bar, rate) in enumerate(zip(bars, plot_df['Positive_Rate'])):
            if rate > global_rate * 1.2:
                bar.set_color(COLORS['success'])
            elif rate < global_rate * 0.8:
                bar.set_color(COLORS['danger'])
        
        # Ajouter les valeurs et tailles d'échantillon
        for idx, (rate, size) in enumerate(zip(plot_df['Positive_Rate'], plot_df['Sample_Size'])):
            ax.text(idx, rate, f'{rate:.1f}%\n(n={size})', 
                   ha='center', va='bottom', fontsize=9)
        
        # Ligne de référence (taux global)
        ax.axhline(y=global_rate, color='red', linestyle='--', 
                  label=f'Taux global: {global_rate:.1f}%', linewidth=2)
        
        ax.set_xlabel('Sous-groupe Intersectionnel', fontsize=12, fontweight='bold')
        ax.set_ylabel('Taux de Labels Positifs (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Analyse Intersectionnelle: {attr1} × {attr2}', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels(plot_df['Group'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return fig
    
    def plot_performance_fairness_tradeoff(self, results: List[Dict],
                                          fairness_metric: str = 'disparate_impact',
                                          performance_metric: str = 'accuracy',
                                          save_name: str = 'pareto_tradeoff.png'):
        """Graphique du trade-off performance vs fairness (Pareto)"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        fairness_values = []
        performance_values = []
        labels = []
        
        for result in results:
            model_name = result.get('name', 'Unknown')
            fairness = result.get('fairness', {}).get(fairness_metric, 0)
            performance = result.get('performance', {}).get(performance_metric, 0)
            
            # Transformer fairness pour que "plus haut = mieux"
            if 'ratio' in fairness_metric or 'impact' in fairness_metric:
                # Pour disparate impact, 1.0 est parfait
                fairness_score = 1.0 - abs(1.0 - fairness)
            else:
                # Pour les différences, 0 est parfait
                fairness_score = 1.0 - abs(fairness)
            
            fairness_values.append(fairness_score)
            performance_values.append(performance)
            labels.append(model_name)
        
        # Scatter plot
        scatter = ax.scatter(fairness_values, performance_values, 
                           s=200, alpha=0.6, c=range(len(labels)), 
                           cmap='viridis', edgecolors='black', linewidth=2)
        
        # Ajouter les labels
        for i, label in enumerate(labels):
            ax.annotate(label, (fairness_values[i], performance_values[i]),
                       xytext=(10, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
        
        # Frontière de Pareto approximative
        from scipy.spatial import ConvexHull
        if len(fairness_values) >= 3:
            points = np.column_stack([fairness_values, performance_values])
            try:
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    ax.plot(points[simplex, 0], points[simplex, 1], 'r--', alpha=0.3)
            except:
                pass
        
        ax.set_xlabel(f'Fairness Score ({fairness_metric})', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Performance ({performance_metric})', fontsize=12, fontweight='bold')
        ax.set_title('Trade-off Performance vs Fairness (Pareto Frontier)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Zone optimale (haut-droite)
        ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.3, label='Zone optimale')
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return fig


class PerformanceVisualizer:
    """Visualisations pour les métriques de performance"""
    
    def __init__(self, save_dir='reports/figures'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_performance_comparison(self, perf_dict: Dict[str, Dict],
                                   save_name: str = 'performance_comparison.png'):
        """Compare les performances entre modèles"""
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(metrics))
        width = 0.8 / len(perf_dict)
        
        for idx, (model_name, perf) in enumerate(perf_dict.items()):
            values = [perf.get(m, 0) for m in metrics]
            offset = (idx - len(perf_dict)/2 + 0.5) * width
            
            bars = ax.bar(x + offset, values, width, label=model_name, alpha=0.8)
            
            # Ajouter les valeurs
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Métrique', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Comparaison des Performances', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return fig
    
    def plot_stratified_performance(self, stratified_metrics: Dict,
                                   metric_name: str = 'f1_score',
                                   save_name: str = 'stratified_performance.png'):
        """Performance stratifiée par groupe"""
        
        groups = list(stratified_metrics.keys())
        values = [stratified_metrics[g][metric_name] for g in groups]
        sizes = [stratified_metrics[g]['sample_size'] for g in groups]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(groups, values, color=COLORS['primary'], alpha=0.7, edgecolor='black')
        
        # Colorer selon la performance
        mean_value = np.mean(values)
        for bar, val in zip(bars, values):
            if val < mean_value * 0.9:
                bar.set_color(COLORS['danger'])
            elif val > mean_value * 1.1:
                bar.set_color(COLORS['success'])
        
        # Ajouter valeurs et tailles
        for idx, (val, size) in enumerate(zip(values, sizes)):
            ax.text(idx, val, f'{val:.3f}\n(n={size})', 
                   ha='center', va='bottom', fontsize=10)
        
        ax.axhline(y=mean_value, color='red', linestyle='--', 
                  label=f'Moyenne: {mean_value:.3f}', linewidth=2)
        
        ax.set_xlabel('Groupe', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_title(f'Performance Stratifiée: {metric_name.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return fig


if __name__ == "__main__":
    print("Test du module de visualisation...")
    
    # Créer des données de test
    from src.data_processing import generate_sample_data
    
    df = generate_sample_data(n_samples=5000)
    
    viz = FairnessVisualizer()
    
    # Test 1: Distribution démographique
    print("\n=== Test Distribution Démographique ===")
    viz.plot_demographic_distribution(df, ['gender', 'race'], 'label')
    
    # Test 2: Métriques de fairness
    print("\n=== Test Comparaison Fairness ===")
    metrics_dict = {
        'baseline': {
            'demographic_parity_difference': 0.25,
            'disparate_impact': 0.65,
            'average_odds_difference': 0.18,
            'equal_opportunity_difference': 0.20
        },
        'debiased': {
            'demographic_parity_difference': 0.08,
            'disparate_impact': 0.92,
            'average_odds_difference': 0.05,
            'equal_opportunity_difference': 0.06
        }
    }
    viz.plot_fairness_metrics_comparison(metrics_dict)
    
    print("\n✓ Module de visualisation testé avec succès!")
