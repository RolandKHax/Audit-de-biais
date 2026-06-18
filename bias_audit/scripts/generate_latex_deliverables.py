"""Génère le rapport LaTeX et les slides Beamer à partir des résultats d'audit."""

from __future__ import annotations

import json
from pathlib import Path


def load_results(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Résultats introuvables: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value, digits: int = 3) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def tex_escape(value) -> str:
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def baseline_table(data: dict, limit: int = 12) -> str:
    rows = data["baseline_evaluation"].get("aggregate_summary", [])
    if not rows:
        rows = data["baseline_evaluation"].get("summary", [])
    body = []
    for row in rows[:limit]:
        body.append(
            " & ".join([
                tex_escape(row.get("feature_policy", "")),
                tex_escape(row.get("model", "")),
                fmt(row.get("accuracy_mean", row.get("accuracy", 0))),
                fmt(row.get("f1_score_mean", row.get("f1_score", 0))),
                fmt(row.get("roc_auc_mean", row.get("roc_auc", 0))),
            ]) + r" \\"
        )
    return "\n".join(body)


def fairness_table(data: dict) -> str:
    rows = []
    for attr, metrics in data["baseline_evaluation"]["fairness"].items():
        for group, values in metrics["group_metrics"].items():
            rows.append(
                " & ".join([
                    tex_escape(attr),
                    tex_escape(group),
                    str(values["sample_size"]),
                    fmt(values["base_rate"]),
                    fmt(values["selection_rate"]),
                    fmt(values["tpr"]),
                    fmt(values["fpr"]),
                    fmt(values["precision"]),
                ]) + r" \\"
            )
    return "\n".join(rows)


def mitigation_table(data: dict) -> str:
    rows = []
    for attr, methods in data.get("debiasing_results", {}).items():
        for method, result in methods.items():
            if method == "adversarial_pytorch":
                for lambda_value, lambda_result in result.items():
                    if isinstance(lambda_result, dict) and "fairness" in lambda_result:
                        rows.append(_mitigation_row(attr, f"adversarial lambda={lambda_value}", lambda_result))
                continue
            if isinstance(result, dict) and "fairness" in result:
                rows.append(_mitigation_row(attr, method, result))
            else:
                rows.append(
                    f"{tex_escape(attr)} & {tex_escape(method)} & -- & -- & -- & -- \\\\"
                )
    return "\n".join(rows)


def _mitigation_row(attr: str, method: str, result: dict) -> str:
    fairness = result["fairness"][attr]
    perf = result["performance"]
    return " & ".join([
        tex_escape(attr),
        tex_escape(method),
        fmt(perf["accuracy"]),
        fmt(perf["f1_score"]),
        fmt(fairness["demographic_parity_difference"]),
        fmt(fairness["equalized_odds_difference"]),
    ]) + r" \\"


def recommendations(data: dict) -> str:
    items = []
    for rec in data.get("recommendations", []):
        items.append(
            r"\item \textbf{" + tex_escape(rec["priority"]) + " -- " +
            tex_escape(rec["category"]) + r"} : " +
            tex_escape(rec["recommendation"])
        )
    return "\n".join(items)


def build_report(data: dict) -> str:
    shape = data["data_analysis"]["shape"]
    primary = data["baseline_evaluation"]["primary"]
    perf = data["baseline_evaluation"]["performance"]
    rows_count = shape["rows"]
    columns_count = shape["columns"]
    primary_model = tex_escape(primary["model"])
    primary_policy = tex_escape(primary["policy"])
    primary_seed = primary["seed"]
    primary_accuracy = fmt(perf["accuracy"])
    primary_f1 = fmt(perf["f1_score"])
    primary_auc = fmt(perf.get("roc_auc", 0))

    return rf"""\documentclass[12pt,a4paper]{{report}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage[french]{{babel}}
\usepackage{{geometry}}
\geometry{{a4paper, top=2.5cm, bottom=2.5cm, left=3cm, right=2.5cm, headheight=15pt}}
\usepackage{{graphicx}}
\usepackage{{fancyhdr}}
\usepackage{{titlesec}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{float}}
\usepackage[most]{{tcolorbox}}
\usepackage{{amsmath,amssymb}}

\definecolor{{primarycolor}}{{RGB}}{{37,99,235}}
\definecolor{{primarydark}}{{RGB}}{{29,78,186}}
\definecolor{{primarylight}}{{RGB}}{{219,234,254}}
\definecolor{{secondarycolor}}{{RGB}}{{100,116,139}}
\definecolor{{warningcolor}}{{RGB}}{{245,158,11}}
\definecolor{{successcolor}}{{RGB}}{{16,185,129}}
\definecolor{{dangercolor}}{{RGB}}{{239,68,68}}

\hypersetup{{colorlinks=true, linkcolor=primarycolor, urlcolor=primarydark, citecolor=primarycolor,
pdftitle={{Audit de biais COMPAS}}, pdfauthor={{A renseigner}}}}

\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{\small\color{{secondarycolor}}\leftmark}}
\fancyhead[R]{{\small\color{{secondarycolor}}ENSA Béni Mellal}}
\fancyfoot[C]{{\color{{secondarycolor}}\small\thepage}}
\fancyfoot[R]{{\color{{secondarycolor}}\small Audit de biais}}

\titleformat{{\chapter}}[display]{{\normalfont\huge\bfseries}}{{\color{{primarycolor}}\chaptertitlename\ \thechapter}}{{16pt}}{{\color{{primarydark}}\Huge\bfseries}}
\titleformat{{\section}}{{\normalfont\Large\bfseries\color{{primarycolor}}}}{{\thesection}}{{1em}}{{}}[\color{{primarycolor!40}}\titlerule]
\titleformat{{\subsection}}{{\normalfont\large\bfseries\color{{primarydark}}}}{{\thesubsection}}{{1em}}{{}}

\tcbset{{mybox/.style={{enhanced, arc=4pt, boxrule=0.8pt, left=8pt, right=8pt, top=6pt, bottom=6pt}},
summarybox/.style={{mybox, colback=primarylight, colframe=primarycolor}},
warningbox/.style={{mybox, colback=warningcolor!15, colframe=warningcolor}},
successbox/.style={{mybox, colback=successcolor!12, colframe=successcolor}}}}
\newtcolorbox{{warningbox}}{{warningbox}}
\newtcolorbox{{summaryinfobox}}{{summarybox}}

\begin{{document}}
\begin{{titlepage}}
\thispagestyle{{empty}}
\begin{{center}}
\IfFileExists{{ensabm.png}}{{\includegraphics[height=3cm]{{ensabm.png}}}}{{\fbox{{\Large Logo ENSA à renseigner}}}}
\vspace{{2.5cm}}
\begin{{tcolorbox}}[summarybox]
\centering
{{\large MINI-PROJET -- 2025/2026}}\\[8pt]
{{\Huge\bfseries Audit de biais d'un modèle de classification}}\\[8pt]
{{\Large Étude du dataset COMPAS ProPublica}}\\[12pt]
Module : Deep Learning / Fairness algorithmique
\end{{tcolorbox}}
\vspace{{2cm}}
\begin{{tabular}}{{ll}}
\textbf{{Réalisé par :}} & À renseigner \\
\textbf{{Encadré par :}} & À renseigner \\
\textbf{{Filière :}} & Intelligence Artificielle et Cybersécurité \\
\textbf{{Année universitaire :}} & 2025--2026
\end{{tabular}}
\end{{center}}
\end{{titlepage}}

\chapter*{{Résumé exécutif}}
\addcontentsline{{toc}}{{chapter}}{{Résumé exécutif}}
\begin{{summaryinfobox}}
Ce rapport audite un modèle de classification binaire appliqué au dataset COMPAS de ProPublica. L'objectif est de mesurer et d'atténuer les disparités liées à l'origine (\texttt{{race}}) et au genre (\texttt{{sex}}) avec des métriques de fairness : demographic parity, disparate impact, equalized odds, TPR/FPR et selection rate par groupe.
\end{{summaryinfobox}}

Le dataset traité contient \textbf{{{rows_count} observations}} et \textbf{{{columns_count} variables}}. La baseline principale est \texttt{{{primary_model}}}, politique \texttt{{{primary_policy}}}, seed {primary_seed}. Ses scores sont : accuracy={primary_accuracy}, F1={primary_f1}, ROC-AUC={primary_auc}.

\tableofcontents
\listoffigures
\listoftables

\chapter{{Introduction}}
Les modèles de classification utilisés dans des contextes sensibles peuvent être performants globalement tout en produisant des erreurs inégalement distribuées entre groupes. Dans COMPAS, la décision étudiée est la prédiction de récidive à deux ans. Les faux positifs sont particulièrement critiques : ils peuvent correspondre à des personnes prédites à risque alors qu'elles ne récidivent pas.

\chapter{{État de l'art}}
\section{{Fairness algorithmique}}
La fairness n'a pas une définition unique. La parité démographique impose des taux de sélection comparables, tandis qu'equalized odds exige des taux d'erreur comparables, notamment TPR et FPR.
\[
P(\hat{{Y}}=1 \mid A=a) = P(\hat{{Y}}=1 \mid A=b)
\]
\[
TPR_a \approx TPR_b \quad \text{{et}} \quad FPR_a \approx FPR_b
\]

\section{{Mitigation}}
Le projet compare des méthodes de pré-processing (reweighting, resampling), d'in-processing (Fairlearn reductions, adversarial debiasing PyTorch) et de post-processing (seuils par groupe).

\chapter{{Données COMPAS}}
Le dataset principal est \texttt{{compas-scores-two-years.csv}}. Les variables centrales sont \texttt{{two\_year\_recid}} pour la cible, \texttt{{race}} et \texttt{{sex}} pour l'audit, ainsi que \texttt{{age}}, \texttt{{priors\_count}}, \texttt{{c\_charge\_degree}} et les scores COMPAS.

\begin{{warningbox}}
COMPAS est un cas réel à fort impact humain. Le rapport ne prétend pas que le label soit une vérité neutre : il peut contenir des biais de mesure, de collecte ou de procédure judiciaire.
\end{{warningbox}}

\chapter{{Analyse exploratoire orientée biais}}
\IfFileExists{{../figures/eda_demographics.png}}{{\begin{{figure}}[H]\centering\includegraphics[width=.9\textwidth]{{../figures/eda_demographics.png}}\caption{{Distribution des labels par attribut sensible}}\end{{figure}}}}{{}}

\chapter{{Méthodologie}}
Le protocole utilise un split train/validation/test stratifié par label et attributs sensibles quand les effectifs le permettent. Les expériences sont répétées avec les seeds 0, 1, 2, 3 et 4. Les variables sensibles sont conservées pour l'audit, même lorsqu'elles sont retirées des features du modèle.

\chapter{{Résultats baseline}}
\begin{{longtable}}{{llrrr}}
\toprule
Politique & Modèle & Accuracy & F1 & ROC-AUC \\
\midrule
{baseline_table(data)}
\bottomrule
\caption{{Résumé des baselines, moyenne multi-seeds quand disponible}}
\end{{longtable}}

\IfFileExists{{../figures/baseline_confusion_matrices.png}}{{\begin{{figure}}[H]\centering\includegraphics[width=.95\textwidth]{{../figures/baseline_confusion_matrices.png}}\caption{{Matrices de confusion par groupe}}\end{{figure}}}}{{}}
\IfFileExists{{../figures/baseline_roc_curves.png}}{{\begin{{figure}}[H]\centering\includegraphics[width=.8\textwidth]{{../figures/baseline_roc_curves.png}}\caption{{Courbes ROC par groupe}}\end{{figure}}}}{{}}

\chapter{{Métriques par groupe}}
\begin{{longtable}}{{llrrrrrr}}
\toprule
Attribut & Groupe & n & Base rate & Selection & TPR & FPR & Precision \\
\midrule
{fairness_table(data)}
\bottomrule
\caption{{Métriques de fairness par groupe}}
\end{{longtable}}

\chapter{{Atténuation des biais}}
\begin{{longtable}}{{llrrrr}}
\toprule
Attribut & Méthode & Accuracy & F1 & DP diff & EO diff \\
\midrule
{mitigation_table(data)}
\bottomrule
\caption{{Comparaison des méthodes de mitigation}}
\end{{longtable}}

\IfFileExists{{../figures/debiasing_comparison.png}}{{\begin{{figure}}[H]\centering\includegraphics[width=.9\textwidth]{{../figures/debiasing_comparison.png}}\caption{{Comparaison fairness avant/après mitigation}}\end{{figure}}}}{{}}

\chapter{{Discussion}}
\section{{Compromis performance/fairness}}
Les résultats montrent que l'accuracy globale ne suffit pas. Les écarts de FPR, TPR et selection rate doivent être analysés groupe par groupe. Equalized odds est privilégiée pour COMPAS car elle explicite les erreurs potentiellement lourdes dans un contexte judiciaire.

\section{{Limites}}
Les limites principales sont la nature observationnelle du dataset, les critiques méthodologiques sur la construction des labels COMPAS, la sensibilité juridique des seuils par groupe et l'impossibilité de satisfaire simultanément toutes les définitions de fairness.

\chapter{{Recommandations}}
\begin{{itemize}}
{recommendations(data)}
\end{{itemize}}

\chapter{{Conclusion}}
Le projet fournit un audit reproductible de COMPAS, compare plusieurs modèles, mesure les biais par groupe et teste plusieurs familles de mitigation. La recommandation principale est de ne jamais déployer un modèle sensible sur la seule base de l'accuracy globale.

\appendix
\chapter{{Annexes techniques}}
Les scripts principaux sont \texttt{{scripts/download\_compas.py}}, \texttt{{scripts/train\_baseline.py}}, \texttt{{scripts/evaluate\_fairness.py}}, \texttt{{scripts/train\_adversarial.py}} et \texttt{{scripts/run\_all.sh}}.

\begin{{thebibliography}}{{9}}
\bibitem{{propublica}} ProPublica, \emph{{Machine Bias / COMPAS analysis dataset}}, 2016.
\bibitem{{hardt}} Hardt, Price, Srebro, \emph{{Equality of Opportunity in Supervised Learning}}, NeurIPS 2016.
\bibitem{{kamiran}} Kamiran, Calders, \emph{{Data preprocessing techniques for classification without discrimination}}, Knowledge and Information Systems, 2012.
\bibitem{{zhang}} Zhang, Lemoine, Mitchell, \emph{{Mitigating Unwanted Biases with Adversarial Learning}}, AIES 2018.
\bibitem{{nist}} NIST, \emph{{AI Risk Management Framework}}, 2023.
\end{{thebibliography}}
\end{{document}}
"""


def build_slides(data: dict) -> str:
    perf = data["baseline_evaluation"]["performance"]
    primary = data["baseline_evaluation"]["primary"]
    race = data["baseline_evaluation"]["fairness"].get("race", {})
    sex = data["baseline_evaluation"]["fairness"].get("sex", {})
    race_eo = race.get("equalized_odds_difference", 0)
    sex_eo = sex.get("equalized_odds_difference", 0)

    return rf"""\documentclass[aspectratio=169,11pt]{{beamer}}
\usepackage[utf8]{{inputenc}}
\usepackage[french]{{babel}}
\usepackage[T1]{{fontenc}}
\usepackage{{graphicx}}
\usepackage{{tikz}}
\usetikzlibrary{{shapes,arrows,positioning,shadows,calc}}
\usepackage{{booktabs}}
\usepackage{{xcolor}}
\usetheme{{Madrid}}
\definecolor{{primarycolor}}{{RGB}}{{37,99,235}}
\definecolor{{secondarycolor}}{{RGB}}{{71,85,105}}
\definecolor{{successcolor}}{{RGB}}{{16,185,129}}
\definecolor{{warningcolor}}{{RGB}}{{245,158,11}}
\definecolor{{dangercolor}}{{RGB}}{{239,68,68}}
\setbeamercolor{{palette primary}}{{bg=primarycolor,fg=white}}
\setbeamercolor{{structure}}{{fg=primarycolor}}
\setbeamercolor{{frametitle}}{{bg=primarycolor,fg=white}}
\setbeamertemplate{{navigation symbols}}{{}}
\setbeamertemplate{{footline}}[frame number]

\title[Audit de biais COMPAS]{{\LARGE\bfseries Audit de biais d'un modèle de classification}}
\subtitle{{Étude du dataset COMPAS ProPublica}}
\institute{{Filière Intelligence Artificielle et Cybersécurité}}
\author{{\textbf{{Réalisé par :}} À renseigner\\\textbf{{Encadré par :}} À renseigner}}
\date{{\today}}
\titlegraphic{{\IfFileExists{{ensabm.png}}{{\includegraphics[height=1.5cm]{{ensabm.png}}}}{{}}}}

\begin{{document}}
\begin{{frame}}[plain]\titlepage\end{{frame}}
\begin{{frame}}{{Sommaire}}\tableofcontents\end{{frame}}

\section{{Contexte}}
\begin{{frame}}{{Problématique}}
\begin{{block}}{{Question centrale}}
Le modèle prend-il des décisions équitables entre groupes sensibles, et comment réduire les disparités sans détruire la performance ?
\end{{block}}
\begin{{itemize}}
\item Cas réel : prédiction de récidive à deux ans.
\item Attributs audités : \texttt{{race}} et \texttt{{sex}}.
\item Risque majeur : faux positifs différenciés selon les groupes.
\end{{itemize}}
\end{{frame}}

\begin{{frame}}{{Dataset COMPAS}}
\begin{{itemize}}
\item Source : ProPublica, \texttt{{compas-scores-two-years.csv}}.
\item Cible : \texttt{{two\_year\_recid}}.
\item Variables : âge, antécédents, type d'accusation, score COMPAS.
\item Limite : les labels judiciaires ne sont pas une vérité sociale neutre.
\end{{itemize}}
\end{{frame}}

\section{{Méthodologie}}
\begin{{frame}}{{Pipeline expérimental}}
\begin{{enumerate}}
\item Nettoyage COMPAS et EDA orientée biais.
\item Split train/validation/test stratifié.
\item Baselines : Logistic Regression, Random Forest, MLP.
\item Audit : DP, DI, EO, TPR/FPR, selection rate.
\item Mitigation : reweighting, resampling, Fairlearn, PyTorch adversarial, seuils.
\end{{enumerate}}
\end{{frame}}

\begin{{frame}}{{Pourquoi garder les attributs sensibles ?}}
\begin{{alertblock}}{{Point critique}}
Retirer \texttt{{race}} ou \texttt{{sex}} des features ne prouve pas l'absence de biais : d'autres variables peuvent agir comme proxys.
\end{{alertblock}}
\end{{frame}}

\section{{Résultats}}
\begin{{frame}}{{Performance baseline}}
\begin{{center}}
\begin{{tabular}}{{lr}}
\toprule
Métrique & Valeur \\
\midrule
Accuracy & {fmt(perf["accuracy"])} \\
F1-score & {fmt(perf["f1_score"])} \\
ROC-AUC & {fmt(perf.get("roc_auc", 0))} \\
Baseline & \texttt{{{tex_escape(primary["model"])}}} \\
\bottomrule
\end{{tabular}}
\end{{center}}
\end{{frame}}

\begin{{frame}}{{Fairness baseline}}
\begin{{center}}
\begin{{tabular}}{{lrr}}
\toprule
Attribut & EO diff & DP diff \\
\midrule
race & {fmt(race_eo)} & {fmt(race.get("demographic_parity_difference", 0))} \\
sex & {fmt(sex_eo)} & {fmt(sex.get("demographic_parity_difference", 0))} \\
\bottomrule
\end{{tabular}}
\end{{center}}
\end{{frame}}

\begin{{frame}}{{Visualisations}}
\IfFileExists{{../reports/figures/eda_demographics.png}}{{\includegraphics[width=.48\textwidth]{{../reports/figures/eda_demographics.png}}}}{{}}
\IfFileExists{{../reports/figures/debiasing_comparison.png}}{{\includegraphics[width=.48\textwidth]{{../reports/figures/debiasing_comparison.png}}}}{{}}
\end{{frame}}

\section{{Mitigation}}
\begin{{frame}}{{Méthodes comparées}}
\begin{{itemize}}
\item \textbf{{Reweighting}} : pondération par couples groupe/label.
\item \textbf{{Resampling}} : équilibrage des sous-groupes.
\item \textbf{{Fairlearn}} : contraintes demographic parity/equalized odds.
\item \textbf{{Adversarial PyTorch}} : représentation moins informative sur l'attribut sensible.
\item \textbf{{Post-processing}} : seuils par groupe, à discuter éthiquement.
\end{{itemize}}
\end{{frame}}

\begin{{frame}}{{Recommandations}}
\begin{{itemize}}
\item Ne jamais se limiter à l'accuracy globale.
\item Auditer systématiquement TPR/FPR par groupe.
\item Privilégier equalized odds pour les erreurs en contexte judiciaire.
\item Documenter les limites COMPAS et surveiller la dérive après déploiement.
\end{{itemize}}
\end{{frame}}

\section{{Conclusion}}
\begin{{frame}}{{Conclusion}}
\begin{{exampleblock}}{{Message clé}}
Un modèle peut être statistiquement utile et socialement problématique. L'audit doit donc combiner performance, fairness, mitigation et gouvernance.
\end{{exampleblock}}
\end{{frame}}
\end{{document}}
"""


def main():
    data = load_results(Path("results/metrics/audit_metrics.json"))
    report_path = Path("reports/latex/rapport_audit_biais.tex")
    slides_path = Path("slides/presentation.tex")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    slides_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(data), encoding="utf-8")
    slides_path.write_text(build_slides(data), encoding="utf-8")
    print(f"Rapport LaTeX généré: {report_path}")
    print(f"Slides Beamer générées: {slides_path}")


if __name__ == "__main__":
    main()
