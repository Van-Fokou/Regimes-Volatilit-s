# -*- coding: utf-8 -*-
"""
Modélisation complète : Logistique, XGBoost, Random Forest
---------------------------------------------------------
Pipeline :
1) Chargement CSV, parsing des dates
2) Estimation MS-2 (MarkovRegression) sur log(RealizedVol_10d) -> proba de régime élevé
3) Construction de la cible Regime_High_{t+1}
4) Préparation des features (z-scores si besoin)
5) Split temporel 80/20
6) Entraînement :
   - Régression logistique (statsmodels)
   - XGBoost (avec early stopping)
   - Random Forest (class_weight=balanced_subsample)
7) Sorties :
   - Métriques (AUC, Accuracy), matrices de confusion
   - Graphiques ROC (3) : logit, XGB, RF
   - Importances : XGB (gain), RF (Gini)
   - SHAP : summary + bar (si librarie dispo)
   - Table LaTeX (métriques XGB) en option
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import statsmodels.api as sm

from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, roc_curve, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ------------------------------------------------------------
# 0) Paramètres I/O
# ------------------------------------------------------------
CSV_PATH = "weekly_combined_with_inflation.csv"
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)
RANDOM_STATE = 42

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def to_z(x):
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

def plot_roc(y_true, y_proba, title, outpath):
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    return auc

def save_confusion(cm, outpath_txt, model_name):
    with open(outpath_txt, "a", encoding="utf-8") as f:
        f.write(f"\n=== {model_name} ===\n")
        f.write("Confusion matrix (rows=true, cols=pred):\n")
        f.write(str(cm) + "\n")

def safe_series(df, name, fallback=None):
    return df[name] if name in df.columns else fallback

# ------------------------------------------------------------
# 1) Charger la base & préparer la série
# ------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
date_col = 'Unnamed: 0'
if date_col not in df.columns:
    raise ValueError(f"La colonne date '{date_col}' est introuvable dans le CSV.")
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
df = df.rename(columns={date_col: 'date'}).set_index('date')

# Colonnes potentiellement utiles
col_realized = 'RealizedVol_10d'
col_vix      = 'VIX'
col_sent     = 'Sentiment_Composite'
col_gpr      = 'GPR_Global'
col_epu      = 'EPU'
col_infl     = 'Infl_Exp_10Y_US'

if col_realized not in df.columns:
    raise ValueError(f"'{col_realized}' est requis pour construire la cible via MS-2.")

# ------------------------------------------------------------
# 2) Estimation MS-2 -> proba de régime élevé
# ------------------------------------------------------------
work = df[[col_realized]].dropna().copy()
work['log_realized'] = np.log(work[col_realized])

mod = MarkovRegression(work['log_realized'].astype(float),
                       k_regimes=2, trend='c', switching_variance=True)
res = mod.fit(em_iter=50, search_reps=20, maxiter=1000, disp=False)

smoothed = res.smoothed_marginal_probabilities
probs_df = pd.DataFrame({'P_regime0': smoothed[0], 'P_regime1': smoothed[1]},
                        index=work.index)

# Identifier le régime de plus forte variance
param_map = dict(zip(res.model.param_names, res.params))
sigma2_0 = param_map.get('sigma2[0]', np.nan)
sigma2_1 = param_map.get('sigma2[1]', np.nan)
high_reg = 1 if np.nan_to_num(sigma2_1) > np.nan_to_num(sigma2_0) else 0

probs_df['P_high_MS'] = probs_df[f'P_regime{high_reg}']

# ------------------------------------------------------------
# 3) Construire la cible Regime_High_{t+1}
# ------------------------------------------------------------
df = df.join(probs_df['P_high_MS'], how='left')
df['Regime_High_t']   = (df['P_high_MS'] > 0.5).astype(float)
df['Regime_High_t+1'] = df['Regime_High_t'].shift(-1)

# ------------------------------------------------------------
# 4) Préparer les features
#    - on génère les z-scores si absents
#    - on retient un sous-ensemble robuste
# ------------------------------------------------------------
if 'GPR_Global_z' not in df.columns and col_gpr in df.columns:
    df['GPR_Global_z'] = to_z(df[col_gpr])
if 'EPU_z' not in df.columns and col_epu in df.columns:
    df['EPU_z'] = to_z(df[col_epu])
if 'Infl_Exp_10Y_US_z' not in df.columns and col_infl in df.columns:
    df['Infl_Exp_10Y_US_z'] = to_z(df[col_infl])
if 'VIX_z' not in df.columns and col_vix in df.columns:
    df['VIX_z'] = to_z(df[col_vix])

feature_candidates = [
    'GPR_Global_z', 'EPU_z', 'Infl_Exp_10Y_US_z',
    col_vix, col_realized, col_sent
]
features = [c for c in feature_candidates if c in df.columns]

data_model = df[features + ['Regime_High_t+1']].dropna().copy()
if data_model.empty:
    raise ValueError("Dataset final vide après dropna. Vérifie la présence des colonnes features et de la cible.")

# ------------------------------------------------------------
# 5) Split temporel 80/20
# ------------------------------------------------------------
n = data_model.shape[0]
split = int(n * 0.8)
train = data_model.iloc[:split].copy()
test  = data_model.iloc[split:].copy()

X_train = train.drop(columns='Regime_High_t+1').values
y_train = train['Regime_High_t+1'].values.astype(int)
X_test  = test.drop(columns='Regime_High_t+1').values
y_test  = test['Regime_High_t+1'].values.astype(int)
feature_names = train.drop(columns='Regime_High_t+1').columns.tolist()

# ============================================================
# 6) RÉGRESSION LOGISTIQUE (statsmodels)
# ============================================================
X_train_const = sm.add_constant(X_train)
X_test_const  = sm.add_constant(X_test)

logit_model = sm.Logit(y_train, X_train_const)
logit_res   = logit_model.fit(disp=0)

# Probabilités & métriques
proba_logit = logit_res.predict(X_test_const)
pred_logit  = (proba_logit >= 0.5).astype(int)

auc_logit = roc_auc_score(y_test, proba_logit)
acc_logit = accuracy_score(y_test, pred_logit)
cm_logit  = confusion_matrix(y_test, pred_logit)

# ROC logit
plot_roc(y_test, proba_logit,
         "Courbe ROC – Régression logistique (test)",
         os.path.join(OUTDIR, "logit_roc.png"))

# Table des coefficients + odds ratios
summary_table = logit_res.summary2().tables[1].copy()
summary_table['Odds_Ratio'] = np.exp(summary_table['Coef.'])
summary_table.to_csv(os.path.join(OUTDIR, "logit_coeffs.csv"), encoding="utf-8")

with open(os.path.join(OUTDIR, "logit_metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"AUC: {auc_logit:.3f}\nAccuracy: {acc_logit:.3f}\n")
    f.write(str(classification_report(y_test, pred_logit, digits=3)) + "\n")
save_confusion(cm_logit, os.path.join(OUTDIR, "logit_metrics.txt"), "Logit")

# ============================================================
# 7) XGBOOST (avec early stopping)
# ============================================================
split2 = int(len(X_train)*0.8)
X_tr, X_val = X_train[:split2], X_train[split2:]
y_tr, y_val = y_train[:split2], y_train[split2:]

xgb = XGBClassifier(
    n_estimators=400,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective='binary:logistic',
    random_state=RANDOM_STATE,
    eval_metric='auc'
)
xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

proba_xgb = xgb.predict_proba(X_test)[:, 1]
pred_xgb  = (proba_xgb >= 0.5).astype(int)

auc_xgb = roc_auc_score(y_test, proba_xgb)
acc_xgb = accuracy_score(y_test, pred_xgb)
cm_xgb  = confusion_matrix(y_test, pred_xgb)

# ROC XGB
plot_roc(y_test, proba_xgb,
         "Courbe ROC – XGBoost (test)",
         os.path.join(OUTDIR, "xgb_roc.png"))

# Importance (gain)
importance = xgb.get_booster().get_score(importance_type='gain')
imp_series = pd.Series(importance).sort_values(ascending=False)
# Remapper f0,f1,... vers noms de colonnes
imp_series.index = [f"Feature_{int(s[1:])}" for s in imp_series.index]
mapping = {f"Feature_{i}": name for i, name in enumerate(feature_names)}
imp_series.index = [mapping.get(ix, ix) for ix in imp_series.index]

plt.figure(figsize=(7,5))
imp_series.iloc[::-1].plot(kind='barh')
plt.title("Importance des variables (gain) – XGBoost")
plt.xlabel("Gain moyen (unités relatives)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "xgb_importance.png"), dpi=160)
plt.close()

with open(os.path.join(OUTDIR, "xgb_metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"AUC: {auc_xgb:.3f}\nAccuracy: {acc_xgb:.3f}\n")
    f.write(str(classification_report(y_test, pred_xgb, digits=3)) + "\n")
save_confusion(cm_xgb, os.path.join(OUTDIR, "xgb_metrics.txt"), "XGBoost")

# Table LaTeX (optionnelle)
metrics_latex = (
    "\\begin{table}[H]\n"
    "\\centering\n"
    "\\caption{Performances XGBoost (jeu de test)}\\label{tab:xgb_metrics}\n"
    "\\begin{tabular}{lcc}\n"
    "\\toprule\n"
    " & AUC & Accuracy \\\\\n"
    "\\midrule\n"
    f"XGBoost & {auc_xgb:.3f} & {acc_xgb:.3f} \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{table}\n"
)
with open(os.path.join(OUTDIR, "xgb_metrics_table.tex"), "w", encoding="utf-8") as f:
    f.write(metrics_latex)

# SHAP (si disponible)
try:
    import shap
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)

    # SHAP summary
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title("SHAP Summary – XGBoost (test)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "shap_summary.png"), dpi=160)
    plt.close()

    # SHAP bar (|SHAP| moyen)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type='bar', show=False)
    plt.title("Importance globale (|SHAP| moyen) – XGBoost (test)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "shap_bar.png"), dpi=160)
    plt.close()

except Exception as e:
    with open(os.path.join(OUTDIR, "xgb_metrics.txt"), "a", encoding="utf-8") as f:
        f.write(f"[INFO] SHAP non généré : {e}\n")

# ============================================================
# 8) RANDOM FOREST
# ============================================================
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=5,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train, y_train)

proba_rf = rf.predict_proba(X_test)[:, 1]
pred_rf  = (proba_rf >= 0.5).astype(int)

auc_rf = roc_auc_score(y_test, proba_rf)
acc_rf = accuracy_score(y_test, pred_rf)
cm_rf  = confusion_matrix(y_test, pred_rf)

# ROC RF
plot_roc(y_test, proba_rf,
         "Courbe ROC – Random Forest (test)",
         os.path.join(OUTDIR, "rf_roc.png"))

# Importances RF (Gini)
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
plt.figure(figsize=(7,5))
importances.iloc[::-1].plot(kind='barh')
plt.title("Importance des variables – Random Forest (Gini)")
plt.xlabel("Importance (Gini)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "rf_importance.png"), dpi=160)
plt.close()

with open(os.path.join(OUTDIR, "rf_metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"AUC: {auc_rf:.3f}\nAccuracy: {acc_rf:.3f}\n")
    f.write(str(classification_report(y_test, pred_rf, digits=3)) + "\n")
save_confusion(cm_rf, os.path.join(OUTDIR, "rf_metrics.txt"), "Random Forest")

# ------------------------------------------------------------
# 9) Récapitulatif global
# ------------------------------------------------------------
summary = pd.DataFrame({
    'Modèle': ['Logit', 'XGBoost', 'RandomForest'],
    'AUC':    [auc_logit, auc_xgb, auc_rf],
    'Accuracy': [acc_logit, acc_xgb, acc_rf]
})
summary.to_csv(os.path.join(OUTDIR, "summary_metrics.csv"), index=False, encoding="utf-8")

print("\n=== Résumé des performances (test) ===")
print(summary)
print(f"\nFichiers enregistrés dans : {os.path.abspath(OUTDIR)}")
