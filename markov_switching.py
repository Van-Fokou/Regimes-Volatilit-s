# -*- coding: utf-8 -*-
"""
Pipeline complet:
- Estimation MS-2 (Markov) sur log-volatilité réalisée -> P_high_MS_t
- Construction de la cible Regime_High_{t+1}
- Préparation des features (z-scores si besoin)
- Entraînement XGBoost (split temporel)
- Graphiques: ROC, importance (gain), SHAP (si dispo)
- Sauvegardes dans outputs/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 0) I/O
# ------------------------------------------------------------
CSV_PATH = "weekly_combined_with_inflation.csv"  # adapte le chemin si besoin
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# ------------------------------------------------------------
# 1) Charger la base & préparer les séries
# ------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
date_col = 'Unnamed: 0'
if date_col in df.columns:
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    df = df.rename(columns={date_col: 'date'})
else:
    raise ValueError("La colonne de date 'Unnamed: 0' est introuvable.")

df = df.set_index('date')

# Colonnes attendues (noms typiques)
col_realized = 'RealizedVol_10d'
col_vix      = 'VIX'
col_vstoxx   = 'VSTOXX'
col_gpr      = 'GPR_Global'
col_epu      = 'EPU'
col_infl_exp = 'Infl_Exp_10Y_US'
col_sent     = 'Sentiment_Composite'

# Vérifications souples: certaines colonnes peuvent manquer; on ajuste
for c in [col_realized, col_vix, col_gpr, col_epu, col_infl_exp]:
    if c not in df.columns:
        print(f"[AVERTISSEMENT] Colonne absente: {c}")

# ------------------------------------------------------------
# 2) Estimation du MS-2 (Markov) sur log(RealizedVol_10d)
# ------------------------------------------------------------
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

work = df.copy()
# garder les observations avec realized vol disponible
work = work[[col_realized]].dropna().copy()
work['log_realized'] = np.log(work[col_realized])

# MS-2 : variance en changement de régime, constante par régime
mod = MarkovRegression(work['log_realized'].astype(float), k_regimes=2,
                       trend='c', switching_variance=True)
res = mod.fit(em_iter=50, search_reps=20, maxiter=1000, disp=False)

# Probabilités lissées par régime
smoothed = res.smoothed_marginal_probabilities
probs_df = pd.DataFrame({
    'P_regime0': smoothed[0],
    'P_regime1': smoothed[1],
}, index=work.index)

# Identifier le régime "élevé" via les variances estimées
param_map = dict(zip(res.model.param_names, res.params))
sigma2_0 = param_map.get('sigma2[0]', np.nan)
sigma2_1 = param_map.get('sigma2[1]', np.nan)
high_reg = 1 if np.nan_to_num(sigma2_1) > np.nan_to_num(sigma2_0) else 0
probs_df['P_high_MS'] = probs_df[f'P_regime{high_reg}']

# ------------------------------------------------------------
# 3) Construire la cible: Regime_High_{t+1} (seuil 0.5)
# ------------------------------------------------------------
df = df.join(probs_df['P_high_MS'], how='left')
df['Regime_High_t']   = (df['P_high_MS'] > 0.5).astype(float)
df['Regime_High_t+1'] = df['Regime_High_t'].shift(-1)

# ------------------------------------------------------------
# 4) Features: z-scores si versions _z manquent
# ------------------------------------------------------------
def to_z(x):
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

# Colonnes z si non présentes
if 'GPR_Global_z' not in df.columns and col_gpr in df.columns:
    df['GPR_Global_z'] = to_z(df[col_gpr])
if 'EPU_z' not in df.columns and col_epu in df.columns:
    df['EPU_z'] = to_z(df[col_epu])
if 'Infl_Exp_10Y_US_z' not in df.columns and col_infl_exp in df.columns:
    df['Infl_Exp_10Y_US_z'] = to_z(df[col_infl_exp])

# Sélection des features disponibles
feature_candidates = [
    'GPR_Global_z', 'EPU_z', 'Infl_Exp_10Y_US_z',
    col_vix, col_realized,  # niveaux, utiles économiquement
    col_sent                # sentiment composite si présent
]
features = [c for c in feature_candidates if c in df.columns]

# Dataset final (dropna sur features + target)
data_model = df[features + ['Regime_High_t+1']].dropna().copy()
print(f"Dataset pour XGB: {data_model.shape[0]} obs, {len(features)} features.")

# ------------------------------------------------------------
# 5) Split temporel 80/20 + XGBoost
# ------------------------------------------------------------
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve

n = data_model.shape[0]
split = int(n * 0.8)
train = data_model.iloc[:split].copy()
test  = data_model.iloc[split:].copy()

X_train = train.drop(columns='Regime_High_t+1').values
y_train = train['Regime_High_t+1'].values.astype(int)
X_test  = test.drop(columns='Regime_High_t+1').values
y_test  = test['Regime_High_t+1'].values.astype(int)
feature_names = train.drop(columns='Regime_High_t+1').columns.tolist()

# Validation interne pour early stopping
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
    random_state=42,
    eval_metric='auc'
)
xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

# ------------------------------------------------------------
# 6) Prédictions & Métriques
# ------------------------------------------------------------
proba_test = xgb.predict_proba(X_test)[:, 1]
pred_test  = (proba_test >= 0.5).astype(int)

auc = roc_auc_score(y_test, proba_test)
acc = accuracy_score(y_test, pred_test)
cm  = confusion_matrix(y_test, pred_test)

print(f"AUC (test) = {auc:.3f} | Accuracy (test) = {acc:.3f}")
print("Confusion matrix (rows=true, cols=pred):\n", cm)

# Sauvegarde métriques texte
with open(os.path.join(OUTDIR, "xgb_metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"AUC: {auc:.3f}\nAccuracy: {acc:.3f}\nConfusion matrix (rows=true, cols=pred):\n{cm}\n")

# ------------------------------------------------------------
# 7) Courbe ROC
# ------------------------------------------------------------
fpr, tpr, thr = roc_curve(y_test, proba_test)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"XGBoost (AUC={auc:.3f})")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.title("Courbe ROC – XGBoost (jeu de test)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "xgb_roc.png"), dpi=160)
plt.close()

# ------------------------------------------------------------
# 8) Importance des variables (gain)
# ------------------------------------------------------------
importance = xgb.get_booster().get_score(importance_type='gain')
imp_series = pd.Series(importance).sort_values(ascending=False)

# Remapper f0, f1, ... vers noms de colonnes
imp_series.index = [f"Feature_{int(s[1:])}" for s in imp_series.index]  # s="f0","f1",...
mapping = {f"Feature_{i}": name for i, name in enumerate(feature_names)}
imp_series.index = [mapping.get(ix, ix) for ix in imp_series.index]

plt.figure(figsize=(7,5))
imp_series.iloc[::-1].plot(kind='barh')
plt.title("Importance des variables (gain) – XGBoost")
plt.xlabel("Gain moyen (unités relatives)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "xgb_importance.png"), dpi=160)
plt.close()

# ------------------------------------------------------------
# 9) SHAP (optionnel) – interprétabilité locale/globale
# ------------------------------------------------------------
try:
    import shap
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)

    # SHAP summary
    plt.figure()  # nouvelle figure
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title("SHAP Summary – XGBoost (test)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "shap_summary.png"), dpi=160)
    plt.close()

    # SHAP bar (|SHAP| moyen)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      plot_type='bar', show=False)
    plt.title("Importance globale (|SHAP| moyen) – XGBoost (test)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "shap_bar.png"), dpi=160)
    plt.close()

except Exception as e:
    print("[INFO] SHAP non généré (librairie absente ou environnement limitatif) :", e)

print(f"✅ Fichiers enregistrés dans: {os.path.abspath(OUTDIR)}")
