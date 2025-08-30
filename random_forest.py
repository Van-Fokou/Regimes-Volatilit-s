# -*- coding: utf-8 -*-
"""
Random Forest pour prédire RegimeHigh_{t+1} à partir d'indicateurs macro/market.
Pipeline:
1) Chargement CSV, parsing des dates
2) Estimation MS-2 sur log(RealizedVol_10d) -> probas de régime élevé
3) Construction de la cible Regime_High_{t+1}
4) Préparation des features (z-scores si besoin)
5) Split temporel 80/20
6) Entraînement RandomForest (class_weight=balanced_subsample)
7) Graphiques: ROC, importances (Gini)
8) Sauvegarde des métriques
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve

# ------------------------------------------------------------
# 0) I/O
# ------------------------------------------------------------
CSV_PATH = "weekly_combined_with_inflation.csv"   # <-- adapte le chemin si besoin
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# ------------------------------------------------------------
# 1) Chargement des données & préparation
# ------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

date_col = "Unnamed: 0"  # colonne date typique du fichier fourni
if date_col not in df.columns:
    raise ValueError(f"Colonne de date '{date_col}' introuvable dans le CSV.")
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
df = df.rename(columns={date_col: "date"}).set_index("date")

# Colonnes attendues/possibles
col_realized = "RealizedVol_10d"
col_vix      = "VIX"
col_sent     = "Sentiment_Composite"
col_gpr      = "GPR_Global"
col_epu      = "EPU"
col_infl     = "Infl_Exp_10Y_US"

# ------------------------------------------------------------
# 2) Modèle markovien MS-2 sur log-volatilité réalisée -> P_high_MS_t
# ------------------------------------------------------------
if col_realized not in df.columns:
    raise ValueError(f"'{col_realized}' est requis pour construire la cible via MS-2.")

work = df[[col_realized]].dropna().copy()
work["log_realized"] = np.log(work[col_realized])

mod = MarkovRegression(work["log_realized"].astype(float),
                       k_regimes=2, trend="c", switching_variance=True)
res = mod.fit(em_iter=50, search_reps=20, maxiter=1000, disp=False)

smoothed = res.smoothed_marginal_probabilities
probs_df = pd.DataFrame({"P_regime0": smoothed[0],
                         "P_regime1": smoothed[1]}, index=work.index)

# Identifier le régime de plus forte variance
param_map = dict(zip(res.model.param_names, res.params))
sigma2_0 = param_map.get("sigma2[0]", np.nan)
sigma2_1 = param_map.get("sigma2[1]", np.nan)
high_reg = 1 if np.nan_to_num(sigma2_1) > np.nan_to_num(sigma2_0) else 0

probs_df["P_high_MS"] = probs_df[f"P_regime{high_reg}"]

# ------------------------------------------------------------
# 3) Cible Regime_High_{t+1} (seuil 0.5)
# ------------------------------------------------------------
df = df.join(probs_df["P_high_MS"], how="left")
df["Regime_High_t"]   = (df["P_high_MS"] > 0.5).astype(float)
df["Regime_High_t+1"] = df["Regime_High_t"].shift(-1)

# ------------------------------------------------------------
# 4) Features (z-scores si versions _z absentes)
# ------------------------------------------------------------
def to_z(x):
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

if "GPR_Global_z" not in df.columns and col_gpr in df.columns:
    df["GPR_Global_z"] = to_z(df[col_gpr])
if "EPU_z" not in df.columns and col_epu in df.columns:
    df["EPU_z"] = to_z(df[col_epu])
if "Infl_Exp_10Y_US_z" not in df.columns and col_infl in df.columns:
    df["Infl_Exp_10Y_US_z"] = to_z(df[col_infl])

feature_candidates = [
    "GPR_Global_z",
    "EPU_z",
    "Infl_Exp_10Y_US_z",
    col_vix,
    col_realized,
    col_sent
]
features = [c for c in feature_candidates if c in df.columns]

data_model = df[features + ["Regime_High_t+1"]].dropna().copy()
if data_model.empty:
    raise ValueError("Dataset final vide après dropna. Vérifie les colonnes/features disponibles.")

# ------------------------------------------------------------
# 5) Split temporel 80/20 (préservation de l’ordre)
# ------------------------------------------------------------
n = data_model.shape[0]
split = int(n * 0.8)
train = data_model.iloc[:split].copy()
test  = data_model.iloc[split:].copy()

X_train = train.drop(columns="Regime_High_t+1").values
y_train = train["Regime_High_t+1"].values.astype(int)
X_test  = test.drop(columns="Regime_High_t+1").values
y_test  = test["Regime_High_t+1"].values.astype(int)
feature_names = train.drop(columns="Regime_High_t+1").columns.tolist()

# ------------------------------------------------------------
# 6) Random Forest (paramètres parcimonieux + balancing)
# ------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=5,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ------------------------------------------------------------
# 7) Prédictions & métriques
# ------------------------------------------------------------
proba_test = rf.predict_proba(X_test)[:, 1]
pred_test  = (proba_test >= 0.5).astype(int)

auc = roc_auc_score(y_test, proba_test)
acc = accuracy_score(y_test, pred_test)
cm  = confusion_matrix(y_test, pred_test)

print(f"AUC (test) = {auc:.3f} | Accuracy (test) = {acc:.3f}")
print("Matrice de confusion (rows=true, cols=pred):\n", cm)

with open(os.path.join(OUTDIR, "rf_metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"AUC: {auc:.3f}\nAccuracy: {acc:.3f}\n")
    f.write("Confusion matrix (rows=true, cols=pred):\n")
    f.write(str(cm) + "\n")

# ------------------------------------------------------------
# 8) Courbe ROC
# ------------------------------------------------------------
fpr, tpr, thr = roc_curve(y_test, proba_test)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"Random Forest (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.title("Courbe ROC – Random Forest (jeu de test)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "rf_roc.png"), dpi=160)
plt.close()

# ------------------------------------------------------------
# 9) Importances des variables (Gini)
# ------------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(7, 5))
importances.iloc[::-1].plot(kind="barh")
plt.title("Importance des variables – Random Forest")
plt.xlabel("Importance (Gini)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "rf_importance.png"), dpi=160)
plt.close()

print(f"✅ Fichiers enregistrés dans: {os.path.abspath(OUTDIR)}")
# -> rf_roc.png, rf_importance.png, rf_metrics.txt
