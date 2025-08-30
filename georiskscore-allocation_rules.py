# -*- coding: utf-8 -*-
"""
Construction d'un score de risque géopolitique & Stratégies d'allocation dynamiques
-----------------------------------------------------------------------------------
Ce script :
1) charge les données hebdomadaires,
2) construit un GeoRiskScore = w1*GPR_z + w2*EPU_z + w3*GT_agg_z + w4*VIX_z,
3) estime un modèle MS-2 sur log(RealizedVol_10d) et récupère P_high_MS,
4) définit des règles d'allocation dynamiques (seuils MS-2, puis règle combinée avec GeoRiskScore),
5) calcule les performances (buy&hold vs stratégies dynamiques), métriques et figures.

Sorties :
- outputs/georisk_score.png (série + MA13 + seuil p85 + ombrage)
- outputs/backtest_allocation_ms2.png (BH vs stratégie MS-2, ombrage stress)
- outputs/backtest_allocation_combo.png (BH vs stratégie combinée, ombrage stress)
- outputs/performance_summary.csv (métriques)
- outputs/weights_and_probabilities.png (poids et P_high_MS dans le temps)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# ------------- Paramètres généraux -------------
CSV_PATH = "weekly_combined_with_inflation.csv"
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# Pondérations du GeoRiskScore (modifiable)
W_GEO = {
    "GPR_Global_z": 0.30,
    "EPU_z":        0.10,
    "GT_agg_z":     0.20,
    "VIX_z":        0.40,
}
# Seuil d'alerte du score (percentile)
SCORE_PCTL = 85

# Seuils de probabilité MS-2 pour la règle 3 bandes
P_LOW  = 0.40
P_HIGH = 0.60

# Coût de transaction en points de base appliqué au turnover hebdo (optionnel)
TC_BPS = 0.0  # ex: 5.0 -> 5 bps = 0.05%

# ------------- Fonctions utilitaires -------------
def to_z(x):
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

def annualized_metrics(simple_rets, freq_per_year=52):
    """Retourne rendement annualisé (géométrique), vol annualisée, Sharpe (rf=0), max drawdown."""
    r = simple_rets.dropna().values
    if len(r) == 0:
        return np.nan, np.nan, np.nan, np.nan

    # Courbe de valeur
    eq = np.cumprod(1.0 + r)
    # Rendement annualisé géométrique
    ann_ret = (eq[-1])**(freq_per_year / len(r)) - 1.0
    # Vol annualisée
    ann_vol = np.std(r, ddof=1) * np.sqrt(freq_per_year)
    # Sharpe (rf=0)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    # Max drawdown
    roll_max = np.maximum.accumulate(eq)
    dd = eq / roll_max - 1.0
    max_dd = dd.min() if len(dd) else np.nan
    return ann_ret, ann_vol, sharpe, max_dd

def estimate_ms2_probabilities(log_realized):
    """Estime un MS-2 sur log_vol; retourne une série P_high_MS alignée à l'index."""
    mod = MarkovRegression(log_realized.astype(float), k_regimes=2,
                           trend='c', switching_variance=True)
    res = mod.fit(em_iter=50, search_reps=20, maxiter=1000, disp=False)

    smoothed = res.smoothed_marginal_probabilities
    probs = pd.DataFrame({'P0': smoothed[0], 'P1': smoothed[1]}, index=log_realized.index)

    # Identifier le régime "élevé" via la variance régimique
    param_map = dict(zip(res.model.param_names, res.params))
    s2_0 = param_map.get('sigma2[0]', np.nan)
    s2_1 = param_map.get('sigma2[1]', np.nan)
    high = 1 if np.nan_to_num(s2_1) > np.nan_to_num(s2_0) else 0

    probs['P_high_MS'] = probs[f'P{high}']
    return probs['P_high_MS']

def allocation_weight_ms2(p):
    """Règle 3 bandes selon P_high_MS : 1.0 / 0.6 / 0.2."""
    if p < P_LOW:
        return 1.0
    elif p <= P_HIGH:
        return 0.6
    else:
        return 0.2

def allocation_weight_combo(p, score, score_thr):
    """
    Règle combinée MS-2 + Score :
      - Si score > seuil (alerte) -> limiter à 0.2
      - Sinon appliquer la règle MS-2 3 bandes.
    """
    if score > score_thr:
        return 0.2
    return allocation_weight_ms2(p)

def apply_transaction_costs(w_series, gross_rets, tc_bps=0.0):
    """
    Applique un coût de transaction proportionnel au turnover hebdo:
      cost_t = tc * |w_t - w_{t-1}|
    Retourne la série de coûts (négatifs) alignée aux rendements.
    """
    if tc_bps <= 0.0:
        return pd.Series(0.0, index=gross_rets.index)
    tc = tc_bps / 1e4
    dw = w_series.diff().abs().fillna(0.0)
    costs = -tc * dw  # coût déduit comme un "retour" négatif
    return costs.reindex(gross_rets.index).fillna(0.0)

# ------------- 1) Charger les données -------------
df = pd.read_csv(CSV_PATH)
df['date'] = pd.to_datetime(df['Unnamed: 0'], errors='coerce')
df = df.dropna(subset=['date']).sort_values('date').set_index('date')

# ------------- 2) Construire GeoRiskScore -------------
# Composantes : GPR_Global_z, EPU_z, VIX_z ; agrégat GT_agg_z = moyenne des *_z "thématiques"
if 'GPR_Global_z' not in df.columns and 'GPR_Global' in df.columns:
    df['GPR_Global_z'] = to_z(df['GPR_Global'])
if 'EPU_z' not in df.columns and 'EPU' in df.columns:
    df['EPU_z'] = to_z(df['EPU'])
if 'VIX_z' not in df.columns and 'VIX' in df.columns:
    df['VIX_z'] = to_z(df['VIX'])

# Agrégat Google Trends : moyenne de tous les *_z sauf macro_z explicites
macro_z_exclude = {
    'GPR_Global_z','EPU_z','OAT_10Y_z','Bund_10Y_z','Infl_Exp_10Y_US_z','Spread_OAT_Bund_z','VIX_z'
}
gt_cols = [c for c in df.columns if c.endswith('_z') and c not in macro_z_exclude]
df['GT_agg_z'] = df[gt_cols].mean(axis=1) if len(gt_cols) else np.nan

# Calcul du score (avec gestion des composantes manquantes -> 0)
for k in W_GEO:
    if k not in df.columns:
        df[k] = np.nan
score_components = df[list(W_GEO.keys())].copy()
score_components = score_components.fillna(0.0)
df['GeoRiskScore'] = (score_components * pd.Series(W_GEO)).sum(axis=1)

# Moyenne mobile 13 semaines pour lisibilité
df['GeoRiskScore_MA13'] = df['GeoRiskScore'].rolling(window=13, min_periods=5).mean()
score_thr = np.nanpercentile(df['GeoRiskScore'].dropna(), SCORE_PCTL)

# ------------- 3) Estimation MS-2 sur log(RealizedVol_10d) -------------
if 'RealizedVol_10d' not in df.columns:
    raise ValueError("La colonne 'RealizedVol_10d' est requise pour MS-2.")
work = df[['RealizedVol_10d']].dropna().copy()
work['log_realized'] = np.log(work['RealizedVol_10d'])
p_high_ms = estimate_ms2_probabilities(work['log_realized'])

# Joindre P_high_MS à df (ffill pour combler les trous éventuels)
df = df.join(p_high_ms.rename('P_high_MS'), how='left')
df['P_high_MS'] = df['P_high_MS'].ffill()

# ------------- 4) Série de rendements actions (STOXX 600 de préférence) -------------
ret_candidates = ['STOXX600_ret', 'SX5E_ret', 'EUROSTOXX_ret', 'CAC40_ret', 'DAX_ret']
ret_col = next((c for c in ret_candidates if c in df.columns), None)
if ret_col is None:
    raise ValueError("Aucune série de rendement actions trouvée (ex: 'STOXX600_ret').")

rets = df[ret_col].dropna()
align_idx = rets.index

# ------------- 5) Poids d'allocation -------------
# Règle MS-2 3 bandes
w_ms2 = df.loc[align_idx, 'P_high_MS'].apply(allocation_weight_ms2)

# Règle combinée (MS-2 + alerte score > seuil)
w_combo = pd.Series(
    [allocation_weight_combo(p, s, score_thr)
     for p, s in zip(df.loc[align_idx, 'P_high_MS'], df.loc[align_idx, 'GeoRiskScore'])],
    index=align_idx
)

# ------------- 6) Backtests (BH vs dynamiques) -------------
# Buy & Hold
bh = (1.0 + rets).cumprod()

# Stratégie MS-2 seule
dyn_ms2_gross = 1.0 + w_ms2 * rets
cost_ms2 = apply_transaction_costs(w_ms2, rets, tc_bps=TC_BPS)
dyn_ms2 = (dyn_ms2_gross + cost_ms2).cumprod()

# Stratégie combinée
dyn_combo_gross = 1.0 + w_combo * rets
cost_combo = apply_transaction_costs(w_combo, rets, tc_bps=TC_BPS)
dyn_combo = (dyn_combo_gross + cost_combo).cumprod()

# ------------- 7) Graphiques -------------
# (a) GeoRiskScore + MA13 + seuil p85 + ombrage > seuil
score_df = df[['GeoRiskScore','GeoRiskScore_MA13']].dropna().copy()
thr = np.nanpercentile(df['GeoRiskScore'].dropna(), SCORE_PCTL)

plt.figure(figsize=(12, 4.2))
plt.plot(score_df.index, score_df['GeoRiskScore'], linewidth=1.0, label='GeoRiskScore (hebdo)')
plt.plot(score_df.index, score_df['GeoRiskScore_MA13'], linewidth=1.4, label='MA 13 sem.')
plt.axhline(thr, linestyle='--', linewidth=1.0, label=f'Seuil p{SCORE_PCTL}')
# Ombrage au-dessus du seuil
above = score_df['GeoRiskScore'] > thr
in_seg, start = False, None
for i, (dt, up) in enumerate(above.items()):
    if up and not in_seg:
        start, in_seg = dt, True
    elif not up and in_seg:
        end = above.index[i-1]
        plt.axvspan(start, end, alpha=0.15)
        in_seg = False
if in_seg:
    plt.axvspan(start, above.index[-1], alpha=0.15)
plt.title("Score de risque géopolitique — Série, MA13 et seuil p85")
plt.ylabel("Score (z-normalisé pondéré)")
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "georisk_score.png"), dpi=160)
plt.close()

# (b) Backtest MS-2 (BH vs dynamique) + ombrage périodes stress (P_high_MS > 0.6)
plt.figure(figsize=(12, 4.6))
plt.plot(bh.index, bh.values, linewidth=1.2, label="Buy & Hold")
plt.plot(dyn_ms2.index, dyn_ms2.values, linewidth=1.2, label="Allocation dynamique (MS-2)")
# Ombrage des périodes stress
ph = df.loc[dyn_ms2.index, 'P_high_MS'] > P_HIGH
in_seg, start = False, None
for i, (dt, flag) in enumerate(ph.items()):
    if flag and not in_seg:
        start, in_seg = dt, True
    elif not flag and in_seg:
        end = ph.index[i-1]
        plt.axvspan(start, end, alpha=0.15)
        in_seg = False
if in_seg:
    plt.axvspan(start, ph.index[-1], alpha=0.15)
plt.title("Backtest allocation dynamique (MS-2)")
plt.ylabel("Valeur cumulée (base 1)")
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "backtest_allocation_ms2.png"), dpi=160)
plt.close()

# (c) Backtest combiné (BH vs dynamique combo) + ombrage stress
plt.figure(figsize=(12, 4.6))
plt.plot(bh.index, bh.values, linewidth=1.2, label="Buy & Hold")
plt.plot(dyn_combo.index, dyn_combo.values, linewidth=1.2, label="Allocation dynamique (MS-2 + Score)")
ph = df.loc[dyn_combo.index, 'P_high_MS'] > P_HIGH
in_seg, start = False, None
for i, (dt, flag) in enumerate(ph.items()):
    if flag and not in_seg:
        start, in_seg = dt, True
    elif not flag and in_seg:
        end = ph.index[i-1]
        plt.axvspan(start, end, alpha=0.15)
        in_seg = False
if in_seg:
    plt.axvspan(start, ph.index[-1], alpha=0.15)
plt.title("Backtest allocation dynamique (MS-2 + GeoRiskScore)")
plt.ylabel("Valeur cumulée (base 1)")
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "backtest_allocation_combo.png"), dpi=160)
plt.close()

# (d) Tracé des poids et de P_high_MS (diagnostic)
ax_idx = rets.index
fig, ax1 = plt.subplots(figsize=(12, 4.2))
ax1.plot(ax_idx, w_ms2.loc[ax_idx].values, linewidth=1.0, label='Poids MS-2')
ax1.plot(ax_idx, w_combo.loc[ax_idx].values, linewidth=1.0, label='Poids combiné')
ax1.set_ylabel("Poids actions")
ax2 = ax1.twinx()
ax2.plot(ax_idx, df.loc[ax_idx, 'P_high_MS'].values, linewidth=1.0, label='P_high_MS', alpha=0.7)
ax2.set_ylabel("Probabilité régime élevé (MS-2)")
fig.suptitle("Évolution des poids et de la probabilité MS-2")
fig.legend(loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "weights_and_probabilities.png"), dpi=160)
plt.close(fig)

# ------------- 8) Métriques & export -------------
perf = []
# Séries de rendements simples de chaque stratégie
r_bh     = rets
r_ms2    = dyn_ms2.pct_change().dropna()
r_combo  = dyn_combo.pct_change().dropna()

for name, r in [
    ("Buy&Hold", r_bh),
    ("Dyn_MS2",  r_ms2),
    ("Dyn_Combo", r_combo),
]:
    ann_ret, ann_vol, sharpe, max_dd = annualized_metrics(r)
    perf.append([name, ann_ret, ann_vol, sharpe, max_dd])

perf_df = pd.DataFrame(perf, columns=["Stratégie", "Ann. Return", "Ann. Vol", "Sharpe", "Max Drawdown"])
perf_df.to_csv(os.path.join(OUTDIR, "performance_summary.csv"), index=False, encoding="utf-8")

print("=== RÉSUMÉ DES PERFORMANCES (hebdo -> annualisé) ===")
print(perf_df.to_string(index=False))
print(f"\nFigures & tableaux enregistrés dans : {os.path.abspath(OUTDIR)}")
