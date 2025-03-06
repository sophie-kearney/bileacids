###
# IMPORTS
###

import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
pd.set_option('display.max_columns', None)

###
# CONSTANTS
###

cohort = "pMCIiAD" # pHCiAD, pMCIiAD
trained_model = "seed32_MaskedGRU_0.8267"

###
# LOAD DATA
###

predictions = pd.read_csv(f"processed/{cohort}/{trained_model}_predictions.csv")
print(predictions)

###
# FIT DATA
###

# kmf = KaplanMeierFitter()
# plt.figure(figsize=(8, 6))
#
# colors = {
#     "High": "#d8b709",
#     "Low": "#0c775e"
# }
#
# for group in predictions["RiskGroup"].unique():
#     mask = predictions["RiskGroup"] == group
#     kmf.fit(predictions.loc[mask, "ADConversionTime"].fillna(predictions["ADConversionTime"].max()), event_observed=predictions.loc[mask, "AD"])
#     kmf.plot_survival_function(label=f"{group} BARS", color=colors.get(group, "#000000"))
#
# plt.title("Kaplan Meier Curve of MCI to AD Conversion")
# plt.xlabel("Time (months)")
# plt.ylabel("Survival Probability")
# plt.legend(title="Risk Group")
# plt.show()

###
# STRATIFY BY PRS
###

# kmf = KaplanMeierFitter()
# plt.figure(figsize=(8, 6))
#
# colors = {
#     ("High", "High"): "#f09c06",
#     ("High", "Low"): "#fc0007",
#     ("Low", "High"): "#4dadcc",
#     ("Low", "Low"): "#149076"
# }
#
# for risk_group in predictions["RiskGroup"].unique():
#     for prs_group in predictions["PRSRiskGroup"].unique():
#         mask = (predictions["RiskGroup"] == risk_group) & (predictions["PRSRiskGroup"] == prs_group)
#         label = f"{risk_group} BARS, {prs_group} PRS"
#         kmf.fit(predictions.loc[mask, "ADConversionTime"].fillna(predictions["ADConversionTime"].max()), event_observed=predictions.loc[mask, "AD"])
#         kmf.plot_survival_function(label=label, color=colors.get((risk_group, prs_group), "#000000"))
#
# plt.title("Kaplan Meier Curve of MCI to AD Conversion")
# plt.xlabel("Time (months)")
# plt.ylabel("Survival Probability")
# plt.legend(title="Risk Group")
# plt.show()

###
# GET ODDS RATIO TABLE
###

table = pd.crosstab([predictions["RiskGroup"], predictions["PRSRiskGroup"]], predictions["AD"])
odds_ratios = {}
for idx, row in table.iterrows():
    if row.shape[0] == 2:
        odds_ratio, _ = fisher_exact([[row[0], row[1]], [row[1], row[0]]])
        odds_ratios[idx] = odds_ratio
    else:
        odds_ratios[idx] = np.nan
    print(f"RiskGroup: {idx[0]}, PRSRiskGroup: {idx[1]}, Odds Ratio: {odds_ratios[idx]}")
print(odds_ratios)

heatmap_data = pd.DataFrame(index=['Low', 'High'], columns=['Low', 'High'])
for (exp1, exp2), or_value in odds_ratios.items():
    heatmap_data.loc[exp1, exp2] = or_value

plt.figure(figsize=(8, 6))
norm = TwoSlopeNorm(vmin=odds_ratios.min(), vcenter=1.0, vmax=odds_ratios.max())
sns.heatmap(heatmap_data.astype(float), annot=True, cmap='coolwarm', cbar_kws={'label': 'Odds Ratio'})
plt.title('Odds Ratio Heatmap for AD Outcomes')
plt.xlabel('BARS Groups')
plt.ylabel('PRS Groups')
plt.show()