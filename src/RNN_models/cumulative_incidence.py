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
# plt.figure(figsize=(7, 5))
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
# plt.ylim(0, 1)
# plt.ylabel("Survival Probability")
# plt.legend(title="Risk Group")
# plt.show()

###
# STRATIFY BY PRS
###

# kmf = KaplanMeierFitter()
# plt.figure(figsize=(7, 5))
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
# plt.ylim(0, 1)
# plt.ylabel("Survival Probability")
# plt.legend(title="Risk Group")
# plt.show()

kmf = KaplanMeierFitter()
plt.figure(figsize=(7, 5))
colors = {
    "High": "#d8b709",
    "Low": "#0c775e",
    "Intermediate": "#f09c06"
}

for group in predictions["PRSRiskGroup"].unique():
    if group == "Intermediate":
        continue
    mask = predictions["PRSRiskGroup"] == group
    kmf.fit(predictions.loc[mask, "ADConversionTime"].fillna(predictions["ADConversionTime"].max()), event_observed=predictions.loc[mask, "AD"])
    kmf.plot_survival_function(label=f"{group} 20% PRS", color=colors.get(group, "#000000"))

plt.title("Kaplan Meier Curve of MCI to AD Conversion")
plt.xlabel("Time (months)")
plt.ylim(0, 1)
plt.ylabel("Survival Probability")
plt.legend(title="PRS Risk Group Quintiles")
plt.show()

###
# GET ODDS RATIO TABLE
###

# table = pd.crosstab([predictions["RiskGroup"], predictions["PRSRiskGroup"]], predictions["AD"])
# odds_ratios = {}
# for idx, row in table.iterrows():
#     if row.shape[0] == 2:
#         odds_ratio, _ = fisher_exact([[row[0], row[1]], [row[1], row[0]]])
#         odds_ratios[idx] = odds_ratio
#     else:
#         odds_ratios[idx] = np.nan
#     print(f"RiskGroup: {idx[0]}, PRSRiskGroup: {idx[1]}, Odds Ratio: {odds_ratios[idx]}")
# print(odds_ratios)

###
# STRATIFY BY APOE
###
#
# kmf = KaplanMeierFitter()
# plt.figure(figsize=(7, 5))
#
# colors = {
#     (23.0): "#f09c06",
#     (24.0): "#c5cdf5",
#     (33.0): "#4dadcc",
#     (34.0): "#149076",
#     (44.0): "#f32200"
# }
#
# for group in predictions["APOE"].dropna().unique():
#     if group == -9:
#         continue
#     mask = predictions["APOE"] == group
#     kmf.fit(predictions.loc[mask, "ADConversionTime"].fillna(predictions["ADConversionTime"].max()),
#             event_observed=predictions.loc[mask, "AD"])
#     kmf.plot_survival_function(label=f"{group}", color=colors.get(group, "#000000"))
#
# plt.title("Kaplan Meier Curve of MCI to AD Conversion")
# plt.xlabel("Time (months)")
# plt.ylim(0, 1)
# plt.ylabel("Survival Probability")
# handles, labels = plt.gca().get_legend_handles_labels()
# sorted_handles_labels = sorted(zip(labels, handles), key=lambda x: x[0])
# sorted_labels, sorted_handles = zip(*sorted_handles_labels)
# plt.legend(sorted_handles, sorted_labels, title="APOE")
# plt.show()
#
# # facet by risk group BA
#
# plt.figure(figsize=(6, 5))
# pred_high = predictions[predictions["RiskGroup"] == "High"]
# print(pred_high)
# pred_low = predictions[predictions["RiskGroup"] == "Low"]
#
# for group in pred_high["APOE"].dropna().unique():
#     if group == -9:
#         continue
#     mask = pred_high["APOE"] == group
#     kmf.fit(pred_high.loc[mask, "ADConversionTime"].fillna(pred_high["ADConversionTime"].max()),
#             event_observed=pred_high.loc[mask, "AD"])
#     kmf.plot_survival_function(label=f"{group}", color=colors.get(group, "#000000"))
#
# plt.title("BARS High")
# plt.xlabel("Time (months)")
# plt.ylabel("Survival Probability")
# plt.ylim(0, 1)
# handles, labels = plt.gca().get_legend_handles_labels()
# sorted_handles_labels = sorted(zip(labels, handles), key=lambda x: x[0])
# sorted_labels, sorted_handles = zip(*sorted_handles_labels)
# plt.legend(sorted_handles, sorted_labels, title="APOE")
# plt.show()
#
# plt.figure(figsize=(6, 5))
# for group in pred_low["APOE"].dropna().unique():
#     if group == -9:
#         continue
#     mask = pred_low["APOE"] == group
#     kmf.fit(pred_low.loc[mask, "ADConversionTime"].fillna(pred_low["ADConversionTime"].max()),
#             event_observed=pred_low.loc[mask, "AD"])
#     kmf.plot_survival_function(label=f"{group}", color=colors.get(group, "#000000"))
#
# plt.title("BARS Low")
# plt.xlabel("Time (months)")
# plt.ylabel("Survival Probability")
# plt.ylim(0, 1)
# handles, labels = plt.gca().get_legend_handles_labels()
# sorted_handles_labels = sorted(zip(labels, handles), key=lambda x: x[0])
# sorted_labels, sorted_handles = zip(*sorted_handles_labels)
# plt.legend(sorted_handles, sorted_labels, title="APOE")
# plt.show()
