###
# IMPORTS
###

import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

###
# CONSTANTS
###

cohort = "pMCIiAD" # pHCiAD, pMCIiAD
trained_model = "MaskedGRU_0.8215"

###
# LOAD DATA
###

predictions = pd.read_csv(f"processed/{cohort}/{trained_model}_predictions.csv")
# print(predictions)

###
# FIT DATA
###

kmf = KaplanMeierFitter()
plt.figure(figsize=(8, 6))

colors = {
    "High": "#d8b709",
    "Low": "#0c775e"
}

for group in predictions["RiskGroup"].unique():
    mask = predictions["RiskGroup"] == group
    kmf.fit(predictions.loc[mask, "ADConversionTime"].fillna(predictions["ADConversionTime"].max()), event_observed=predictions.loc[mask, "AD"])
    kmf.plot_survival_function(label=group, color=colors.get(group, "#000000"))

plt.title("Kaplan Meier Curve of MCI to AD Conversion")
plt.xlabel("Time (months)")
plt.ylabel("Survival Probability")
plt.legend(title="Risk Group")
plt.show()