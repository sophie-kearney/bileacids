###
# IMPORTS
###
import os
import pandas as pd
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LassoCV
from tabulate import tabulate
import shap
pd.set_option('display.max_columns', None)

###
# CONSTANTS
###

plot = False
feature_analysis = False
cohort = "HCAD"

###
# READ DATA
###

project_path = os.getcwd()

data = pd.read_csv(f'{project_path}/processed/{cohort}.csv')

# isolate BL
data = data[data['EXAMDATE_RANK'] == 0]

###
# GET BILE ACIDS
###

species_df = pd.read_csv(f"{project_path}/raw/class.csv")
all_class = species_df['class'].unique()

# BA
bile_species = species_df[species_df['class'].isin(["primary", "secondary", "primary_conjugated",
                                                    "secondary_conjugated"])]['bile_acids'].to_list()

# nonBA
# bile_species = species_df[~species_df['class'].isin(["primary", "secondary", "primary_conjugated",
#                                                      "secondary_conjugated", "ratio"])]['bile_acids'].to_list()

# include covariates
bile_species += ['AGE', 'PTGENDER', 'BMI', 'fast', 'APOE_e2e4']

###
# SPLIT DATA
###

# removing 77 patients with missing values
all_rows = bile_species + ['DX_VALS']
subset_data = data[all_rows].dropna()

X = subset_data[bile_species].values
y = subset_data["DX_VALS"].values
y = [0 if x == 1 else x for x in y]
y = [1 if x == 4 else x for x in y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# X_test = scaler.transform(X_test)
# X_train = -np.log10(X_train)
# X_test = -np.log10(X_test)

###
# LR
###

logr = linear_model.LogisticRegression()
logr.fit(X_train,y_train)

pred_probs = logr.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, pred_probs)
print(auroc)

fpr, tpr, _ = roc_curve(y_test, pred_probs)

if plot:
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{cohort}")
    plt.legend(loc="lower right")

    plt.savefig(f'{project_path}/figures/LR_ROC_{cohort}_{auroc:.2f}.png')
    plt.show()

if feature_analysis:

    feature_importance = list(zip(bile_species, logr.coef_[0]))
    sorted_importance = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)

    explainer = shap.Explainer(logr, X_train)
    shap_values = explainer(X_train)

    mean_abs_shap_values = [np.mean(np.absolute(shap_values[:, i].values), axis=0) for i in range(len(bile_species))]

    combined_importance = [
        (feature, coef, mean_abs_shap) for feature, coef, mean_abs_shap in
        zip(bile_species, logr.coef_[0], mean_abs_shap_values)
    ]
    all = sorted(combined_importance, key=lambda x: abs(x[1]), reverse=True)

    print(tabulate(all, headers=["Feature", "Coefficient", "Mean Abs SHAP"], tablefmt="grid"))

    shap.summary_plot(shap_values, X_train, feature_names=bile_species)

###
# MIXED MODEL
###

# data.columns = data.columns.str.replace('.', '', regex=False)
# data.columns = data.columns.str.replace('-', '', regex=False)
# data.columns = data.columns.str.replace('_', '', regex=False)
#
# train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
#
# formula = 'y ~ Sphingosine'  # Replace with your features
# for i in data.iloc[:, 12:52].columns.to_list():
#     formula += " + " + i
#
# model = smf.mixedlm(formula, train_data, groups=train_data['RID']).fit()
# print(model.summary())
#
# pred_probs = model.predict(test_data)
#
# auroc = roc_auc_score(test_data['y'], pred_probs)
# print(f"AUROC: {auroc}")
#
# if plot:
#     fpr, tpr, _ = roc_curve(test_data['y'], pred_probs)
#     plt.plot(fpr, tpr, label=f"AUROC = {auroc:.2f}")
#     plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("ROC Curve")
#     plt.legend(loc="lower right")
#     plt.show()

###
# LASSO REGRESSION
###

# lasso = LassoCV(cv=5, random_state=42)
# lasso.fit(X_train, y_train)
#
# feature_names = bile_species
# print(feature_names)
# important_features = [(name, coef) for name, coef in zip(feature_names, lasso.coef_) if coef != 0]
#
# important_features = sorted(important_features, key=lambda x: abs(x[1]), reverse=True)
#
# for name, coef in important_features:
#     print(f"{name} {coef}")
#
# y_pred = lasso.predict(X_test)
# auroc = roc_auc_score(y_test, y_pred)
# print(f"AUROC: {auroc}")
#
# pred_probs = lasso.predict(X_test)
# fpr, tpr, _ = roc_curve(y_test, pred_probs)
# auroc = roc_auc_score(y_test, pred_probs)
#
# if plot:
#     plt.plot(fpr, tpr, label=f"AUROC = {auroc:.2f}")
#     plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("ROC Curve")
#     plt.legend(loc="lower right")
#     plt.show()