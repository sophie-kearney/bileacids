import torch, shap, time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score, roc_auc_score, auc, precision_recall_curve, r2_score, roc_curve, confusion_matrix
import seaborn as sns
from lifelines import KaplanMeierFitter

def test_model(imputed, model_choice, model, test_loader, seed, cohort):
    model.eval()
    y_true = []
    y_pred = []
    all_probs = []

    with torch.no_grad():
        for test_batch in test_loader:
            if imputed:
                X_test_batch, y_test_batch, mask_test_batch, time_missing_test_batch = test_batch
            else:
                X_test_batch, y_test_batch = test_batch
            if model_choice == "MaskedGRU":
                logits, _ = model(X_test_batch, time_missing_test_batch, mask_test_batch)
                probs = torch.softmax(logits, dim=1)[:, 1]
                predicted_labels = torch.argmax(logits, dim=1)
            else:
                output = model(X_test_batch)
                probs = torch.sigmoid(output).squeeze()
                predicted_labels = (probs >= 0.5).float()

            y_true.append(y_test_batch.numpy())
            y_pred.append(predicted_labels.numpy())
            all_probs.append(probs.numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    all_probs = np.concatenate(all_probs)

    accuracy = accuracy_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, all_probs)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, all_probs)
    aproc = auc(recall, precision)
    r2 = r2_score(y_true, all_probs)

    print(f"  ACCURACY: {accuracy:.4f}")
    print(f"  ROC: {roc_auc:.4f}")
    print(f"  AUPRC: {aproc:.4f}")
    print(f"  R^2: {r2:.4f}")

    plt.figure()
    plt.plot(fpr, tpr, color='navy', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('BARS Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

    file_path = f"models/{cohort}/{seed}_{model_choice}_{time.strftime('%m%d%H%M')}_{roc_auc:.4f}"
    if not imputed:
        file_path += "_noImp"
    torch.save(model.state_dict(), f"{file_path}")
    print(f"> Model saved to {seed}_{model_choice}_{time.strftime('%m%d%H%M')}_{roc_auc:.4f}")

def test_long_model(model, test_loader):
    model.eval()

    y_true = []
    y_pred = []
    all_probs = []

    with torch.no_grad():
        for test_batch in test_loader:
            X_test_batch, y_test_batch = test_batch

            output = model(X_test_batch)
            probs = torch.sigmoid(output).squeeze()
            predicted_labels = (probs >= 0.5).float()

            y_true.append(y_test_batch.numpy())
            y_pred.append(predicted_labels.numpy())
            all_probs.append(probs.numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    all_probs = np.concatenate(all_probs)

    accuracy = accuracy_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, all_probs)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, all_probs)
    aproc = auc(recall, precision)
    r2 = r2_score(y_true, all_probs)

    print(f"  ACCURACY: {accuracy:.4f}")
    print(f"  ROC: {roc_auc:.4f}")
    print(f"  AUPRC: {aproc:.4f}")
    print(f"  R^2: {r2:.4f}")

    plt.figure()
    plt.plot(fpr, tpr, color='navy', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Longitudinal Covariates Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

def generate_predictions(model, long_model, X, y, is_missing, time_missing, static_covariates, long_cov):
    dataset = TensorDataset(X, y, is_missing, time_missing, static_covariates, long_cov)
    loader = DataLoader(dataset, shuffle=False)
    model.eval()

    # --- get BARS ---
    all_probs = []
    all_covs = []
    with torch.no_grad():
        for test_batch in loader:
            X_test_batch, y_test_batch, mask_test_batch, time_missing_test_batch, cov_test_batch, long_cov_test = test_batch
            logits, _ = model(X_test_batch, time_missing_test_batch, mask_test_batch)
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_probs.append(probs.numpy())
            all_covs.append(cov_test_batch)

        all_probs = np.concatenate(all_probs)
        all_covs = np.concatenate(all_covs)

    data = pd.DataFrame(all_covs, columns=["rid", "AGE", "PTGENDER", "APOE_e2e4"])
    data["ADRiskScore"] = all_probs
    data["rid"] = data["rid"].astype(int)

    # --- get times ---
    rids = sorted(data["rid"])
    master_data = pd.read_csv("processed/master_data.csv").sort_values(by=["RID", "EXAMDATE_RANK"])
    times = {}
    for rid, patient in master_data.groupby("RID"):
        if int(rid) in rids:
            if 4 in patient["DX_VALS"].values:
                first_ad_row = patient[patient["DX_VALS"] == 4].iloc[0]
                first_ad_viscode2 = int(first_ad_row["VISCODE2"].replace("m", "").replace("bl", "0"))
                times[rid] = first_ad_viscode2
            else:
                times[rid] = np.nan
    data["ADConversionTime"] = data["rid"].map(times)

    threshold = data["ADRiskScore"].median()
    data["RiskGroup"] = data["ADRiskScore"].apply(lambda x: "High" if x >= threshold else "Low")
    data["AD"] = data["ADConversionTime"].notna().astype(int)

    # --- get long covariate scores
    long_model.eval()
    all_probs = []

    with torch.no_grad():
        for test_batch in loader:
            X_test_batch, y_test_batch, mask_test_batch, time_missing_test_batch, cov_test_batch, long_cov_test = test_batch

            output = long_model(long_cov_test)
            probs = torch.sigmoid(output).squeeze()

            all_probs.append(probs.numpy())
        data["LongCovRiskScore"] = all_probs

    # --- get PRS
    # get other columns that are needed from the megafile
    usecols = ["RID", "SCORE", "APOE"]
    all_data = pd.read_csv("raw/gene_level_directed_merge_pupdated_apoe_prsnorm.csv", usecols=usecols)
    all_data = all_data[["RID", "SCORE", "APOE"]].rename(columns={"SCORE": "PRS"}) # rename PRS column
    # merge into our predictions dataframe
    data = data.merge(all_data, left_on="rid", right_on="RID", how="left").drop(columns=["RID"])

    # stratify by PRS using 50th percentile
    threshold = data["PRS"].median()
    data["PRSRiskGroup"] = data["PRS"].apply(lambda x: "High" if x >= threshold else "Low")

    # make sure APOE is an integer
    data["APOE"] = pd.to_numeric(data["APOE"], errors='coerce').astype('Int64')

    return data

def logistic_regression_embeddings(data, lr_fi):
    # TODO - may want to remove APOE here bc of NA values
    data = data[["AGE", "PTGENDER", "APOE_e2e4", "ADRiskScore", "AD", "LongCovRiskScore", "PRS"]]
    data = data.dropna()

    X = data[["AGE", "PTGENDER", "APOE_e2e4", "ADRiskScore", "LongCovRiskScore", "PRS"]]
    y = data["AD"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logr = linear_model.LogisticRegression()
    logr.fit(X_train, y_train)

    pred_probs = logr.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, pred_probs)

    fpr, tpr, _ = roc_curve(y_test, pred_probs)
    pred_labels = (pred_probs > 0.5).astype(int)

    accuracy = accuracy_score(y_test, pred_labels)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, pred_labels)
    aproc = auc(recall, precision)
    r2 = r2_score(y_test, pred_labels)

    print(f"  ACCURACY: {accuracy:.4f}")
    print(f"  ROC: {roc_auc:.4f}")
    print(f"  AUPRC: {aproc:.4f}")
    print(f"  R^2: {r2:.4f}")

    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"LR AUROC using RNN Embeddings and Covariates")
    plt.legend(loc="lower right")

    plt.show()

    cm = confusion_matrix(y_test, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('LR with Embeddings and Covariates Confusion Matrix')
    plt.show()

    if lr_fi == "y":
        print("> Generating LR feature importance...")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': logr.coef_[0]
        })

        feature_importance['Importance'] = feature_importance['Coefficient'].abs()
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.show()

        explainer = shap.Explainer(logr, X_train)
        shap_values = explainer(X_test)

        shap_values_array = np.array(shap_values.values, dtype=np.float32)
        shap.summary_plot(shap_values_array, X_test, feature_names=X.columns)

def kmplot(data, var1, var2=None):
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(7, 5))
    color_set = ["#f09c06", "#0c775e", "#fc0007", "#4dadcc", "#149076"]

    # only one variable
    if var2 is None:
        groups = data[var1].dropna().unique()
        groups = [group for group in groups if group != -9]
        colors = {}
        for i in range(len(groups)):
            colors[groups[i]] = color_set[i]

        for group in groups:
            mask = data[var1] == group
            kmf.fit(data.loc[mask, "ADConversionTime"].fillna(data["ADConversionTime"].max()),
                    event_observed=data.loc[mask, "AD"])
            kmf.plot_survival_function(label=f"{group} {var1}", color=colors.get(group, "#000000"))

        title = f"{var1} Kaplan Meier Curve of MCI to AD Conversion"

    # two variables
    else:
        group1 = data[var1].unique()
        group2 = data[var2].unique()
        colors = {}
        counter = 0
        for i in range(len(group1)):
            for j in range(len(group2)):
                colors[(group1[i], group2[j])] = color_set[counter]
                counter += 1

        for g1 in group1:
            for g2 in group2:
                mask = (data[var1] == g1) & (data[var2] == g2)
                label = f"{g1} {var1}, {g2} {var2}"
                kmf.fit(data.loc[mask, "ADConversionTime"].fillna(data["ADConversionTime"].max()), event_observed=data.loc[mask, "AD"])
                kmf.plot_survival_function(label=label, color=colors.get((g1, g2), "#000000"))

        title = f"{var1} {var2} Kaplan Meier Curve of MCI to AD Conversion"

    plt.title(title)
    plt.xlabel("Time (months)")
    plt.ylim(0, 1)
    plt.ylabel("Survival Probability")
    plt.legend(title="Risk Group")
    plt.show()

def kmplots(predictions):
    kmplot(data=predictions, var1="RiskGroup", var2=None)
    kmplot(data=predictions, var1="RiskGroup", var2="PRSRiskGroup")
    kmplot(data=predictions, var1="APOE", var2=None)

    pred_high = predictions[predictions["RiskGroup"] == "High"]
    pred_low = predictions[predictions["RiskGroup"] == "Low"]
    kmplot(data=pred_high, var1="APOE", var2=None)
    kmplot(data=pred_low, var1="APOE", var2=None)

def MaskedGRU_feature_importance(X, time_missing, is_missing, model):
    hidden_size = 111
    seq_size = 13

    model.eval()
    with torch.no_grad():
        W_o = model.fc.weight  # (num_classes, hidden_size) = (2, 111)
        b_o = model.fc.bias  # (num_classes,) = (2,)
        outputs, h_t = model(X, time_missing, is_missing)  # (batch_size, seq_len, num_classes) = (586, 13, 2)
        # print(f"h_t shape: {h_t.shape}")

        c_t_d = []  # (hidden_size, seq_size, batch_size) = (111, 13, 586)
        for d in range(hidden_size):
            W_o_d = W_o[1, d]  # get the weight of the positive class
            b_o_d = b_o[1]  # get the bias of the positive class

            c_d = []
            for t in range(seq_size):
                h_d_t = h_t[:, t, d]  # (batch_size,) = (586,)
                c_d.append(W_o_d * h_d_t + b_o_d)

            c_t_d.append(c_d)

        # print(f"c_t_d shape: ({len(c_t_d)}, {len(c_t_d[0])}, {len(c_t_d[0][1])})")
        u_d = np.mean(c_t_d, axis=1)  # (hidden_size, seq_size, batch_size) -> (hidden_size, batch_size)
        # (111, 13, 586) -> (111, 586)
        # print("u_d shape: ", u_d.shape)

        u_d_abs = np.abs(u_d)
        mean_abs_u_d = np.mean(u_d_abs, axis=1)  # (hidden_size, batch_size) -> (hidden_size,)
        # (111, 586) -> (111,)
        # print("mean_abs_u_d shape: ", mean_abs_u_d.shape)

    importance = pd.DataFrame({
        'ndx': np.arange(111),
        'contribution': mean_abs_u_d
    })

    feature_names = pd.read_csv("processed/feature_names.csv")
    importance = importance.merge(feature_names, on='ndx')
    importance = importance.sort_values(by='contribution', ascending=False)

    top_20_features = importance.sort_values(by='contribution', ascending=False).head(20)

    plt.figure(figsize=(8, 6))
    plt.barh(top_20_features['bileacid'], top_20_features['contribution'], color='skyblue', align='center')
    plt.xlabel('Contribution')
    plt.ylabel('Feature')
    plt.title('Top 20 Features by Contribution')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()