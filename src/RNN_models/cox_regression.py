###
# IMPORT PACKAGES
###

import pandas as pd
import seaborn as sns
import numpy as np
from dateutil.relativedelta import relativedelta

from lifelines.fitters.coxph_fitter import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from lifelines import KaplanMeierFitter

###
# LOAD DATA
###

data = pd.read_csv("processed/master_data.csv")

# find correct columns
begin_met = data.columns.get_loc("L_HISTIDINE")
end_met = data.columns.get_loc("BUDCA") + 1
begin_BA = data.columns.get_loc("CA")
begin_BA_ratio = data.columns.get_loc("CA_CDCA")
end_BA_ratio = data.columns.get_loc("GLCA_CDCA") + 1

# log 10 scale
data.iloc[:, begin_met:end_met] = np.log10(data.iloc[:, begin_met:end_met].replace(0, np.nan))
data.iloc[:, begin_BA_ratio:begin_BA_ratio] = np.log10(data.iloc[:, begin_BA_ratio:begin_BA_ratio].replace(0, np.nan))
data.fillna(0, inplace=True)

# isolate pMCI vs iAD
pMCI = data.groupby("RID").filter(lambda x: x["DX_VALS"].isin([2, 3]).all())
y = [0] * pMCI.shape[0] # 374 RIDs

# isolate MCI->AD
data["DX_VALS"] = data["DX_VALS"].replace(3, 2)
rids = [] # 259 RIDs
for rid, patient in data.groupby("RID"):
    dxs = patient["DX_VALS"].values
    if 2 in dxs:
        idx_2 = list(dxs).index(2)
        if 4 in dxs[idx_2:]:
            idx_4 = list(dxs[idx_2:]).index(4) + idx_2
            if 2 not in dxs[idx_4:] and 1 not in dxs[idx_2:idx_4]:
                rids.append(rid)

# remove specific cases that fail previous filtering
rids.remove(162)
rids.remove(166)
rids.remove(1123)
iAD = data[data["RID"].isin(rids)]
print(len(rids))
# combine AD labels
y += [1] * iAD.shape[0]

# combine data together
pMCI_iAD = pd.concat([pMCI, iAD], ignore_index=True)
# pMCI_iAD['y'] = y

# get time for each event
time = []
AD = 0
MCI = 0
for rid, patient in pMCI_iAD.groupby("RID"):
    dxs = patient["DX_VALS"].values
    dates = pd.to_datetime(patient["EXAMDATE"].values)
    if 4 not in dxs:
        deltat = (relativedelta(dates[-1], dates[0]).years * 12) + relativedelta(dates[-1], dates[0]).months
        time.append(deltat)
        MCI += 1
    else:
        # find the first occurance of AD in the list of diagnosis
        first_ad = list(dxs).index(4)
        deltat = (relativedelta(dates[first_ad], dates[0]).years * 12) + relativedelta(dates[-1], dates[0]).months
        time.append(deltat)
        AD += 1

print(AD, MCI)
print(time)

###
# COX REGRESSION
###

# cph = CoxPHFitter(alpha=0.05)
# cph.fit(survival_pd, 'tenure', 'churn')
