import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# pd.set_option('display.max_columns', None)
project_path = os.getcwd()

KEYCOLS = ['RID','VISCODE2','VISCODE2_DX','EXAMDATE','EXAMDATE_RANK','y','fast','BMI','hdl','chol','trig']

###
# LOAD LIPID DATA
###
lipid_data = pd.read_csv("raw/ADMCGUTMETABOLITESLONG_12_13_21_28Jan2025.csv", sep=",")
lipid_data['RID'] = lipid_data['RID'].astype(int)
# for each sample, rank data by EXAMDATE. set to EXAMDATE_RANK
lipid_data['EXAMDATE'] = pd.to_datetime(lipid_data['EXAMDATE'], format='%Y-%m-%d')
lipid_data['EXAMDATE'] = lipid_data['EXAMDATE'].dt.date
gr = lipid_data.sort_values(by='EXAMDATE').groupby('RID', as_index=False)
lipid_data['EXAMDATE_RANK'] = gr.cumcount()

###
# LOAD DX DATA - get diagnosis for each sample
###

dx = pd.read_excel(pd.ExcelFile("raw/ADNI_DX.xlsx"), 'ADNI_DX')
dx['RID'] = dx['RID'].astype(int)
dx.drop(dx.columns[2:22], axis=1, inplace=True)
dx.drop(dx.columns[0], axis=1, inplace=True)

dx_cols = dx.columns.str.replace('_DXGrp', '', regex=False).str.lower().to_list()

dx_vals = []
for _, row in lipid_data.iterrows():
    rid = row['RID']
    viscode = row['VISCODE2']

    vis_ndx = dx_cols.index(viscode)
    dx_val = dx.loc[dx['RID'] == rid].iloc[0, vis_ndx]
    if pd.isna(dx_val):
        dx_val = -1
    dx_vals.append(int(dx_val))
lipid_data['DX_VALS'] = dx_vals

###
# GET CLINICAL DATA
###

clin = pd.read_csv("raw/ClinicalInformation.csv", sep=",")
clin[['RID','VISCODE2']] = clin['RID_visit'].str.split('_',expand=True)
clin.drop(clin.columns[0], axis=1, inplace=True)
clin['RID'] = clin['RID'].astype(int)
clin['fast'] = clin['fast'].apply(lambda x: 1 if x=='Yes' else 0 if x=='No' else None)

data = lipid_data.merge(clin, on=['RID','VISCODE2'])

###
# PROCESS CSF DATA
###

csf = pd.read_csv("raw/ADNI1GO23_CSF_Biomarkers.csv", sep=",")
# get the columns names but segment out the VISCODE2
csf_cols_key = [col.split('_')[0].lower() for col in csf.columns[1:]]

abeta42 = []
ptau = []

for _, row in data.iterrows():
    rid = row['RID']
    viscode = row['VISCODE2']
    # find the first occurance of the right VISCODE2 from the column names
    ndx = csf_cols_key.index(viscode) + 1
    csf_vals = csf.loc[csf['RID'] == rid, csf.columns[ndx+1:ndx+3]].values
    ab = csf_vals[0][0]
    pt = csf_vals[0][1]

    # convert to float if not nan
    if pd.isna(ab):
        ab = np.nan
    elif isinstance(ab, str):
        ab = ab.replace("<","")
        ab = float(ab)

    if pd.isna(pt):
        pt = np.nan
    elif isinstance(pt, str):
        pt = pt.replace("<","")
        pt = float(pt)

    abeta42.append(ab)
    ptau.append(pt)

data['ABeta42'] = abeta42
data['pTau'] = ptau

###
# ADD RATIOS
###

data['CA_CDCA'] = data['CA'] / data['CDCA']
data['DCA_CA'] = data['DCA'] / data['CA']
data['TDCA_CA'] = data['TDCA'] / data['CA']
data['GDCA_CA'] = data['GDCA'] / data['CA']
data['TDCA_DCA'] = data['TDCA'] / data['DCA']
data['GDCA_DCA'] = data['GDCA'] / data['DCA']
data['GLCA_CDCA'] = data['GLCA'] / data['CDCA']

###
# APOE STATUS
###

cov_all = pd.read_csv("raw/gene_level_directed_merge_pupdated_apoe_prsnorm.csv")
cov = cov_all[["RID","AGE","PTGENDER","ABETA","TAU","APOE_e2", "APOE_e4", "APOE_e2e4", "FDG", "WholeBrain", "AV45"]]
data = pd.merge(cov, data, on="RID", how="right")

###
# CONVERT TO NUMERICAL
###

data['TAU'] = pd.to_numeric(data['TAU'], errors='coerce')
data['ABETA'] = pd.to_numeric(data['ABETA'], errors='coerce')
data['AV45'] = pd.to_numeric(data['AV45'], errors='coerce')

###
# SAVE DATA
###

# move EXAMDATE_RANK and DX to the front
data = data[['RID', 'EXAMDATE_RANK', 'DX_VALS'] + [col for col in data.columns if col not in ['RID', 'EXAMDATE_RANK', 'DX_VALS']]]
data.to_csv("processed/master_data.csv", index=False)