import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

data = pd.read_csv("processed/master_data.csv")
data['TAU'] = pd.to_numeric(data['TAU'], errors='coerce')
data['ABETA'] = pd.to_numeric(data['ABETA'], errors='coerce')
data['AV45'] = pd.to_numeric(data['AV45'], errors='coerce')

# fill empty
data.iloc[:, 132:139] = data.iloc[:, 132:139].replace([np.inf, -np.inf], np.nan)

#  minmax
# scaler = MinMaxScaler()
# data.iloc[:, 96:123] = scaler.fit_transform(data.iloc[:, 96:123])
# data.iloc[:, 131:138] = scaler.fit_transform(data.iloc[:, 131:138])
# print(data.iloc[:, 131:138])

# -log10
data.iloc[:, 97:124] = np.log10(data.iloc[:, 97:124].replace(0, np.nan))
data.iloc[:, 132:139] = np.log10(data.iloc[:, 132:139].replace(0, np.nan))

# scaler = StandardScaler()
# data.iloc[:, 96:123] = scaler.fit_transform(data.iloc[:, 96:123])
# data.iloc[:, 131:138] = scaler.fit_transform(data.iloc[:, 131:138])

data.to_csv("processed/master_data_log10.csv", index=False)