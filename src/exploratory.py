import pandas as pd

data = pd.read_csv("processed/master_data.csv")
data['TAU'] = pd.to_numeric(data['TAU'], errors='coerce')
data['ABETA'] = pd.to_numeric(data['ABETA'], errors='coerce')

print(data)