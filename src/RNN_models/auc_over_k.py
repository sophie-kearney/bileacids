import subprocess, os, sys

accuracy_values = []
auroc_values = []
aproc_values = []

model_choice = "MaskedGRU"
imputed = 1
cohort = "pMCIiAD"

for k in range(11, 14):
    # Set the environment variable or pass k as an argument if needed
    os.environ['MODEL'] = model_choice
    os.environ['COHORT'] = cohort
    os.environ['K'] = str(k)
    os.environ['IMPUTED'] = str(imputed)
    try:
        subprocess.run(['python', 'src/RNN_models/create_RNN_data.py'], check=True)
        result = subprocess.run(['python', 'src/RNN_models/imputed_RNN.py'], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred for k={k}: {e.stderr}")
        sys.exit()

    vals = result.stdout.strip().split(' ')

    # Assuming imputed_RNN.py prints the AUROC value
    accuracy = float(vals[0])
    auroc = float(vals[1])
    aproc = float(vals[2])
    accuracy_values.append(accuracy)
    auroc_values.append(auroc)
    aproc_values.append(aproc)
    print(k, accuracy, auroc, aproc)

# with open(f'processed/{cohort}/{model_choice}_aucoverk.csv', 'w') as f:
#     f.write("k,accuracy,auroc,aproc\n")
#     for i in range(0,13):
#         f.write(f'{i+1},{accuracy_values[i]},{auroc_values[i]},{aproc_values[i]}\n')

