import subprocess, os

accuracy_values = []
auroc_values = []
aproc_values = []

model_choice = "MaskedGRU"
imputed = 1
cohort = "pHCiAD"

for k in range(1, 14):
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
        break

    vals = result.stdout.strip().split(' ')

    # Assuming imputed_RNN.py prints the AUROC value
    accuracy = float(vals[0])
    auroc = float(vals[1])
    aproc = float(vals[2])
    accuracy_values.append(accuracy)
    auroc_values.append(auroc)
    aproc_values.append(aproc)
    print(k, accuracy, auroc, aproc)

# accuracy_values = [0.6441,0.6695,0.6441,0.6525,0.6186,0.5593,0.6102,0.7034,0.6441,0.6186,0.6695,0.6695,0.5085]
# auroc_values = [0.6494,0.7045,0.652,0.6709,0.676,0.5941,0.6731,0.7358,0.7159,0.5946,0.7085,0.7313,0.6151]
# aproc_values = [0.4651,0.5946,0.5257,0.4928,0.6885,0.4871,0.6794,0.6749,0.6618,0.4783,0.5568,0.6327,0.5046]
with open(f'processed/{cohort}/{model_choice}_aucoverk.csv', 'w') as f:
    f.write("k,accuracy,auroc,aproc\n")
    for i in range(0,13):
        f.write(f'{i+1},{accuracy_values[i]},{auroc_values[i]},{aproc_values[i]}\n')

