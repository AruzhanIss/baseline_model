import pandas as pd
import numpy as np
np.random.seed(42)
n_samples = 1000

amount = np.random.randint(1, 5000, n_samples)  
oldbalanceOrg = np.random.randint(0, 10000, n_samples)  
newbalanceOrig = oldbalanceOrg - amount + np.random.randint(-50, 50, n_samples)  
oldbalanceDest = np.random.randint(0, 20000, n_samples) 
newbalanceDest = oldbalanceDest + amount + np.random.randint(-50, 50, n_samples)  

is_fraud = (amount > 4000).astype(int)
is_fraud[np.random.choice(n_samples, 50, replace=False)] = 1

df = pd.DataFrame({
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "newbalanceOrig": newbalanceOrig,
    "oldbalanceDest": oldbalanceDest,
    "newbalanceDest": newbalanceDest,
    "is_fraud": is_fraud
})

csv_path = "fraud_data.csv"
df.to_csv(csv_path, index=False)

print(f"✅ Dataset saved to {csv_path}, shape = {df.shape}")
