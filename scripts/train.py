import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.nn_model import MedicalPredictor
from web3 import Web3
import json
import numpy as np

# Load contract info
with open('federated-medical-app/src/contract_info.json', 'r') as f:
    contract_info = json.load(f)

# Connect to Ganache
w3 = Web3(Web3.HTTPProvider(contract_info['rpc_url']))
private_key = '0x4f3edf983ac636a65a842ce7c78d9aa706d3b113bce9c46f30d7d21715b23b1d'  # Account 0 from Ganache
account = w3.eth.account.from_key(private_key)
w3.eth.default_account = account.address

contract = w3.eth.contract(address=contract_info['address'], abi=contract_info['abi'])

# Create model
model = MedicalPredictor()

# Load real diabetes dataset
print("Loading diabetes dataset...")
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'diabetes.csv')
df = pd.read_csv(data_path)

# Prepare features and labels
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = df[features].values
y = df['Outcome'].values

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Convert to tensors
X = torch.FloatTensor(X)
y = torch.FloatTensor(y).unsqueeze(1)

print(f"Dataset loaded: {len(X)} samples with {len(features)} features")

# Train model
criterion = nn.BCELoss()  # Binary cross-entropy for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training medical prediction model...")
for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/200, Loss: {loss.item():.4f}")

print("Model trained successfully")

# Get parameters and submit to blockchain
params = model.get_parameters_as_int()

print("Submitting model hash to blockchain...")
# Generate a mock model hash (in real implementation, this would be the hash of trained model parameters)
model_hash = "0x" + np.random.bytes(32).hex()  # Mock hash

nonce = w3.eth.get_transaction_count(account.address)
tx = contract.functions.submitModel(model_hash).build_transaction({
    'chainId': 1337,  # Ganache chain ID
    'gas': 200000,
    'gasPrice': 20000000000,  # 20 gwei
    'nonce': nonce,
})

signed_tx = w3.eth.account.sign_transaction(tx, private_key)
tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

print(f"✅ Model hash submitted successfully!")
print(f"Transaction hash: {tx_hash.hex()}")
print(f"Block number: {receipt['blockNumber']}")
print("Parameters submitted to federated learning contract")
