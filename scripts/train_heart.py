import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.heart_model import HeartDiseasePredictor
from web3 import Web3
import json
import numpy as np

# Load contract info
with open('contract_info.json', 'r') as f:
    contract_info = json.load(f)

# Connect to Sepolia
w3 = Web3(Web3.HTTPProvider(contract_info['rpc_url']))
private_key = '067bd2c3138b1c1a9671f2da7b0acd48e7b4896a94eacf823322773a94065620'
account = w3.eth.account.from_key(private_key)
w3.eth.default_account = account.address

contract = w3.eth.contract(address=contract_info['address'], abi=contract_info['abi'])

# Create model
model = HeartDiseasePredictor()

# Load real heart disease dataset
print("Loading heart disease dataset...")
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'heart_disease', 'heart.csv')
df = pd.read_csv(data_path)

# Prepare features and labels
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = df[features].values
y = df['target'].values

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Convert to tensors
X = torch.FloatTensor(X)
y = torch.FloatTensor(y).unsqueeze(1)

print(f"Dataset loaded: {len(X)} samples with {len(features)} features")

# Train model
criterion = nn.BCELoss()  # Binary cross-entropy for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training heart disease prediction model...")
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

print("Submitting parameters to blockchain...")
for layer, values in params.items():
    if 'weight' in layer:
        nonce = w3.eth.get_transaction_count(account.address)
        tx = contract.functions.submitWeights(layer, values).build_transaction({
            'chainId': 11155111,
            'gas': 500000,
            'gasPrice': w3.eth.gas_price * 3,
            'nonce': nonce,
        })
        signed_tx = w3.eth.account.sign_transaction(tx, private_key)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"Submitted weights for {layer}")
    elif 'bias' in layer:
        nonce = w3.eth.get_transaction_count(account.address)
        tx = contract.functions.submitBiases(layer, values).build_transaction({
            'chainId': 11155111,
            'gas': 500000,
            'gasPrice': w3.eth.gas_price * 3,
            'nonce': nonce,
        })
        signed_tx = w3.eth.account.sign_transaction(tx, private_key)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"Submitted biases for {layer}")

print("Parameters submitted to federated learning contract")
