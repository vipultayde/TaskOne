import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.nn_model import MedicalPredictor
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
model = MedicalPredictor()

# Generate synthetic medical data for training
np.random.seed(42)
torch.manual_seed(42)

# Features: glucose, blood_pressure, insulin, bmi, age, pregnancies, skin_thickness, diabetes_pedigree
num_samples = 500
X = torch.randn(num_samples, 8)
# Generate binary labels (0 or 1) based on some pattern
y = (X[:, 0] + X[:, 3] + X[:, 4] > 0.5).float().unsqueeze(1)

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
