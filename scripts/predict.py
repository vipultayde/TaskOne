import torch
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
contract = w3.eth.contract(address=contract_info['address'], abi=contract_info['abi'])

# Create model
model = MedicalPredictor()

# First, trigger averaging on the blockchain
private_key = '067bd2c3138b1c1a9671f2da7b0acd48e7b4896a94eacf823322773a94065620'
account = w3.eth.account.from_key(private_key)
w3.eth.default_account = account.address

print("Triggering parameter averaging on blockchain...")
nonce = w3.eth.get_transaction_count(account.address)
tx = contract.functions.averageParameters('fc1.weight').build_transaction({
    'chainId': 11155111,
    'gas': 1000000,
    'gasPrice': w3.eth.gas_price * 3,
    'nonce': nonce,
})
signed_tx = w3.eth.account.sign_transaction(tx, private_key)
tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
w3.eth.wait_for_transaction_receipt(tx_hash)

# Average other layers too
layers = ['fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']
for layer in layers:
    nonce = w3.eth.get_transaction_count(account.address)
    tx = contract.functions.averageParameters(layer).build_transaction({
        'chainId': 11155111,
        'gas': 1000000,
        'gasPrice': w3.eth.gas_price * 3,
        'nonce': nonce,
    })
    signed_tx = w3.eth.account.sign_transaction(tx, private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    w3.eth.wait_for_transaction_receipt(tx_hash)

print("Parameters averaged from all participants")

# Fetch averaged parameters from blockchain
params_dict = {}
all_available = True
for name, _ in model.named_parameters():
    if 'weight' in name:
        params = contract.functions.getAveragedWeights(name).call()
        if len(params) == 0:
            print(f"Warning: No averaged weights available for {name}")
            all_available = False
            break
        params_dict[name] = params
    elif 'bias' in name:
        params = contract.functions.getAveragedBiases(name).call()
        if len(params) == 0:
            print(f"Warning: No averaged biases available for {name}")
            all_available = False
            break
        params_dict[name] = params

if all_available:
    # Set parameters in model
    model.set_parameters_from_int(params_dict)
    print("Averaged parameters loaded from blockchain")
else:
    print("Warning: Some averaged parameters not available. Using default model parameters.")

print("Averaged parameters loaded from blockchain")

# Make prediction with sample medical data
sample_data = {
    'glucose': 120.0,
    'blood_pressure': 70.0,
    'insulin': 80.0,
    'bmi': 25.0,
    'age': 35.0,
    'pregnancies': 2.0,
    'skin_thickness': 20.0,
    'diabetes_pedigree': 0.5
}

input_tensor = MedicalPredictor.preprocess_input(sample_data)

with torch.no_grad():
    prediction = model(input_tensor)
    probability = prediction.item()
    risk_level = "HIGH RISK" if probability > 0.5 else "LOW RISK"

print(f"Sample Medical Data: {sample_data}")
print(f"Diabetes Risk Probability: {probability:.4f}")
print(f"Risk Assessment: {risk_level}")

# This can be run repeatedly to get real-time predictions with updated federated parameters
