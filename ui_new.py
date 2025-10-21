import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from model.nn_model import SimpleNN
from web3 import Web3
import json
import numpy as np
import subprocess
import sys
import os

# Load contract info
with open('contract_info.json', 'r') as f:
    contract_info = json.load(f)

w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))
contract = w3.eth.contract(address=contract_info['address'], abi=contract_info['abi'])
w3.eth.default_account = w3.eth.accounts[0]

st.title("Neural Network & Blockchain Integration UI")

st.header("Model Training")
if st.button("Train Model and Upload to Blockchain"):
    with st.spinner("Training model..."):
        # Run train.py
        result = subprocess.run([sys.executable, 'scripts/train.py'], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("Model trained and parameters uploaded to blockchain!")
            st.code(result.stdout)
        else:
            st.error("Error during training:")
            st.code(result.stderr)

st.header("Real-time Prediction")
if st.button("Make Prediction Using Blockchain Parameters"):
    with st.spinner("Fetching parameters and making prediction..."):
        # Run predict.py
        result = subprocess.run([sys.executable, 'scripts/predict.py'], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("Prediction made using real-time blockchain parameters!")
            st.code(result.stdout)
        else:
            st.error("Error during prediction:")
            st.code(result.stderr)

st.header("Blockchain Status")
if st.button("Check Blockchain Connection"):
    if w3.is_connected():
        st.success("Connected to blockchain")
        st.write(f"Contract Address: {contract_info['address']}")
        st.write(f"Current Block: {w3.eth.block_number}")
    else:
        st.error("Not connected to blockchain")

st.header("Model Parameters on Blockchain")
col1, col2 = st.columns(2)
with col1:
    if st.button("View Weights"):
        try:
            weights = contract.functions.getWeights('fc1.weight').call()
            st.write("fc1.weight:", weights[:10], "...")  # Show first 10
            weights = contract.functions.getWeights('fc2.weight').call()
            st.write("fc2.weight:", weights[:10], "...")
        except:
            st.error("Error fetching weights")

with col2:
    if st.button("View Biases"):
        try:
            biases = contract.functions.getBiases('fc1.bias').call()
            st.write("fc1.bias:", biases)
            biases = contract.functions.getBiases('fc2.bias').call()
            st.write("fc2.bias:", biases)
        except:
            st.error("Error fetching biases")
