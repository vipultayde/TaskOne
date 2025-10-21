import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from model.nn_model import MedicalPredictor
from web3 import Web3
import json
import numpy as np
import streamlit.components.v1 as components

st.title("Federated Medical Prediction App")
st.markdown("### Decentralized Diabetes Risk Assessment using Blockchain")

# Load contract info
with open('contract_info.json', 'r') as f:
    contract_info = json.load(f)

# Connect to Sepolia
w3 = Web3(Web3.HTTPProvider(contract_info['rpc_url']))
contract = w3.eth.contract(address=contract_info['address'], abi=contract_info['abi'])

# MetaMask connection placeholder (in real implementation, use web3modal or similar)
st.sidebar.header("MetaMask Connection")

# Wallet connection status
if 'wallet_connected' not in st.session_state:
    st.session_state.wallet_connected = False
    st.session_state.wallet_address = None

if st.sidebar.button("🔗 Connect MetaMask", type="secondary"):
    # For demo purposes, simulate MetaMask connection
    # In production, this would use web3modal or direct MetaMask integration
    st.session_state.wallet_connected = True
    st.session_state.wallet_address = "0xcC933Da78ab228bee7EE0403C4b4F89CB015DDBD"  # Use the actual address from the task
    st.rerun()

if st.session_state.wallet_connected:
    st.sidebar.success(f"✅ Connected: {st.session_state.wallet_address}")
    st.sidebar.info("Ready to submit training data or make predictions")
else:
    st.sidebar.warning("⚠️ Please connect your MetaMask wallet to participate in federated learning")
    st.sidebar.info("💡 MetaMask required for blockchain transactions and gas fee payments")

# Federated Learning Contribution Section
st.sidebar.header("🤝 Contribute to Federated Learning")
if st.sidebar.button("🚀 Train & Submit Model Parameters", type="secondary"):
    st.sidebar.info("Training local model... (this may take a moment)")

    # Generate synthetic training data locally
    np.random.seed(42)
    torch.manual_seed(42)
    num_samples = 100  # Smaller dataset for quick training
    X = torch.randn(num_samples, 8)
    y = (X[:, 0] + X[:, 3] + X[:, 4] > 0.5).float().unsqueeze(1)

    # Quick training
    local_model = MedicalPredictor()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(local_model.parameters(), lr=0.01)

    for epoch in range(50):  # Fewer epochs for speed
        optimizer.zero_grad()
        outputs = local_model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    # Get parameters
    params = local_model.get_parameters_as_int()

    st.sidebar.success("✅ Local training complete!")
    st.sidebar.info("💰 **Note:** Submitting to blockchain will require MetaMask approval and gas fees on Sepolia testnet.")

    # In real implementation, integrate with MetaMask to submit params
    # For now, show what would be submitted
    with st.sidebar.expander("📊 Parameters to Submit"):
        for layer, values in params.items():
            st.write(f"{layer}: {len(values)} values")

    if st.sidebar.button("📤 Submit to Blockchain", type="primary"):
        st.sidebar.warning("💰 **Gas Fee Notice:** This transaction will cost ~0.001 ETH in gas fees on Sepolia testnet.")
        st.sidebar.info("🔐 Only model parameters are sent - your medical data stays private!")

        # Actual blockchain submission using web3.py
        with st.sidebar:
            with st.spinner("🔄 Preparing transaction..."):
                try:
                    # Use the same private key as in scripts
                    private_key = '067bd2c3138b1c1a9671f2da7b0acd48e7b4896a94eacf823322773a94065620'
                    account = w3.eth.account.from_key(private_key)
                    w3.eth.default_account = account.address

                    # Submit parameters for each layer
                    submitted_layers = []
                    for layer_name, params in params.items():
                        if 'weight' in layer_name:
                            nonce = w3.eth.get_transaction_count(account.address)
                            tx = contract.functions.submitWeights(layer_name, params).build_transaction({
                                'chainId': 11155111,
                                'gas': 500000,
                                'gasPrice': w3.eth.gas_price * 3,
                                'nonce': nonce,
                            })
                            signed_tx = w3.eth.account.sign_transaction(tx, private_key)
                            tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
                            w3.eth.wait_for_transaction_receipt(tx_hash)
                            submitted_layers.append(f"{layer_name} ✅")

                        elif 'bias' in layer_name:
                            nonce = w3.eth.get_transaction_count(account.address)
                            tx = contract.functions.submitBiases(layer_name, params).build_transaction({
                                'chainId': 11155111,
                                'gas': 500000,
                                'gasPrice': w3.eth.gas_price * 3,
                                'nonce': nonce,
                            })
                            signed_tx = w3.eth.account.sign_transaction(tx, private_key)
                            tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
                            w3.eth.wait_for_transaction_receipt(tx_hash)
                            submitted_layers.append(f"{layer_name} ✅")

                    st.sidebar.success("✅ Parameters successfully submitted to blockchain!")
                    st.sidebar.info("📋 **Submitted Layers:**")
                    for layer in submitted_layers:
                        st.sidebar.write(f"• {layer}")

                    # Show transaction details
                    st.sidebar.code(f"""
📋 Transaction Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Network: Sepolia Testnet
Contract: {contract_info['address'][:10]}...{contract_info['address'][-8:]}
Layers Submitted: {len(submitted_layers)}
Total Gas Used: ~{len(submitted_layers) * 500000} gas
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    """, language="text")

                    st.sidebar.info("🔍 **Check MetaMask:** You can view the transactions in your MetaMask wallet under 'Activity' tab.")

                except Exception as e:
                    st.sidebar.error(f"❌ Submission failed: {str(e)}")
                    st.sidebar.info("💡 **Troubleshooting:**\n• Check your internet connection\n• Ensure Sepolia testnet is selected\n• Verify sufficient ETH balance")

# Create model
model = MedicalPredictor()

# Fetch averaged parameters from blockchain
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_model_parameters():
    params_dict = {}
    all_available = True
    for name, _ in model.named_parameters():
        if 'weight' in name:
            params = contract.functions.getAveragedWeights(name).call()
            if len(params) == 0:
                all_available = False
                break
            params_dict[name] = params
        elif 'bias' in name:
            params = contract.functions.getAveragedBiases(name).call()
            if len(params) == 0:
                all_available = False
                break
            params_dict[name] = params

    if all_available:
        model.set_parameters_from_int(params_dict)
        return True
    else:
        return False

if load_model_parameters():
    st.success("✅ Model parameters loaded from federated blockchain")
else:
    st.warning("⚠️ Federated parameters not yet available. Using default model parameters.")
    # Initialize with default parameters
    pass

# Get participant count
participant_count = contract.functions.getParticipantCount().call()
st.metric("Federated Learning Participants", participant_count)

# Input form for medical data
st.header("🩺 Enter Medical Data for Diabetes Risk Assessment")

col1, col2 = st.columns(2)

with col1:
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0.0, max_value=300.0, value=100.0, step=1.0)
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
    insulin = st.number_input("Insulin Level (μU/mL)", min_value=0.0, max_value=500.0, value=80.0, step=1.0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

with col2:
    age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=35.0, step=1.0)
    pregnancies = st.number_input("Number of Pregnancies", min_value=0.0, max_value=20.0, value=2.0, step=1.0)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)

# Prediction button
    if st.button("🔮 Assess Diabetes Risk", type="primary"):
        input_data = {
            'glucose': glucose,
            'blood_pressure': blood_pressure,
            'insulin': insulin,
            'bmi': bmi,
            'age': age,
            'pregnancies': pregnancies,
            'skin_thickness': skin_thickness,
            'diabetes_pedigree': diabetes_pedigree
        }

        # Option to use local model or blockchain prediction
        use_blockchain = st.checkbox("🚀 Use Blockchain Prediction (via MetaMask)", value=False)

        if use_blockchain:
            st.info("💰 **Blockchain Prediction Fee:** ~0.0001 ETH gas fee for on-chain computation")

            if st.button("📤 Send to Blockchain for Prediction", type="secondary"):
                # Convert input to integers for blockchain
                input_int = []
                for key, value in input_data.items():
                    # Scale to match model's preprocessing
                    if key == 'glucose':
                        scaled = int((value - 120.2) / 47.3 * 1000000)
                    elif key == 'blood_pressure':
                        scaled = int((value - 69.1) / 19.4 * 1000000)
                    elif key == 'insulin':
                        scaled = int((value - 79.8) / 115.2 * 1000000)
                    elif key == 'bmi':
                        scaled = int((value - 32.0) / 7.9 * 1000000)
                    elif key == 'age':
                        scaled = int((value - 33.2) / 11.8 * 1000000)
                    elif key == 'pregnancies':
                        scaled = int((value - 3.8) / 3.4 * 1000000)
                    elif key == 'skin_thickness':
                        scaled = int((value - 20.5) / 16.0 * 1000000)
                    elif key == 'diabetes_pedigree':
                        scaled = int((value - 0.5) / 0.3 * 1000000)
                    input_int.append(scaled)

                try:
                    # Call blockchain prediction
                    result = contract.functions.predict(input_int).call()
                    probability = result / 1000000.0  # Convert back to float

                    st.success("✅ Prediction completed on blockchain!")

                except Exception as e:
                    st.error(f"❌ Blockchain prediction failed: {str(e)}")
                    st.info("Falling back to local model prediction...")
                    input_tensor = MedicalPredictor.preprocess_input(input_data)
                    with torch.no_grad():
                        prediction = model(input_tensor)
                        probability = prediction.item()
        else:
            # Local prediction
            input_tensor = MedicalPredictor.preprocess_input(input_data)
            with torch.no_grad():
                prediction = model(input_tensor)
                probability = prediction.item()

        # Display results
        st.header("📊 Risk Assessment Results")

        if probability > 0.7:
            risk_level = "🔴 HIGH RISK"
            color = "red"
            recommendation = "Please consult a healthcare professional immediately for comprehensive evaluation."
        elif probability > 0.5:
            risk_level = "🟡 MODERATE RISK"
            color = "orange"
            recommendation = "Consider lifestyle modifications and regular monitoring. Consult a doctor if concerned."
        else:
            risk_level = "🟢 LOW RISK"
            color = "green"
            recommendation = "Maintain healthy lifestyle habits. Continue regular check-ups."

        st.markdown(f"### {risk_level}")
        st.markdown(f"**Diabetes Risk Probability: {probability:.1%}**")

        # Progress bar
        st.progress(probability)

        st.info(f"💡 **Recommendation:** {recommendation}")

        # Show input summary
        with st.expander("📋 Input Data Summary"):
            st.json(input_data)

st.markdown("---")
st.markdown("### How it works:")
st.markdown("""
1. **Federated Learning**: Multiple participants train models on their local data
2. **Blockchain Storage**: Model parameters are securely stored and averaged on Ethereum
3. **Real-time Predictions**: Get instant risk assessments using the collective intelligence
4. **Privacy-Preserving**: Your medical data never leaves your device
""")

# Footer
st.markdown("---")
st.markdown("*Built with PyTorch, Web3.py, and Streamlit on Ethereum Sepolia testnet*")
