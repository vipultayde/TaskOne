# Decentralized Federated Learning for Medical Prediction

A blockchain-powered federated learning system for diabetes risk assessment using neural networks on Ethereum Sepolia testnet.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- MetaMask wallet with Sepolia ETH (get free test ETH from https://sepoliafaucet.com/)
- Node.js & npm (for React frontend)
- Git

### 1. Install Python Dependencies

```bash
pip install torch web3 streamlit numpy
```

### 2. Deploy Smart Contract

```bash
python scripts/deploy_sepolia.py
```

This creates `contract_info.json` with deployment details.

### 3. Train & Submit Parameters (Backend)

For **Diabetes Prediction**:

```bash
python scripts/train.py
```

For **Heart Disease Prediction**:

```bash
python scripts/train_heart.py
```

Trains local NN model and submits parameters to blockchain.

### 4. Average Parameters

```bash
python scripts/predict.py
```

Triggers on-chain parameter averaging from all participants.

### 5. Launch Streamlit Web UI

```bash
streamlit run ui_new.py
```

Access at http://localhost:8503

### 6. Launch React Frontend (Alternative UI)

```bash
cd federated-medical-app
npm install
npm start
```

Access React app at http://localhost:3000 with MetaMask integration

## 🎯 React Frontend Setup (Detailed Steps)

### Prerequisites for React App

- Node.js 16+ and npm
- MetaMask browser extension
- Sepolia test ETH in your MetaMask wallet

### Step-by-Step React Setup:

1. **Navigate to React App Directory:**

   ```bash
   cd federated-medical-app
   ```

2. **Install Dependencies:**

   ```bash
   npm install
   ```

   This installs ethers.js, axios, and other required packages.

3. **Start Development Server:**

   ```bash
   npm start
   ```

   The app will open at http://localhost:3000

4. **MetaMask Setup:**

   - Ensure MetaMask is installed and connected to Sepolia testnet
   - Make sure you have test ETH (faucet: https://sepoliafaucet.com/)

5. **Using the React App:**
   - Click "Connect MetaMask" (automatically switches to Sepolia)
   - Click "🚀 Train & Submit Model Parameters" to participate
   - View participant count and transaction confirmations

### React App Features:

- 🔗 MetaMask wallet integration
- 🌐 Automatic Sepolia network switching
- 🧠 Local neural network training simulation
- ⛓️ Real-time blockchain parameter submission
- 👥 Live participant count updates
- 📊 Transaction status monitoring

## 📁 Project Structure

```
├── contracts/
│   └── FederatedLearning.sol      # Smart contract for parameter storage/averaging
├── data/
│   ├── diabetes.csv               # Pima Indians Diabetes Dataset (768 samples)
│   └── heart_disease/
│       └── heart.csv              # Cleveland Heart Disease Dataset (303 samples)
├── model/
│   ├── nn_model.py                # PyTorch neural network for diabetes prediction
│   └── heart_model.py             # PyTorch neural network for heart disease prediction
├── scripts/
│   ├── deploy_sepolia.py          # Contract deployment to Sepolia
│   ├── train.py                   # Diabetes training & blockchain submission
│   ├── train_heart.py             # Heart disease training & blockchain submission
│   └── predict.py                 # Parameter averaging & prediction testing
├── federated-medical-app/         # React frontend with MetaMask integration
├── ui_new.py                      # Streamlit web interface
├── contract_info.json             # Deployed contract details
└── README.md
```

## 🔧 How It Works

1. **Local Training**: Users train NN models on their private medical data
2. **Parameter Submission**: Trained weights/biases sent to blockchain (gas fees apply)
3. **Federated Averaging**: Smart contract averages parameters from all participants
4. **Global Model**: UI loads averaged parameters for real-time predictions
5. **Privacy**: Medical data never leaves user devices

## 💰 Gas Fees

- **Parameter Submission**: ~0.001 ETH per submission
- **Network**: Sepolia testnet (free faucet ETH available)
- **Privacy**: Only mathematical parameters sent, no medical data

## 🔐 Privacy & Security

- ✅ Medical data stays local
- ✅ Only NN parameters shared on blockchain
- ✅ Decentralized parameter averaging
- ✅ No central data collection

## 🏥 Medical Features

### Diabetes Prediction Model

The diabetes model uses the **Pima Indians Diabetes Dataset** with 8 features:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration (mg/dL)
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age (years)

**Dataset Source**: The Pima Indians Diabetes Dataset is a publicly available dataset from the UCI Machine Learning Repository, containing 768 samples from Pima Indian women aged 21 and above.

### Heart Disease Prediction Model

The heart disease model uses the **Cleveland Heart Disease Dataset** with 13 features:

- **Age**: Age in years
- **Sex**: Sex (1 = male; 0 = female)
- **CP**: Chest pain type (0-3)
- **Trestbps**: Resting blood pressure (mm Hg)
- **Chol**: Serum cholesterol (mg/dl)
- **FBS**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- **Restecg**: Resting electrocardiographic results (0-2)
- **Thalach**: Maximum heart rate achieved
- **Exang**: Exercise induced angina (1 = yes; 0 = no)
- **Oldpeak**: ST depression induced by exercise relative to rest
- **Slope**: Slope of the peak exercise ST segment (0-2)
- **CA**: Number of major vessels colored by fluoroscopy (0-3)
- **Thal**: Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect)

**Dataset Source**: The Cleveland Heart Disease Dataset is a publicly available dataset from the UCI Machine Learning Repository, containing 303 samples for heart disease diagnosis.

## 🛠️ Technical Details

- **Blockchain**: Ethereum Sepolia testnet
- **Smart Contract**: Solidity with parameter averaging logic
- **ML Framework**: PyTorch neural network
- **Web3**: Python web3.py for blockchain interaction
- **UI**: Streamlit with MetaMask integration

## 📊 Usage Flow

1. Deploy contract to Sepolia
2. Multiple users run `train.py` to submit parameters
3. Run `predict.py` to average parameters
4. Launch UI for predictions using federated model

## 🤝 Contributing

Users contribute by training local models and submitting parameters, improving the global model's accuracy through federated learning.

---

Built with ❤️ for privacy-preserving AI in healthcare
