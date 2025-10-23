# Decentralized Federated Learning for Medical Prediction

A blockchain-powered federated learning system for diabetes risk assessment using neural networks on Ethereum Sepolia testnet.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Low-Level Design (LLD)](#low-level-design-lld)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Deployment](#deployment)
- [Security](#security)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a **privacy-preserving federated learning system** for medical prediction using blockchain technology. Users can train neural network models on their private medical data locally, then contribute only the mathematical parameters (weights and biases) to a shared model on Ethereum, without ever sharing their actual medical data.

### Key Benefits:

- ✅ **Privacy-First**: Medical data never leaves user devices
- ✅ **Decentralized**: No central authority controls the data
- ✅ **Transparent**: All model updates are recorded on blockchain
- ✅ **Collaborative**: Multiple participants improve global model accuracy
- ✅ **Cost-Effective**: Uses testnet for development, low gas fees

## 🏗️ Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Device   │    │   Smart Contract│    │   Blockchain    │
│                 │    │                 │    │   (Sepolia)     │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │                 │
│ │Local Dataset│ │    │ │Federated    │ │    │ ┌─────────────┐ │
│ │(Private)    │ │    │ │Learning     │ │    │ │Transaction  │ │
│ └─────────────┘ │    │ │Contract     │ │    │ │History      │ │
│                 │    │ └─────────────┘ │    │ └─────────────┘ │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │Neural       │ │    │ │Model Params │◄────┼──►│Global Model│ │
│ │Network      │◄────┼──►│Storage      │ │    │ │Aggregation  │ │
│ │Training     │ │    │ └─────────────┘ │    │ └─────────────┘ │
│ └─────────────┘ │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Architecture

1. **Frontend Layer (React + MetaMask)**

   - User interface for model selection and training
   - MetaMask wallet integration
   - Real-time blockchain interaction

2. **Backend Layer (Python + PyTorch)**

   - Neural network training scripts
   - Parameter serialization and submission
   - Local model evaluation

3. **Blockchain Layer (Solidity + Web3)**

   - Smart contract for parameter storage
   - Decentralized parameter aggregation
   - Transparent transaction history

4. **Data Layer**
   - Local medical datasets (diabetes, heart disease)
   - Off-chain parameter storage
   - Secure data preprocessing

## 📐 Low-Level Design (LLD)

### Smart Contract Design

```solidity
contract FederatedLearning {
    // State Variables
    mapping(address => string) public modelHashes;        // User -> Model Hash
    address[] public participants;                        // List of participants
    mapping(address => bool) public hasSubmitted;         // Submission tracking
    uint256 public submissionCount;                       // Total submissions
    mapping(address => uint256) public predictions;       // Stored predictions

    // Events
    event ModelSubmitted(address user, string modelHash);
    event PredictionMade(address user, uint256 prediction);

    // Functions
    function submitModel(string memory modelHash) public
    function getParticipantCount() public view returns (uint256)
    function getModelHash(address user) public view returns (string memory)
    function storePrediction(uint256 prediction) public
}
```

### Neural Network Architecture

#### Diabetes Prediction Model

```python
class MedicalPredictor(nn.Module):
    def __init__(self, input_size=8, hidden_size=16, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)      # Input: 8 features
        self.fc2 = nn.Linear(16, 8)      # Hidden layer
        self.fc3 = nn.Linear(8, 1)       # Output: probability
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc3(x))
```

**Input Features (8):**

- Pregnancies, Glucose, BloodPressure, SkinThickness
- Insulin, BMI, DiabetesPedigreeFunction, Age

**Output:** Diabetes risk probability (0-1)

#### Heart Disease Prediction Model

```python
class HeartPredictor(nn.Module):
    def __init__(self, input_size=13, hidden_size=32, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(13, 32)     # Input: 13 features
        self.fc2 = nn.Linear(32, 16)     # Hidden layer
        self.fc3 = nn.Linear(16, 1)      # Output: probability
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc3(x))
```

**Input Features (13):**

- Age, Sex, Chest Pain Type, Resting BP, Cholesterol
- Fasting Blood Sugar, Resting ECG, Max Heart Rate
- Exercise Angina, ST Depression, ST Slope, Major Vessels, Thalassemia

**Output:** Heart disease risk probability (0-1)

### Data Flow Diagram

```
1. User selects model type (diabetes/heart)
2. Local training on private dataset
3. Parameter extraction and serialization
4. Hash generation for integrity
5. MetaMask transaction creation
6. Smart contract parameter storage
7. Global model aggregation
8. Prediction using federated model
```

### Database Schema (Off-chain Storage)

```json
{
  "contract_info": {
    "address": "0x738702AcF262E8BC2Bf37cdFe8e53d39F573B04b",
    "abi": [...],
    "network": "sepolia",
    "rpc_url": "https://sepolia.infura.io/v3/...",
    "chain_id": 11155111
  },
  "model_parameters": {
    "user_address": "0x...",
    "model_hash": "0x...",
    "parameters": {
      "fc1.weight": [[...], [...]],
      "fc1.bias": [...],
      "fc2.weight": [[...], [...]],
      "fc2.bias": [...]
    },
    "timestamp": 1234567890,
    "model_type": "diabetes|heart"
  }
}
```

## ✨ Features

### Core Features

- 🔐 **Privacy-Preserving**: Medical data stays local
- 🌐 **Decentralized**: Blockchain-based parameter sharing
- 🤖 **AI-Powered**: PyTorch neural networks
- 💰 **Cost-Effective**: Sepolia testnet deployment
- 📱 **User-Friendly**: React web interface
- 🔄 **Real-Time**: Live blockchain updates

### Advanced Features

- 📊 **Multi-Model Support**: Diabetes and heart disease
- 🎯 **Model Aggregation**: Federated averaging
- 📈 **Progress Tracking**: Training and prediction progress
- 🔍 **Transaction Monitoring**: MetaMask integration
- 📋 **Tutorial System**: Onboarding for new users
- 🎨 **Responsive UI**: Mobile-friendly design

## 📋 Prerequisites

### System Requirements

- **OS**: Windows 10/11, macOS, or Linux
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Network**: Stable internet connection

### Software Dependencies

- **Python**: 3.8 or higher
- **Node.js**: 16.x or higher
- **npm**: 7.x or higher
- **Git**: 2.x or higher
- **MetaMask**: Browser extension

### Blockchain Requirements

- **Wallet**: MetaMask with Sepolia testnet configured
- **Test ETH**: Free from [Sepolia Faucet](https://sepoliafaucet.com/)
- **Network**: Sepolia testnet (chain ID: 11155111)

## 🚀 Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/federated-medical-ai.git
cd federated-medical-ai
```

### 2. Python Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch web3 streamlit numpy pandas scikit-learn
```

### 3. Smart Contract Deployment

```bash
# Deploy to Sepolia testnet
python scripts/deploy_sepolia.py
```

This creates `contract_info.json` with deployment details.

### 4. React Frontend Setup

```bash
cd federated-medical-app
npm install
npm start
```

Access at http://localhost:3000

### 5. MetaMask Configuration

1. Install MetaMask extension
2. Switch to Sepolia testnet
3. Get free test ETH from faucet
4. Import your account if needed

## 📖 Usage

### Basic Workflow

1. **Select Model**: Choose diabetes or heart disease prediction
2. **Connect Wallet**: Link MetaMask to the application
3. **Train Locally**: AI trains on your data (privacy preserved)
4. **Submit Parameters**: Share model improvements on blockchain
5. **Get Predictions**: Use enhanced global model for assessments

### Training Scripts

#### Diabetes Model Training

```bash
python scripts/train.py
```

#### Heart Disease Model Training

```bash
python scripts/train_heart.py
```

### Web Interface Usage

#### React Frontend (http://localhost:3000)

- Click "Connect MetaMask" (auto-switches to Sepolia)
- Select medical model type
- Click "Train & Submit Model" to participate
- Enter medical data for risk assessment
- View prediction results

#### Streamlit UI (http://localhost:8501)

```bash
streamlit run ui_new.py
```

## 🔧 API Reference

### Smart Contract Functions

#### `submitModel(string modelHash)`

Submits a model hash to the blockchain.

**Parameters:**

- `modelHash`: SHA-256 hash of model parameters

**Events Emitted:**

- `ModelSubmitted(address user, string modelHash)`

#### `getParticipantCount() → uint256`

Returns the total number of participants.

#### `getModelHash(address user) → string`

Returns the model hash for a specific user.

#### `storePrediction(uint256 prediction)`

Stores a prediction result on-chain.

### Python Classes

#### `MedicalPredictor`

Neural network class for medical prediction.

**Methods:**

- `__init__(input_size, hidden_size, output_size)`: Initialize network
- `forward(x)`: Forward pass
- `get_parameters_as_int()`: Serialize parameters for blockchain
- `set_parameters_from_int(params)`: Deserialize parameters

## 🧪 Testing

### Unit Tests

```bash
# Run Python tests
python -m pytest tests/

# Run contract tests
truffle test
```

### Integration Tests

```bash
# Test full federated learning cycle
python scripts/test_integration.py
```

### Manual Testing Checklist

- [ ] MetaMask connection works
- [ ] Network switches to Sepolia automatically
- [ ] Model training completes successfully
- [ ] Parameter submission transactions confirm
- [ ] Prediction functionality works
- [ ] UI updates in real-time

## 🚢 Deployment

### Local Development

```bash
# Start all services
npm start              # React frontend (port 3000)
streamlit run ui_new.py # Streamlit UI (port 8501)
```

### Production Deployment

```bash
# Build React app
cd federated-medical-app
npm run build

# Deploy to web server
# Copy build/ folder to your web server
```

### Contract Deployment

```bash
# Deploy to different networks
python scripts/deploy_mainnet.py    # Ethereum mainnet
python scripts/deploy_polygon.py    # Polygon network
```

## 🔒 Security

### Privacy Measures

- **Local Training**: All data processing happens client-side
- **Parameter Only**: Only mathematical weights shared, not raw data
- **Hash Verification**: Model integrity through cryptographic hashing
- **No Data Storage**: Medical data never stored on blockchain

### Smart Contract Security

- **Access Control**: Only authorized functions can modify state
- **Input Validation**: All inputs validated before processing
- **Gas Optimization**: Efficient gas usage to minimize costs
- **Event Logging**: All state changes logged for transparency

### Frontend Security

- **MetaMask Integration**: Secure wallet communication
- **Input Sanitization**: All user inputs validated
- **HTTPS Only**: Secure communication channels
- **CORS Protection**: Cross-origin request protection

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 for Python code
- Use ESLint for JavaScript/React code
- Write comprehensive tests
- Update documentation
- Ensure cross-platform compatibility

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Datasets**: UCI Machine Learning Repository
- **PyTorch**: For neural network framework
- **Ethereum**: For blockchain infrastructure
- **MetaMask**: For wallet integration
- **Open Source Community**: For invaluable tools and libraries

## 📞 Support

For support and questions:

- 📧 Email: support@federatedmedical.ai
- 💬 Discord: [Join our community](https://discord.gg/federatedmedical)
- 📖 Documentation: [Full docs](https://docs.federatedmedical.ai)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/federated-medical-ai/issues)

---

**Built with ❤️ for privacy-preserving healthcare AI**
