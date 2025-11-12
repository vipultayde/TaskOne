from web3 import Web3
import json
import solcx

# Sepolia testnet configuration
SEPOLIA_RPC_URL = 'https://sepolia.infura.io/v3/7f9be7585b0942df9ae022fc6e5c2ffb'
PRIVATE_KEY = '0xedf6222a88939d510cadd36536e27efb1d396958e637d62c8cc203924fce78f1'  # User's private key
ACCOUNT_ADDRESS = '0xcC933Da78ab228bee7EE0403C4b4F89CB015DDBD'  # User's address

# Compile contract
compiled_sol = solcx.compile_source(
    open('contracts/FederatedLearning.sol').read(),
    output_values=['abi', 'bin'],
    solc_version='0.8.0'
)

contract_id, contract_interface = compiled_sol.popitem()
abi = contract_interface['abi']
bytecode = contract_interface['bin']

# Connect to Sepolia testnet
w3 = Web3(Web3.HTTPProvider(SEPOLIA_RPC_URL))

# Set up account
account = w3.eth.account.from_key(PRIVATE_KEY)
w3.eth.default_account = account.address

# Get nonce
nonce = w3.eth.get_transaction_count(account.address)

# Check account balance first
balance = w3.eth.get_balance(account.address)
balance_eth = w3.from_wei(balance, 'ether')
print(f"Account balance: {balance_eth} ETH")

if balance == 0:
    print("❌ Account has no ETH. Please fund your Sepolia account with test ETH from https://sepoliafaucet.com/")
    exit(1)

# Deploy contract
FederatedLearning = w3.eth.contract(abi=abi, bytecode=bytecode)
tx = FederatedLearning.constructor().build_transaction({
    'chainId': 11155111,  # Sepolia chain ID
    'gas': 3000000,
    'gasPrice': 20000000000,  # 20 gwei - lower gas price
    'nonce': nonce,
})

# Sign and send transaction
signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

# Save contract info
contract_info = {
    'address': tx_receipt.contractAddress,
    'abi': abi,
    'network': 'sepolia',
    'rpc_url': SEPOLIA_RPC_URL
}

with open('contract_info.json', 'w') as f:
    json.dump(contract_info, f)

print(f"Contract deployed on Sepolia at: {tx_receipt.contractAddress}")
print(f"Transaction hash: {tx_hash.hex()}")
