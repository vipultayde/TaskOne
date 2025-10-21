from web3 import Web3
from solcx import compile_source
import json

# Connect to local blockchain (Ganache)
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))

# Check connection
if not w3.is_connected():
    raise Exception("Cannot connect to blockchain")

# Compile contract
with open('contracts/ParameterStorage.sol', 'r') as file:
    contract_source = file.read()

compiled_sol = compile_source(contract_source, solc_version='0.8.0')
contract_interface = compiled_sol['<stdin>:ParameterStorage']

# Deploy contract
w3.eth.default_account = w3.eth.accounts[0]
ParameterStorage = w3.eth.contract(abi=contract_interface['abi'], bytecode=contract_interface['bin'])
tx_hash = ParameterStorage.constructor().transact()
tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

contract_address = tx_receipt.contractAddress
print(f"Contract deployed at: {contract_address}")

# Save contract address and ABI
with open('contract_info.json', 'w') as f:
    json.dump({
        'address': contract_address,
        'abi': contract_interface['abi']
    }, f)
