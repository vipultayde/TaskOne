from web3 import Web3
import json
import solcx

# Ganache local network configuration
GANACHE_RPC_URL = 'http://127.0.0.1:7545'  # Updated to match user's RPC server
# Use the user's provided Ganache account
PRIVATE_KEY = '0x8bc316605eef23969de300774045274d7414b37a5805f637511a91a1a7425262'
ACCOUNT_ADDRESS = '0x427e70dBE6072b288E945f4ea4B5A2073A51ee36'

def deploy_to_ganache():
    try:
        # Compile contract
        print("Compiling contract...")
        compiled_sol = solcx.compile_source(
            open('contracts/FederatedLearning.sol').read(),
            output_values=['abi', 'bin'],
            solc_version='0.8.0'
        )

        contract_id, contract_interface = compiled_sol.popitem()
        abi = contract_interface['abi']
        bytecode = contract_interface['bin']

        # Connect to Ganache
        print("Connecting to Ganache...")
        w3 = Web3(Web3.HTTPProvider(GANACHE_RPC_URL))

        if not w3.is_connected():
            raise Exception("Failed to connect to Ganache. Make sure Ganache is running on http://127.0.0.1:7545")

        # Set up account
        account = w3.eth.account.from_key(PRIVATE_KEY)
        w3.eth.default_account = account.address

        # Check account balance
        balance = w3.eth.get_balance(account.address)
        balance_eth = w3.from_wei(balance, 'ether')
        nonce = w3.eth.get_transaction_count(account.address)
        print(f"Account: {account.address}")
        print(f"Balance: {balance_eth} ETH")
        print(f"Nonce: {nonce}")

        # Deploy contract
        print("Deploying contract...")
        FederatedLearning = w3.eth.contract(abi=abi, bytecode=bytecode)
        # Get current block gas limit
        block_gas_limit = w3.eth.get_block('latest')['gasLimit']
        print(f"Block gas limit: {block_gas_limit}")

        tx = FederatedLearning.constructor().build_transaction({
            'chainId': 1337,  # Ganache default chain ID
            'gas': int(block_gas_limit * 0.8),  # Use 80% of block gas limit
            'gasPrice': 1000000000,  # 1 gwei - very low gas price for Ganache
            'nonce': nonce,
        })

        # Sign and send transaction
        signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        print(f"Transaction hash: {tx_hash.hex()}")

        # Wait for transaction receipt
        print("Waiting for transaction confirmation...")
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

        contract_address = tx_receipt.contractAddress
        print(f"✅ Contract deployed successfully at: {contract_address}")

        # Save contract info
        contract_info = {
            'address': contract_address,
            'abi': abi,
            'network': 'ganache',
            'rpc_url': GANACHE_RPC_URL,
            'chain_id': 1337
        }

        with open('federated-medical-app/src/contract_info.json', 'w') as f:
            json.dump(contract_info, f, indent=2)

        print("Contract info saved to federated-medical-app/src/contract_info.json")
        print("\n🚀 Deployment completed successfully!")
        print(f"📄 Contract Address: {contract_address}")
        print("🔗 Network: Ganache Local")
        return contract_address

    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        raise

if __name__ == "__main__":
    deploy_to_ganache()
