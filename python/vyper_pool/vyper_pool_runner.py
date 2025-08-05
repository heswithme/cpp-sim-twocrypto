#!/usr/bin/env python3
"""
Vyper pool benchmark runner - all-in-one module for deploying and testing Vyper pools
"""
import json
import os
import sys
from typing import Dict, List, Any, Tuple
import boa


class VyperPoolRunner:
    """Complete Vyper pool deployment and testing infrastructure."""
    
    def __init__(self, contracts_path: str):
        """Initialize with path to Vyper contracts."""
        self.contracts_path = contracts_path
        self.factory = None
        self.math_contract = None
        self.views_contract = None
        self.amm_implementation = None
        self.owner = None
        
    def deploy_infrastructure(self):
        """Deploy all required Vyper contracts."""
        # Contract paths
        math_path = os.path.join(self.contracts_path, "contracts/main/StableswapMath.vy")
        views_path = os.path.join(self.contracts_path, "contracts/main/TwocryptoView.vy")
        pool_path = os.path.join(self.contracts_path, "contracts/main/Twocrypto.vy")
        factory_path = os.path.join(self.contracts_path, "contracts/main/TwocryptoFactory.vy")
        
        print("Deploying Vyper infrastructure...")
        
        # Deploy math contract
        self.math_contract = boa.load(math_path)
        print(f"  ✓ Math contract: {self.math_contract.address}")
        
        # Deploy views contract  
        self.views_contract = boa.load(views_path)
        print(f"  ✓ Views contract: {self.views_contract.address}")
        
        # Deploy pool implementation as blueprint
        pool_contract = boa.load_partial(pool_path)
        self.amm_implementation = pool_contract.deploy_as_blueprint()
        print("  ✓ Pool implementation deployed")
        
        # Deploy factory
        self.factory = boa.load(factory_path)
        print(f"  ✓ Factory: {self.factory.address}")
        
        # Initialize factory
        fee_receiver = boa.env.generate_address()
        self.owner = boa.env.generate_address()  # Store owner for later use
        self.factory.initialise_ownership(fee_receiver, self.owner)
        
        # Set implementations
        with boa.env.prank(self.owner):
            self.factory.set_pool_implementation(self.amm_implementation, 0)
            self.factory.set_views_implementation(self.views_contract.address)
            self.factory.set_math_implementation(self.math_contract.address)
        
        print("  ✓ Infrastructure ready")
        
    def deploy_mock_tokens(self) -> Tuple[Any, Any]:
        """Deploy two mock ERC20 tokens for testing."""
        # Use our custom mock with snekmate
        mock_path = os.path.join(os.path.dirname(__file__), "mock_erc20.vy")
        
        # Deploy tokens - snekmate should be available via uv
        token0 = boa.load(mock_path, "Token0", "TK0")
        token1 = boa.load(mock_path, "Token1", "TK1")
        
        # Mint initial supply to deployer
        deployer = boa.env.eoa
        large_amount = 10**9 * 10**18  # 1 billion tokens
        token0.mint(deployer, large_amount)
        token1.mint(deployer, large_amount)
        
        return token0, token1
    
    def deploy_pool(self, params: Dict[str, str], token0: Any, token1: Any) -> Any:
        """Deploy a pool with given parameters."""
        # Deploy pool through factory
        pool_address = self.factory.deploy_pool(
            "Test Pool",                         # name
            "TEST",                              # symbol
            [token0.address, token1.address],    # coins
            0,                                   # implementation_id
            int(params["A"]),                    # A
            int(params["gamma"]),                # gamma
            int(params["mid_fee"]),              # mid_fee
            int(params["out_fee"]),              # out_fee
            int(params["fee_gamma"]),            # fee_gamma
            int(params["allowed_extra_profit"]), # allowed_extra_profit
            int(params["adjustment_step"]),      # adjustment_step
            int(params["ma_time"]),              # ma_exp_time
            int(params["initial_price"]),        # initial_price
        )
        
        # Load the pool contract at the deployed address
        pool_path = os.path.join(self.contracts_path, "contracts/main/Twocrypto.vy")
        pool = boa.load_partial(pool_path).at(pool_address)
        
        # Set periphery using the owner we saved during initialization
        with boa.env.prank(self.owner):
            pool.set_periphery(self.views_contract.address, self.math_contract.address)
        
        return pool
    
    def add_initial_liquidity(self, pool: Any, tokens: Tuple[Any, Any], amounts: List[str]) -> None:
        """Add initial liquidity to pool."""
        token0, token1 = tokens
        user = boa.env.generate_address()
        
        # Mint tokens
        token0.mint(user, int(amounts[0]))
        token1.mint(user, int(amounts[1]))
        
        # Approve pool
        with boa.env.prank(user):
            token0.approve(pool.address, 2**256 - 1)
            token1.approve(pool.address, 2**256 - 1)
            
            # Add liquidity
            pool.add_liquidity([int(amounts[0]), int(amounts[1])], 0)
    
    def take_pool_snapshot(self, pool: Any) -> Dict[str, Any]:
        """Take snapshot of Vyper pool state."""
        # Get balances
        balances = [pool.balances(0), pool.balances(1)]
        
        # Get price scale
        price_scale = pool.price_scale()
        
        # Calculate xp - simplified version
        xp = [balances[0], balances[1] * price_scale // 10**18]
        
        return {
            "balances": [str(b) for b in balances],
            "xp": [str(xp[0]), str(xp[1])],
            "D": str(pool.D()),
            "virtual_price": str(pool.virtual_price()),
            "xcp_profit": str(pool.xcp_profit()),
            "price_scale": str(price_scale),
            "price_oracle": str(pool.price_oracle()),
            "last_prices": str(pool.last_prices()),
            "totalSupply": str(pool.totalSupply()),
            "timestamp": boa.env.timestamp
        }
    
    def execute_actions(self, pool: Any, tokens: Tuple[Any, Any], 
                       actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute action sequence on Vyper pool and collect snapshots."""
        token0, token1 = tokens
        user = boa.env.generate_address()
        snapshots = []
        
        # Take initial snapshot after deployment
        snapshots.append(self.take_pool_snapshot(pool))
        
        for i, action in enumerate(actions):
            # Apply time delta if present
            if action.get("time_delta", 0) > 0:
                boa.env.timestamp = boa.env.timestamp + action["time_delta"]
            
            success = True
            error = None
            
            try:
                if action["type"] == "exchange":
                    # Mint tokens for exchange
                    if action["i"] == 0:
                        token0.mint(user, int(action["dx"]))
                        with boa.env.prank(user):
                            token0.approve(pool.address, 2**256 - 1)
                    else:
                        token1.mint(user, int(action["dx"]))
                        with boa.env.prank(user):
                            token1.approve(pool.address, 2**256 - 1)
                    
                    # Execute exchange
                    with boa.env.prank(user):
                        pool.exchange(action["i"], action["j"], int(action["dx"]), 0)
                        
            except Exception as e:
                success = False
                error = str(e)
                print(f"  ! Action {i} failed: {e}")
            
            # Take snapshot after action
            snapshot = self.take_pool_snapshot(pool)
            snapshot["action_success"] = success
            if error:
                snapshot["error"] = error
            snapshots.append(snapshot)
        
        return snapshots
    
    def run_benchmark(self, pool_configs_file: str, sequences_file: str) -> Dict:
        """Run complete benchmark with given configurations."""
        # Load configurations
        with open(pool_configs_file, 'r') as f:
            pools_data = json.load(f)["pools"]
        
        with open(sequences_file, 'r') as f:
            sequences_data = json.load(f)["sequences"]
        
        # Deploy infrastructure once
        self.deploy_infrastructure()
        
        # Results storage
        results = []
        total_tests = len(pools_data) * len(sequences_data)
        test_num = 0
        
        # Run each test combination
        for pool_config in pools_data:
            for sequence in sequences_data:
                test_num += 1
                print(f"\n[{test_num}/{total_tests}] Testing {pool_config['name']} with {sequence['name']}")
                
                # Deploy mock tokens (reuse existing infrastructure)
                token0, token1 = self.deploy_mock_tokens()
                
                # Deploy pool with config (uses existing infrastructure)
                pool = self.deploy_pool(pool_config, token0, token1)
                
                # Add initial liquidity
                self.add_initial_liquidity(pool, (token0, token1), pool_config["initial_liquidity"])
                
                # Execute actions and collect snapshots
                snapshots = self.execute_actions(pool, (token0, token1), sequence["actions"])
                
                # Store result
                results.append({
                    "pool_config": pool_config["name"],
                    "sequence": sequence["name"],
                    "result": {
                        "success": all(s.get("action_success", True) for s in snapshots[1:]),
                        "states": snapshots
                    }
                })
        
        return {"results": results}


def run_vyper_pool(pool_configs_file: str, sequences_file: str, output_file: str) -> Dict:
    """Main entry point for running Vyper pool benchmark."""
    # Path to Vyper contracts
    contracts_path = "/Users/michael/Documents/projects/cpp-twocrypto/twocrypto-ng"
    
    # Create runner and execute benchmark
    runner = VyperPoolRunner(contracts_path)
    results = runner.run_benchmark(pool_configs_file, sequences_file)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Vyper benchmark complete - {len(results['results'])} tests")
    print(f"✓ Results saved to {output_file}")
    
    return results


def main():
    """Command-line interface."""
    if len(sys.argv) != 4:
        print("Usage: python vyper_pool_runner.py <pool_configs.json> <sequences.json> <output.json>")
        return 1
    
    pool_configs = sys.argv[1]
    sequences = sys.argv[2]
    output = sys.argv[3]
    
    # Check files exist
    if not os.path.exists(pool_configs):
        print(f"❌ Pool configs not found: {pool_configs}")
        return 1
    
    if not os.path.exists(sequences):
        print(f"❌ Sequences not found: {sequences}")
        return 1
    
    # Run benchmark
    try:
        run_vyper_pool(pool_configs, sequences, output)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())