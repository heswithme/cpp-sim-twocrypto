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
        
        # Get cached price oracle directly from storage (not calculated view)
        # The price_oracle() view function calculates EMA on-the-fly, but we want
        # the actual stored value for accurate state comparison
        cached_price_oracle = pool.eval("self.cached_price_oracle")
        
        # Calculate xp using internal helper for exact parity with contract logic
        xp = pool.internal._xp([balances[0], balances[1]], price_scale)
        
        # Donation shares
        donation_shares = pool.donation_shares()
        donation_duration = pool.donation_duration()
        last_donation_release_ts = pool.last_donation_release_ts()
        donation_protection_expiry_ts = pool.donation_protection_expiry_ts()
        donation_protection_period = pool.donation_protection_period()
        
        # Compute unlocked donation shares directly via internal function
        # This mirrors _donation_shares(True) exactly
        donation_shares_unlocked = pool.internal._donation_shares()
        
        return {
            "balances": [str(b) for b in balances],
            "xp": [str(xp[0]), str(xp[1])],
            "D": str(pool.D()),
            "virtual_price": str(pool.virtual_price()),
            "xcp_profit": str(pool.xcp_profit()),
            "price_scale": str(price_scale),
            "price_oracle": str(cached_price_oracle),
            "last_prices": str(pool.last_prices()),
            "totalSupply": str(pool.totalSupply()),
            "timestamp": boa.env.timestamp,
            "donation_shares": str(donation_shares),
            "donation_shares_unlocked": str(donation_shares_unlocked),
            "donation_protection_expiry_ts": str(donation_protection_expiry_ts),
            "last_donation_release_ts": str(last_donation_release_ts)
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
                elif action["type"] == "add_liquidity":
                    amts = action["amounts"]
                    donation = bool(action.get("donation", False))
                    # Mint and approve for user
                    token0.mint(user, int(amts[0]))
                    token1.mint(user, int(amts[1]))
                    with boa.env.prank(user):
                        token0.approve(pool.address, 2**256 - 1)
                        token1.approve(pool.address, 2**256 - 1)
                        if donation:
                            # receiver must be empty for donations
                            zero = "0x0000000000000000000000000000000000000000"
                            pool.add_liquidity([int(amts[0]), int(amts[1])], 0, zero, True)
                        else:
                            pool.add_liquidity([int(amts[0]), int(amts[1])], 0, user, False)
                elif action["type"] == "time_travel":
                    # Relative preferred; fallback to absolute if provided
                    secs = int(action.get("seconds", 0))
                    if secs > 0:
                        boa.env.timestamp = boa.env.timestamp + secs
                    elif "timestamp" in action:
                        boa.env.timestamp = int(action["timestamp"])
                        
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
    
    def run_benchmark(self, pool_configs_file: str, sequences_file: str, pool_names: List[str] | None = None) -> Dict:
        """Run complete benchmark with given configurations."""
        # Load configurations
        with open(pool_configs_file, 'r') as f:
            pools_all = json.load(f)["pools"]
        # Optional pool filtering by name
        if pool_names:
            wanted = set(pool_names)
            pools_data = [p for p in pools_all if p.get("name") in wanted]
        else:
            pools_data = pools_all
        
        with open(sequences_file, 'r') as f:
            sequences_data = json.load(f)["sequences"]
        # Expect exactly one sequence; use the first
        if not sequences_data:
            raise RuntimeError("No sequences found in sequences.json")
        sequence = sequences_data[0]
        
        # Pool isolation handled by filtering above
        
        # Deploy infrastructure once
        self.deploy_infrastructure()
        
        # Results storage
        results = []
        total_tests = len(pools_data)
        test_num = 0
        
        # Optionally save only the final state to reduce output size
        save_last_only = os.getenv("SAVE_LAST_ONLY") == "1"

        # Run each test combination
        for pool_config in pools_data:
            test_num += 1
            print(f"\n[{test_num}/{total_tests}] Testing {pool_config['name']}")

            # Deploy mock tokens (reuse existing infrastructure)
            token0, token1 = self.deploy_mock_tokens()

            # Set start timestamp if provided BEFORE pool deploy (important for init state)
            start_ts = int(sequence.get("start_timestamp", boa.env.timestamp))
            boa.env.timestamp = start_ts

            # Deploy pool with config (uses existing infrastructure)
            pool = self.deploy_pool(pool_config, token0, token1)

            # Add initial liquidity
            self.add_initial_liquidity(pool, (token0, token1), pool_config["initial_liquidity"])

            # Execute actions and collect snapshots
            snapshots = self.execute_actions(pool, (token0, token1), sequence["actions"])

            # Store result
            if save_last_only:
                final_state = snapshots[-1] if snapshots else self.take_pool_snapshot(pool)
                results.append({
                    "pool_config": pool_config["name"],
                    "result": {
                        "success": all(s.get("action_success", True) for s in snapshots[1:]) if snapshots else True,
                        "final_state": final_state
                    }
                })
            else:
                results.append({
                    "pool_config": pool_config["name"],
                    "result": {
                        "success": all(s.get("action_success", True) for s in snapshots[1:]),
                        "states": snapshots
                    }
                })
        
        return {"results": results}


def run_vyper_pool(pool_configs_file: str, sequences_file: str, output_file: str, pool_names: List[str] | None = None) -> Dict:
    """Main entry point for running Vyper pool benchmark."""
    # Path to Vyper contracts (repo-relative)
    from pathlib import Path
    contracts_path = str(Path(__file__).resolve().parents[2] / "twocrypto-ng")
    
    # Create runner and execute benchmark
    runner = VyperPoolRunner(contracts_path)
    results = runner.run_benchmark(pool_configs_file, sequences_file, pool_names)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Vyper benchmark complete - {len(results['results'])} tests")
    print(f"✓ Results saved to {output_file}")
    
    return results


def main():
    """Command-line interface."""
    import argparse
    ap = argparse.ArgumentParser(description="Run Vyper pool benchmark (optionally filter pools)")
    ap.add_argument("pool_configs", help="Path to pools.json")
    ap.add_argument("sequences", help="Path to sequences.json")
    ap.add_argument("output", help="Path to write output results.json")
    ap.add_argument("--pools", default=None, help="Comma-separated list of pool names to run")
    args = ap.parse_args()

    pool_configs = args.pool_configs
    sequences = args.sequences
    output = args.output

    # Check files exist
    if not os.path.exists(pool_configs):
        print(f"❌ Pool configs not found: {pool_configs}")
        return 1
    if not os.path.exists(sequences):
        print(f"❌ Sequences not found: {sequences}")
        return 1

    pools_filter = None
    if args.pools:
        pools_filter = [p for p in (s.strip() for s in args.pools.split(",")) if p]

    # Run benchmark
    try:
        run_vyper_pool(pool_configs, sequences, output, pools_filter)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
