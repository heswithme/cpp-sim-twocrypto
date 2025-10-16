# IDR/USD Parameter Investigation Log

## Overview

Investigation of IDR/USD stablecoin pair parameter optimization following CHF/USD methodology. IDR data spans 2020-2025 with ~935k minute candles. Initial price ~0.00007 USD per IDR.

## Statistical Summary: IDR/USD vs Other Currency Pairs

### Dataset Overview

| Currency | Data Points | Time Range (Days) | Start Date | End Date |
|----------|-------------|-------------------|------------|----------|
| IDR/USD | 935,187 | 1768 | 2020-12-13 | 2025-10-16 |
| CHF/USD | 2,086,452 | 2109 | 2020-01-01 | 2025-10-10 |
| EUR/USD | 2,071,085 | 2081 | 2020-01-01 | 2025-09-12 |

### Price Characteristics

| Currency | Min Price | Max Price | Mean Price | Median Price | CV |
|----------|-----------|----------|------------|--------------|----|
| IDR/USD | 0.000054 | 0.000072 | 0.000065 | 0.000064 | 0.0482 |
| CHF/USD | 0.985513 | 1.276992 | 1.105439 | 1.097779 | 0.0519 |
| EUR/USD | 0.953820 | 1.234870 | 1.109531 | 1.096080 | 0.0521 |

### Corrected Volatility Analysis

| Currency | Daily Vol | Annual Vol | Max Daily Return | Min Daily Return |
|----------|-----------|------------|------------------|------------------|
| IDR/USD | 0.0360 | 0.6880 | 5.6928 | -5.6948 |
| CHF/USD | 0.0056 | 0.1068 | 0.4054 | -0.3968 |
| EUR/USD | 0.0052 | 0.0999 | 0.3932 | -0.4859 |

### Notes

- IDR/USD appears more volatile than EUR/CHF from summary stats, although the coefficient of variation suggests this may be small numbers playing tricks on things?  Worth keeping an eye out either way
- May require more conservative A values to maintain price tracking
- Big daily moves of +/- 5.6% notable


## Sweep 1: A vs mid_fee

Consider sweep of A versus mid_fee

**Parameter Ranges**:
- A: 10,000 - 200,000 (32 points, logarithmic)
- mid_fee: 1 - 100 bps (32 points, linear)
- Fixed parameters: out_fee=20bps, ma_time=1h, fee_gamma=0.001, donation_apy=5%

**Commands Used**:
```bash
python python/arb_sim/generate_pools_idr.py && \
python python/arb_sim/arb_sim.py --dustswapfreq 600 -n 10 --events && \
python python/arb_sim/add_geometric_mean.py python/arb_sim/run_data/arb_run_1.json && \
python python/arb_sim/plot_heatmap.py --metrics "apy,apy_coin0,apy_coin0_boost,xcp_profit,vp,vp_boosted,avg_pool_fee,n_rebalances,trades,total_notional_coin0,apy_geom_mean,apy_geom_mean_net,avg_rel_bps,tw_slippage,tw_liq_density" --ncol 5 --font-size 28 --out "heatmaps_A_vs_mid_fee_log.png"
```

### Sweep 1 Results: A vs mid_fee

- `apy_geom_mean` sees a range of 2-10%, with `apy_geom_mean_net` showing APY after donations can range from -1% to 6.8%
- Very few rebalances in the upper right quadrant `A` and `mid_fee`, where slippage spikes locally
- `avg_rel_bps` minimized at `mid_fee` under 40 bps for lower values of `A` and under 20 bps in the upper band
- That red stripe arcing across raw number of `trades` echoes cold zones in several other graphs

### Sweep 2 Results: A vs donation_apy

We run further sweeps using fixed mid_fee of 20 bps

- Interesting sawtooth pattern shows up on many APY charts (`apy`, `apy_geom_mean`, `apy_coin0`, `vp`), suggests the pool's performance may be sensitive to a donation tipping point beyond which extra donation is surplus, must watch this parameter carefully
- Otherwise, other parameters exhibit a similar diagonal curve from low to high values of `A` / `donation_apy` (`apy_coin_0`, `apy_geom_mean_net`, `avg_rel_bps`, `n_rebalances`, `trades` )running diagonally from the low to high value 

### Sweep 3 Results: A vs ma_time

- The sawtooth pattern from Sweep 2 reoccurs here, around an `ma_time` of ~4 hours APY for values of `A` as low as 60 performs similar to values of `A` around 200 below this threshold.
- Heavy rebalances for `ma_time` below 4 hours, tapering off a bit with higher `A` values (pattern echoed in `avg_rel_bps`)

### Sweep 4 Results: A vs ma_time

- Some slightly higher `apy` for mid-tier values of `A` at low `fee_gamma`
- Most parameters show little effect, but lowering `fee_gamma`  of `trades`, `n_rebalances`, 
