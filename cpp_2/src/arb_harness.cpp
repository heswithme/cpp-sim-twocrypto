#include <boost/json.hpp>
#include <boost/json/src.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "arbitrageur.hpp"
#include "pool_runner.hpp"
#include "real_type.hpp"
#include "trader.hpp"

namespace json = boost::json;
namespace fs = std::filesystem;

using sim::RealT;

namespace {

struct PoolInit {
    std::array<RealT, 2> precisions{RealT(1), RealT(1)};
    std::array<RealT, 2> initial_liq{RealT(0), RealT(0)};
    RealT A{static_cast<RealT>(100000)};
    RealT gamma{static_cast<RealT>(0)};
    RealT mid_fee{static_cast<RealT>(3e-4)};
    RealT out_fee{static_cast<RealT>(5e-4)};
    RealT fee_gamma{static_cast<RealT>(0.23)};
    RealT allowed_extra{static_cast<RealT>(1e-3)};
    RealT adj_step{static_cast<RealT>(1e-3)};
    RealT ma_time{static_cast<RealT>(600)};
    RealT initial_price{static_cast<RealT>(1)};
    uint64_t start_ts{0};
};

struct PoolSpec {
    std::string tag;
    PoolInit init;
    sim::Costs costs;
};

struct HarnessInput {
    std::vector<PoolSpec> pools;
    fs::path data_path;
    fs::path config_dir;
    std::string raw_data_path;
};


RealT parse_scaled_1e18(const json::value& v) {
    if (v.is_string()) return static_cast<RealT>(std::strtold(v.as_string().c_str(), nullptr) / 1e18L);
    if (v.is_double()) return static_cast<RealT>(v.as_double() / 1e18);
    if (v.is_int64())  return static_cast<RealT>(static_cast<long double>(v.as_int64()) / 1e18L);
    if (v.is_uint64()) return static_cast<RealT>(static_cast<long double>(v.as_uint64()) / 1e18L);
    return RealT(0);
}

RealT parse_fee_1e10(const json::value& v) {
    if (v.is_string()) return static_cast<RealT>(std::strtold(v.as_string().c_str(), nullptr) / 1e10L);
    if (v.is_double()) return static_cast<RealT>(v.as_double() / 1e10);
    if (v.is_int64())  return static_cast<RealT>(static_cast<long double>(v.as_int64()) / 1e10L);
    if (v.is_uint64()) return static_cast<RealT>(static_cast<long double>(v.as_uint64()) / 1e10L);
    return RealT(0);
}

RealT parse_plain_real(const json::value& v) {
    if (v.is_string()) return static_cast<RealT>(std::strtold(v.as_string().c_str(), nullptr));
    if (v.is_double()) return static_cast<RealT>(v.as_double());
    if (v.is_int64())  return static_cast<RealT>(v.as_int64());
    if (v.is_uint64()) return static_cast<RealT>(v.as_uint64());
    return RealT(0);
}

uint64_t parse_timestamp(const json::value& v) {
    uint64_t ts = 0;
    if (v.is_uint64()) ts = v.as_uint64();
    else if (v.is_int64()) ts = static_cast<uint64_t>(v.as_int64());
    else if (v.is_double()) ts = static_cast<uint64_t>(v.as_double());
    else if (v.is_string()) ts = static_cast<uint64_t>(std::strtoll(v.as_string().c_str(), nullptr, 10));
    if (ts > 10000000000ULL) ts /= 1000ULL;
    return ts;
}

std::string read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open file: " + path.string());
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

void parse_pool_entry(const json::object& entry, PoolSpec& spec) {
    const json::object* pool_obj = nullptr;
    if (entry.contains("pool")) {
        pool_obj = &entry.at("pool").as_object();
    } else {
        pool_obj = &entry;
    }
    if (!pool_obj) throw std::runtime_error("pool entry missing 'pool' object");

    if (auto* v = pool_obj->if_contains("initial_liquidity")) {
        const auto& arr = v->as_array();
        if (arr.size() >= 2) {
            spec.init.initial_liq[0] = parse_scaled_1e18(arr[0]);
            spec.init.initial_liq[1] = parse_scaled_1e18(arr[1]);
        }
    }
    if (auto* v = pool_obj->if_contains("A")) spec.init.A = parse_plain_real(*v);
    if (auto* v = pool_obj->if_contains("gamma")) spec.init.gamma = parse_scaled_1e18(*v);
    if (auto* v = pool_obj->if_contains("mid_fee")) spec.init.mid_fee = parse_fee_1e10(*v);
    if (auto* v = pool_obj->if_contains("out_fee")) spec.init.out_fee = parse_fee_1e10(*v);
    if (auto* v = pool_obj->if_contains("fee_gamma")) spec.init.fee_gamma = parse_scaled_1e18(*v);
    if (auto* v = pool_obj->if_contains("allowed_extra_profit")) spec.init.allowed_extra = parse_scaled_1e18(*v);
    if (auto* v = pool_obj->if_contains("adjustment_step")) spec.init.adj_step = parse_scaled_1e18(*v);
    if (auto* v = pool_obj->if_contains("ma_time")) spec.init.ma_time = parse_plain_real(*v);
    if (auto* v = pool_obj->if_contains("initial_price")) spec.init.initial_price = parse_scaled_1e18(*v);
    if (auto* v = pool_obj->if_contains("start_timestamp")) spec.init.start_ts = static_cast<uint64_t>(parse_plain_real(*v));

    if (auto* costs_obj = entry.if_contains("costs")) {
        const auto& co = costs_obj->as_object();
        if (auto* v = co.if_contains("arb_fee_bps")) spec.costs.arb_fee_bps = parse_plain_real(*v);
        if (auto* v = co.if_contains("gas_coin0")) spec.costs.gas_coin0 = parse_plain_real(*v);
        if (auto* v = co.if_contains("use_volume_cap")) spec.costs.use_volume_cap = v->as_bool();
        if (auto* v = co.if_contains("volume_cap_mult")) spec.costs.volume_cap_mult = parse_plain_real(*v);
    }

    if (auto* v = entry.if_contains("tag")) {
        spec.tag = v->as_string().c_str();
    }
}

HarnessInput load_pool_file(const fs::path& path) {
    HarnessInput input;
    input.config_dir = path.parent_path();

    const std::string contents = read_file(path);
    const json::value root = json::parse(contents);

    const json::object* obj_root = root.is_object() ? &root.as_object() : nullptr;
    if (obj_root) {
        if (obj_root->contains("meta")) {
            const auto& meta = obj_root->at("meta").as_object();
            if (auto* df = meta.if_contains("datafile")) {
                input.raw_data_path = df->as_string().c_str();
                fs::path raw = input.raw_data_path;
                if (!raw.is_absolute()) {
                    raw = input.config_dir / raw;
                }
                input.data_path = fs::absolute(raw);
            }
        }
        if (obj_root->contains("pools")) {
            const auto& arr = obj_root->at("pools").as_array();
            input.pools.reserve(arr.size());
            for (const auto& entry : arr) {
                PoolSpec spec;
                parse_pool_entry(entry.as_object(), spec);
                if (spec.tag.empty()) spec.tag = "pool_" + std::to_string(input.pools.size());
                input.pools.push_back(std::move(spec));
            }
            return input;
        }
        if (obj_root->contains("pool")) {
            PoolSpec spec;
            parse_pool_entry(*obj_root, spec);
            if (spec.tag.empty()) spec.tag = "pool_0";
            input.pools.push_back(std::move(spec));
            return input;
        }
    }

    if (root.is_array()) {
        const auto& arr = root.as_array();
        input.pools.reserve(arr.size());
        for (const auto& entry : arr) {
            PoolSpec spec;
            parse_pool_entry(entry.as_object(), spec);
            if (spec.tag.empty()) spec.tag = "pool_" + std::to_string(input.pools.size());
            input.pools.push_back(std::move(spec));
        }
        return input;
    }

    throw std::runtime_error("pool_config must be object with 'pools' array or array of entries");
}


struct Candle {
    uint64_t ts{0};
    RealT open{0};
    RealT high{0};
    RealT low{0};
    RealT close{0};
    RealT volume{0};
};

RealT to_real(const json::value& v) {
    if (v.is_double()) return static_cast<RealT>(v.as_double());
    if (v.is_int64())  return static_cast<RealT>(v.as_int64());
    if (v.is_uint64()) return static_cast<RealT>(v.as_uint64());
    if (v.is_string()) return static_cast<RealT>(std::strtold(v.as_string().c_str(), nullptr));
    return RealT(0);
}

std::vector<Candle> parse_candles(const json::value& root, size_t limit) {
    const json::array* arr = nullptr;
    if (root.is_array()) {
        arr = &root.as_array();
    } else if (root.is_object()) {
        const auto& obj = root.as_object();
        for (const char* key : {"data", "candles"}) {
            if (obj.contains(key) && obj.at(key).is_array()) {
                arr = &obj.at(key).as_array();
                break;
            }
        }
    }
    if (!arr) throw std::runtime_error("expected candles array");
    const size_t count = limit ? std::min(limit, arr->size()) : arr->size();
    std::vector<Candle> out;
    out.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const auto& row = (*arr)[i];
        if (!row.is_array()) continue;
        const auto& a = row.as_array();
        if (a.size() < 6) continue;
        Candle c;
        c.ts = parse_timestamp(a[0]);
        c.open = to_real(a[1]);
        c.high = to_real(a[2]);
        c.low  = to_real(a[3]);
        c.close = to_real(a[4]);
        c.volume = to_real(a[5]);
        out.push_back(c);
    }
    return out;
}

std::vector<sim::PriceEvent> candles_to_events(const std::vector<Candle>& candles) {
    std::vector<sim::PriceEvent> events;
    events.reserve(candles.size() * 2);
    for (const auto& c : candles) {
        const RealT path_low  = std::abs(c.open - c.low)  + std::abs(c.high - c.close);
        const RealT path_high = std::abs(c.open - c.high) + std::abs(c.low  - c.close);
        const bool first_low = path_low <= path_high;
        sim::PriceEvent ev1{c.ts, first_low ? c.low : c.high, c.volume / RealT(2)};
        sim::PriceEvent ev2{c.ts + 10, first_low ? c.high : c.low, c.volume / RealT(2)};
        events.push_back(ev1);
        events.push_back(ev2);
    }
    std::sort(events.begin(), events.end(), [](const auto& a, const auto& b) { return a.timestamp < b.timestamp; });
    return events;
}

std::vector<sim::PriceEvent> parse_events_array(const json::value& root, size_t limit) {
    const json::array* arr = nullptr;
    if (root.is_array()) {
        arr = &root.as_array();
    } else if (root.is_object()) {
        const auto& obj = root.as_object();
        for (const char* key : {"events", "data"}) {
            if (obj.contains(key) && obj.at(key).is_array()) {
                arr = &obj.at(key).as_array();
                break;
            }
        }
    }
    if (!arr) throw std::runtime_error("expected events array");
    const size_t count = limit ? std::min(limit, arr->size()) : arr->size();
    std::vector<sim::PriceEvent> events;
    events.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const auto& row = (*arr)[i];
        if (row.is_array()) {
            const auto& a = row.as_array();
            if (a.size() < 3) continue;
            sim::PriceEvent ev{};
            ev.timestamp = parse_timestamp(a[0]);
            ev.cex_price = to_real(a[1]);
            ev.volume = (a.size() >= 3) ? to_real(a[2]) : RealT(0);
            if (ev.timestamp && ev.cex_price > RealT(0)) events.push_back(ev);
        } else if (row.is_object()) {
            const auto& o = row.as_object();
            sim::PriceEvent ev{};
            if (auto* v = o.if_contains("ts")) {
                ev.timestamp = parse_timestamp(*v);
            } else if (auto* vt = o.if_contains("timestamp")) {
                ev.timestamp = parse_timestamp(*vt);
            }
            if (auto* vp = o.if_contains("price")) ev.cex_price = to_real(*vp);
            else if (auto* vc = o.if_contains("close")) ev.cex_price = to_real(*vc);
            if (auto* vv = o.if_contains("volume")) ev.volume = to_real(*vv);
            if (ev.timestamp && ev.cex_price > RealT(0)) events.push_back(ev);
        }
    }
    std::sort(events.begin(), events.end(), [](const auto& a, const auto& b) { return a.timestamp < b.timestamp; });
    return events;
}

std::vector<sim::PriceEvent> load_price_events(const fs::path& path, bool treat_as_events, size_t limit) {
    const std::string contents = read_file(path);
    json::value root = json::parse(contents);
    if (!treat_as_events) {
        auto candles = parse_candles(root, limit);
        return candles_to_events(candles);
    }
    return parse_events_array(root, limit);
}

sim::SimpleArbitrageur::Config build_trader_config(RealT threshold, RealT trade_fraction, RealT min_swap_frac, RealT max_swap_frac) {
    sim::SimpleArbitrageur::Config cfg;
    cfg.threshold = threshold;
    cfg.trade_fraction = trade_fraction;
    cfg.min_trade_frac = std::max(min_swap_frac, static_cast<RealT>(1e-9));
    cfg.max_trade_frac = std::max(max_swap_frac, cfg.min_trade_frac * RealT(10));
    if (cfg.max_trade_frac > RealT(1)) cfg.max_trade_frac = RealT(1);
    return cfg;
}

struct Options {
    fs::path config_path;
    std::optional<fs::path> data_override;
    bool treat_input_as_events{false};
    bool use_optimal_trader{false};
    size_t limit_events{0};
    size_t threads{std::max<size_t>(1, std::thread::hardware_concurrency())};
    RealT threshold{static_cast<RealT>(0.001)};
    RealT trade_fraction{static_cast<RealT>(0.01)};
    RealT min_swap_frac{static_cast<RealT>(1e-6)};
    RealT max_swap_frac{static_cast<RealT>(0.1)};
};

Options parse_cli(int argc, char** argv) {
    Options opts;
    bool explicit_path = (argc >= 2 && argv[1][0] != '-');
    if (explicit_path) {
        opts.config_path = fs::absolute(fs::path(argv[1]));
    } else {
        opts.config_path = fs::absolute(fs::path("python/arb_sim/run_data/pool_config.json"));
    }
    int start_idx = explicit_path ? 2 : 1;
    for (int i = start_idx; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](RealT& dst) {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
            dst = static_cast<RealT>(std::strtod(argv[++i], nullptr));
        };
        auto next_size = [&](size_t& dst) {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
            dst = static_cast<size_t>(std::stoll(argv[++i]));
        };
        if (arg == "--data" && i + 1 < argc) {
            opts.data_override = fs::absolute(fs::path(argv[++i]));
        } else if (arg == "--events") {
            opts.treat_input_as_events = true;
        } else if (arg == "--optimal") {
            opts.use_optimal_trader = true;
        } else if (arg == "--limit" && i + 1 < argc) {
            next_size(opts.limit_events);
        } else if (arg == "--threads" && i + 1 < argc) {
            next_size(opts.threads);
            if (opts.threads == 0) opts.threads = 1;
        } else if (arg == "--threshold" && i + 1 < argc) {
            next(opts.threshold);
        } else if (arg == "--trade-frac" && i + 1 < argc) {
            next(opts.trade_fraction);
        } else if (arg == "--min-swap" && i + 1 < argc) {
            next(opts.min_swap_frac);
        } else if (arg == "--max-swap" && i + 1 < argc) {
            next(opts.max_swap_frac);
        }
    }
    return opts;
}

sim::PoolRunner make_runner(const PoolInit& init) {
    twocrypto::TwoCryptoPoolT<RealT> pool({init.precisions[0], init.precisions[1]},
                                          init.A,
                                          init.gamma,
                                          init.mid_fee,
                                          init.out_fee,
                                          init.fee_gamma,
                                          init.allowed_extra,
                                          init.adj_step,
                                          init.ma_time,
                                          init.initial_price);
    if (init.start_ts) {
        pool.set_block_timestamp(init.start_ts);
    }
    pool.add_liquidity({init.initial_liq[0], init.initial_liq[1]}, RealT(0));
    return sim::PoolRunner(std::move(pool));
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = parse_cli(argc, argv);
        HarnessInput input = load_pool_file(opts.config_path);
        if (input.pools.empty()) {
            throw std::runtime_error("no pools found in config");
        }
        if (opts.data_override) {
            input.data_path = *opts.data_override;
            input.raw_data_path = opts.data_override->string();
        }

        if (input.data_path.empty()) {
            throw std::runtime_error("pool_config missing meta.datafile and no --data override provided");
        }
        if (!fs::exists(input.data_path) && !input.raw_data_path.empty()) {
            fs::path alt = fs::absolute(fs::path(input.raw_data_path));
            if (fs::exists(alt)) {
                input.data_path = alt;
            }
        }
        if (!fs::exists(input.data_path)) {
            throw std::runtime_error("data file not found: " + input.data_path.string());
        }
        const auto events = load_price_events(input.data_path, opts.treat_input_as_events, opts.limit_events);
        if (events.empty()) {
            throw std::runtime_error("no price events loaded");
        }
        std::cout << "Loaded " << input.pools.size() << " pools and " << events.size()
                  << " events from " << input.data_path << "\n";

        const size_t thread_count = std::max<size_t>(1, opts.threads);
        std::vector<json::object> run_summaries(input.pools.size());
        std::atomic<size_t> next_idx{0};
        std::mutex io_mu;

        auto trader_cfg = build_trader_config(opts.threshold, opts.trade_fraction, opts.min_swap_frac, opts.max_swap_frac);
        sim::OptimalArbitrageur::Config optimal_cfg;
        optimal_cfg.min_trade_frac = std::max(opts.min_swap_frac, static_cast<RealT>(1e-9));
        optimal_cfg.max_trade_frac = std::max(opts.max_swap_frac, optimal_cfg.min_trade_frac * RealT(2));
        if (optimal_cfg.max_trade_frac > RealT(1)) optimal_cfg.max_trade_frac = RealT(1);

        auto worker = [&](size_t worker_id) {
            while (true) {
                const size_t idx = next_idx.fetch_add(1);
                if (idx >= input.pools.size()) break;
                const auto& spec = input.pools[idx];
                auto runner = make_runner(spec.init);
                sim::RunResult result;
                if (opts.use_optimal_trader) {
                    sim::OptimalArbitrageur trader(optimal_cfg, spec.costs);
                    result = runner.run(events, trader);
                } else {
                    sim::SimpleArbitrageur trader(trader_cfg, spec.costs);
                    result = runner.run(events, trader);
                }

                json::object run;
                run["tag"] = spec.tag;
                run["trades"] = static_cast<uint64_t>(result.trades);
                run["notional_coin0"] = static_cast<double>(result.notional_coin0);
                run["profit_coin0"] = static_cast<double>(result.profit_coin0);
                run["final_price"] = static_cast<double>(result.final_price);
                json::array balances;
                balances.push_back(static_cast<double>(result.final_balances[0]));
                balances.push_back(static_cast<double>(result.final_balances[1]));
                run["final_balances"] = balances;
                run_summaries[idx] = std::move(run);
                {
                    std::lock_guard<std::mutex> lk(io_mu);
                    std::cout << "completed " << spec.tag << " (" << (idx + 1) << "/" << input.pools.size() << ")\n";
                }
            }
        };

        std::vector<std::thread> threads;
        threads.reserve(thread_count);
        for (size_t t = 0; t < thread_count; ++t) {
            threads.emplace_back(worker, t);
        }
        for (auto& th : threads) th.join();

        json::object meta;
        meta["config"] = opts.config_path.string();
        meta["data_file"] = input.data_path.string();
        meta["events"] = static_cast<uint64_t>(events.size());
        meta["threads"] = static_cast<uint64_t>(thread_count);
        meta["trader"] = opts.use_optimal_trader ? "optimal" : "simple";

        json::object output;
        output["metadata"] = meta;
        json::array runs;
        runs.reserve(run_summaries.size());
        for (auto& r : run_summaries) runs.push_back(r);
        output["runs"] = runs;

        std::cout << json::serialize(output) << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
