// Events module - implementation (non-templated)
#include "events/loader.hpp"

#include <boost/json.hpp>
#include <boost/json/src.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace json = boost::json;

namespace arb {

std::vector<Candle> load_candles(const std::string& path,
                                  size_t max_candles,
                                  double squeeze_frac) {
    std::vector<Candle> out;
    out.reserve(1024);

    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open candles file: " + path);

    std::ostringstream oss;
    oss << in.rdbuf();
    const std::string s = oss.str();

    json::value val = json::parse(s);
    if (!val.is_array()) throw std::runtime_error("Candles JSON must be an array of arrays");

    const auto& arr = val.as_array();
    const size_t limit = (max_candles ? std::min(max_candles, arr.size()) : arr.size());
    out.reserve(limit);

    auto to_d = [](const json::value& v) -> double {
        if (v.is_double()) return v.as_double();
        if (v.is_int64())  return static_cast<double>(v.as_int64());
        if (v.is_uint64()) return static_cast<double>(v.as_uint64());
        return 0.0;
    };

    for (size_t idx = 0; idx < limit; ++idx) {
        const auto& a = arr[idx].as_array();
        if (a.size() < 6) continue;

        Candle c{};
        uint64_t ts = 0;
        const auto& tsv = a[0];
        if (tsv.is_uint64()) ts = tsv.as_uint64();
        else if (tsv.is_int64()) ts = static_cast<uint64_t>(tsv.as_int64());
        else if (tsv.is_double()) ts = static_cast<uint64_t>(tsv.as_double());
        if (ts > 10000000000ULL) ts /= 1000ULL; // ms->s
        c.ts = ts;

        c.open   = to_d(a[1]);
        c.high   = to_d(a[2]);
        c.low    = to_d(a[3]);
        c.close  = to_d(a[4]);
        c.volume = to_d(a[5]);

        // Apply squeeze filter
        if (squeeze_frac > 0.0) {
            const double oc_mid = 0.5 * (c.open + c.close);
            if (oc_mid > 0) {
                const double max_h = oc_mid * (1.0 + squeeze_frac);
                const double min_l = oc_mid * (1.0 - squeeze_frac);
                if (c.high > max_h) c.high = max_h;
                if (c.low  < min_l) c.low  = min_l;
            }
        }
        out.push_back(c);
    }
    return out;
}

std::vector<Event> gen_events(const std::vector<Candle>& cs) {
    std::vector<Event> evs;
    evs.reserve(cs.size() * 2);

    for (const auto& c : cs) {
        // Choose path: "low first" vs "high first" based on which is shorter
        const double path1 = std::abs(c.open - c.low)  + std::abs(c.high - c.close);
        const double path2 = std::abs(c.open - c.high) + std::abs(c.low  - c.close);
        const bool first_low = path1 < path2;

        evs.push_back(Event{c.ts,      first_low ? c.low  : c.high, c.volume / 2.0, c});
        evs.push_back(Event{c.ts + 10, first_low ? c.high : c.low,  c.volume / 2.0, c});
    }

    std::sort(evs.begin(), evs.end(), [](const Event& a, const Event& b) {
        return a.ts < b.ts;
    });
    return evs;
}

std::vector<Event> load_events(const std::string& path, size_t max_events) {
    std::vector<Event> out;
    out.reserve(1024);

    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open events file: " + path);

    std::ostringstream oss;
    oss << in.rdbuf();
    const std::string s = oss.str();

    json::value val = json::parse(s);
    const json::array* arr = nullptr;

    if (val.is_array()) {
        arr = &val.as_array();
    } else if (val.is_object()) {
        const auto& o = val.as_object();
        if (o.contains("data") && o.at("data").is_array())
            arr = &o.at("data").as_array();
        else if (o.contains("events") && o.at("events").is_array())
            arr = &o.at("events").as_array();
    }
    if (!arr) {
        throw std::runtime_error(
            "Events JSON must be an array of [ts,price,vol] or object with 'data'/'events' array");
    }

    const size_t limit = (max_events ? std::min(max_events, arr->size()) : arr->size());
    out.reserve(limit);

    auto to_d = [](const json::value& v) -> double {
        if (v.is_double()) return v.as_double();
        if (v.is_int64())  return static_cast<double>(v.as_int64());
        if (v.is_uint64()) return static_cast<double>(v.as_uint64());
        if (v.is_string()) return std::strtold(v.as_string().c_str(), nullptr);
        return 0.0;
    };

    for (size_t i = 0; i < limit; ++i) {
        const auto& e = (*arr)[i];

        if (e.is_array()) {
            const auto& a = e.as_array();
            if (a.size() < 3) continue;

            uint64_t ts = 0;
            const auto& tsv = a[0];
            if (tsv.is_uint64()) ts = tsv.as_uint64();
            else if (tsv.is_int64()) ts = static_cast<uint64_t>(tsv.as_int64());
            else if (tsv.is_double()) ts = static_cast<uint64_t>(tsv.as_double());
            else if (tsv.is_string()) ts = static_cast<uint64_t>(std::strtoll(tsv.as_string().c_str(), nullptr, 10));
            if (ts > 10000000000ULL) ts /= 1000ULL; // ms->s guard

            if (a.size() >= 6) {
                // Candles: [ts, o, h, l, c, v] -> use close and volume
                Event ev{ts, to_d(a[4]), to_d(a[5])};
                out.push_back(ev);
            } else {
                // Events: [ts, price, volume]
                Event ev{ts, to_d(a[1]), to_d(a[2])};
                out.push_back(ev);
            }
        } else if (e.is_object()) {
            const auto& o = e.as_object();
            uint64_t ts = 0;
            if (auto* v = o.if_contains("ts")) {
                if (v->is_uint64()) ts = v->as_uint64();
                else if (v->is_int64()) ts = static_cast<uint64_t>(v->as_int64());
                else if (v->is_double()) ts = static_cast<uint64_t>(v->as_double());
            }
            if (ts > 10000000000ULL) ts /= 1000ULL;

            double price = 0.0, vol = 0.0;
            if (auto* v = o.if_contains("price")) price = to_d(*v);
            else if (auto* v = o.if_contains("close")) price = to_d(*v);
            else if (auto* v = o.if_contains("c")) price = to_d(*v);
            if (auto* v = o.if_contains("volume")) vol = to_d(*v);
            else if (auto* v = o.if_contains("v")) vol = to_d(*v);

            if (ts && price > 0) out.push_back(Event{ts, price, vol});
        }
    }

    std::sort(out.begin(), out.end(), [](const Event& A, const Event& B) {
        return A.ts < B.ts;
    });
    return out;
}

} // namespace arb
