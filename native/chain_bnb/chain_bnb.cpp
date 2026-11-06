// Exact fixed-order Chain/Atomic BnB over immutable endpoint-state data.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <list>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using Ints = std::vector<int>;
using Matrix = std::vector<std::vector<double>>;
using Clock = std::chrono::steady_clock;
constexpr double tol = 1e-9;
constexpr double inf = std::numeric_limits<double>::infinity();

struct Problem {
    std::vector<Ints> parents, candidates, groups, group_candidates, group_owner_keys;
    Matrix services, group_durations, pair, first, bound_durations;
    std::vector<double> speeds, optimistic;
    std::vector<bool> has_positions;
    // [robot][task][previous endpoint-state] -> (distance, next state)
    std::vector<std::vector<std::vector<std::pair<double, int>>>> routes;
    Ints tie_order, seed;
    bool seed_present, use_exact_prefix;
    double chain_distance, source_distance;
    int n, nr, owner_count;

    explicit Problem(const py::dict& d) {
#define GET(field, type) field = d[#field].cast<type>()
        GET(parents, std::vector<Ints>); GET(candidates, std::vector<Ints>);
        GET(groups, std::vector<Ints>); GET(group_candidates, std::vector<Ints>);
        GET(group_owner_keys, std::vector<Ints>); GET(owner_count, int); GET(use_exact_prefix, bool);
        GET(services, Matrix); GET(group_durations, Matrix); GET(pair, Matrix);
        GET(first, Matrix); GET(bound_durations, Matrix);
        GET(speeds, std::vector<double>); GET(optimistic, std::vector<double>);
        GET(has_positions, std::vector<bool>); GET(tie_order, Ints); GET(seed, Ints);
        GET(seed_present, bool); GET(chain_distance, double); GET(source_distance, double);
        routes = d["routes"].cast<decltype(routes)>();
#undef GET
        n = static_cast<int>(parents.size()); nr = static_cast<int>(speeds.size());
        validate();
    }
    void validate() const {
        auto require = [](bool condition) {
            if (!condition) throw std::invalid_argument("Malformed native assignment data");
        };
        auto scalar = [&](double x) { require(std::isfinite(x) && x >= 0.); };
        require(candidates.size() == parents.size() && services.size() == parents.size()
            && pair.size() == parents.size() && bound_durations.size() == parents.size()
            && optimistic.size() == parents.size() && has_positions.size() == parents.size()
            && tie_order.size() == parents.size() && first.size() == speeds.size()
            && routes.size() == speeds.size() && groups.size() == group_candidates.size()
            && groups.size() == group_durations.size() && groups.size() == group_owner_keys.size()
            && owner_count >= 0);
        scalar(chain_distance); scalar(source_distance);
        Ints coverage(n, 0), ties(n, 0);
        for (int i = 0; i < n; ++i) {
            require(!candidates[i].empty() && services[i].size() == speeds.size()
                && bound_durations[i].size() == speeds.size() && pair[i].size() == parents.size());
            for (int parent : parents[i]) require(parent >= 0 && parent < i);
            Ints seen(nr, 0);
            for (int r : candidates[i]) {
                require(r >= 0 && r < nr); require(++seen[r] == 1);
                scalar(services[i][r]); require(services[i][r] > 0.);
                scalar(bound_durations[i][r]);
            }
            for (double x : pair[i]) scalar(x);
            scalar(optimistic[i]);
            require(tie_order[i] >= 0 && tie_order[i] < n);
            require(++ties[tie_order[i]] == 1);
        }
        for (int r = 0; r < nr; ++r) {
            scalar(speeds[r]); require(speeds[r] > 0.);
            require(first[r].size() == parents.size() && routes[r].size() == parents.size());
            for (double x : first[r]) scalar(x);
            const size_t ns = n ? routes[r][0].size() : 0;
            if (n) require(ns > 0);
            for (const auto& row : routes[r]) {
                require(row.size() == ns);
                for (const auto& arc : row) {
                    if (arc.second == -1) require(arc.first == -1.);
                    else { require(arc.second >= 0 && static_cast<size_t>(arc.second) < ns); scalar(arc.first); }
                }
            }
        }
        for (size_t g = 0; g < groups.size(); ++g) {
            for (int key : group_owner_keys[g]) require(key >= 0 && key < owner_count);
            require(!groups[g].empty() && !group_candidates[g].empty()
                && group_durations[g].size() == speeds.size());
            Ints seen(nr, 0);
            for (int r : group_candidates[g]) {
                require(r >= 0 && r < nr); require(++seen[r] == 1);
                scalar(group_durations[g][r]);
            }
            for (int i : groups[g]) {
                require(i >= 0 && i < n); require(++coverage[i] == 1);
                for (int r : group_candidates[g])
                    require(std::find(candidates[i].begin(), candidates[i].end(), r) != candidates[i].end());
            }
        }
        for (int count : coverage) require(count == 1);
        if (seed_present) {
            Ints owners(owner_count, -1);
            require(seed.size() == parents.size());
            for (int i = 0; i < n; ++i)
                require(std::find(candidates[i].begin(), candidates[i].end(), seed[i]) != candidates[i].end());
            for (const auto& group : groups)
                for (int i : group) require(seed[i] == seed[group[0]]);
            for (size_t g = 0; g < groups.size(); ++g)
                for (int key : group_owner_keys[g]) {
                    int r = seed[groups[g][0]];
                    require(owners[key] < 0 || owners[key] == r);
                    owners[key] = r;
                }
        }
    }
};

struct Prefix {
    std::vector<double> finish, available;
    Ints states;
    double distance = 0., makespan = 0.;
    bool feasible = true;
};

class Search {
    const Problem& p;
    Ints assignment, best, best_rank, owners;
    std::vector<double> load;
    double incumbent_t = inf, incumbent_d = inf, best_t = inf, best_d = inf;
    bool found = false, timed_out = false;
    Clock::time_point deadline;
    bool limited;
    std::map<std::string, uint64_t> stats;
    struct Cached { Prefix value; std::list<Ints>::iterator age; };
    std::map<Ints, Cached> cache;
    std::list<Ints> ages;

    bool stopped(bool force = false) {
        if (timed_out) return true;
        if (force || stats["expanded_nodes"] % 128 == 0) {
            // The DFS holds no Python objects/GIL. Poll signals periodically
            // so the benchmark's existing watchdog and Ctrl-C remain usable.
            { py::gil_scoped_acquire acquire;
              if (PyErr_CheckSignals() != 0) throw py::error_already_set(); }
            if (limited && Clock::now() >= deadline) timed_out = true;
        }
        return timed_out;
    }

    Prefix prefix(const Ints& key) {
        auto hit = cache.find(key);
        if (hit != cache.end()) {
            ++stats["exact_prefix_cache_hits"];
            ages.splice(ages.begin(), ages, hit->second.age);
            return hit->second.value;
        }
        ++stats["exact_prefix_cache_misses"];
        Prefix state;
        if (key.empty()) {
            state.available.assign(p.nr, 0.); state.states.assign(p.nr, 0);
        } else {
            Ints previous(key.begin(), key.end()-1);
            state = prefix(previous);
            if (state.feasible) {
                int i = static_cast<int>(key.size())-1, r = key.back();
                const auto& arc = p.routes[r][i][state.states[r]];
                if (arc.second < 0) state.feasible = false;
                else {
                    double dependency = 0.;
                    for (int parent : p.parents[i]) dependency = std::max(dependency, state.finish[parent]);
                    double finish = std::max(state.available[r], dependency)
                        + arc.first / p.speeds[r] + p.services[i][r];
                    state.finish.push_back(finish);
                    state.available[r] = finish; state.states[r] = arc.second;
                    state.distance += arc.first; state.makespan = std::max(state.makespan, finish);
                }
            }
        }
        // Match the Python prefix cache's bounded memory (4096 entries).
        if (cache.size() == 4096) { cache.erase(ages.back()); ages.pop_back(); }
        ages.push_front(key);
        cache.emplace(key, Cached{state, ages.begin()});
        return state;
    }

    std::pair<double, double> bound() {
        std::vector<double> finish(p.n, 0.), service(p.nr, 0.), travel(p.nr, 0.);
        Ints previous(p.nr, -1), active;
        std::vector<bool> seen(p.nr, false);
        double makespan = 0.;
        for (int i = 0; i < p.n; ++i) {
            int r = assignment[i];
            double dependency = 0.;
            for (int parent : p.parents[i]) dependency = std::max(dependency, finish[parent]);
            finish[i] = dependency + (r < 0 ? p.optimistic[i] : p.bound_durations[i][r]);
            makespan = std::max(makespan, finish[i]);
            if (r >= 0) {
                if (!seen[r]) { active.push_back(r); seen[r] = true; }
                service[r] += p.services[i][r];
                travel[r] += previous[r] < 0 ? p.first[r][i] : p.pair[previous[r]][i];
                if (p.has_positions[i]) previous[r] = i;
            }
        }
        double distance = 0.;
        for (int r : active) {
            makespan = std::max(makespan, service[r] + travel[r] / p.speeds[r]);
            distance += travel[r];
        }
        distance = std::max({p.chain_distance, distance, p.source_distance});
        if (!p.use_exact_prefix) return {makespan, distance};
        Ints committed;
        for (int r : assignment) { if (r < 0) break; committed.push_back(r); }
        Prefix exact = prefix(committed);
        if (!exact.feasible) return {inf, inf};
        return {std::max(makespan, exact.makespan), std::max(distance, exact.distance)};
    }

    void evaluate(const Ints& candidate, bool greedy) {
        if (!greedy && p.seed_present && candidate == p.seed) {
            ++stats["duplicate_complete_skipped"]; return;
        }
        ++stats[greedy ? "greedy_evaluated" : "complete_evaluated"];
        Prefix value = prefix(candidate);
        if (!value.feasible) return;
        ++stats["feasible_complete_evaluated"];
        double t = value.makespan, d = value.distance;
        if (t < incumbent_t - tol || (std::abs(t-incumbent_t) <= tol && d < incumbent_d-tol)) {
            incumbent_t = t; incumbent_d = d;
        }
        Ints rank;
        for (int i : p.tie_order) rank.push_back(static_cast<int>(
            std::find(p.candidates[i].begin(), p.candidates[i].end(), candidate[i])-p.candidates[i].begin()));
        if (!found || t < best_t || (t == best_t && (d < best_d || (d == best_d && rank < best_rank)))) {
            found = true; best = candidate; best_rank = std::move(rank); best_t = t; best_d = d;
        }
    }

    void dfs(size_t g) {
        if (stopped()) return;
        if (g == p.groups.size()) { evaluate(assignment, false); return; }
        Ints candidates = p.group_candidates[g];
        // stable_sort retains input candidate rank for equal loads.
        std::stable_sort(candidates.begin(), candidates.end(), [&](int a, int b) {
            return load[a]+p.group_durations[g][a] < load[b]+p.group_durations[g][b];
        });
        for (int r : candidates) {
            if (stopped()) return;
            Ints claimed;
            bool violates = false;
            for (int key : p.group_owner_keys[g]) {
                if (owners[key] >= 0 && owners[key] != r) { violates = true; break; }
                if (owners[key] < 0) { owners[key] = r; claimed.push_back(key); }
            }
            if (violates) {
                ++stats["handoff_pruned"];
                for (int key : claimed) owners[key] = -1;
                continue;
            }
            for (int i : p.groups[g]) assignment[i] = r;
            double old_load = load[r]; load[r] += p.group_durations[g][r];
            ++stats["expanded_nodes"];
            auto lb = bound();
            bool prune = lb.first > incumbent_t+tol ||
                (std::abs(lb.first-incumbent_t) <= tol && lb.second >= incumbent_d-tol);
            if (prune) {
                ++stats["bound_pruned"];
                if (std::abs(lb.first-incumbent_t) <= tol && lb.second >= incumbent_d-tol)
                    ++stats["distance_bound_pruned"];
            } else dfs(g+1);
            for (int i : p.groups[g]) assignment[i] = -1;
            load[r] = old_load;
            for (int key : claimed) owners[key] = -1;
        }
    }

public:
    Search(const Problem& problem, double seconds) : p(problem), assignment(p.n, -1), owners(p.owner_count, -1), load(p.nr, 0.), limited(seconds >= 0.) {
        if (!std::isfinite(seconds)) throw std::invalid_argument("Invalid native time limit");
        if (limited) deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        for (const char* name : {"expanded_nodes", "bound_pruned", "distance_bound_pruned", "handoff_pruned",
            "complete_evaluated", "feasible_complete_evaluated", "greedy_evaluated", "duplicate_complete_skipped",
            "exact_prefix_cache_hits", "exact_prefix_cache_misses"}) stats[name] = 0;
    }
    void run() {
        if (stopped(true)) return;
        if (p.seed_present) evaluate(p.seed, true);
        dfs(0); stopped(true);
    }
    py::dict statistics() const { return py::cast(stats); }
    py::dict result() const {
        py::dict out;
        out["found"] = found; out["timed_out"] = timed_out;
        out["assignment"] = best; out["T"] = best_t; out["D"] = best_d;
        out["stats"] = statistics();
        return out;
    }
};

PYBIND11_MODULE(_chain_bnb_native, module) {
    module.doc() = "Deterministic fixed-order Chain/Atomic BnB over frozen oracle data";
    module.attr("revision") = "bnb-cpp-chain-atomic-v2";
    module.def("solve", [](const py::dict& data, double seconds, py::dict diagnostics) {
        Problem problem(data);
        Search search(problem, seconds);
        try {
            py::gil_scoped_release release;
            search.run();
        } catch (const py::error_already_set&) {
            diagnostics.attr("update")(search.statistics());
            throw;
        }
        return search.result();
    }, py::arg("data"), py::arg("time_limit") = -1., py::arg("diagnostics") = py::dict());
}
