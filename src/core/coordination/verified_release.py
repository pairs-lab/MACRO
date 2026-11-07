"""Opt-in bounded executor verification of ambiguous DAG timing predictions.

Dynamic reservations screen the unchanged feasible pool. Deterministic
representatives then share one executor-prefix-plus-remaining-DAG model.
Neither global optimality nor full candidate enumeration is claimed.
"""
import math
import time

from .dag_lookahead import refine_candidates, replay_candidate
from .dynamic_release import refine_dynamic_release_events
from .relocation_search import _candidate_stable_key, _select_resolution_candidate


def refine_verified_release(candidates, *, robots, tasks, task_status, objects,
                            navigation_cells, robot_radius, grid_size, speed,
                            participants, stats, search_deadline, baseline_finishes,
                            baseline_distance, horizon=5., verification_budget_s=5.,
                            verification_enabled=True):
    screened = refine_dynamic_release_events(
        candidates, robots=robots, tasks=tasks, task_status=task_status,
        objects=objects, navigation_cells=navigation_cells, robot_radius=robot_radius,
        grid_size=grid_size, speed=speed, participants=participants, stats=stats,
        search_deadline=search_deadline, baseline_finishes=baseline_finishes,
        baseline_distance=baseline_distance, horizon=horizon, budget_s=5.,
        use_release_guards=True)
    # Censored encounters and uncertain continuation paths are estimates, not
    # evidence that a robot can actually finish/wait as the reservation says.
    if not verification_enabled:
        stats['executor_verification'] = dict(status='disabled_after_incomplete_batch', wall_time_s=0.)
        return screened
    ambiguous = any(c.get('release_prediction', {}).get('censored', False)
                    or c.get('release_prediction', {}).get('release_events', {}).get('unknown_agents')
                    for c in screened)
    if not ambiguous:
        stats['executor_verification'] = dict(status='skipped_unambiguous', wall_time_s=0.)
        return screened
    remaining = (math.inf if search_deadline is None else
                 max(0., search_deadline-time.perf_counter()))
    # One common deadline; any incomplete prefix rolls back the whole shortlist
    # transaction, never mixes replayed and optimistic analytical estimates.
    route_cache = {}
    results = []
    def evaluate(candidate, deadline):
        result = replay_candidate(
            candidate, robots=robots, tasks=tasks, task_status=task_status,
            objects=objects, navigation_cells=navigation_cells, robot_radius=robot_radius,
            grid_size=grid_size, speed=speed, participants=participants,
            deadline=deadline, horizon_s=30., max_repairs=4, route_cache=route_cache,
            recourse_policy='dag_aware',
            recourse_release_prediction_model='dynamic_events_guarded')
        results.append(result)
        return result
    selected_pool = refine_candidates(
        screened, evaluator=evaluate, stable_key=_candidate_stable_key,
        select=lambda pool: _select_resolution_candidate(pool, 'dag_aware'), stats=stats,
        budget_s=min(verification_budget_s, remaining),
        baseline_finishes=baseline_finishes, baseline_distance=baseline_distance,
        representatives_per_root=2, diagnostic_key='executor_verification')
    info = stats['executor_verification']
    info.update(model='guarded-reservations-then-bounded-executor-prefix-v2',
                recourse_policy='dag_aware',
                recourse_release_prediction_model='dynamic_events_guarded',
                nested_verification=False,
                max_recourse_repairs=4, representatives_per_root=2,
                prefix_task_limit_per_robot=2, global_search=False,
                recourse_repairs_observed=[r['recourse_repairs'] for r in results])
    return selected_pool
