"""Independent PathAware-rule adaptation; not the upstream Java engine.

Source: erancihan/NegotiationForMAPF, commit
7e0e4ac09ee2604c23ba9c9851af6373cfc1ed7f. Token/path ratio and increasing
stakes on repeated insistence follow PathAwareAgent and Action. Feasible
complete intervention bids and unanimous multi-robot acceptance are adaptations.
No remaining DAG, task successors, assignment objective or mission score enters
this selector. Planning is pure; balances commit only on accepted live plans.
"""
from copy import deepcopy

from .planner import path_length

INITIAL_TOKENS = 10
MAX_ROUNDS = 128
UPSTREAM_COMMIT = '7e0e4ac09ee2604c23ba9c9851af6373cfc1ed7f'


class NegotiationFailure(RuntimeError):
    pass


def new_state(robot_ids):
    return dict(tokens={r:INITIAL_TOKENS for r in robot_ids},initial_path_lengths={})


def make_selector(robots, state, speed, participants=None):
    """Snapshot only the active task's route, balance and initial route length."""
    state = deepcopy(state or new_state(r.robot_id for r in robots))
    agents, nominal = {}, {}
    for robot in robots:
        if participants is not None and robot.robot_id not in participants:
            continue
        if robot.current_task is None or robot.phase not in {'moving','preview','waiting'}:
            continue
        rid = robot.robot_id
        route = list(robot.path[robot.path_index:])
        if not route or route[0] != robot.cell:
            route.insert(0,robot.cell)
        nominal[rid] = route
        key = f'{rid}:{robot.current_task["id"]}'
        remaining = path_length(route)
        initial = state['initial_path_lengths'].get(key,remaining)
        agents[rid] = dict(tokens=state['tokens'].get(rid,INITIAL_TOKENS),
                          path_rate=max(remaining,1.)/max(initial,1.))
    def select(candidates, stats):
        return negotiate(candidates,agents,nominal,speed,stats)
    return select


def negotiate(candidates, agents, nominal_paths, speed, stats, max_rounds=MAX_ROUNDS):
    """Round-robin offers with unanimous acceptance; bounded, deterministic.

    Each agent orders complete feasible bids by its own current-route travel
    plus intervention wait. Greedy agents insist/increase stakes or enumerate
    alternatives. A conceding agent accepts a feasible standing offer, as in
    the upstream rule. Equal preferred bids also agree. Multi-party payments
    conservatively require enough balance for every accepting opponent.
    """
    from .relocation_search import _candidate_stable_key
    if not candidates:
        raise NegotiationFailure('negotiation_no_feasible_bid')
    ids = sorted(agents)
    before = {rid:int(agents[rid]['tokens']) for rid in ids}
    if any(t<0 for t in before.values()):
        raise ValueError('Negative negotiation token balance')
    trace = []
    def diagnostic(status, **extra):
        stats['negotiation'] = dict(status=status,adapter_revision='pathaware-feasible-bids-v1',
            upstream_commit=UPSTREAM_COMMIT,implementation='independent_python_adaptation',
            initial_tokens=INITIAL_TOKENS,max_rounds=max_rounds,tokens_before=before,
            tokens_after=before,trace=trace,dag_information_used=False,**extra)
    if len(candidates)==1:
        diagnostic('sole_feasible_bid',selected=candidates[0].get('candidate_id'))
        return candidates[0]
    if not ids:
        diagnostic('failed_no_active_agents')
        raise NegotiationFailure('negotiation_no_active_agents')
    def route(candidate,rid):
        return candidate.get('preferred_paths',{}).get(rid,nominal_paths[rid])
    def own_cost(candidate,rid):
        # Staging is already included in delay_by_robot; never sum it twice.
        return (path_length(route(candidate,rid))/speed +
                float(candidate.get('delay_by_robot',{}).get(rid,0.)))
    preferences = {rid:sorted(candidates,key=lambda c:(own_cost(c,rid),_candidate_stable_key(c)))
                   for rid in ids}
    bid_costs = {rid:[dict(candidate_id=c.get('candidate_id'),cost_s=own_cost(c,rid))
                      for c in preferences[rid]] for rid in ids}
    stakes = {rid:0 for rid in ids}
    history = {rid:set() for rid in ids}
    alternative = {rid:0 for rid in ids}
    accepted = set()
    offered, proposer = None, None
    def finish():
        after = dict(before)
        for rid in ids:
            if rid == proposer:
                continue
            payment = max(0,stakes[proposer]-stakes[rid])
            after[proposer] -= payment
            after[rid] += payment
        if any(t<0 for t in after.values()) or sum(after.values())!=sum(before.values()):
            raise NegotiationFailure('negotiation_invalid_settlement')
        diagnostic('agreement',selected=offered.get('candidate_id'),proposer=proposer,
                   rounds=len(trace),stakes=stakes,bid_costs=bid_costs,
                   path_rates={rid:agents[rid]['path_rate'] for rid in ids})
        stats['negotiation']['tokens_after'] = after
        return offered
    for turn in range(max_rounds):
        rid = ids[turn % len(ids)]
        ratio = ((before[rid]-stakes[rid])/INITIAL_TOKENS)/agents[rid]['path_rate']
        greedy = ratio >= 1.
        preferred = preferences[rid][0]
        if offered is not None and (not greedy or offered is preferred or rid in accepted):
            accepted.add(rid)
            trace.append(dict(robot_id=rid,action='accept',mode='greedy' if greedy else 'concede',
                              candidate_id=offered.get('candidate_id'),ratio=ratio))
            if len(accepted)==len(ids):
                return finish()
            continue
        proposals = [preferred] if greedy else []
        proposals += preferences[rid][alternative[rid]:]
        proposal = None
        proposed_stake = stakes[rid]
        for candidate in proposals:
            key = _candidate_stable_key(candidate)
            repeated = key in history[rid]
            broadcast = tuple(route(candidate,rid))==tuple(nominal_paths[rid])
            stake = stakes[rid] + int(repeated or broadcast)
            liability = sum(max(0,stake-stakes[other]) for other in ids if other!=rid)
            if stake <= before[rid] and liability <= before[rid]:
                proposal,proposed_stake = candidate,stake
                break
        if proposal is None:
            if offered is not None:
                accepted.add(rid)
                trace.append(dict(robot_id=rid,action='accept',mode='bid_space_exhausted',
                                  candidate_id=offered.get('candidate_id'),ratio=ratio))
                if len(accepted)==len(ids):
                    return finish()
                continue
            diagnostic('failed_no_affordable_bid')
            raise NegotiationFailure('negotiation_no_affordable_bid')
        stakes[rid] = proposed_stake
        history[rid].add(_candidate_stable_key(proposal))
        alternative[rid] = min(len(preferences[rid]),preferences[rid].index(proposal)+1)
        offered,proposer = proposal,rid
        accepted = {rid}
        trace.append(dict(robot_id=rid,action='offer',mode='greedy' if greedy else 'concede',
                          candidate_id=offered.get('candidate_id'),stake=proposed_stake,ratio=ratio))
        if len(ids)==1:
            return finish()
    diagnostic('failed_round_limit',rounds=len(trace))
    raise NegotiationFailure('negotiation_round_limit')


def commit(state, info):
    """Only affected agents are updated; disjoint async groups do not overwrite."""
    before,after = info['tokens_before'],info['tokens_after']
    if any(state['tokens'].get(r,INITIAL_TOKENS)!=t for r,t in before.items()):
        raise NegotiationFailure('negotiation_stale_token_snapshot')
    if any(t<0 for t in after.values()) or sum(before.values())!=sum(after.values()):
        raise NegotiationFailure('negotiation_invalid_settlement')
    state['tokens'].update(after)
