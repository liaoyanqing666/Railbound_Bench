"""Simulation-guided constraint search for immutable-board rail placement.

The variable ordering follows the first cell a vehicle can inspect. Earlier
choices cannot be altered by assigning an as-yet uninspected ordinary rail.
Optional signals are decided before their first attempted entry; their static
connected-junction requirement is assigned at the same time when necessary.
Every returned layout is checked by the public validator. A simulation/resource
limit is never used as an infeasibility proof.
"""
from collections import Counter, defaultdict, deque
from copy import deepcopy
import math
import time

from validator.api import check_level, check_solution, parse_json, ValidationError
from validator.answers import encode_answer, normalize_answer
from validator.simulation import simulate, ROUTES, DESTINATIONS, TUNNEL_EXIT, REVERSE
from validator.tiles import (DIRECTIONS, PLACEABLE_TRACKS, SIMPLE_TRACKS,
                             NUMBERED_MODIFIERS, PORTS)


_RAILS = tuple(sorted(PLACEABLE_TRACKS))
_JUNCTIONS = tuple(range(9, 17))


class _LimitReached(Exception):
    pass


class _Search:
    def __init__(self, level, budget, timeout, node_limit, max_ticks, optimize):
        self.started = time.perf_counter()
        self.cpu_started = time.process_time()
        self.deadline = self.started + timeout if timeout is not None else None
        self.node_limit, self.max_ticks, self.optimize = node_limit, max_ticks, optimize
        self.level = check_level(level)
        self.level['tracks'] = budget if budget is not None else self.level['tracks']
        self.budget, self.signal_budget = self.level['tracks'], self.level['semaphores']
        # Ordinary rails are observed through ROUTES on entry. Fixed signals
        # additionally observe junction/port geometry at their immediate rail
        # neighbors, which must therefore keep concrete assignments. New
        # signals can occur anywhere, so their puzzles remain fully concrete.
        self.lazy_domains = self.signal_budget == 0
        self._signal_geometry = set()
        for y, row in enumerate(self.level['mods']):
            for x, modifier in enumerate(row):
                if modifier == 8:
                    for direction in PORTS[self.level['board'][y][x]]:
                        dy, dx = DIRECTIONS[direction]
                        self._signal_geometry.add((y + dy, x + dx))
        self.base = deepcopy(self.level)
        self.base.pop('metadata', None)
        for y, row in enumerate(self.base['mods']):
            for x, modifier in enumerate(row):
                if modifier not in NUMBERED_MODIFIERS:
                    self.base['mod_nums'][y][x] = 0
        self.height, self.width = len(self.base['board']), len(self.base['board'][0])
        self.empty = tuple((y, x) for y, row in enumerate(self.base['board']) for x, tile in enumerate(row) if tile == 0)
        self.empty_set = set(self.empty)
        self.best = None
        self.best_trace_steps = None
        self.best_cost = self.budget + 1
        self.best_signals = None
        self.lower_bound = 0
        self.mandatory = set()
        self.unresolved = False
        self.stop_reason = None
        self._bound_cache = {}
        self.stats = Counter(search_nodes=0, simulations=0, simulation_ticks=0,
            bound_checks=0, bound_cache_hits=0, pruned_by_bound=0, pruned_by_failure=0,
            rail_decisions=0, rail_refinements=0, signal_decisions=0, maximum_depth=0,
            incumbent_validations=0, unresolved_simulations=0, solutions_found=0)
        self._graph = self._potential_graph({})
        self._components = self._component_labels(self._graph)
        self._requirements = []
        self._station_targets = {}
        self._destination_targets = {}
        self._target_distance_cache = {}
        self._hint_targets = ()
        self._visited_stops = defaultdict(set)
        for car in self.level['cars']:
            kind = car['type']
            if kind == 'DECOY':
                continue
            station_mod = 6 if kind == 'NORMAL' else 11
            stations = {(y, x) for y, row in enumerate(self.base['mods']) for x, mod in enumerate(row)
                        if mod == station_mod and self.base['mod_nums'][y][x] == car['num']}
            goals = {(y, x) for y, row in enumerate(self.base['board']) for x, tile in enumerate(row)
                     if tile in DESTINATIONS and DESTINATIONS[tile][0] == kind}
            start = tuple(car['pos'])
            self._station_targets[kind, car['num']] = stations
            self._destination_targets[kind] = goals
            self._requirements.append((start, stations, goals, self._components.get(start)))

    def check_limit(self, *, nodes=False):
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            self.stop_reason = 'time_limit'
            raise _LimitReached
        if nodes and self.node_limit is not None and self.stats['search_nodes'] >= self.node_limit:
            self.stop_reason = 'node_limit'
            raise _LimitReached

    def inside(self, pos):
        return 0 <= pos[0] < self.height and 0 <= pos[1] < self.width

    def neighbor(self, pos, heading):
        dy, dx = DIRECTIONS[heading]
        return pos[0] + dy, pos[1] + dx

    def _potential_graph(self, assignments):
        graph = {}
        for y, row in enumerate(self.base['board']):
            for x, tile in enumerate(row):
                pos = (y, x)
                if tile in (4, 34, 35, 36, 37) or assignments.get(pos) == 0:
                    continue
                graph[pos] = []
        for pos in graph:
            graph[pos].extend(other for direction in DIRECTIONS
                              if (other := self.neighbor(pos, direction)) in graph)
        groups = defaultdict(list)
        for y, row in enumerate(self.base['board']):
            for x, tile in enumerate(row):
                if tile in TUNNEL_EXIT:
                    groups[self.base['mod_nums'][y][x]].append((y, x))
        for a, b in groups.values():
            graph[a].append(b)
            graph[b].append(a)
        return graph

    @staticmethod
    def _component_labels(graph):
        labels = {}
        for pos in graph:
            if pos in labels:
                continue
            label = len(labels)
            labels[pos] = label
            todo = [pos]
            while todo:
                for other in graph[todo.pop()]:
                    if other not in labels:
                        labels[other] = label
                        todo.append(other)
        return labels

    @staticmethod
    def _reachable(graph, start, blocked=None):
        if start == blocked or start not in graph:
            return set()
        seen, todo = {start}, [start]
        while todo:
            for other in graph[todo.pop()]:
                if other != blocked and other not in seen:
                    seen.add(other)
                    todo.append(other)
        return seen

    def initial_bounds(self):
        # Each cut vertex that every possible completion must traverse costs a
        # rail. Ignoring directions/gates only adds possibilities, so this graph
        # is an optimistic relaxation rather than a replacement simulator.
        for start, stations, goals, _ in self._requirements:
            reached = self._reachable(self._graph, start)
            if not stations <= reached or not goals & reached:
                return None
        for pos in self.empty:
            self.check_limit()
            for start, stations, goals, _ in self._requirements:
                reached = self._reachable(self._graph, start, pos)
                if not stations <= reached or not goals & reached:
                    self.mandatory.add(pos)
                    break
        return self.bound({})

    def bound(self, assignments):
        self.stats['bound_checks'] += 1
        self.check_limit()
        key = (frozenset(assignments), frozenset(pos for pos, tile in assignments.items() if tile == 0))
        if key in self._bound_cache:
            self.stats['bound_cache_hits'] += 1
            return self._bound_cache[key]
        value = self._compute_bound(assignments)
        if len(self._bound_cache) >= 8192:
            self._bound_cache.clear()
        self._bound_cache[key] = value
        return value

    def _compute_bound(self, assignments):
        graph = self._graph if not any(tile == 0 for tile in assignments.values()) else self._potential_graph(assignments)
        used = sum(tile != 0 for tile in assignments.values())
        required = self.mandatory - assignments.keys()
        component_costs = defaultdict(int)
        for start, stations, goals, component in self._requirements:
            self.check_limit()
            dist, todo = {start: 0}, deque([start])
            while todo:
                current = todo.popleft()
                for other in graph.get(current, ()):
                    weight = int(other in self.empty_set and other not in assignments and other not in required)
                    value = dist[current] + weight
                    if value < dist.get(other, math.inf):
                        dist[other] = value
                        (todo.append if weight else todo.appendleft)(other)
            if any(pos not in dist for pos in stations):
                return None
            choices = [dist[pos] for pos in goals if pos in dist]
            if not choices:
                return None
            cost = max([min(choices)] + [dist[pos] for pos in stations])
            component_costs[component] = max(component_costs[component], cost)
        return used + len(required) + sum(component_costs.values())

    def make_candidate(self, assignments, signals):
        candidate = deepcopy(self.base)
        for (y, x), tile in assignments.items():
            if isinstance(tile, tuple):
                preferred = self.best['board'][y][x] if self.best is not None else None
                tile = preferred if preferred in tile else tile[0]
            candidate['board'][y][x] = tile
        for (y, x), placed in signals.items():
            if placed:
                candidate['mods'][y][x] = 8
        return candidate

    def run_simulation(self, candidate, *, incumbent=False, assignments=None, signals=None):
        self.check_limit()
        self._visited_stops = defaultdict(set)
        self.stats['simulations'] += 1
        if incumbent:
            self.stats['incumbent_validations'] += 1
        hook = None if incumbent else lambda cars: self.query_cars(cars, candidate, assignments, signals)
        result = simulate(self.level, candidate, self.max_ticks, record_trace=False, decision_hook=hook)
        self.stats['simulation_ticks'] += result.get('tick', 0)
        return result

    def accept(self, candidate, result):
        cost = sum(self.base['board'][y][x] == 0 and tile != 0
                   for y, row in enumerate(candidate['board']) for x, tile in enumerate(row))
        signals = sum(self.base['mods'][y][x] == 0 and mod == 8
                      for y, row in enumerate(candidate['mods']) for x, mod in enumerate(row))
        if cost < self.best_cost or cost == self.best_cost and (self.best_signals is None or signals < self.best_signals):
            self.best, self.best_cost, self.best_signals = candidate, cost, signals
            self.best_trace_steps = result['tick']
        self.stats['solutions_found'] += 1

    def add_incumbent(self, incumbent):
        if incumbent is None:
            return
        value = parse_json(incumbent)
        value.pop('metadata', None)
        value = normalize_answer(self.level, value)
        candidate = check_solution(self.level, value)
        result = self.run_simulation(candidate, incumbent=True)
        if result['success']:
            self.accept(candidate, result)
        elif result['error_type'] == 'resource_limit':
            self.unresolved = True
            self.stats['unresolved_simulations'] += 1
        else:
            raise ValidationError('The supplied incumbent is not a valid solution: ' + result['error_type'])

    def signal_connectors(self, pos, candidate, assignments):
        """All ways to satisfy a new signal's immediate static rail connection."""
        y, x = pos
        unknown = []
        for direction in PORTS[candidate['board'][y][x]]:
            other = self.neighbor(pos, direction)
            if not self.inside(other):
                continue
            tile = candidate['board'][other[0]][other[1]]
            if tile in _JUNCTIONS and REVERSE[direction] in PORTS[tile]:
                return [()]
            if other in self.empty_set and other not in assignments:
                unknown.extend(((other, code),) for code in _JUNCTIONS if REVERSE[direction] in PORTS[code])
        return unknown

    def query_cars(self, cars, candidate, assignments, signals):
        remaining = self.signal_budget - sum(signals.values())
        for car in cars:
            key = car['type'], car['num']
            stops = self._station_targets.get(key, set())
            if tuple(car['pos']) in stops:
                self._visited_stops[key].add(tuple(car['pos']))
            if car['status'] in ('delivered', 'crashed') or car['dwell']:
                continue
            self._hint_targets = tuple(sorted(stops - self._visited_stops[key] or
                                              self._destination_targets.get(car['type'], ())))
            pos = self.neighbor(tuple(car['pos']), car['direction'])
            if not self.inside(pos):
                continue
            if pos in self.empty_set and pos not in assignments:
                return 'rail', pos, car['direction'], car['type'], None
            domain = assignments.get(pos)
            if isinstance(domain, tuple) and len({ROUTES[tile].get(car['direction']) for tile in domain}) > 1:
                return 'refine', pos, car['direction'], car['type'], None
            y, x = pos
            if (remaining and pos not in signals and self.base['mods'][y][x] == 0
                    and candidate['board'][y][x] in SIMPLE_TRACKS):
                connectors = self.signal_connectors(pos, candidate, assignments)
                if connectors:
                    # Waiting a later delivery car beside an existing junction
                    # often lets its predecessor pass first. This only changes
                    # branch order; both signal choices remain in the search.
                    self._prefer_signal = connectors == [()] and any(
                        other['type'] == car['type'] and other['num'] < car['num']
                        and other['status'] != 'delivered' for other in cars)
                    return 'signal', pos, car['direction'], car['type'], connectors
        return None

    def possible_next(self, pos, heading, assignments, kind):
        other = self.neighbor(pos, heading)
        if not self.inside(other):
            return False
        tile = assignments.get(other, self.base['board'][other[0]][other[1]])
        if other in self.empty_set and other not in assignments:
            return True
        if isinstance(tile, tuple):
            return any(heading in ROUTES[code] for code in tile)
        if tile in ROUTES:
            return heading in ROUTES[tile]
        if tile in TUNNEL_EXIT:
            return heading == REVERSE[TUNNEL_EXIT[tile]]
        return tile in DESTINATIONS and DESTINATIONS[tile] == (kind, heading)

    def rail_options(self, pos, heading, kind, assignments):
        existing = assignments.get(pos)
        choices = list(existing) if isinstance(existing, tuple) else list(_RAILS)
        if kind != 'DECOY':
            choices = [tile for tile in choices if heading in ROUTES[tile]
                       and self.possible_next(pos, ROUTES[tile][heading], assignments, kind)]
        elif existing is None:
            choices.insert(0, 0)  # An intentional trolley wreck is a legal decision.
        preferred = self.best['board'][pos[0]][pos[1]] if self.best is not None else None
        distances = self.target_distances(self._hint_targets) if kind != 'DECOY' else {}
        def preference(tile):
            outgoing = ROUTES.get(tile, {}).get(heading)
            distance = distances.get(self.neighbor(pos, outgoing), math.inf) if outgoing else math.inf
            # A decoy may already have pressed every useful control. Try its
            # legal wreck before spending rails extending its route; retain
            # every rail option in case the decoy still has work to do.
            return kind == 'DECOY' and tile != 0, tile != preferred, distance, tile not in SIMPLE_TRACKS, tile
        choices.sort(key=preference)
        if self.lazy_domains and pos not in self._signal_geometry:
            groups = {}
            for tile in choices:
                key = ('empty',) if tile == 0 else ROUTES[tile].get(heading)
                groups.setdefault(key, []).append(tile)
            return [0 if values == [0] else tuple(values) for values in groups.values()]
        return choices

    def target_distances(self, targets):
        """An ordering hint only; ignores directions and timing, never prunes."""
        if targets not in self._target_distance_cache:
            distances = {pos: 0 for pos in targets}
            todo = deque(targets)
            while todo:
                pos = todo.popleft()
                for other in self._graph.get(pos, ()):
                    if other not in distances:
                        distances[other] = distances[pos] + 1
                        todo.append(other)
            self._target_distance_cache[targets] = distances
        return self._target_distance_cache[targets]

    def search(self):
        stack = [({}, {})]
        while stack:
            self.check_limit(nodes=True)
            assignments, signals = stack.pop()
            self.stats['search_nodes'] += 1
            self.stats['maximum_depth'] = max(self.stats['maximum_depth'], len(assignments) + len(signals))
            cost_bound = self.bound(assignments)
            if cost_bound is None or cost_bound > self.budget or self.optimize and cost_bound >= self.best_cost:
                self.stats['pruned_by_bound'] += 1
                continue
            candidate = self.make_candidate(assignments, signals)
            result = self.run_simulation(candidate, assignments=assignments, signals=signals)
            query = result.get('decision')
            if query is not None:
                kind, pos, heading, car_type, connectors = query
                if kind in ('rail', 'refine'):
                    self.stats['rail_decisions' if kind == 'rail' else 'rail_refinements'] += 1
                    for tile in reversed(self.rail_options(pos, heading, car_type, assignments)):
                        cost = (sum(value != 0 for value in assignments.values()) + bool(tile)
                                - bool(assignments.get(pos, 0)))
                        if cost > self.budget:
                            continue
                        child = dict(assignments)
                        child[pos] = tile
                        stack.append((child, signals))
                else:
                    self.stats['signal_decisions'] += 1
                    chosen = dict(signals)
                    chosen[pos] = False
                    children = [(assignments, chosen)]
                    for additions in reversed(connectors):
                        child = dict(assignments)
                        child.update(additions)
                        if sum(tile != 0 for tile in child.values()) > self.budget:
                            continue
                        chosen = dict(signals)
                        chosen[pos] = True
                        children.insert(1, (child, chosen))
                    if self._prefer_signal:
                        children = children[1:] + children[:1]
                    stack.extend(reversed(children))
                continue
            if result['success']:
                self.accept(check_solution(self.level, candidate), result)
                if not self.optimize or self.best_cost == self.lower_bound:
                    self.stop_reason = 'solution_found' if not self.optimize else 'lower_bound_matched'
                    return self.optimize and self.best_cost == self.lower_bound
            elif result['error_type'] == 'resource_limit':
                self.unresolved = True
                self.stats['unresolved_simulations'] += 1
            else:
                self.stats['pruned_by_failure'] += 1
        self.stop_reason = 'simulation_limit' if self.unresolved else 'search_exhausted'
        return not self.unresolved

    def result(self, complete):
        found = self.best is not None
        optimal = bool(found and self.optimize and (complete or self.best_cost == self.lower_bound))
        status = 'OPTIMAL' if optimal else 'SAT' if found else 'UNSAT' if complete else 'UNKNOWN'
        stats = dict(self.stats)
        stats.update(elapsed_seconds=time.perf_counter() - self.started, complete_search=bool(complete),
                     cpu_seconds=time.process_time() - self.cpu_started + self.stats.get('native_cpu_seconds', 0),
                     search_exhausted=self.stop_reason == 'search_exhausted',
                     time_limit_reached=self.stop_reason == 'time_limit',
                     node_limit_reached=self.stop_reason == 'node_limit')
        return dict(status=status, success=found, optimality_proven=optimal,
                    search_mode='route_domains' if self.lazy_domains or
                    self.stats.get('native_backend') else 'concrete_rails',
                    backend='native' if self.stats.get('native_backend') else 'python',
                    minimum_tracks=self.best_cost if optimal else None,
                    tracks_used=self.best_cost if found else None,
                    semaphores_used=self.best_signals if found else None,
                    lower_bound=self.best_cost if optimal else max(self.lower_bound, self.budget + 1)
                    if status == 'UNSAT' else self.lower_bound,
                    track_budget=self.budget, semaphore_budget=self.signal_budget,
                    simulation_steps=self.best_trace_steps,
                    solution=deepcopy(self.best), answer=encode_answer(self.level, self.best) if found else None,
                    reason=self.stop_reason,
                    infeasibility_scope='The immutable board, track budget and semaphore budget in this result.',
                    statistics=stats)


def solve(level, track_budget=None, *, timeout=600.0, node_limit=None,
          optimize=True, incumbent=None, max_ticks=10000, backend='auto'):
    """Find a valid layout, and prove its minimum track count when possible.

    ``level`` accepts a dict, JSON text, bytes or explicit pathlib.Path.
    ``track_budget`` overrides only a copied problem's track limit. A returned
    full solution uses that same limit. Signals retain the level's budget.
    ``incumbent`` is an optional additions answer, such as a reference file or
    a previous result's ``answer``, always independently checked before use. Time limits are cooperative between
    simulations; ``max_ticks`` limits each simulator call. Neither limit proves
    UNSAT. The default search-time limit is ten minutes with no node cap.
    Set ``timeout``/``node_limit`` to None to disable those limits. The native
    backend also propagates route domains with a positive semaphore budget;
    placing a signal fixes physical geometry along every observed port.
    The Python backend uses domains only with zero new-semaphore budget.
    ``backend='auto'`` uses the optional native C++14 accelerator when g++ or
    clang++ is available and otherwise retains the Python implementation.
    ``backend='native'`` requires it; ``backend='python'`` selects the portable
    reference implementation. Native witnesses always receive a Python replay.
    """
    if track_budget is not None and (type(track_budget) is not int or track_budget < 0):
        raise ValueError('track_budget must be a nonnegative integer or None')
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                                or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('timeout must be a positive finite number or None')
    if node_limit is not None and (type(node_limit) is not int or node_limit < 1):
        raise ValueError('node_limit must be a positive integer or None')
    if type(max_ticks) is not int or max_ticks < 1:
        raise ValueError('max_ticks must be a positive integer')
    if type(optimize) is not bool:
        raise ValueError('optimize must be a boolean')
    if backend not in ('auto', 'native', 'python'):
        raise ValueError('backend must be auto, native or python')
    search = _Search(level, track_budget, timeout, node_limit, max_ticks, optimize)
    complete = False
    try:
        search.add_incumbent(incumbent)
        if search.best is not None and not optimize:
            search.stop_reason = 'incumbent_verified'
            return search.result(False)
        bound = search.initial_bounds()
        if bound is None or bound > search.budget:
            if search.best is not None:
                raise RuntimeError('A valid incumbent contradicts the static relaxation')
            search.lower_bound = search.budget + 1 if bound is None else bound
            search.stop_reason = 'unreachable' if bound is None else 'budget_lower_bound'
            complete = True
        else:
            search.lower_bound = bound
            if search.best is not None and search.best_cost == bound:
                search.stop_reason = 'lower_bound_matched'
                complete = True
            else:
                native_complete = None
                if backend != 'python':
                    from .native import run
                    native_complete = run(search)
                    if native_complete is None and backend == 'native':
                        raise RuntimeError('The native backend needs g++ or clang++ with C++14 support')
                complete = search.search() if native_complete is None else native_complete
    except _LimitReached:
        pass
    return search.result(complete)
