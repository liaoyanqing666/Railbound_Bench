"""Deterministic synchronous Railbound simulation.

One tick is a movement opportunity. Runtime directions are outgoing headings,
selected on entry to a tile. This module never mutates either input.
"""
from collections import defaultdict
from copy import deepcopy


VECTORS = {'LEFT': (0, -1), 'RIGHT': (0, 1), 'DOWN': (1, 0), 'UP': (-1, 0)}
REVERSE = {'LEFT': 'RIGHT', 'RIGHT': 'LEFT', 'DOWN': 'UP', 'UP': 'DOWN'}
# Incoming movement heading -> outgoing heading; a missing key is a derailment.
ROUTES = {
    1: {'LEFT': 'LEFT', 'RIGHT': 'RIGHT'},
    2: {'DOWN': 'DOWN', 'UP': 'UP'},
    5: {'LEFT': 'DOWN', 'UP': 'RIGHT'},
    6: {'RIGHT': 'DOWN', 'UP': 'LEFT'},
    7: {'LEFT': 'UP', 'DOWN': 'RIGHT'},
    8: {'RIGHT': 'UP', 'DOWN': 'LEFT'},
    9: {'LEFT': 'DOWN', 'RIGHT': 'RIGHT', 'UP': 'RIGHT'},
    10: {'LEFT': 'DOWN', 'DOWN': 'DOWN', 'UP': 'RIGHT'},
    11: {'LEFT': 'LEFT', 'RIGHT': 'DOWN', 'UP': 'LEFT'},
    12: {'RIGHT': 'DOWN', 'DOWN': 'DOWN', 'UP': 'LEFT'},
    13: {'LEFT': 'UP', 'RIGHT': 'RIGHT', 'DOWN': 'RIGHT'},
    14: {'LEFT': 'UP', 'DOWN': 'RIGHT', 'UP': 'UP'},
    15: {'LEFT': 'LEFT', 'RIGHT': 'UP', 'DOWN': 'LEFT'},
    16: {'RIGHT': 'UP', 'DOWN': 'LEFT', 'UP': 'UP'},
}
SWAPPED = {9: 11, 11: 9, 10: 14, 14: 10, 12: 16, 16: 12, 13: 15, 15: 13}
TUNNEL_EXIT = {17: 'LEFT', 18: 'RIGHT', 19: 'DOWN', 20: 'UP'}
DESTINATIONS = {
    3: ('NORMAL', 'RIGHT'), 29: ('NORMAL', 'LEFT'),
    30: ('NORMAL', 'DOWN'), 31: ('NORMAL', 'UP'),
    21: ('NUMERAL', 'RIGHT'), 22: ('NUMERAL', 'LEFT'),
    32: ('NUMERAL', 'DOWN'), 33: ('NUMERAL', 'UP'),
}


def _advance(pos, direction):
    dy, dx = VECTORS[direction]
    return (pos[0] + dy, pos[1] + dx)


def _ports(tile):
    return {REVERSE[d] for d in ROUTES.get(tile, {})}


def simulate(level, solution, max_ticks=10000, *, record_trace=True, decision_hook=None):
    """Run the shared rules, optionally stopping before an unresolved move.

    The validator keeps every tick by default. Layout search can omit snapshots
    and inspect live cars through a read-only hook before each movement tick.
    A non-None hook result stops execution as ``incomplete_layout``; it is a
    search decision, never an infeasibility judgment.
    """
    if type(max_ticks) is not int or max_ticks < 1:
        raise ValueError('max_ticks must be a positive integer')
    if type(record_trace) is not bool:
        raise ValueError('record_trace must be a boolean')
    if decision_hook is not None and not callable(decision_hook):
        raise ValueError('decision_hook must be callable or None')
    board = deepcopy(solution['board'])
    mods = deepcopy(solution['mods'])
    nums = deepcopy(solution['mod_nums'])
    cars = deepcopy(level['cars'])
    height, width = len(board), len(board[0])
    for index, car in enumerate(cars):
        car.update(id=index, status='moving', dwell=0, stalled=False)
    delivered = {'NORMAL': [], 'NUMERAL': []}
    # Entries are processed at the beginning of the following movement tick.
    entries = list(range(len(cars)))
    pending_close = set()
    desired_gate = {(y, x): mods[y][x] == 4 for y in range(height)
                    for x in range(width) if mods[y][x] in (3, 4)}
    changing_tracks = [(y, x) for y in range(height) for x in range(width) if mods[y][x] in (5, 7)]
    changing_modifiers = [(y, x) for y in range(height) for x in range(width) if mods[y][x] in (3, 4, 6, 8, 11)]
    tunnel_groups = defaultdict(list)
    for y in range(height):
        for x in range(width):
            if board[y][x] in TUNNEL_EXIT:
                tunnel_groups[nums[y][x]].append((y, x))
    tunnels = {}
    for positions in tunnel_groups.values():
        if len(positions) != 2:
            return dict(success=False, error_type='invalid_tunnel',
                        message='Each tunnel identifier must have exactly two mouths.',
                        trace=[], error_positions=[list(p) for p in positions])
        a, b = positions
        tunnels[a], tunnels[b] = b, a
    victory_countdown = None
    trace = []
    tick = 0

    def inside(pos):
        return 0 <= pos[0] < height and 0 <= pos[1] < width

    def snapshot(tick, errors=None):
        state = dict(tick=tick, board=[row[:] for row in board], mods=[row[:] for row in mods],
                     mod_nums=[row[:] for row in nums],
                     cars=[dict(car, pos=list(car['pos'])) for car in cars],
                     delivered={kind: list(values) for kind, values in delivered.items()}, pending_entries=list(entries),
                     pending_gate_closures=[list(p) for p in sorted(pending_close)],
                     gate_targets=[dict(pos=list(p), open=opened)
                                   for p, opened in sorted(desired_gate.items())],
                     victory_countdown=victory_countdown)
        if errors:
            state['error_positions'] = [list(p) for p in sorted(set(errors)) if inside(p)]
        return state

    def signature():
        # Every future-affecting value is represented; tick is only display data.
        # Other tiles, modifiers and IDs are constants within this execution.
        return (tuple(board[y][x] for y, x in changing_tracks),
                tuple(mods[y][x] for y, x in changing_modifiers),
                tuple((tuple(car['pos']), car['direction'],
                       car['status'], car['dwell'], car['stalled']) for car in cars),
                tuple(delivered['NORMAL']), tuple(delivered['NUMERAL']), tuple(entries),
                tuple(sorted(pending_close)), tuple(sorted(desired_gate.items())), victory_countdown)

    def result(success, kind, message, positions=(), **extra):
        extra.setdefault('tick', tick)
        return dict(success=success, error_type=kind, message=message, trace=trace,
                    error_positions=[list(p) for p in sorted(set(positions)) if inside(p)],
                    **extra)

    if record_trace:
        trace.append(snapshot(0))
    seen = {signature(): 0}
    for tick in range(1, max_ticks + 1):
        if decision_hook is not None:
            decision = decision_hook(cars)
            if decision is not None:
                return result(False, 'incomplete_layout', 'The layout needs another search decision.',
                              decision=decision, tick=tick - 1)
        occupied = {tuple(car['pos']) for car in cars if car['status'] != 'delivered'}
        for pos in list(pending_close):
            if pos not in occupied:
                mods[pos[0]][pos[1]] = 3
                pending_close.remove(pos)

        button_counts = defaultdict(int)
        entered_positions = []
        for index in entries:
            car = cars[index]
            if car['status'] in ('delivered', 'crashed'):
                continue
            y, x = car['pos']
            entered_positions.append((tuple(car['pos']), car['direction']))
            if mods[y][x] == 1:
                button_counts[nums[y][x]] += 1
            elif mods[y][x] == 7:
                board[y][x] = SWAPPED.get(board[y][x], board[y][x])
        for y in range(height):
            for x in range(width):
                if button_counts[nums[y][x]] % 2:
                    if (y, x) in desired_gate:
                        opened = not desired_gate[y, x]
                        desired_gate[y, x] = opened
                        if opened:
                            mods[y][x] = 4
                            pending_close.discard((y, x))
                        elif (y, x) in occupied:
                            pending_close.add((y, x))
                        else:
                            mods[y][x] = 3
                    elif mods[y][x] == 5:
                        board[y][x] = SWAPPED.get(board[y][x], board[y][x])
                if mods[y][x] == 8:
                    # A car passing a connected adjacent junction releases the
                    # signal permanently; a car heading toward it cannot release itself.
                    for port in _ports(board[y][x]):
                        neighbor = _advance((y, x), port)
                        if not inside(neighbor) or board[neighbor[0]][neighbor[1]] not in SWAPPED:
                            continue
                        if REVERSE[port] not in _ports(board[neighbor[0]][neighbor[1]]):
                            continue
                        if any(pos == neighbor and heading != REVERSE[port]
                               for pos, heading in entered_positions):
                            mods[y][x] = 9
                            break

        entries = []
        proposals = []
        failures = []
        previously_stalled = {i for i, c in enumerate(cars) if c['stalled']}
        for index, car in enumerate(cars):
            pos = tuple(car['pos'])
            proposed = dict(pos=pos, direction=car['direction'], moved=False,
                            tunnel=None, entrance=None, deliver=False, wreck=False)
            proposals.append(proposed)
            if car['status'] in ('delivered', 'crashed'):
                continue
            car['stalled'] = False
            if car['dwell']:
                car['dwell'] -= 1
                car['status'] = 'boarding' if car['dwell'] else 'moving'
                continue
            dest = _advance(pos, car['direction'])
            if inside(dest) and mods[dest[0]][dest[1]] in (3, 8):
                car['stalled'] = True
                car['status'] = 'waiting'
                continue
            car['status'] = 'moving'
            if not inside(dest):
                if car['type'] == 'DECOY':
                    proposed['wreck'] = True
                else:
                    failures.append(('derailment', index, pos, 'A car ran outside the board.'))
                continue
            tile = board[dest[0]][dest[1]]
            direction = car['direction']
            if tile in DESTINATIONS:
                kind, required_direction = DESTINATIONS[tile]
                if direction != required_direction:
                    if car['type'] == 'DECOY':
                        proposed['wreck'] = True
                    else:
                        failures.append(('derailment', index, dest, 'A locomotive was approached from the wrong side.'))
                elif car['type'] == 'DECOY':
                    failures.append(('decoy_locomotive', index, dest, 'A trolley reached a locomotive.'))
                elif car['type'] != kind:
                    failures.append(('wrong_locomotive', index, dest, 'A car reached the wrong locomotive type.'))
                elif car['num'] != len(delivered[kind]):
                    failures.append(('wrong_order', index, dest, 'Cars must reach each locomotive in number order.'))
                elif any(mods[y][x] == (6 if kind == 'NORMAL' else 11) and nums[y][x] == car['num']
                         for y in range(height) for x in range(width)):
                    failures.append(('unvisited_station', index, dest, 'A car reached its locomotive before visiting every matching station.'))
                else:
                    proposed.update(pos=dest, moved=True, deliver=True)
                continue
            if tile in TUNNEL_EXIT:
                if direction != REVERSE[TUNNEL_EXIT[tile]]:
                    if car['type'] == 'DECOY':
                        proposed['wreck'] = True
                    else:
                        failures.append(('derailment', index, dest, 'A car entered the closed side of a tunnel.'))
                    continue
                exit_pos = tunnels[dest]
                proposed.update(pos=exit_pos, moved=True, entrance=dest,
                                tunnel=tuple(sorted((dest, exit_pos))),
                                direction=TUNNEL_EXIT[board[exit_pos[0]][exit_pos[1]]])
                continue
            outgoing = ROUTES.get(tile, {}).get(direction)
            if outgoing is None:
                if car['type'] == 'DECOY':
                    proposed['wreck'] = True
                else:
                    failures.append(('derailment', index, dest, 'A car reached missing or incompatible rails.'))
                continue
            proposed.update(pos=dest, direction=outgoing, moved=True)

        # A head-on exchange at a tunnel mouth happens before either vehicle
        # can transfer to the remote exit.
        blocked_transfers = set()
        for a in range(len(cars)):
            for b in range(a + 1, len(cars)):
                pa, pb = proposals[a], proposals[b]
                first_a = pa['entrance'] if pa['entrance'] is not None else pa['pos']
                first_b = pb['entrance'] if pb['entrance'] is not None else pb['pos']
                if (pa['moved'] and pb['moved'] and first_a == tuple(cars[b]['pos'])
                        and first_b == tuple(cars[a]['pos'])):
                    blocked_transfers.update(index for index in (a, b) if proposals[index]['entrance'] is not None)
        for index in blocked_transfers:
            proposals[index].update(pos=proposals[index]['entrance'], direction=cars[index]['direction'], tunnel=None)

        # Resolve trolley wrecks simultaneously, including a vehicle that was
        # following a trolley which now stops. Recheck to a fixed point so car
        # array order cannot change whether a chain collision happens.
        while True:
            # A newly stopped trolley changes the simultaneous proposals.
            # Keep only contacts in the stable configuration, rather than
            # retaining contacts with its discarded moving proposal.
            collision_failures = {}
            collision_pairs = {}
            new_wrecks = set()
            for a in range(len(cars)):
                if cars[a]['status'] == 'delivered':
                    continue
                for b in range(a + 1, len(cars)):
                    if cars[b]['status'] == 'delivered':
                        continue
                    pa, pb = proposals[a], proposals[b]
                    old_a, old_b = tuple(cars[a]['pos']), tuple(cars[b]['pos'])
                    tunnel_crash = pa['tunnel'] is not None and pa['tunnel'] == pb['tunnel']
                    collision = (pa['pos'] == pb['pos'] or
                                 (pa['moved'] and pb['moved'] and pa['pos'] == old_b and pb['pos'] == old_a) or
                                 (a in previously_stalled and pb['moved'] and pb['pos'] == old_a) or
                                 (b in previously_stalled and pa['moved'] and pa['pos'] == old_b) or
                                 (pa['entrance'] == pb['pos']) or (pb['entrance'] == pa['pos']))
                    if not (collision or tunnel_crash):
                        continue
                    edge_swap = (pa['moved'] and pb['moved'] and pa['pos'] == old_b and pb['pos'] == old_a)
                    rear_end = (cars[a]['direction'] == cars[b]['direction'] or
                                a in previously_stalled or b in previously_stalled or
                                not pa['moved'] or not pb['moved'])
                    geometry = ('tunnel_opposite' if tunnel_crash else 'edge_swap' if edge_swap else
                                'rear_end' if rear_end else 'same_cell')
                    if edge_swap:
                        impact = [(old_a[0] + old_b[0]) / 2, (old_a[1] + old_b[1]) / 2]
                    elif a in previously_stalled and pb['pos'] == old_a:
                        impact = list(old_a)
                    elif b in previously_stalled and pa['pos'] == old_b:
                        impact = list(old_b)
                    elif pa['entrance'] is not None and pa['entrance'] == pb['pos']:
                        impact = list(pa['entrance'])
                    elif pb['entrance'] is not None and pb['entrance'] == pa['pos']:
                        impact = list(pb['entrance'])
                    else:
                        impact = list(pa['pos'])
                    event = dict(kind='tunnel_collision' if tunnel_crash else 'collision',
                                 geometry=geometry, car_ids=[a, b], impact=impact,
                                 positions=[list(pa['pos']), list(pb['pos'])])
                    if rear_end:
                        event['leader_id'] = (a if not pa['moved'] or pb['pos'] == old_a or a in previously_stalled else b)
                    if tunnel_crash:
                        event['positions'] = [list(pos) for pos in pa['tunnel']]
                        event['tunnel_id'] = nums[pa['tunnel'][0][0]][pa['tunnel'][0][1]]
                    if cars[a]['type'] == cars[b]['type'] == 'DECOY':
                        new_wrecks.update(index for index in (a, b) if not proposals[index]['wreck'])
                    else:
                        collision_pairs.setdefault((a, b), event)
                        kind = 'tunnel_collision' if tunnel_crash else 'collision'
                        for index in (a, b):
                            collision_failures[index] = (kind, index, proposals[index]['pos'], 'Two vehicles collided.')
            if not new_wrecks:
                break
            for index in new_wrecks:
                proposals[index].update(pos=tuple(cars[index]['pos']), moved=False,
                                        wreck=True, tunnel=None, entrance=None)
        failures.extend(collision_failures[index] for index in sorted(collision_failures))

        error_positions = []
        for index, (car, proposed) in enumerate(zip(cars, proposals)):
            if car['status'] == 'delivered':
                continue
            if proposed['wreck']:
                car.update(status='crashed', stalled=False, dwell=0)
            elif proposed['moved']:
                car['pos'] = list(proposed['pos'])
                car['direction'] = proposed['direction']
                if proposed['deliver']:
                    car['status'] = 'delivered'
                    delivered[car['type']].append(car['num'])
                else:
                    entries.append(index)
                    y, x = car['pos']
                    if (car['type'] == 'NORMAL' and mods[y][x] == 6 or
                            car['type'] == 'NUMERAL' and mods[y][x] == 11) and nums[y][x] == car['num']:
                        car.update(dwell=2, status='boarding')
                        mods[y][x] = 9
        for kind, index, pos, message in failures:
            cars[index]['status'] = 'crashed'
            if kind == 'unvisited_station':
                mod = 6 if cars[index]['type'] == 'NORMAL' else 11
                error_positions.extend((y, x) for y, row in enumerate(mods) for x, value in enumerate(row)
                                       if value == mod and nums[y][x] == cars[index]['num'])
            else:
                error_positions.extend((tuple(cars[index]['pos']), pos))
        if failures:
            events = []
            if record_trace:
                from .conflicts import describe_conflicts
                previous = trace[-1]
                events, motion = describe_conflicts(previous, cars, proposals, failures, collision_pairs,
                                                    board, mods, nums, previous['delivered'])
                events.sort(key=lambda event: event['kind'] != failures[0][0])
                failed_state = snapshot(tick, error_positions)
                failed_state.update(previous_tick=previous['tick'], previous_cars=deepcopy(previous['cars']),
                                    motion=motion, conflicts=events)
                trace.append(failed_state)
            kind, _, _, message = failures[0]
            return result(False, kind, message, error_positions, conflicts=events)

        all_delivered = all(c['status'] == 'delivered' for c in cars if c['type'] != 'DECOY')
        if all_delivered:
            victory_countdown = 2 if victory_countdown is None else victory_countdown - 1
        if record_trace:
            trace.append(snapshot(tick))
        if all_delivered and (victory_countdown == 0 or not any(
                c['type'] == 'DECOY' and c['status'] != 'crashed' for c in cars)):
            return result(True, None, 'All required cars reached their matching locomotives in order.')
        state_key = signature()
        if state_key in seen:
            positions = [tuple(c['pos']) for c in cars if c['status'] not in ('delivered', 'crashed')]
            return result(False, 'loop', 'The simulation entered a loop; this layout cannot finish.',
                          positions, loop_start=seen[state_key], loop_end=tick)
        seen[state_key] = tick
    return result(False, 'resource_limit', 'The simulation reached max_ticks without a result.')
