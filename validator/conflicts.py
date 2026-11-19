"""Describe a failed movement for self-contained visual feedback."""
from copy import deepcopy

from .tiles import DIRECTIONS, ENDINGS, TUNNELS


def describe_conflicts(previous, cars, proposals, failures, pairs, board, mods, numbers, delivered):
    events = [deepcopy(value) for _, value in sorted(pairs.items())]
    for kind, index, pos, _ in failures:
        if kind in ('collision', 'tunnel_collision'):
            continue
        car = previous['cars'][index]
        dy, dx = DIRECTIONS[car['direction']]
        target = [car['pos'][0] + dy, car['pos'][1] + dx]
        inside = 0 <= target[0] < len(board) and 0 <= target[1] < len(board[0])
        tile = board[target[0]][target[1]] if inside else None
        geometry = kind
        if kind == 'derailment':
            geometry = ('outside_board' if not inside else 'missing_track' if tile == 0 else
                        'obstacle' if tile == 4 else 'wrong_tunnel_side' if tile in TUNNELS else
                        'wrong_locomotive_side' if tile in ENDINGS else 'incompatible_track')
        event = dict(kind=kind, geometry=geometry, car_ids=[index], target=target,
                     impact=[float(target[0]), float(target[1])], positions=[list(pos)])
        if kind == 'unvisited_station':
            mod = 6 if car['type'] == 'NORMAL' else 11
            event['stations'] = [[y, x] for y, row in enumerate(mods) for x, value in enumerate(row)
                                 if value == mod and numbers[y][x] == car['num']]
            event['positions'] = deepcopy(event['stations'])
        if kind == 'wrong_order':
            event['expected_number'] = len(delivered[car['type']])
        events.append(event)
    affected = {index for event in events for index in event['car_ids']}
    motion = []
    for index in sorted(affected):
        car, current = previous['cars'][index], cars[index]
        dy, dx = DIRECTIONS[car['direction']]
        attempted = [car['pos'][0] + dy, car['pos'][1] + dx]
        reason = None
        if car['status'] == 'crashed':
            reason = 'wreck'
        elif car['dwell']:
            reason = 'post_office' if car['type'] == 'NUMERAL' else 'passenger_stop'
        elif current.get('stalled') or car.get('stalled'):
            y, x = attempted
            if 0 <= y < len(board) and 0 <= x < len(board[0]):
                reason = 'semaphore' if previous['mods'][y][x] == 8 else 'gate'
        motion.append(dict(id=index, type=car['type'], num=car['num'],
                           **{'from': list(car['pos']), 'to': list(proposals[index]['pos'])},
                           attempted=attempted, direction=car['direction'],
                           outgoing_direction=proposals[index]['direction'],
                           entrance=list(proposals[index]['entrance']) if proposals[index]['entrance'] else None,
                           status_before=car['status'], waiting_reason=reason))
    return events, motion
