"""Expand additions answers into a complete layout for the shared validator."""
from copy import deepcopy

from .api import parse_json, ValidationError
from .tiles import PLACEABLE_TRACKS


def load_solution(level, answer):
    """Read a reference file or an answer and expand its additions.

    Reference files may attach descriptive metadata. Model answers are checked
    by ``normalize_answer``, which rejects any key outside the answer format.
    """
    value = parse_json(answer)
    annotation = value.pop('metadata', None)
    if annotation is not None and type(annotation) is not dict:
        raise ValidationError('metadata must be an object')
    return normalize_answer(level, value)


def _position(values, height, width):
    if any(type(v) is not int or v < 0 for v in values):
        raise ValidationError('Coordinates and track codes must be nonnegative JSON integers')
    y, x = values[:2]
    if not (y < height and x < width):
        raise ValidationError('Position outside the original board', 'illegal_edit', [[y, x]])
    return y, x


def normalize_answer(level, answer):
    """Expand an additions answer without trusting it to supply cars, limits or dimensions."""
    value = parse_json(answer)
    if set(value) - {'add_tracks', 'add_semaphores'} or 'add_tracks' not in value:
        raise ValidationError('Expected keys: add_tracks' + ('; optional add_semaphores' if level['semaphores'] else ''))
    result = deepcopy(level)
    result.pop('metadata', None)
    height, width = len(level['board']), len(level['board'][0])
    additions = value['add_tracks']
    if not isinstance(additions, list): raise ValidationError('add_tracks must be an array')
    seen = set()
    for item in additions:
        if not isinstance(item, list) or len(item) != 3:
            raise ValidationError('Each add_tracks entry must be [row,column,track_code]')
        y, x = _position(item, height, width)
        if (y, x) in seen: raise ValidationError('Duplicate track position')
        seen.add((y, x))
        if item[2] not in PLACEABLE_TRACKS:
            raise ValidationError('Only ordinary rails and junctions may be added', 'illegal_edit', [[y, x]])
        if level['board'][y][x] != 0:
            raise ValidationError('add_tracks must refer only to originally empty cells', 'illegal_edit', [[y, x]])
        result['board'][y][x] = item[2]
    signals = value.get('add_semaphores', [])
    if not isinstance(signals, list): raise ValidationError('add_semaphores must be an array')
    seen = set()
    for item in signals:
        if not isinstance(item, list) or len(item) != 2:
            raise ValidationError('Each add_semaphores entry must be [row,column]')
        y, x = _position(item, height, width)
        if (y, x) in seen: raise ValidationError('Duplicate semaphore position')
        seen.add((y, x))
        if level['mods'][y][x] != 0:
            raise ValidationError('add_semaphores must refer only to cells without an original modifier', 'illegal_edit', [[y, x]])
        result['mods'][y][x] = 8
    return result


def encode_answer(level, solution):
    """Encode a completed layout as additions to the original puzzle."""
    signals = [[y, x] for y, row in enumerate(level['mods']) for x, old in enumerate(row)
               if old == 0 and solution['mods'][y][x] == 8]
    result = {'add_tracks': [[y, x, solution['board'][y][x]] for y, row in enumerate(level['board'])
                             for x, old in enumerate(row) if old == 0 and solution['board'][y][x] != 0]}
    if signals or level['semaphores']: result['add_semaphores'] = signals
    return result
