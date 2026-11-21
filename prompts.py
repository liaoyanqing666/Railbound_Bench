"""English puzzle prompts: image inputs or a rail grid with movement rules."""
import json
from validator.tiles import TRACK_NAMES,MOD_NAMES,mechanisms

PROMPT_MODES = ('image_rules', 'image_no_rules', 'text_rules')

SYSTEM_PROMPT=(
    'Solve this railway-routing puzzle using your own reasoning only. You have no tools, '
    'code execution, browsing, agents, or reference solutions. Return exactly one JSON '
    'object, without Markdown fences or commentary. Do not request or simulate tool calls.'
)

NATIVE_TEXT_CONTINUATION_PROMPT = (
    'Continue solving the same puzzle using your preserved reasoning. '
    'Return one complete solution JSON, not a fragment or an edit to an earlier answer.'
)

RULES={
 'junctions': 'Junction rules: A three-way rail has a preferred curved pair and a straight branch. '
    'The first two direction words in its name are its curved endpoints; the third is the extra straight endpoint. '
    'Entering a curved endpoint follows the curve; entering the extra branch continues straight. '
    'Ordinary junctions keep their orientation. Any newly occupied rail cell costs one track, including a junction.',
 'tunnels': 'Tunnel rules: Matching tunnel IDs pair two portals. Travel is bidirectional: either portal can be the entrance, '
    'and the other becomes the exit. Approach each portal from its open mouth (the named facing side), not its solid back or sides. '
    'Entering teleports the car to the paired portal during that same move, facing outward there, with zero extra waiting ticks. '
    'Opposite simultaneous traversal of one pair collides.',
 'switches_and_gates': 'Switch and gate rules: A car crossing a button toggles every gate and button-switched junction with the same ID. '
    'Effects apply at the next movement opportunity. A car waits before a closed gate until it opens; the delay is event-dependent, not a fixed number. '
    'A car already on a junction keeps its previously selected outgoing heading when the rails switch. A car stalled in front of a gate still blocks a rear car '
    'in the tick in which the gate opens.',
 'self_switching_junctions': 'Automatic junction rules: A self-switching junction changes its preferred curve after a new arrival has selected '
    'its outgoing heading, before the next movement opportunity, with zero extra waiting ticks. A waiting car does not repeatedly switch it.',
 'passenger_stations': 'Passenger station rules: Each station ID is assigned to exactly the NORMAL car with that zero-based number. '
    'Other cars pass without collecting it. On movement entry, its matching car stops for TWO additional movement opportunities, '
    'then moves on. Each passenger station must be served once before its car reaches its locomotive. Multiple stations may belong to one car.',
 'post_offices': 'Post office rules: A post office serves the NUMERAL freight car with the same zero-based number. '
    'It works like a passenger station: movement entry causes TWO additional stationary movement opportunities at each unserved matching office. '
    'Every matching office must be served before delivery; other cars pass without serving it.',
 'semaphores': 'Semaphore rules: You may add at most the stated semaphore budget, only to a simple straight/curved rail with no existing modifier '
    'and adjoining a connected three-way junction. It initially stops cars before its tile. It permanently opens when another car passes '
    'a rail-connected neighboring three-way junction facing away from the signal; opening occurs at the next movement opportunity. '
    'Waiting duration depends on that event, not a fixed timer. Set its modifier to 8; preserve numbered mechanism IDs.',
 'multiple_trains': 'Multiple train rules: NORMAL cars (Arabic labels) and NUMERAL cars (Roman labels) have separate passenger/freight '
    'locomotives and independent delivery order 0,1,2,... . A car must enter a locomotive of its own type.',
 'decoy_cars': 'Decoy rules: DECOY cars (yellow D) need not reach a locomotive. They can operate buttons. Derailing a decoy leaves a stationary '
    'wreck at its last valid rail cell; it can obstruct other cars. It may not deliver to a locomotive. After all real cars are delivered, '
    'the simulator checks two further movement opportunities when decoys remain. '
    'Two decoys may collide and leave wrecks without failing; a collision involving any NORMAL or NUMERAL car fails. '
    'A decoy approaching a locomotive from an incompatible direction wrecks before it; '
    'entering from that locomotive\'s valid delivery side is forbidden.',
}


BASE_RULES = (
    'All cars move simultaneously by one cell each tick unless waiting. A car direction is its next movement direction. '
    'Reach the matching locomotive in number order, starting with num 0. Delivery removes that car at the end of its move, '
    'with no extra global coupling pause; other cars continue. Two cars occupying the same cell, exchanging adjacent cells '
    'head-on, or entering invalid/missing rail crash. Following a moving car into its vacated cell is allowed. '
    'Numbered delivery order is global for each car type across all of its matching locomotives. '
    'All required cars must be delivered. Entering a loop fails.'
)


def _visual_encoding(level, active):
    """Names and glyph decoding only: never disclose how a mechanism behaves."""
    parts = [
        'Read the complete starting board from the image. Column coordinates are printed above the grid; row coordinates are on the left. '
        'Every grid square is one cell. Empty terrain is board code 0; a rock is code 4. '
        'Steel-colored rails are original; amber rails in later feedback are additions. '
        'For a straight/curved rail, its drawn endpoints determine its board code. For a three-way rail, '
        'the solid curved rail gives the first two endpoints in its code name, and the third word gives its other endpoint. '
        'In code names, top=UP and bottom=DOWN. This description specifies the drawing and numeric encoding.',
        'Vehicles have a blunt rear, a pointed front and a white arrow on the nose. The arrow determines direction '
        '(LEFT, RIGHT, UP, DOWN). Blue passenger cars use Arabic labels 1, 2, ... and type NORMAL; '
        'their num is the visible label minus 1. A blue locomotive has a cab, boiler and chimney, '
        'and its board code passenger_end_DIRECTION uses its arrow direction. '
        'A locomotive is a board tile, never an entry in cars. A car may share its starting cell with a drawn mechanism. '
        'Modifier 10 marks a plain car start; it does not replace a mechanism shown in that cell. '
        'The rail visible around the vehicle still determines that cell\'s board code.'
    ]
    if any(car['type']=='NUMERAL' for car in level['cars']):
        parts.append('Purple freight cars use Roman labels I, II, ... and type NUMERAL; num is the Roman value minus 1. '
                     'A purple locomotive uses a freight_end_DIRECTION board code.')
    if 'decoy_cars' in active:
        parts.append('Yellow open trolleys are type DECOY; labels D1, D2, ... encode num 0, 1, ... .')
    if 'tunnels' in active:
        parts.append('A stone tunnel portal uses a tunnel_DIRECTION board code: DIRECTION is the side of its open mouth. '
                     'Its modifier is 2, and its printed ID is mod_nums verbatim.')
    mods={value for row in level['mods'] for value in row}
    if mods & {1,3,4,5}:
        parts.append('A round colored press plate is modifier 1 (button); a striped barrier across its rail is modifier 3 '
                     '(closed_gate), while a barrier lifted beside the rail is modifier 4 (open_gate). '
                     'A physical points lever joined to the rail tongue by a metal rod, with a numbered colored handle, '
                     'is modifier 5 (button_switched_junction). Its current bend and straight rail are solid and have identical colors. '
                     'The inactive bend is drawn as dashed rails; use only the solid curved rail to encode the selected pair. '
                     'For these symbols, copy the printed group ID into mod_nums without subtracting 1.')
    if 7 in mods:
        parts.append('A physical points lever joined to the rail tongue, with curved arrows on an unnumbered green handle, '
                     'is modifier 7 (self_switching_junction). Its current bend and straight rail are solid and have identical colors. '
                     'The inactive bend is drawn as dashed rails; use only the solid curved rail to encode the selected pair.')
    if 6 in mods:
        parts.append('A cream platform beside the rail, with boarding steps, a waving person carrying a suitcase '
                     'and an Arabic number sign, is modifier 6 (passenger_station). '
                     'It belongs to the rail cell containing the platform; mod_nums is its label minus 1.')
    if 11 in mods:
        parts.append('A cream platform with a parcel and Roman label is modifier 11 (post_office). '
                     'It belongs to the rail cell containing the platform; mod_nums is its Roman value minus 1.')
    if 'semaphores' in active:
        parts.append('An unnumbered two-lamp railway signal with its red lamp lit and a solid red/white arm '
                     'lowered across the rail is modifier 8 (closed_semaphore). '
                     'An opened signal in feedback has its green lamp lit and a green/white arm raised clear of the rail. '
                     'These signals have no striped gate barrier or group number.')
    if {value for row in level['board'] for value in row} & {34,35,36,37}:
        parts.append('A roofed station building occupies a station_building_DIRECTION board tile; '
                     'its door faces DIRECTION. It is separate from a platform modifier on a rail cell.')
    parts.append('Unmarked cells have modifier 0. Unnumbered modifiers and cells without a printed ID use mod_nums 0. '
                 'A served platform in feedback is empty, with a green check in place of its person or parcel; consumed modifiers use code 9. '
                 'Red outlines and exclamation badges mark feedback error cells. '
                 'In a failed frame, translucent vehicles show previous positions; solid vehicles depict the failure instant. '
                 'Red motion trails show the last attempted movement, and white/orange bursts show contact points. '
                 'These overlays are not additional vehicles, rails or modifiers.')
    return '\n'.join(parts)




def _text_board(level):
    """One rail grid, a mechanism list and the initial cars."""
    parts = ['board grid (each row lists columns from left to right):\n' + '\n'.join(
        'row %d: %s' % (y, ' '.join(map(str, row))) for y, row in enumerate(level['board']))]
    modifiers = [{'pos': [y, x], 'modifier': mod, 'id': level['mod_nums'][y][x]}
                 for y, row in enumerate(level['mods']) for x, mod in enumerate(row) if mod not in (0, 10)]
    starts = [[y, x] for y, row in enumerate(level['mods']) for x, mod in enumerate(row) if mod == 10]
    parts.append('Mechanisms, listed by position: ' + json.dumps(modifiers, separators=(',', ':')) +
                 '\ncar_start markers (modifier 10): ' + json.dumps(starts, separators=(',', ':')) +
                 '. All remaining unlisted cells have modifier 0. Initial cars retain the listed modifier at their cell. '
                 'Unnumbered modifiers use ID 0.')
    parts.append('Initial cars:\n' + json.dumps(level['cars'], separators=(',', ':')))
    return '\n\n'.join(parts)


def _movement_rules(level):
    """Geometry and movement lookups shared by both rule-explaining inputs."""
    from validator.simulation import ROUTES, DESTINATIONS, SWAPPED
    parts = []
    codes = {1, 2, 5, 6, 7, 8} | {tile for row in level['board'] for tile in row if tile in ROUTES}
    if level['tracks']:
        codes.update(range(9, 17))
    parts.append('Explicit rail movement lookup. Incoming is the car movement direction BEFORE entering the tile, '
                 'not its entry side: moving RIGHT enters from the LEFT. Outgoing is the heading selected on entry '
                 'and used for the NEXT move. Missing incoming directions are incompatible. top=UP; bottom=DOWN.\n' +
                 '\n'.join('%d: %s' % (code, ', '.join('%s -> %s' % pair for pair in ROUTES[code].items()))
                           for code in sorted(codes)))
    parts.append('Locomotive entry requirements: ' + json.dumps({str(tile): {
        'car_type': DESTINATIONS[tile][0], 'incoming_direction': DESTINATIONS[tile][1]}
        for row in level['board'] for tile in row if tile in DESTINATIONS}, separators=(',', ':')) +
        '. A RIGHT-facing locomotive is entered by moving RIGHT from its LEFT neighbor.')
    if any(mod in (5, 7) for row in level['mods'] for mod in row):
        parts.append('A switching event replaces a junction code using ' + json.dumps(SWAPPED, separators=(',', ':')) + '.')
    return '\n\n'.join(parts)


def _starting_rules(level):
    """Explain only the initial-placement exceptions present in this board."""
    starting = [(car, level['mods'][car['pos'][0]][car['pos'][1]],
                 level['mod_nums'][car['pos'][0]][car['pos'][1]]) for car in level['cars']]
    parts = []
    controls = {mod for car, mod, number in starting} & {1, 7}
    if controls:
        names = ['button' if mod == 1 else 'automatic junction' for mod in sorted(controls)]
        parts.append('Initial placement: a car starting on a ' + ' or '.join(names) +
                     ' activates it once before the first move. Its supplied outgoing heading is retained. '
                     'Waiting there does not activate it again; a later entry can.')
    if any((mod == 6 and car['type'] == 'NORMAL' or mod == 11 and car['type'] == 'NUMERAL')
           and number == car['num'] for car, mod, number in starting):
        parts.append('Service requires a later movement entry into a matching stop. '
                     'Initially standing or waiting on its assigned stop does not serve it: '
                     'the car must leave and return before delivery.')
    if (level['semaphores'] or any(8 in row for row in level['mods'])) and any(
            level['board'][car['pos'][0]][car['pos'][1]] in range(9, 17) for car in level['cars']):
        parts.append('Initial occupancy of a qualifying connected junction also counts as a passage '
                     'for semaphore release before the first move.')
    return '\n'.join(parts)


def _answer_requirements(level):
    parts = ['Output format: exactly one JSON object with required key add_tracks, an array of '
             '[row,column,track_code] entries for NEW rail cells only.']
    example = {'add_tracks': [[0, 0, 1]]}
    if level['semaphores']:
        parts.append('Optional key add_semaphores is an array of [row,column] positions for new signals. '
                     'Omit it or use [] if no signals are added.')
        example['add_semaphores'] = []
    parts.append('Example syntax only: ' + json.dumps(example, separators=(',', ':')) + '.')
    parts.append('Coordinates are zero-based [row,column] within the original board. Use JSON integers, never booleans. '
                 'Do not list original rails, cars, dimensions, budgets, metadata or mechanism IDs. '
                 'Everything not listed is inherited unchanged from the ORIGINAL level. '
                 'Each round return your complete proposed additions relative to that original level, '
                 'not an incremental edit to your previous answer. No duplicate positions, extra keys, '
                 'comments, Markdown fences or commentary.')
    return ' '.join(parts)


def initial_prompt(level, mode='image_rules'):
    """Keep goals, decoding and output identical; only hide mechanism laws in image_no_rules."""
    if mode not in PROMPT_MODES:
        raise ValueError('Unknown prompt mode: ' + str(mode))
    active = mechanisms(level)
    height, width = len(level['board']), len(level['board'][0])
    tiles = {0, 1, 2, 4, 5, 6, 7, 8} | {tile for row in level['board'] for tile in row}
    if level['tracks']:
        tiles.update(range(9, 17))
    mods = {0} | {value for row in level['mods'] for value in row}
    if level['semaphores']:
        mods.add(8)
    parts = [SYSTEM_PROMPT,
        'Solve level ' + json.dumps(level['level_id']) + '. The board has %d rows and %d columns. ' % (height, width) +
        'Coordinates are zero-based [row,column], row 0 at the top and column 0 at the left. '
        'Valid rows are 0 through %d and columns 0 through %d. Never add cells outside this board.' % (height - 1, width - 1),
        'Preserve every original nonzero board tile, every original nonzero modifier and numbered mechanism ID. '
        'Initial cars, directions, types, numbers, dimensions and budgets are fixed. Add rails only to empty cells' +
        (', and add signals only where permitted.' if level['semaphores'] else '.'),
        'Maximum number of new track cells: %d. Each added straight, curved or three-way rail cell costs one track. ' % level['tracks'] +
        'This is an upper bound, not an exact requirement; unused tracks are allowed. Original rails do not consume this budget.',
        'Tile code names: ' + json.dumps({str(k): TRACK_NAMES[k] for k in sorted(tiles)}, separators=(',', ':')),
        'Modifier codes: ' + json.dumps({str(k): MOD_NAMES[k] for k in sorted(mods)}, separators=(',', ':')),
        _answer_requirements(level)]
    if 'semaphores' in active:
        parts.append('Maximum number of new semaphores: %d. Unused semaphores are allowed; preserve existing signals.' % level['semaphores'])
    if level['semaphores']:
        parts.append('A new signal changes modifier 0 to 8, only on an unmodified simple straight/curved rail '
                     'connected to an adjacent three-way rail. All other modifiers and numbered mechanism IDs are fixed.')
    else:
        parts.append('All modifiers and numbered mechanism IDs remain unchanged.')
    if mode.startswith('image_'):
        parts.append(_visual_encoding(level, active))
        parts.append('Infer all original cells and vehicle positions, identities and directions from the image. '
                     'No serialized original board is supplied.')
    else:
        parts.append(_text_board(level))
    parts.append(BASE_RULES)
    if 'decoy_cars' in active:
        parts.append('Goal: DECOY trolleys do not need delivery; all NORMAL and NUMERAL cars do.')
    if 'passenger_stations' in active or 'post_offices' in active:
        parts.append('Goal: before delivery, every required car must visit every stop assigned to its type and zero-based number. '
                     'Passenger stations belong to NORMAL cars; post offices belong to NUMERAL cars.')
    if mode != 'image_no_rules':
        parts.append(_movement_rules(level))
        parts.extend(RULES[name] for name in active)
        starting_rules = _starting_rules(level)
        if starting_rules:
            parts.append(starting_rules)
    else:
        parts.append('Mechanism transition rules and waiting durations are not supplied. '
                     'Use the shared goal, visual symbols, output format and observed execution feedback.')
    return '\n\n'.join(parts)


def visible_feedback_state(state):
    """Project a validator snapshot without exposing its internal timers or event queues.

    The validator's original trace is unchanged. Visible poses, rail/modifier
    states, delivery identities, and drawn collision/history geometry remain.
    """
    from copy import deepcopy
    projected = {key: deepcopy(state[key]) for key in ('tick', 'board', 'delivered', 'error_positions', 'previous_tick') if key in state}
    projected['mechanisms'] = [
        {'pos': [y, x], 'modifier': mod, 'id': state['mod_nums'][y][x]}
        for y, row in enumerate(state.get('mods', [])) for x, mod in enumerate(row) if mod not in (0, 10)]
    fields = ('id', 'type', 'num', 'pos', 'direction')
    def pose(car):
        return {key: deepcopy(car[key]) for key in fields if key in car}
    hidden = {'delivered', 'finished', 'removed'}
    projected['cars'] = [pose(car) for car in state.get('cars', []) if car.get('status') not in hidden]
    if state.get('motion'):
        motion_fields = ('id', 'type', 'num', 'from', 'to', 'attempted', 'direction', 'outgoing_direction', 'entrance')
        projected['motion'] = [{key: deepcopy(item[key]) for key in motion_fields if key in item}
                               for item in state['motion']]
        drawn_ids = {item['id'] for item in state['motion']}
        projected['previous_cars'] = [pose(car) for car in state.get('previous_cars', [])
                                      if car.get('id') in drawn_ids and car.get('status') not in hidden]
    if state.get('conflicts'):
        event_fields = ('kind', 'geometry', 'car_ids', 'target', 'impact', 'positions', 'stations', 'leader_id')
        projected['conflicts'] = [{key: deepcopy(event[key]) for key in event_fields if key in event}
                                 for event in state['conflicts']]
    return projected


def feedback_prompt(result, mode='visual', *, prompt_mode='image_rules'):
    if mode not in ('binary', 'textual', 'visual'):
        raise ValueError('Unknown feedback mode')
    if prompt_mode not in PROMPT_MODES:
        raise ValueError('Unknown prompt mode')
    if result['success']:
        return 'Success.'
    if result.get('error_type') in ('track_budget', 'semaphore_budget'):
        budget = result.get('budget_violation')
        if budget:
            resource = 'track cells' if budget['resource'] == 'tracks' else 'semaphores'
            details = 'Maximum new %s: %d; your solution uses %d.' % (resource, budget['maximum'], budget['used'])
        else:
            details = result.get('message', 'The solution exceeds its resource budget.')
        return 'Failure: ' + result['error_type'] + '. ' + details + ' Submit another complete additions JSON.'
    diagnostic = 'Failure: ' + str(result.get('error_type')) + '.'
    if result.get('error_positions'):
        diagnostic += ' Affected [row,column] cells: ' + json.dumps(result['error_positions']) + '.'
    if not result.get('trace'):
        return diagnostic + ' ' + result.get('message', '') + '\nSubmit another complete additions JSON.'
    if mode == 'binary':
        return 'The solution failed. Please submit another complete answer.'
    states = result.get('feedback_states') or []
    if mode == 'visual' and states:
        text = diagnostic + ' The attached feedback images are in chronological order at ticks ' + str([s.get('tick') for s in states]) + '. '
        if prompt_mode != 'image_no_rules':
            text += ('Faded vehicles show the preceding pose, red trails show the attempted movement, '
                     'and impact bursts mark contact. An unserved stop is highlighted at its platform. ')
        return text + 'For a loop, the last image shows where the loop begins. Submit another complete additions JSON.'
    text = diagnostic
    if mode == 'textual' and states:
        text += '\nSelected feedback observations (chronological):\n' + json.dumps(
            [visible_feedback_state(state) for state in states], separators=(',', ':'), ensure_ascii=True)
    return text + '\nSubmit another complete additions JSON.'
