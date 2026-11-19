"""Strict input validation followed by deterministic puzzle simulation."""
import copy
import json
import math
from pathlib import Path

from .tiles import TRACK_NAMES, MOD_NAMES, DIRECTIONS, PLACEABLE_TRACKS, SIMPLE_TRACKS, PORTS, TUNNELS, ENDINGS, NUMBERED_MODIFIERS

REQUIRED = {'schema_version','level_id','board','mods','mod_nums','cars','tracks','semaphores'}
ALLOWED = REQUIRED | {'metadata'}


class ValidationError(ValueError):
    def __init__(self, message, kind='invalid_schema', positions=None, *, details=None):
        super().__init__(message)
        self.kind=kind
        self.positions=positions or []
        self.details=details


def _pairs(pairs):
    result={}
    for k,v in pairs:
        if k in result: raise ValidationError('Duplicate JSON key: '+k,'invalid_json')
        result[k]=v
    return result


def parse_json(value):
    """Accept a dict, JSON text/bytes, or explicit Path; never execute content."""
    if isinstance(value,Path):
        try:value=value.read_text(encoding='utf-8-sig')
        except (OSError,UnicodeError) as exc:
            raise ValidationError('Cannot read JSON input: '+str(exc),'invalid_json') from None
    if isinstance(value,(str,bytes)):
        try:
            def bad_constant(x): raise ValidationError('Non-finite JSON number: '+x,'invalid_json')
            def finite_float(x):
                number=float(x)
                if not math.isfinite(number):raise ValidationError('Non-finite JSON number','invalid_json')
                return number
            value=json.loads(value,object_pairs_hook=_pairs,parse_constant=bad_constant,parse_float=finite_float)
        except (json.JSONDecodeError,UnicodeError,RecursionError) as exc:
            raise ValidationError('Cannot parse JSON: '+str(exc),'invalid_json') from None
    if type(value) is not dict: raise ValidationError('The answer must be one JSON object.')
    return copy.deepcopy(value)


def _int(value,where,minimum=0):
    if type(value) is not int or value<minimum:
        raise ValidationError(where+' must be an integer >= '+str(minimum))


def check_schema(value):
    missing=REQUIRED-value.keys()
    if missing: raise ValidationError('Missing fields: '+', '.join(sorted(missing)))
    unknown=value.keys()-ALLOWED
    if unknown: raise ValidationError('Unknown fields: '+', '.join(sorted(unknown)))
    if type(value['schema_version']) is not int or value['schema_version']!=1:
        raise ValidationError('schema_version must be 1')
    if not isinstance(value['level_id'],str) or not value['level_id']:
        raise ValidationError('level_id must be a nonempty string')
    board=value['board']
    if not isinstance(board,list) or not board or not isinstance(board[0],list) or not board[0]:
        raise ValidationError('board must be a nonempty rectangular matrix')
    h,w=len(board),len(board[0])
    if h*w>65536: raise ValidationError('Board exceeds 65536-cell resource limit')
    for field in ('board','mods','mod_nums'):
        matrix=value[field]
        if not isinstance(matrix,list) or len(matrix)!=h:
            raise ValidationError(field+' dimensions must match board')
        for y,row in enumerate(matrix):
            if not isinstance(row,list) or len(row)!=w:
                raise ValidationError(field+' dimensions must match board')
            for x,v in enumerate(row):
                _int(v,field+' cell')
                allowed=TRACK_NAMES if field=='board' else MOD_NAMES if field=='mods' else None
                if allowed is not None and v not in allowed:
                    raise ValidationError('Unknown '+field+' code '+str(v),positions=[[y,x]])
    for field in ('tracks','semaphores'): _int(value[field],field)
    if 'metadata' in value and type(value['metadata']) is not dict:
        raise ValidationError('metadata must be an object')
    cars=value['cars']
    if not isinstance(cars,list) or not cars:
        raise ValidationError('cars must be a nonempty array')
    ids=set(); positions=set(); numbers={kind:[] for kind in ('NORMAL','NUMERAL','DECOY')}
    for car in cars:
        if type(car) is not dict or set(car)!={'pos','direction','num','type'}:
            raise ValidationError('Each car requires exactly pos, direction, num, type')
        if car['type'] not in numbers or car['direction'] not in DIRECTIONS:
            raise ValidationError('Invalid car type or cardinal direction')
        pos=car['pos']
        if not isinstance(pos,list) or len(pos)!=2: raise ValidationError('Car pos must be [row,column]')
        for v in pos: _int(v,'Car coordinate')
        y,x=pos
        if y>=h or x>=w: raise ValidationError('Car outside board')
        _int(car['num'],'Car num')
        ident=(car['type'],car['num'])
        if ident in ids or (y,x) in positions: raise ValidationError('Duplicate car ID or starting position')
        ids.add(ident);positions.add((y,x));numbers[car['type']].append(car['num'])
        if board[y][x] not in PLACEABLE_TRACKS: raise ValidationError('Car must start on a rail')
        if car['direction'] not in PORTS[board[y][x]]:
            raise ValidationError('Car direction must match its starting rail',positions=[[y,x]])
    for kind,nums in numbers.items():
        if sorted(nums)!=list(range(len(nums))): raise ValidationError(kind+' car numbers must be contiguous from zero')
    if not numbers['NORMAL'] and not numbers['NUMERAL']: raise ValidationError('At least one delivery car is required')
    return value


def check_level(level):
    level=check_schema(parse_json(level))
    tunnels={}; station_numbers={'NORMAL':set(),'NUMERAL':set()}
    locomotives={'NORMAL':[],'NUMERAL':[]}
    for row in range(len(level['board'])):
        for col,track in enumerate(level['board'][row]):
            mod=level['mods'][row][col];num=level['mod_nums'][row][col]
            pos=[row,col]
            if track in TUNNELS:
                if mod!=2: raise ValidationError('Tunnel requires paired modifier',positions=[pos])
                tunnels.setdefault(num,[]).append(pos)
            elif mod==2: raise ValidationError('Tunnel modifier requires tunnel tile',positions=[pos])
            if mod in (5,7) and track not in range(9,17):
                raise ValidationError('Switching modifier requires junction',positions=[pos])
            if mod in (1,3,4,6,8,10,11) and track not in PLACEABLE_TRACKS:
                raise ValidationError('Modifier requires rail',positions=[pos])
            if mod==8 and track not in SIMPLE_TRACKS:
                raise ValidationError('Semaphore requires a simple rail',positions=[pos])
            if mod in (3,4) and track not in SIMPLE_TRACKS:
                raise ValidationError('Gate requires a simple rail',positions=[pos])
            if mod in (6,11) and track not in (1,2):
                raise ValidationError('Service platform requires a straight rail',positions=[pos])
            if mod==6: station_numbers['NORMAL'].add(num)
            if mod==11: station_numbers['NUMERAL'].add(num)
            if track in (3,29,30,31): locomotives['NORMAL'].append(pos)
            elif track in (21,22,32,33): locomotives['NUMERAL'].append(pos)
    if any(len(v)!=2 for v in tunnels.values()): raise ValidationError('Each tunnel ID requires exactly two mouths')
    for kind,nums in station_numbers.items():
        available={c['num'] for c in level['cars'] if c['type']==kind}
        if not nums<=available: raise ValidationError('Station has no matching car')
    for positions in locomotives.values():
        if len(positions)>1:
            raise ValidationError('At most one locomotive per car type is allowed',positions=positions)
    for car in level['cars']:
        y,x=car['pos']
        if level['board'][y][x] not in (1,2):
            raise ValidationError('Car must start on a straight rail',positions=[[y,x]])
        if level['mods'][y][x]!=10:
            raise ValidationError('Car start requires modifier 10 and no mechanism',positions=[[y,x]])
        if car['type']!='DECOY' and not locomotives[car['type']]:
            raise ValidationError('Required car has no matching locomotive')
    return level


def _signal_connected(board,y,x):
    for direction in PORTS.get(board[y][x],()):
        dy,dx=DIRECTIONS[direction];ny,nx=y+dy,x+dx
        if 0<=ny<len(board) and 0<=nx<len(board[0]):
            junction=board[ny][nx]
            reverse={'LEFT':'RIGHT','RIGHT':'LEFT','UP':'DOWN','DOWN':'UP'}[direction]
            if junction in range(9,17) and reverse in PORTS[junction]:return True
    return False


def check_solution(level,answer):
    original=check_level(level)
    candidate=check_schema(parse_json(answer))
    for key in ('schema_version','level_id','tracks','semaphores'):
        if candidate[key]!=original[key]: raise ValidationError('Changed immutable '+key,'illegal_edit')
    # Array order is not a visible board property. Preserve each complete car
    # object while allowing image-derived answers to serialize them differently.
    car_key=lambda car:(car['type'],car['num'])
    if sorted(candidate['cars'],key=car_key)!=sorted(original['cars'],key=car_key):
        raise ValidationError('Changed immutable cars','illegal_edit')
    # Descriptive metadata is not part of the board or the puzzle constraints.
    candidate.pop('metadata',None)
    if (len(candidate['board']),len(candidate['board'][0]))!=(len(original['board']),len(original['board'][0])):
        raise ValidationError('Changed immutable board dimensions','illegal_edit')
    added_tracks=added_signals=0
    for y,row in enumerate(original['board']):
        for x,old_track in enumerate(row):
            pos=[[y,x]];new_track=candidate['board'][y][x]
            old_mod=original['mods'][y][x];new_mod=candidate['mods'][y][x]
            if old_track and old_track!=new_track:
                raise ValidationError('Changed immutable original tile','illegal_edit',pos)
            if not old_track and new_track:
                if new_track not in PLACEABLE_TRACKS: raise ValidationError('Only rails may be added','illegal_edit',pos)
                added_tracks+=1
            if old_mod in NUMBERED_MODIFIERS and candidate['mod_nums'][y][x]!=original['mod_nums'][y][x]:
                raise ValidationError('Changed immutable modifier ID','illegal_edit',pos)
            if old_mod not in NUMBERED_MODIFIERS:
                # Car numbers live in cars[].num. Empty cells, car starts,
                # automatic junctions and signals have no mechanism ID.
                candidate['mod_nums'][y][x]=0
            if old_mod!=new_mod:
                if old_mod!=0 or new_mod!=8 or new_track not in SIMPLE_TRACKS:
                    raise ValidationError('Changed immutable modifier or invalid semaphore','illegal_edit',pos)
                # A signal must adjoin a junction along one of its rail ports.
                if not _signal_connected(candidate['board'],y,x):
                    raise ValidationError('Semaphore must adjoin a connected junction','illegal_edit',pos)
                added_signals+=1
    if added_tracks>original['tracks']:
        raise ValidationError('Track budget exceeded: %d > %d'%(added_tracks,original['tracks']),'track_budget',
                              details={'resource':'tracks','maximum':original['tracks'],'used':added_tracks})
    if added_signals>original['semaphores']:
        raise ValidationError('Semaphore budget exceeded: %d > %d'%(added_signals,original['semaphores']),'semaphore_budget',
                              details={'resource':'semaphores','maximum':original['semaphores'],'used':added_signals})
    return candidate


def validate(level,answer,history_steps=None,history_gap=1,max_ticks=10000):
    """Return success/error plus full trace and a sampled feedback window."""
    if history_steps is not None and (type(history_steps) is not int or history_steps<1):
        raise ValueError('history_steps must be None or a positive integer')
    if type(history_gap) is not int or history_gap<1: raise ValueError('history_gap must be positive')
    if type(max_ticks) is not int or max_ticks<1: raise ValueError('max_ticks must be positive')
    from .answers import load_solution
    try:
        original=check_level(level)
    except (ValueError,TypeError,KeyError,RecursionError) as exc:
        return dict(success=False,error_type='invalid_level',message=str(exc),trace=[],feedback_states=[],error_positions=getattr(exc,'positions',[]))
    try:
        expanded=load_solution(original,answer)
        solution=check_solution(original,expanded)
    except (ValueError,TypeError,KeyError,RecursionError) as exc:
        result=dict(success=False,error_type=getattr(exc,'kind','invalid_schema'),message=str(exc),trace=[],feedback_states=[],error_positions=getattr(exc,'positions',[]))
        if getattr(exc,'details',None) is not None:result['budget_violation']=exc.details
        return result
    from .simulation import simulate
    from .history import select_feedback
    result=simulate(original,solution,max_ticks=max_ticks)
    anchor=result.get('loop_start') if result.get('error_type')=='loop' else None
    result['feedback_states']=select_feedback(result['trace'],history_steps=history_steps,history_gap=history_gap,anchor=anchor)
    result.setdefault('error_positions',[])
    return result
