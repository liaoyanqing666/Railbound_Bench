"""Optional native accelerator, built from the adjacent dependency-free source.

The executable is cached outside the checkout. The public Python validator is
still the authority for every returned witness. No downloaded solver is used.
"""
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


_KINDS = {'NORMAL': 0, 'NUMERAL': 1, 'DECOY': 2}
_DIRECTIONS = {'LEFT': 0, 'RIGHT': 1, 'DOWN': 2, 'UP': 3}
_executable = None
_unavailable = False


def executable():
    """Return the native binary, or None when no supported compiler exists."""
    global _executable, _unavailable
    if _executable is not None:
        return _executable
    if _unavailable:
        return None
    compiler = shutil.which('g++') or shutil.which('clang++')
    if compiler is None:
        _unavailable = True
        return None
    source = Path(__file__).with_name('native.cpp')
    stat = source.stat()
    cache = Path(tempfile.gettempdir()) / 'railbound-bench-native'
    cache.mkdir(exist_ok=True)
    # Source timestamp/size provide cache invalidation without content digests.
    target = cache / ('solver-%s-%s%s' % (stat.st_mtime_ns, stat.st_size,
                                         '.exe' if os.name == 'nt' else ''))
    if not target.exists():
        temporary = target.with_name(target.stem + '-%s' % os.getpid() + target.suffix)
        command = [compiler, '-std=c++14', '-O3', '-DNDEBUG', str(source), '-o', str(temporary)]
        if os.name == 'nt':
            command.extend(['-static', '-static-libgcc', '-static-libstdc++'])
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            _unavailable = True
            return None
        try:
            os.replace(str(temporary), str(target))
        except OSError:
            if not target.exists():
                raise
            temporary.unlink(missing_ok=True)
    _executable = str(target)
    return _executable


def _request(search):
    binary = executable()
    if binary is None:
        return None
    level = search.base
    height, width = len(level['board']), len(level['board'][0])
    if search.max_ticks > 2147483647:
        return None
    values = [height, width, len(level['cars'])]
    for key in ('board', 'mods'):
        values.extend(value for row in level[key] for value in row)
    # Mechanism identifiers are arbitrary Python integers in the public API.
    # Relabel only large IDs injectively; car/station numbers keep their values.
    car_count = len(level['cars'])
    identifiers = sorted({value for row in level['mod_nums'] for value in row
                          if value >= car_count})
    labels = {value: index + car_count for index, value in enumerate(identifiers)}
    values.extend(labels.get(value, value) for row in level['mod_nums'] for value in row)
    for car in level['cars']:
        values.extend((car['pos'][0] * width + car['pos'][1],
                       _DIRECTIONS[car['direction']], car['num'], _KINDS[car['type']]))
    cells = height * width
    values.extend((min(search.budget, cells), min(search.signal_budget, cells), search.max_ticks,
                   int(search.optimize), min(search.node_limit or 0, 9223372036854775807),
                   max(0.000001, search.deadline - time.perf_counter())
                   if search.deadline is not None else 0,
                   search.lower_bound, min(search.best_cost, cells + 1)))
    values.append(len(search.mandatory))
    values.extend(y * width + x for y, x in sorted(search.mandatory))
    values.append(int(search.best is not None))
    if search.best is not None:
        for key in ('board', 'mods'):
            values.extend(value for row in search.best[key] for value in row)
    return binary, values


def run(search):
    """Run the native tree and transfer its checked witness and proof flags."""
    request = _request(search)
    if request is None:
        return None
    binary, values = request
    result = subprocess.run([binary], input=' '.join(map(str, values)),
                            capture_output=True, text=True, check=True)
    tokens = iter(result.stdout.split())
    complete, reason, found = (int(next(tokens)) for _ in range(3))
    native_cpu = float(next(tokens))
    keys = ('search_nodes', 'simulations', 'simulation_ticks', 'bound_checks',
            'pruned_by_bound', 'pruned_by_failure', 'rail_decisions',
            'rail_refinements', 'signal_decisions', 'maximum_depth',
            'unresolved_simulations', 'solutions_found')
    preceding_ticks = search.stats['simulation_ticks']
    for key in keys:
        search.stats[key] += int(next(tokens))
    search.stats['heuristic_search_nodes'] = int(next(tokens))
    search.stats['heuristic_passes'] = int(next(tokens))
    for key in ('checkpoint_resumes', 'reused_prefix_ticks',
                'logical_simulation_ticks', 'full_replays'):
        search.stats[key] = int(next(tokens))
    search.stats['logical_simulation_ticks'] += preceding_ticks
    search.stats['bound_cache_hits'] += int(next(tokens))
    search.stats['native_cpu_seconds'] = native_cpu
    search.stats['native_backend'] = True
    search.stop_reason = ('search_exhausted', 'solution_found', 'lower_bound_matched',
                          'time_limit', 'node_limit', 'simulation_limit')[reason]
    search.unresolved = bool(search.stats['unresolved_simulations'])
    if found:
        candidate = deepcopy(search.base)
        for key in ('board', 'mods'):
            for row in candidate[key]:
                for x in range(len(row)):
                    row[x] = int(next(tokens))
        # Check schema and immutable content, then replay with the independent
        # public transition engine. A mismatch is fatal.
        from validator.api import check_solution
        from validator.simulation import simulate
        candidate = check_solution(search.level, candidate)
        search.stats['simulations'] += 1
        search.stats['witness_validations'] += 1
        verified = simulate(search.level, candidate, search.max_ticks, record_trace=False)
        search.stats['simulation_ticks'] += verified.get('tick', 0)
        search.stats['logical_simulation_ticks'] += verified.get('tick', 0)
        if not verified['success']:
            raise RuntimeError('Native solver witness failed public validation: ' + str(verified['error_type']))
        # The native search already counted each discovered solution.
        discovered = search.stats['solutions_found']
        search.accept(candidate, verified)
        search.stats['solutions_found'] = discovered
    return bool(complete)
