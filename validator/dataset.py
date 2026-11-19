"""Check a dataset's level/solution pairs and image coverage locally."""
import json
from collections import Counter
from pathlib import Path

from .api import check_level, validate
from .tiles import ENDINGS


METADATA_FIELDS = {'mechanisms', 'locomotive_count', 'total_car_count', 'difficulty',
                   'available_tracks', 'available_semaphores', 'decoy_car_count'}


def validate_dataset(level_root, solution_root, max_ticks=10000):
    level_root, solution_root = Path(level_root), Path(solution_root)
    paths = sorted(level_root.glob('*/*.json'))
    failures, difficulties = [], Counter()
    passed = level_images = solution_images = 0
    expected = {path.relative_to(level_root) for path in paths}
    actual = {path.relative_to(solution_root) for path in solution_root.glob('*/*.json')}
    inventories = []
    for folder in (level_root, solution_root):
        manifest = folder / 'manifest.json'
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding='utf-8'))
                if not isinstance(data, dict):
                    raise ValueError('Inventory must be an object')
                ids = data['level_ids']
                if not isinstance(ids, list) or any(not isinstance(value, str) for value in ids):
                    raise ValueError('Inventory level_ids must be an array of strings')
                inventories.append(set(ids))
                if (len(ids) != len(set(ids)) or data['puzzle_count'] != len(ids)
                        or data['world_counts'] != dict(Counter(value.split('-')[0] for value in ids))):
                    raise ValueError('Inconsistent inventory')
            except (ValueError, OSError, KeyError, TypeError):
                failures.append({'level_id': None, 'error_type': 'invalid_inventory'})
    ids_on_disk = {path.stem for path in paths}
    if any(inventory != ids_on_disk for inventory in inventories):
        failures.append({'level_id': None, 'error_type': 'inventory_mismatch'})
    for path in paths:
        relative = path.relative_to(level_root)
        solution_path = solution_root / relative
        try:
            level = check_level(path)
            if level['level_id'] != path.stem:
                raise ValueError('Filename does not match level_id')
            metadata = level.get('metadata', {})
            if set(metadata) != METADATA_FIELDS:
                raise ValueError('Unexpected or missing level metadata')
            if metadata['difficulty'] not in ('easy','medium','hard'):
                raise ValueError('Invalid difficulty label')
            counts = {
                'locomotive_count': sum(tile in ENDINGS for row in level['board'] for tile in row),
                'total_car_count': len(level['cars']), 'available_tracks': level['tracks'],
                'available_semaphores': level['semaphores'],
                'decoy_car_count': sum(car['type'] == 'DECOY' for car in level['cars'])}
            if any(metadata[key] != value for key, value in counts.items()):
                raise ValueError('Metadata counts do not match the puzzle')
            difficulties[metadata['difficulty']] += 1
            result = validate(level, solution_path, max_ticks=max_ticks)
            if result['success']:
                passed += 1
                saved = json.loads(solution_path.read_text(encoding='utf-8'))
                saved_metadata = saved.get('metadata')
                steps = saved_metadata.get('simulation_steps') if isinstance(saved_metadata, dict) else None
                if type(steps) is not int or steps != result['trace'][-1]['tick']:
                    failures.append({'level_id': path.stem, 'error_type': 'invalid_simulation_steps'})
            else:
                failures.append({'level_id': path.stem, 'error_type': result['error_type']})
        except (ValueError, OSError, KeyError, TypeError) as exc:
            failures.append({'level_id': path.stem, 'error_type': type(exc).__name__})
        level_images += (path.parent / 'images' / (path.stem + '.png')).is_file()
        solution_images += (solution_path.parent / 'images' / (path.stem + '.png')).is_file()
    complete = (bool(paths) and not failures and expected == actual
                and passed == level_images == solution_images == len(paths))
    return {'success': complete, 'level_count': len(paths), 'reference_passed': passed,
            'level_images': level_images, 'solution_images': solution_images,
            'difficulty_counts': dict(difficulties),
            'missing_solutions': sorted(str(p) for p in expected - actual),
            'extra_solutions': sorted(str(p) for p in actual - expected), 'failures': failures}
