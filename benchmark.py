"""Tool-free evaluation: compact disk records, full conversation only in memory."""
import argparse
import copy
import json
import math
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from dotenv import load_dotenv
from models import ModelConfig, ProviderError, create_provider, image_part
from prompts import PROMPT_MODES, initial_prompt, feedback_prompt
from validator.answers import normalize_answer

ROOT = Path(__file__).resolve().parent
RETRYABLE = {'timeout', 'transport_error', 'incomplete_response', 'provider_error',
             'output_truncated', 'output_filtered', 'interrupted', 'resource_limit',
             'protocol_error', 'rate_limit'}
_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


def _replace_file(source, target):
    """Allow a short-lived Windows reader to release an atomically replaced file."""
    for attempt in range(9):
        try:
            os.replace(str(source), str(target))
            return
        except PermissionError:
            if os.name != 'nt' or attempt == 8:
                raise
            time.sleep(min(.025 * 2 ** attempt, .2))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        _replace_file(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def safe_name(value):
    name = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value)).strip('._')
    if not name: raise ValueError('Empty output label')
    return name


@dataclass
class RunOptions:
    rounds: int = 5
    feedback: str = None
    history_steps: int = 1
    history_gap: int = 1
    max_ticks: int = 10000
    prompt_mode: str = 'image_rules'
    max_retry: int = 2
    retry_delay: float = 60.0
    retry_max_delay: float = 600.0

    def __post_init__(self):
        for field in ('rounds', 'history_gap', 'max_ticks'):
            if type(getattr(self, field)) is not int or getattr(self, field) < 1:
                raise ValueError(field + ' must be a positive integer')
        if type(self.max_retry) is not int or self.max_retry < 0:
            raise ValueError('max_retry must be a nonnegative integer')
        for field in ('retry_delay', 'retry_max_delay'):
            value = getattr(self, field)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(field + ' must be finite and positive')
        if self.retry_max_delay < self.retry_delay:
            raise ValueError('retry_max_delay must be at least retry_delay')
        if self.history_steps is not None and (type(self.history_steps) is not int or self.history_steps < 1):
            raise ValueError('history_steps must be positive or None')
        if self.prompt_mode not in PROMPT_MODES: raise ValueError('Unknown prompt_mode')
        if self.feedback is None:
            self.feedback = 'textual' if self.prompt_mode == 'text_rules' else 'visual'
        if self.feedback not in ('binary', 'textual', 'visual'): raise ValueError('Unknown feedback mode')


@contextmanager
def _run_lock(path):
    """Serialize writes to the same result within this process; no lock files."""
    key = str(Path(path).resolve()).casefold() if os.name == 'nt' else str(Path(path).resolve())
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(key, threading.RLock())
    with lock:
        yield


def _puzzle_snapshot(level):
    """Copy the complete gameplay input, without descriptive metadata."""
    fields = ('schema_version', 'level_id', 'board', 'mods', 'mod_nums', 'cars', 'tracks', 'semaphores')
    return copy.deepcopy({field: level[field] for field in fields})


def run_contract(config, levels, options):
    return {'model': config.to_dict(), 'options': asdict(options),
            'levels': [_puzzle_snapshot(level) for level in levels]}


def _resume_contract(metadata):
    """Return the saved contract with its options in their complete form."""
    if metadata.get('schema_version') != 2:
        raise ValueError('Expected schema_version 2 experiment records')
    contract = copy.deepcopy(metadata['contract'])
    contract['options'] = asdict(RunOptions(**contract['options']))
    return contract


def _ensure_metadata(provider, level, directory, options):
    path = directory / 'run.json'
    if not path.exists():
        with _run_lock(path):
            if not path.exists():
                limits = provider.resolve_capabilities()
                atomic_json(path, {'schema_version': 2, 'created_at': now(),
                    'contract': run_contract(provider.config, [level], options),
                    'model_capabilities': limits, 'tools_available_to_model': [], 'model_code_execution': False})
    metadata = json.loads(path.read_text(encoding='utf-8'))
    contract = _resume_contract(metadata)
    if (contract['model'] != provider.config.to_dict()
            or contract['options'] != asdict(options)
            or _puzzle_snapshot(level) not in contract['levels']):
        raise ValueError('Resume model, options or puzzle contents differ from run metadata')
    saved = metadata['model_capabilities']
    if getattr(provider, 'capabilities', None) is not None and provider.capabilities != saved:
        raise ValueError('Resume model capabilities differ from saved metadata')
    provider.capabilities = copy.deepcopy(saved)


def retryable(error_type):
    if error_type in RETRYABLE: return True
    if isinstance(error_type, str) and error_type.startswith('runner_error:'): return True
    if isinstance(error_type, str) and re.fullmatch(r'http_\d{3}', error_type):
        status = int(error_type[5:])
        return status in (408, 409, 425, 429) or 500 <= status <= 599
    return False


def _wait_retry(record, save, options, *, stage, failures, error_type):
    """Persist the next attempt time so an interrupted cooldown survives resume."""
    pending = record.get('retry')
    if not pending or (pending['stage'], pending['failures']) != (stage, failures):
        if failures <= options.max_retry:
            delay = min(2 ** min(failures - 1, 5), 30)
        else:
            delay = options.retry_delay
            for _ in range(failures - options.max_retry - 1):
                delay = min(delay * 2, options.retry_max_delay)
                if delay == options.retry_max_delay: break
        pending = {'stage': stage, 'failures': failures, 'error_type': error_type,
                   'delay_seconds': delay,
                   'next_attempt_at': datetime.fromtimestamp(time.time() + delay, timezone.utc).isoformat()}
        record['retry'] = pending
    record['status'] = 'retry_wait'; save()
    deadline = datetime.fromisoformat(pending['next_attempt_at']).timestamp()
    while True:
        remaining = deadline - time.time()
        if remaining <= 0: break
        time.sleep(min(remaining, 30))
    record.pop('retry', None)
    record['status'] = 'running'; save()


def _error_type(exc):
    if exc.status_code is not None: return 'http_' + str(exc.status_code)
    if exc.kind == 'transport_error' and str(exc) in ('ReadTimeout', 'WriteTimeout', 'ConnectTimeout', 'PoolTimeout'):
        return 'timeout'
    return exc.kind


def _reply(completion):
    answer = {'content': completion.text}
    reasoning = (completion.assistant_message or {}).get('reasoning_content')
    if isinstance(reasoning, str): answer['reasoning_content'] = reasoning
    summary = (completion.assistant_message or {}).get('reasoning_summary')
    if isinstance(summary, str) and summary: answer['reasoning_summary'] = summary
    return answer


def _termination_metadata(*, completion=None, error=None, reason=None, elapsed_seconds=None):
    """Keep termination diagnostics and usage, never request bodies or raw streams."""
    raw = completion.raw if completion is not None else getattr(error, 'raw', None)
    raw = raw if isinstance(raw, dict) else {}
    response = raw.get('response') if isinstance(raw.get('response'), dict) else raw
    events = raw.get('events') if isinstance(raw.get('events'), list) else []
    terminal = next((event for event in reversed(events) if isinstance(event, dict)
                     and event.get('type') in ('response.completed', 'response.incomplete', 'response.failed', 'error')), None)
    if terminal and isinstance(terminal.get('response'), dict): response = terminal['response']
    elif not isinstance(raw.get('response'), dict):
        latest = next((event.get('response') for event in reversed(events)
                       if isinstance(event, dict) and isinstance(event.get('response'), dict)), None)
        if latest is not None: response = latest
    finish = completion.finish_reason if completion is not None else None
    choices = response.get('choices')
    if finish is None and isinstance(choices, list) and choices and isinstance(choices[0], dict):
        finish = choices[0].get('finish_reason')
    info = {'reason': reason or (_error_type(error) if error is not None else 'completed'),
            'finish_reason': finish}
    elapsed = completion.elapsed_seconds if completion is not None else elapsed_seconds
    if elapsed is not None: info['elapsed_seconds'] = round(elapsed, 3)
    usage = completion.usage if completion is not None else response.get('usage')
    if isinstance(usage, dict):
        info['usage'] = {key:copy.deepcopy(value) for key,value in usage.items() if key != 'attribution'}
    if isinstance(response.get('id'), str): info['response_id'] = response['id']
    if isinstance(raw.get('stream_done'), bool): info['stream_completed'] = raw['stream_done']
    elif events: info['stream_completed'] = terminal is not None
    if events and isinstance(events[-1], dict) and isinstance(events[-1].get('type'), str):
        info['stream_last_event'] = events[-1]['type']
    if isinstance(response.get('status'), str): info['response_status'] = response['status']
    for key in ('num_turns', 'generation_count'):
        if type(raw.get(key)) is int:
            info[key] = raw[key]
    if isinstance(raw.get('stop_reason'), str):
        info['provider_stop_reason'] = raw['stop_reason']
    if type(raw.get('usage_is_partial')) is bool:
        info['usage_is_partial'] = raw['usage_is_partial']
    for key in ('native_text_continuation', 'max_text_continuations', 'continuation_stopped_reason',
                'final_text_complete', 'no_tools', 'fallback_used'):
        if isinstance(raw.get(key), (str, int, bool)):
            info[key] = raw[key]
    if isinstance(raw.get('generations'), list):
        fields = ('generation_index', 'is_text_continuation', 'request_dispatched', 'requested_api_model',
                  'api_model', 'status', 'stream_done', 'stop_reason', 'finish_reason', 'usage', 'usage_is_partial')
        info['generations'] = [{key: copy.deepcopy(unit[key]) for key in fields if key in unit}
                               for unit in raw['generations'] if isinstance(unit, dict)]
    details = response.get('incomplete_details')
    if isinstance(details, dict) and isinstance(details.get('reason'), str):
        info['incomplete_reason'] = details['reason']
    if error is not None:
        info['error_kind'] = error.kind
        if error.status_code is not None: info['http_status'] = error.status_code
        if isinstance(error.provider_message, str): info['provider_error_message'] = error.provider_message
        provider_error = response.get('error') or (terminal or {}).get('error')
        if not isinstance(provider_error, dict) and terminal and terminal.get('type') == 'error':
            provider_error = terminal
        if isinstance(provider_error, dict) and isinstance(provider_error.get('code'), (str, int)):
            info['provider_error_code'] = provider_error['code']
    return info


def _partial_reply(raw):
    """Extract returned model text from a broken stream, never its raw payload."""
    if not isinstance(raw, dict): return None
    text, reasoning = [], []
    choices = raw.get('choices')
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get('message')
        if isinstance(message, dict):
            if isinstance(message.get('content'), str): text.append(message['content'])
            if isinstance(message.get('reasoning_content'), str): reasoning.append(message['reasoning_content'])
    # Stream snapshots already contain accumulated text. Do not append deltas twice.
    for chunk in raw.get('chunks', []) if not (text or reasoning) and isinstance(raw.get('chunks'), list) else []:
        if not isinstance(chunk, dict): continue
        for choice in chunk.get('choices', []) if isinstance(chunk.get('choices', []), list) else []:
            delta = choice.get('delta', {}) if isinstance(choice, dict) else {}
            if isinstance(delta, dict):
                if isinstance(delta.get('content'), str): text.append(delta['content'])
                if isinstance(delta.get('reasoning_content'), str): reasoning.append(delta['reasoning_content'])
    for event in raw.get('events', []) if isinstance(raw.get('events'), list) else []:
        if isinstance(event, dict) and event.get('type') == 'response.output_text.delta' and isinstance(event.get('delta'), str):
            text.append(event['delta'])
    if not text and not reasoning: return None
    answer = {'content': ''.join(text)}
    if reasoning: answer['reasoning_content'] = ''.join(reasoning)
    return answer


def _renderable_answer(level, response):
    """Only render complete, schema-valid answers with the original dimensions."""
    from validator.api import check_schema
    try:
        candidate = check_schema(normalize_answer(level, response['content']))
        if (len(candidate['board']), len(candidate['board'][0])) != (len(level['board']), len(level['board'][0])):
            return None
        return candidate
    except (ValueError, TypeError, KeyError, RecursionError):
        return None


def _write_image(render_fn, level, path, **kwargs):
    """Replace a PNG atomically, including when a viewer has opened the old file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + '.' + uuid.uuid4().hex + '.png')
    try:
        render_fn(level, temporary, **kwargs)
        _replace_file(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()
    return path


def _validation_options(options):
    return dict(history_steps=options.history_steps, history_gap=options.history_gap,
                max_ticks=options.max_ticks)


def render_attempt(level, attempt, directory, options, *, result=None, render_fn=None, validate_fn=None):
    """Persist the answer board and its terminal failure, without changing scores."""
    candidate = _renderable_answer(level, attempt.get('response'))
    if candidate is None:
        return {'answer_images': 0, 'failure_images': 0, 'unrenderable_answers': 1}
    if render_fn is None:
        from make_images.render import render as render_fn
    if result is None:
        if validate_fn is None:
            from validator import validate as validate_fn
        result = validate_fn(level, attempt['response']['content'], **_validation_options(options))
    stem = '%s-round-%d-attempt-%d' % (safe_name(level['level_id']), attempt['round'], attempt['attempt'])
    images = Path(directory) / 'images'
    _write_image(render_fn, level, images / (stem + '-answer.png'), solution=candidate)
    if result['success']:
        stale = images / (stem + '-failure.png')
        if stale.is_file(): stale.unlink()
        return {'answer_images': 1, 'failure_images': 0, 'unrenderable_answers': 0}
    trace = result.get('trace') or []
    state = copy.deepcopy(trace[-1]) if trace else {}
    state.setdefault('error_positions', result.get('error_positions', []))
    if not trace and result['error_type'] in ('track_budget', 'semaphore_budget'):
        field = 'board' if result['error_type'] == 'track_budget' else 'mods'
        state['error_positions'] = [[y, x] for y, row in enumerate(level[field])
                                    for x, value in enumerate(row) if not value and candidate[field][y][x]]
    annotations = []
    budget = result.get('budget_violation')
    if budget:
        resource = 'tracks' if budget['resource'] == 'tracks' else 'semaphores'
        title = 'Track budget exceeded' if budget['resource'] == 'tracks' else 'Semaphore budget exceeded'
        annotations = [title,
                       'Maximum new %s: %d   |   Used: %d' % (resource, budget['maximum'], budget['used'])]
    _write_image(render_fn, level, images / (stem + '-failure.png'), solution=candidate, state=state,
                 **({'annotations': annotations} if annotations else {}))
    return {'answer_images': 1, 'failure_images': 1, 'unrenderable_answers': 0}


def render_results(directory, *, rejudge=False):
    """Regenerate images locally; optionally replace judgments, retaining replies."""
    from validator import validate
    directory = Path(directory).resolve()
    metadata = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    contract = _resume_contract(metadata)
    options = RunOptions(**contract['options'])
    levels = {level['level_id']: _puzzle_snapshot(level) for level in contract['levels']}
    counts = Counter()
    outcomes = {value['level_id']: value for value in metadata.get('summary', {}).get('outcomes', [])}
    for path in sorted(directory.glob('*.json'), key=_level_sort):
        record = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(record, dict) or 'attempts' not in record: continue
        level = levels[record['level_id']]
        counts['levels'] += 1
        for attempt in record['attempts']:
            if attempt.get('response') is None: continue
            result = validate(level, attempt['response']['content'], **_validation_options(options))
            if attempt['success'] is not None:
                if (result['success'], result['error_type']) != (attempt['success'], attempt['error_type']):
                    if not rejudge: raise ValueError('Saved validator outcome differs: ' + record['level_id'])
                    attempt['success'], attempt['error_type'] = result['success'], result['error_type']
                    counts['judgments_updated'] += 1
            counts.update(render_attempt(level, attempt, directory, options, result=result))
        if rejudge:
            if any(a['success'] is True for a in record['attempts']): record['status'] = 'success'
            elif len({a['round'] for a in record['attempts'] if a['success'] is not None}) >= options.rounds:
                record['status'] = 'exhausted'
            elif record['status'] == 'success': record['status'] = 'running'
            atomic_json(path, record)
            outcomes[record['level_id']] = {'level_id': record['level_id'], 'status': record['status'],
                                          'rounds': sum(a['success'] is not None for a in record['attempts'])}
    if rejudge:
        metadata['summary'] = {'total_levels': len(metadata['contract']['levels']),
                               'finished_levels': sum(v['status'] in ('success', 'exhausted') for v in outcomes.values()),
                               'outcomes': list(outcomes.values())}
        atomic_json(directory / 'run.json', metadata)
    return dict(counts)


def run_level(provider, level, output_dir, options=None, *, validate_fn=None, render_fn=None,
              initial_prompt_fn=None, feedback_prompt_fn=None):
    """Save replies/outcomes only; replay compact records to reconstruct RAM memory.

    max_retry controls short retries, followed by continued delayed retries.
    Request and local processing failures never consume puzzle-answer rounds.
    """
    options = options or RunOptions()
    directory = Path(output_dir).resolve(); directory.mkdir(parents=True, exist_ok=True)
    _ensure_metadata(provider, level, directory, options)
    path = directory / (safe_name(level['level_id']) + '.json')
    with _run_lock(path):
        record = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
            'level_id': level['level_id'], 'status': 'running', 'attempts': []}
        resume_after_fix = record['status'] == 'needs_attention'
        if record['level_id'] != level['level_id']: raise ValueError('Checkpoint level mismatch')
        if validate_fn is None:
            from validator import validate as validate_fn
        if render_fn is None:
            from make_images.render import render as render_fn
        initial = initial_prompt_fn or initial_prompt
        feedback = feedback_prompt_fn or partial(feedback_prompt, prompt_mode=options.prompt_mode)
        scratch = ROOT / 'temp'; scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='railbound-feedback-', dir=str(scratch)) as temporary:
            memory, image_index = [], 0

            def save():
                atomic_json(path, record)

            def local(stage, operation):
                """Retry the same local work, retaining the already received reply."""
                errors = record.get('processing_errors', {})
                failures = errors.get(stage, {}).get('count', 0)
                pending = record.get('retry')
                if pending and pending['stage'] == stage:
                    _wait_retry(record, save, options, stage=stage, failures=pending['failures'],
                                error_type=pending['error_type'])
                while True:
                    try:
                        value = operation()
                    except Exception as exc:
                        failures += 1
                        error_type = _error_type(exc) if isinstance(exc, ProviderError) else 'runner_error:' + type(exc).__name__
                        record.setdefault('processing_errors', {})[stage] = {
                            'error_type': error_type, 'count': failures, 'recovered': False}
                        _wait_retry(record, save, options, stage=stage, failures=failures, error_type=error_type)
                    else:
                        if stage in record.get('processing_errors', {}):
                            record['processing_errors'][stage]['recovered'] = True
                            save()
                        return value

            def attempt_stage(name, attempt):
                return '%s:round-%d:attempt-%d' % (name, attempt['round'], attempt['attempt'])

            def picture(state=None, solution=None):
                nonlocal image_index
                image_index += 1
                image_path = Path(temporary) / ('%04d.png' % image_index)
                local('feedback-image-%d' % image_index,
                      lambda: render_fn(level, image_path, state=state, solution=solution))
                return image_part(image_path)

            def judge(attempt):
                def operation():
                    result = validate_fn(level, attempt['response']['content'], **_validation_options(options))
                    if result['error_type'] in ('resource_limit', 'invalid_level'):
                        raise ProviderError(result['error_type'], 'Validation has no puzzle judgment')
                    if (type(result['success']) is not bool or
                            (not result['success'] and not isinstance(result['error_type'], str)) or
                            (not result['success'] and not result['error_type'])):
                        raise ValueError('Validator must return a boolean and a typed failure')
                    return result
                return local(attempt_stage('validation', attempt), operation)

            def render_answer(attempt, result=None):
                return local(attempt_stage('answer-images', attempt),
                    lambda: render_attempt(level, attempt, directory, options, result=result,
                                           render_fn=render_fn, validate_fn=validate_fn))

            def append_feedback(attempt, result):
                memory.append(dict(role='assistant', **attempt['response']))
                parts = [{'type': 'text', 'text': local(attempt_stage('feedback-text', attempt),
                                                     lambda: feedback(result, options.feedback))}]
                if options.feedback == 'visual' and result.get('trace'):
                    states = result.get('feedback_states') or []
                    for index, state in enumerate(states):
                        state = copy.deepcopy(state)
                        if index == len(states) - 1 and not state.get('error_positions'):
                            state['error_positions'] = result.get('error_positions', [])
                        parts.append(picture(state, _renderable_answer(level, attempt['response'])))
                memory.append({'role': 'user', 'content': parts})

            parts = [{'type': 'text', 'text': local('initial-prompt', lambda: initial(level, mode=options.prompt_mode))}]
            if options.prompt_mode != 'text_rules': parts.append(picture())
            memory.append({'role': 'user', 'content': parts})
            round_number = 1
            for attempt in record['attempts']:
                if attempt['success'] is None: continue
                if attempt['round'] != round_number: raise ValueError('Inconsistent checkpoint rounds')
                result = judge(attempt)
                if result['success'] != attempt['success'] or result['error_type'] != attempt['error_type']:
                    raise ValueError('Saved validator outcome cannot be reproduced')
                render_answer(attempt, result)
                if attempt['success']:
                    record['status'] = 'success'; save(); return record
                if round_number < options.rounds: append_feedback(attempt, result)
                round_number += 1

            while round_number <= options.rounds:
                attempts = [a for a in record['attempts'] if a['round'] == round_number]
                previous = attempts[-1] if attempts else None
                if previous and previous['error_type'] == 'request_pending':
                    previous['error_type'] = 'interrupted'
                    previous['termination'] = {'reason': 'interrupted', 'finish_reason': None}
                    save()
                if previous and previous['error_type'] == 'validation_pending':
                    attempt = previous
                else:
                    if previous and not retryable(previous['error_type']) and not resume_after_fix:
                        record['status'] = 'needs_attention'; save(); return record
                    resume_after_fix = False
                    if previous:
                        _wait_retry(record, save, options, stage='request:round-%d' % round_number,
                                    failures=len(attempts), error_type=previous['error_type'])
                    attempt = {'round': round_number, 'attempt': len(attempts) + 1,
                               'response': None, 'success': None, 'error_type': 'request_pending',
                               'termination': {'reason': 'request_pending', 'finish_reason': None}}
                    record['attempts'].append(attempt); record['status'] = 'running'; save()
                    request_start = time.monotonic()
                    try:
                        completion = provider.complete(copy.deepcopy(memory))
                    except ProviderError as exc:
                        attempt['response'] = _partial_reply(exc.raw)
                        attempt['error_type'] = _error_type(exc)
                        attempt['termination'] = _termination_metadata(error=exc, elapsed_seconds=time.monotonic()-request_start)
                        save()
                        render_answer(attempt)
                        continue
                    except Exception as exc:
                        attempt['error_type'] = 'runner_error:' + type(exc).__name__
                        attempt['termination'] = {'reason': attempt['error_type'], 'finish_reason': None,
                                                  'elapsed_seconds': round(time.monotonic()-request_start, 3)}
                        save(); continue
                    attempt['response'] = _reply(completion)
                    attempt['error_type'] = {'length': 'output_truncated', 'content_filter': 'output_filtered'}.get(
                        completion.finish_reason, 'validation_pending')
                    if attempt['error_type'] == 'validation_pending' and not completion.text.strip():
                        attempt['error_type'] = 'incomplete_response'
                    reason = ('empty_final_answer' if attempt['error_type'] == 'incomplete_response' else
                              'completed' if attempt['error_type'] == 'validation_pending' else attempt['error_type'])
                    attempt['termination'] = _termination_metadata(completion=completion, reason=reason)
                    save()
                    if attempt['error_type'] != 'validation_pending':
                        render_answer(attempt)
                        continue
                result = judge(attempt)
                attempt['success'], attempt['error_type'] = result['success'], result['error_type']; save()
                render_answer(attempt, result)
                if result['success']:
                    record['status'] = 'success'; save(); return record
                if round_number < options.rounds: append_feedback(attempt, result)
                round_number += 1
            record['status'] = 'exhausted'; save(); return record


def _level_sort(path):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', path.as_posix())]


def summarize_run(directory):
    """Summarize saved outcomes without contacting a model or rerunning validation."""
    directory = Path(directory)
    run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    if run.get('schema_version') != 2:
        raise ValueError('Expected schema_version 2 experiment records')
    states, errors, request_errors, processing_errors = Counter(), Counter(), Counter(), Counter()
    pending_attempts, final_errors, successes_by_round = Counter(), Counter(), Counter()
    validated = returned = retries = 0
    records = {data['level_id']: data for path in directory.glob('*.json')
               for data in [json.loads(path.read_text(encoding='utf-8'))]
               if isinstance(data, dict) and 'level_id' in data and 'attempts' in data}
    outcomes = {value['level_id']: value for value in run.get('summary', {}).get('outcomes', [])}
    for level in run['contract']['levels']:
        level_id = level['level_id']
        if level_id not in records:
            states[outcomes.get(level_id, {}).get('status', 'not_started')] += 1
            continue
        record = records[level_id]
        status = record['status']
        first_success = None
        failed_rounds = set()
        last_error = None
        for attempt in record['attempts']:
            returned += attempt['response'] is not None
            retries += attempt['attempt'] > 1
            if attempt['success'] is True:
                validated += 1
                if first_success is None or attempt['round'] < first_success: first_success = attempt['round']
            elif attempt['success'] is False and attempt['error_type']:
                validated += 1
                errors[attempt['error_type']] += 1
                failed_rounds.add(attempt['round'])
                last_error = attempt['error_type']
            elif attempt['error_type'] in ('request_pending', 'validation_pending'):
                pending_attempts[attempt['error_type']] += 1
            else:
                request_errors[attempt['error_type'] or 'unclassified_engineering_error'] += 1
        for error in record.get('processing_errors', {}).values():
            processing_errors[error['error_type']] += error['count']
        if record.get('retry') or any(not error.get('recovered', True)
                                      for error in record.get('processing_errors', {}).values()):
            status = 'retry_wait' if record.get('retry') else 'running'
        elif first_success is not None:
            status = 'success'
        elif len(failed_rounds) >= run['contract']['options']['rounds']:
            status = 'exhausted'
            final_errors[last_error] += 1
        elif status in ('success', 'exhausted'):
            status = 'needs_attention'
        if first_success is not None: successes_by_round[first_success] += 1
        states[status] += 1
    total = len(run['contract']['levels'])
    completed = total > 0 and states['success'] + states['exhausted'] == total
    return {'configured_levels': total, 'states': dict(states),
            'model_replies_recorded': returned, 'validated_answers': validated,
            'extra_retry_attempts': retries, 'error_types': dict(errors),
            'final_error_types': dict(final_errors), 'request_error_types': dict(request_errors),
            'processing_error_types': dict(processing_errors), 'pending_attempts': dict(pending_attempts),
            'success_by_round': dict(successes_by_round), 'completed_campaign': completed,
            'campaign_accuracy': states['success'] / total if completed else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/deepseek.json')
    parser.add_argument('--resolve-model', action='store_true', help='Read model capacities without inference')
    parser.add_argument('--summarize', type=Path, help='Summarize saved results locally without model access')
    parser.add_argument('--render-results', type=Path, help='Regenerate saved answer/failure images locally without model access')
    parser.add_argument('--rejudge-results', type=Path, help='Rejudge saved replies and regenerate images locally without model access')
    parser.add_argument('--levels', default='levels/extended-hard')
    parser.add_argument('--level-id', action='append')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--prompt-mode', choices=PROMPT_MODES, default='image_rules')
    parser.add_argument('--feedback', choices=('binary', 'textual', 'visual'))
    parser.add_argument('--history-steps', type=int, default=1, help='Selected states; 0 means all')
    parser.add_argument('--history-gap', type=int, default=1)
    parser.add_argument('--max-ticks', type=int, default=10000)
    parser.add_argument('--max-retry', '--max_retry', type=int, default=2, help='Short retries before continuing with delayed retries')
    parser.add_argument('--retry-delay', type=float, default=60, help='Initial delayed retry wait in seconds')
    parser.add_argument('--retry-max-delay', type=float, default=600, help='Maximum delayed retry wait in seconds; retries continue')
    parser.add_argument('--workers', type=int, default=16, help='Concurrent levels (default: 16)')
    parser.add_argument('--output', default='output')
    parser.add_argument('--resume', type=Path, help='Resume a compact run with its original configuration')
    args = parser.parse_args(argv)
    if args.summarize:
        print(json.dumps(summarize_run(args.summarize), indent=2))
        return 0
    if args.render_results:
        print(json.dumps(render_results(args.render_results), indent=2))
        return 0
    if args.rejudge_results:
        print(json.dumps(render_results(args.rejudge_results, rejudge=True), indent=2))
        return 0
    if args.workers < 1 or args.limit is not None and args.limit < 1: parser.error('workers and limit must be positive')
    load_dotenv(); config = ModelConfig.load(args.config)

    def resolve():
        provider = create_provider(config)
        try: return provider.resolve_capabilities()
        except ProviderError as exc: parser.error(str(exc))
        finally: provider.close()
    if args.resolve_model:
        print(json.dumps(resolve(), indent=2)); return 0
    try:
        options = RunOptions(rounds=args.rounds, feedback=args.feedback, prompt_mode=args.prompt_mode,
                             history_steps=None if args.history_steps == 0 else args.history_steps,
                             history_gap=args.history_gap, max_ticks=args.max_ticks, max_retry=args.max_retry,
                             retry_delay=args.retry_delay, retry_max_delay=args.retry_max_delay)
    except ValueError as exc: parser.error(str(exc))
    source = Path(args.levels)
    files = [source] if source.is_file() else sorted(source.rglob('*.json'), key=_level_sort)
    levels = []
    for file in files:
        value = json.loads(file.read_text(encoding='utf-8-sig'))
        if isinstance(value, dict) and 'board' in value and 'level_id' in value:
            if not args.level_id or value['level_id'] in args.level_id: levels.append(value)
    if args.limit: levels = levels[:args.limit]
    if not levels: parser.error('No matching levels')
    ids = [safe_name(level['level_id']) for level in levels]
    if len(ids) != len(set(ids)): parser.error('Level IDs must have unique output filenames')
    contract = run_contract(config, levels, options)
    if args.resume:
        directory = args.resume.resolve()
        metadata = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
        try: saved_contract = _resume_contract(metadata)
        except ValueError as exc: parser.error(str(exc))
        if saved_contract != contract:
            parser.error('Resume requires matching model, options and puzzle contents')
        capabilities = metadata['model_capabilities']
    else:
        capabilities = resolve()
        prefix = safe_name(config.model_name + ('_' + config.version_suffix if config.version_suffix else ''))
        directory = Path(args.output).resolve() / (prefix + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
        directory.mkdir(parents=True, exist_ok=False)
        metadata = {'schema_version': 2, 'created_at': now(), 'contract': contract,
                    'model_capabilities': capabilities, 'tools_available_to_model': [], 'model_code_execution': False}
        atomic_json(directory / 'run.json', metadata)
    metadata['workers'] = args.workers
    atomic_json(directory / 'run.json', metadata)
    print('Run directory: ' + str(directory), flush=True)

    authentication_blocked = threading.Event()

    def execute(level):
        if authentication_blocked.is_set():
            saved_path = directory / (safe_name(level['level_id']) + '.json')
            if saved_path.exists():
                saved = json.loads(saved_path.read_text(encoding='utf-8'))
                if saved['status'] in ('success', 'exhausted'):
                    return {'level_id': level['level_id'], 'status': saved['status'],
                            'rounds': sum(a['success'] is not None for a in saved['attempts'])}
            return {'level_id': level['level_id'], 'status': 'pending',
                    'error_type': 'authentication_blocked'}
        provider = create_provider(config, capabilities=capabilities)
        try:
            record = run_level(provider, level, directory, options)
            error_type = record['attempts'][-1].get('error_type') if record['attempts'] else None
            if record['status'] == 'needs_attention' and error_type in ('authentication_error', 'http_401', 'http_403'):
                authentication_blocked.set()
            return {'level_id': level['level_id'], 'status': record['status'],
                    'rounds': sum(a['success'] is not None for a in record['attempts'])}
        except Exception as exc:
            return {'level_id': level['level_id'], 'status': 'needs_attention', 'error_type': 'runner_error:' + type(exc).__name__}
        finally: provider.close()

    outcomes = []
    with _run_lock(directory):
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(execute, level) for level in levels]
            for future in as_completed(futures):
                outcome = future.result(); outcomes.append(outcome)
                if outcome['status'] != 'pending': print(json.dumps(outcome), flush=True)
                metadata['updated_at'] = now()
                metadata['summary'] = {'total_levels': len(levels),
                    'finished_levels': sum(o['status'] in ('success', 'exhausted') for o in outcomes), 'outcomes': outcomes}
                atomic_json(directory / 'run.json', metadata)
    return 1 if any(o['status'] not in ('success', 'exhausted') for o in outcomes) else 0


if __name__ == '__main__':
    sys.exit(main())
