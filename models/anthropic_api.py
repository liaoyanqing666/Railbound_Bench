"""Tool-free requests to the native Anthropic Messages API.

API version and streaming contract:
https://platform.claude.com/docs/en/api/versioning
https://platform.claude.com/docs/en/build-with-claude/streaming
"""
import base64
import copy
import json
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx

from .providers import Completion, OpenAICompatible, ProviderError


_FINISH = {'end_turn': 'stop', 'stop_sequence': 'stop', 'max_tokens': 'length',
           'refusal': 'content_filter', 'model_context_window_exceeded': 'length'}
_ERROR_STATUS = {'rate_limit_error': 429, 'overloaded_error': 529, 'api_error': 500,
                 'authentication_error': 401, 'permission_error': 403,
                 'invalid_request_error': 400, 'not_found_error': 404}


def _redact(value, token):
    if isinstance(value, str):
        return value.replace(token, '[redacted]') if token else value
    if isinstance(value, dict):
        return {_redact(key, token): _redact(item, token) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, token) for item in value]
    return value


def _response_headers(response):
    return {key: value for key, value in response.headers.items()
            if key in ('request-id', 'retry-after') or key.startswith('anthropic-ratelimit-')}


def _retry_after(headers):
    value = headers.get('retry-after')
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (ValueError, TypeError):
        try:
            target = parsedate_to_datetime(value)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return None


def _error_kind(status):
    return ('authentication_error' if status in (401, 403) else 'rate_limit' if status == 429
            else 'provider_error' if status >= 500 or status == 408 else 'request_error')


def _usage(native):
    """Keep native counters; cached input is separate from native input_tokens."""
    result = copy.deepcopy(native)
    if type(native.get('input_tokens')) is int:
        result['prompt_tokens'] = native['input_tokens'] + sum(
            native.get(key, 0) for key in ('cache_creation_input_tokens', 'cache_read_input_tokens'))
    if type(native.get('output_tokens')) is int:
        result['completion_tokens'] = native['output_tokens']
    if 'prompt_tokens' in result and 'completion_tokens' in result:
        result['total_tokens'] = result['prompt_tokens'] + result['completion_tokens']
    if 'cache_read_input_tokens' in native:
        result['prompt_tokens_details'] = {'cached_tokens': native['cache_read_input_tokens']}
    reasoning = native.get('reasoning_tokens')
    if type(reasoning) is int:
        result['completion_tokens_details'] = {'reasoning_tokens': reasoning}
    return result


def _sum_usage(items):
    """Sum reported token counters only; per-call receipts retain missing values."""
    result = {}
    for item in items:
        for key, value in item.items():
            if type(value) is int:
                result[key] = result.get(key, 0) + value
            elif isinstance(value, dict):
                result[key] = _sum_usage([result.get(key, {}), value])
    return result


def _messages(messages, supports_images):
    conversation, system = [], []
    for message in messages:
        if not isinstance(message, dict) or message.get('role') not in ('system', 'developer', 'user', 'assistant'):
            raise ValueError('Only text/image conversation roles are permitted')
        role, content = message['role'], message.get('content')
        if message.get('tool_calls') or message.get('function_call'):
            raise ValueError('Tool calls are not permitted')
        if isinstance(content, str):
            parts = [{'type': 'text', 'text': content}]
        elif isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    raise ValueError('Message parts must be objects')
                if part.get('type') == 'text' and isinstance(part.get('text'), str):
                    parts.append({'type': 'text', 'text': part['text']})
                elif part.get('type') == 'image' and role == 'user':
                    if not supports_images:
                        raise ProviderError('capability_error', 'Configured model does not support images')
                    mime = part.get('mime_type')
                    if mime not in ('image/jpeg', 'image/png', 'image/gif', 'image/webp'):
                        raise ValueError('Unsupported Anthropic image media type')
                    parts.append({'type': 'image', 'source': {'type': 'base64', 'media_type': mime,
                                  'data': base64.b64encode(Path(part['path']).read_bytes()).decode('ascii')}})
                else:
                    raise ValueError('Only text and local user image parts are allowed')
        else:
            raise ValueError('Message content must be a string or text/image array')
        if role in ('system', 'developer'):
            system.extend(parts)
        else:
            # No reasoning, signatures or other provider memory is replayed.
            conversation.append({'role': role, 'content': content if isinstance(content, str) else parts})
    if not conversation or conversation[-1]['role'] != 'user':
        raise ValueError('Anthropic Messages must end with a user turn')
    return conversation, system


class _MessageStream:
    """Accumulate full content while bounding raw diagnostics to 16 events."""
    def __init__(self):
        self.metadata, self.native_usage, self.blocks = {}, {}, []
        self.events, self.event_count = [], 0
        self.started, self.done, self.final_usage = False, False, False
        self.stop_reason, self.stop_sequence = None, None

    def snapshot(self):
        content, text, thinking = [], [], []
        for block in self.blocks:
            kind = block['type']
            original = copy.deepcopy(block['original'])
            if kind == 'text':
                value = ''.join(block['parts'])
                original['text'] = value
                text.append(value)
            elif kind == 'thinking':
                value = ''.join(block['parts'])
                original['thinking'] = value
                if block['signatures'] or 'signature' in original:
                    original['signature'] = ''.join(block['signatures'])
                thinking.append(value)
            content.append(original)
        message = {'role': 'assistant', 'content': ''.join(text)}
        if thinking:
            message['reasoning_content'] = ''.join(thinking)
        usage = _usage(self.native_usage)
        return dict(self.metadata, content=content, choices=[{'index': 0, 'message': message,
                    'finish_reason': _FINISH.get(self.stop_reason)}], usage=usage,
                    native_usage=copy.deepcopy(self.native_usage), stop_reason=self.stop_reason,
                    stop_sequence=self.stop_sequence, status='completed' if self.done else 'incomplete',
                    stream_done=self.done, usage_is_partial=not (
                        self.done and self.final_usage and 'input_tokens' in self.native_usage),
                    native_events=list(self.events), event_count=self.event_count)

    def fail(self, message, kind='protocol_error', status_code=None, detail=None):
        raw = self.snapshot()
        if detail is not None:
            raw['error'] = detail
        error = ProviderError(kind, message, raw=raw, status_code=status_code)
        if isinstance(detail, dict) and isinstance(detail.get('message'), str):
            error.provider_message = detail['message'][:2000]
        raise error

    def merge_usage(self, usage):
        if not isinstance(usage, dict):
            self.fail('Anthropic usage must be an object')
        for key in ('input_tokens', 'output_tokens', 'cache_creation_input_tokens',
                    'cache_read_input_tokens', 'reasoning_tokens'):
            if key in usage and (type(usage[key]) is not int or usage[key] < 0):
                self.fail('Anthropic token usage must contain nonnegative integers')
        # Counts in message_delta are cumulative, not increments.
        self.native_usage.update(copy.deepcopy(usage))

    def consume(self, item):
        self.event_count += 1
        self.events.append(item)
        if len(self.events) > 16:
            del self.events[0]
        if not isinstance(item, dict) or not isinstance(item.get('type'), str):
            self.fail('Anthropic stream event must contain a type')
        kind = item['type']
        if kind == 'ping':
            return
        if self.done:
            self.fail('Anthropic event arrived after message_stop')
        if kind == 'error':
            detail = item.get('error')
            detail = dict(detail) if isinstance(detail, dict) else {'message': 'Malformed API error'}
            code = detail.get('type', 'api_error')
            detail['code'] = code
            status = _ERROR_STATUS.get(code, 500)
            self.fail('Anthropic stream reported an error', _error_kind(status), status, detail)
        if kind == 'message_start':
            message = item.get('message')
            if self.started or not isinstance(message, dict) or message.get('role') != 'assistant':
                self.fail('Invalid Anthropic message_start')
            if not isinstance(message.get('id'), str) or message.get('content') != []:
                self.fail('Anthropic message_start must have an ID and empty content')
            self.started = True
            self.metadata = {key: message[key] for key in ('id', 'model', 'type') if key in message}
            self.merge_usage(message.get('usage', {}))
        elif kind in ('content_block_start', 'content_block_delta', 'content_block_stop'):
            if not self.started or self.stop_reason is not None:
                self.fail('Anthropic content arrived outside an active message')
            index = item.get('index')
            if type(index) is not int or index < 0:
                self.fail('Invalid Anthropic content block index')
            if kind == 'content_block_start':
                block = item.get('content_block')
                if index != len(self.blocks) or not isinstance(block, dict) or any(
                        not previous['closed'] for previous in self.blocks):
                    self.fail('Invalid Anthropic content block start')
                block_type = block.get('type')
                if block_type not in ('text', 'thinking', 'redacted_thinking'):
                    self.fail('Unexpected Anthropic content block; no tools are available')
                value = block.get('text' if block_type == 'text' else 'thinking', '')
                signature, data = block.get('signature', ''), block.get('data', '')
                if not all(isinstance(field, str) for field in (value, signature, data)):
                    self.fail('Invalid Anthropic content block text')
                self.blocks.append({'type': block_type, 'parts': [value],
                                    'signatures': [signature] if signature else [],
                                    'original': copy.deepcopy(block), 'closed': False})
            else:
                if index >= len(self.blocks) or self.blocks[index]['closed']:
                    self.fail('Anthropic delta or stop has no open content block')
                block = self.blocks[index]
                if kind == 'content_block_stop':
                    block['closed'] = True
                    return
                delta = item.get('delta')
                if not isinstance(delta, dict):
                    self.fail('Anthropic content delta must be an object')
                expected = {'text_delta': ('text', 'text', 'parts'),
                            'thinking_delta': ('thinking', 'thinking', 'parts'),
                            'signature_delta': ('thinking', 'signature', 'signatures')}
                update = expected.get(delta.get('type'))
                if update is None or block['type'] != update[0] or not isinstance(delta.get(update[1]), str):
                    self.fail('Unexpected Anthropic content delta; no tools are available')
                block[update[2]].append(delta[update[1]])
        elif kind == 'message_delta':
            if not self.started or any(not block['closed'] for block in self.blocks):
                self.fail('Anthropic message_delta arrived before content completed')
            delta = item.get('delta')
            if not isinstance(delta, dict):
                self.fail('Anthropic message delta must be an object')
            reason = delta.get('stop_reason')
            if reason is not None:
                if reason not in _FINISH:
                    self.fail('Unexpected Anthropic stop reason: ' + str(reason))
                if self.stop_reason is not None and reason != self.stop_reason:
                    self.fail('Conflicting Anthropic stop reasons')
                self.stop_reason, self.stop_sequence = reason, delta.get('stop_sequence')
            if 'usage' in item:
                self.merge_usage(item['usage'])
                self.final_usage = self.final_usage or 'output_tokens' in item['usage']
        elif kind == 'message_stop':
            if not self.started or self.stop_reason is None or any(not block['closed'] for block in self.blocks):
                self.fail('Anthropic stream ended without a complete message', 'incomplete_response')
            self.done = True
        # The versioned protocol permits additional event types; ignore them.

    def read(self, lines):
        pending, name = [], None
        def flush():
            nonlocal name
            if pending:
                try:
                    item = json.loads('\n'.join(pending))
                except ValueError:
                    self.fail('Invalid Anthropic stream JSON', 'incomplete_response')
                if isinstance(item, dict) and name is not None and item.get('type') != name:
                    self.fail('Anthropic SSE event name and data type disagree')
                self.consume(item)
                pending.clear()
            name = None
        for line in lines:
            if not line:
                flush()
            elif line.startswith('event:'):
                name = line[6:].strip()
            elif line.startswith('data:'):
                pending.append(line[5:].lstrip())
        flush()
        if not self.done:
            self.fail('Anthropic stream ended before message_stop', 'incomplete_response')
        return self.snapshot()


class AnthropicMessages(OpenAICompatible):
    """Native Messages inference without SDK agents, tool definitions or fallback."""
    def __init__(self, config, client=None, capabilities=None, max_text_continuations=None):
        if max_text_continuations is None:
            max_text_continuations = config.max_text_continuations
        if type(max_text_continuations) is not int or max_text_continuations < 0:
            raise ValueError('max_text_continuations must be a nonnegative integer')
        super().__init__(config, client=client, capabilities=capabilities)
        self.max_text_continuations = max_text_continuations
        self.last_response_headers = {}

    def resolve_capabilities(self):
        if not self.config.model_limits:
            raise ProviderError('capability_error', 'Anthropic requires documented model_limits for the exact model')
        return super().resolve_capabilities()

    def build_request(self, messages):
        allowed = {'max_tokens', 'max_completion_tokens', 'max_output_tokens', 'reasoning_effort', 'thinking', 'stop'}
        if set(self.config.parameters) - allowed:
            raise ValueError('Unsupported native Anthropic parameters: ' + ', '.join(
                sorted(set(self.config.parameters) - allowed)))
        resolved = self.resolve_capabilities()
        params = self.config.parameters
        thinking = params.get('thinking', {'type': 'adaptive'})
        if thinking not in ({'type': 'adaptive'}, {'type': 'disabled'}):
            raise ValueError('Anthropic thinking must be adaptive or disabled, without a token budget')
        effort = params.get('reasoning_effort')
        if effort is not None and effort not in ('low', 'medium', 'high', 'xhigh', 'max'):
            raise ValueError('Unsupported Anthropic effort level')
        conversation, system = _messages(messages, self.config.supports_images)
        request = {'model': self.config.api_model, 'messages': conversation,
                   'max_tokens': resolved['requested_output_tokens'], 'stream': self.config.stream,
                   'thinking': copy.deepcopy(thinking)}
        if effort is not None:
            request['output_config'] = {'effort': effort}
        if system:
            request['system'] = system
        if 'stop' in params:
            stop = params['stop']
            stop = [stop] if isinstance(stop, str) else stop
            if not isinstance(stop, list) or any(not isinstance(value, str) for value in stop):
                raise ValueError('Anthropic stop must be a string or list of strings')
            request['stop_sequences'] = list(stop)
        return request

    def _complete_request(self, body, before_generation=None, generation_index=1):
        # Retain request settings only. Images and full prompts stay in memory.
        request = {key: copy.deepcopy(value) for key, value in body.items() if key not in ('messages', 'system')}
        request.update(message_count=len(body['messages']), system_provided='system' in body)
        start, aggregate, headers, token = time.monotonic(), _MessageStream(), {}, self._token()
        if before_generation is not None:
            before_generation(generation_index, {
                'generation_index': generation_index, 'is_text_continuation': generation_index > 1,
                'requested_api_model': self.config.api_model, 'api_version': self.config.api_version,
                'max_tokens': body['max_tokens'], 'stream': body['stream'],
                'max_text_continuations': self.max_text_continuations,
                'native_text_continuation': self.max_text_continuations > 0,
                'no_tools': True, 'fallback_used': False,
                'message_count': len(body['messages']), 'system_provided': 'system' in body})
            start = time.monotonic()
        try:
            with self.client.stream('POST', self.config.base_url + '/messages', json=body,
                    headers={'x-api-key': token, 'anthropic-version': self.config.api_version,
                             'content-type': 'application/json'}) as response:
                headers = _response_headers(response)
                self.last_response_headers = _redact(headers, token)
                if response.is_error:
                    response.read()
                    try:
                        payload = response.json()
                    except ValueError:
                        payload = {'error': {'message': response.text[:2000]}}
                    raw = payload if isinstance(payload, dict) else {'error': {'message': 'Invalid error response'}}
                    detail = raw.get('error')
                    if isinstance(detail, dict) and isinstance(detail.get('type'), str):
                        detail['code'] = detail['type']
                    error = ProviderError(_error_kind(response.status_code),
                        'Anthropic returned HTTP %d' % response.status_code, status_code=response.status_code, raw=raw)
                    if isinstance(detail, dict) and isinstance(detail.get('message'), str):
                        error.provider_message = detail['message'][:2000]
                    raise error
                if self.config.stream:
                    raw = aggregate.read(response.iter_lines())
                else:
                    response.read()
                    try:
                        message = response.json()
                    except ValueError:
                        aggregate.fail('Invalid Anthropic response JSON', 'incomplete_response')
                    if not isinstance(message, dict) or not isinstance(message.get('content'), list):
                        aggregate.fail('Anthropic response must contain a content array')
                    initial = dict(message, content=[])
                    aggregate.consume({'type': 'message_start', 'message': initial})
                    for index, block in enumerate(message['content']):
                        aggregate.consume({'type': 'content_block_start', 'index': index, 'content_block': block})
                        aggregate.consume({'type': 'content_block_stop', 'index': index})
                    aggregate.consume({'type': 'message_delta', 'delta': {
                        'stop_reason': message.get('stop_reason'), 'stop_sequence': message.get('stop_sequence')},
                        'usage': message.get('usage', {})})
                    aggregate.consume({'type': 'message_stop'})
                    raw = aggregate.snapshot()
            if raw.get('model') != self.config.api_model:
                aggregate.fail('Anthropic returned a different model than the exact selected API model')
        except (ProviderError, httpx.HTTPError) as exc:
            if isinstance(exc, httpx.HTTPError):
                exc = ProviderError('transport_error', type(exc).__name__, raw=aggregate.snapshot())
            exc.raw = _redact(exc.raw, token)
            if isinstance(exc.raw, dict):
                exc.raw.update(request_id=headers.get('request-id'), response_headers=_redact(headers, token))
            exc.request = request
            exc.args = tuple(_redact(value, token) for value in exc.args)
            if isinstance(exc.provider_message, str):
                exc.provider_message = _redact(exc.provider_message, token)
            exc.retry_after_seconds = _retry_after(headers)
            exc.request_dispatched = True
            exc.elapsed_seconds = time.monotonic() - start
            raise exc from None
        raw.update(request_id=headers.get('request-id'), response_headers=_redact(headers, token))
        message = raw['choices'][0]['message']
        return Completion(message['content'], raw, raw['usage'], request, time.monotonic() - start,
                          message, _FINISH[raw['stop_reason']])

    def _generation(self, raw, elapsed, index, error=None):
        """A durable subreceipt contains no content blocks, signatures or prompts."""
        result = {key: copy.deepcopy(raw.get(key)) for key in (
            'stop_reason', 'status', 'stream_done', 'request_id', 'response_headers')}
        result.update(generation_index=index, is_text_continuation=index > 1,
            request_dispatched=True, requested_api_model=self.config.api_model,
            api_model=raw.get('model'), response_id=raw.get('id'),
            finish_reason=_FINISH.get(raw.get('stop_reason')),
            usage=copy.deepcopy(raw.get('usage') or {}),
            native_usage=copy.deepcopy(raw.get('native_usage') or {}),
            usage_is_partial=raw.get('usage_is_partial', True), elapsed_seconds=elapsed)
        result.update(no_tools=True, fallback_used=False)
        result['stream_done'] = raw.get('stream_done') is True
        result['status'] = raw.get('status') or ('failed' if error is not None else 'incomplete')
        if error is not None:
            result.update(error_kind=error.kind, http_status=error.status_code)
        return result

    def _decorate(self, raw, generations, dispatch_tracking=False):
        result = copy.deepcopy(raw)
        if self.max_text_continuations:
            # Signed blocks are used only by the in-memory continuation request.
            result.pop('content', None)
            result.pop('native_events', None)
        result.update(native_text_continuation=self.max_text_continuations > 0,
            max_text_continuations=self.max_text_continuations,
            generation_count=len(generations), generations=copy.deepcopy(generations),
            no_tools=True, fallback_used=False,
            final_text_complete=raw.get('stream_done') is True and raw.get('stop_reason') == 'end_turn',
            usage=_sum_usage(item['usage'] for item in generations),
            native_usage=_sum_usage(item['native_usage'] for item in generations),
            usage_is_partial=any(item['usage_is_partial'] for item in generations))
        if dispatch_tracking:
            result['dispatch_tracking'] = True
        return result

    def _continuation_blocker(self, raw, body, continuation_prompt):
        content = raw.get('content')
        if not content:
            return 'empty_assistant_content'
        for block in content:
            if block.get('type') == 'thinking' and not block.get('signature'):
                return 'missing_thinking_signature'
            if block.get('type') == 'redacted_thinking' and not block.get('data'):
                return 'missing_redacted_thinking'
        context = self.resolve_capabilities().get('context_window')
        usage = raw.get('usage', {})
        if (type(context) is not int or type(usage.get('prompt_tokens')) is not int
                or type(usage.get('completion_tokens')) is not int or raw.get('usage_is_partial')):
            return 'unknown_context_usage'
        # The short ASCII continuation uses at most one token per UTF-8 byte.
        # Reserve the full next output ceiling and never trim signed history.
        if (usage['prompt_tokens'] + usage['completion_tokens'] + len(continuation_prompt.encode('utf-8'))
                + body['max_tokens'] > context):
            return 'context_window'
        return None

    def complete(self, messages, on_generation=None, before_generation=None):
        body = self.build_request(messages)
        if self.max_text_continuations:
            from prompts import NATIVE_TEXT_CONTINUATION_PROMPT
            continuation_prompt = NATIVE_TEXT_CONTINUATION_PROMPT
        generations, started = [], time.monotonic()
        while True:
            try:
                completion = self._complete_request(body, before_generation, len(generations) + 1)
            except ProviderError as error:
                if getattr(error, 'request_dispatched', False):
                    raw = error.raw if isinstance(error.raw, dict) else {}
                    receipt = self._generation(raw, error.elapsed_seconds, len(generations) + 1, error)
                    generations.append(receipt)
                    if on_generation is not None:
                        on_generation(copy.deepcopy(receipt))
                    error.raw = self._decorate(raw, generations, before_generation is not None)
                    error.request = dict(error.request or {}, generation_count=len(generations),
                                         max_text_continuations=self.max_text_continuations,
                                         generations=copy.deepcopy(generations))
                raise
            raw = completion.raw
            receipt = self._generation(raw, completion.elapsed_seconds, len(generations) + 1)
            generations.append(receipt)
            if on_generation is not None:
                on_generation(copy.deepcopy(receipt))
            blocker = None
            if raw.get('stop_reason') == 'max_tokens' and len(generations) <= self.max_text_continuations:
                blocker = self._continuation_blocker(raw, body, continuation_prompt)
                if blocker is None:
                    body['messages'].extend([{'role': 'assistant', 'content': raw['content']},
                                             {'role': 'user', 'content': continuation_prompt}])
                    continue
            elif raw.get('stop_reason') == 'max_tokens' and self.max_text_continuations:
                blocker = 'continuation_limit'
            completion.raw = self._decorate(raw, generations, before_generation is not None)
            if blocker:
                completion.raw['continuation_stopped_reason'] = blocker
            completion.usage = completion.raw['usage']
            completion.request.update(generation_count=len(generations),
                                      max_text_continuations=self.max_text_continuations,
                                      generations=copy.deepcopy(generations))
            completion.elapsed_seconds = time.monotonic() - started
            return completion
