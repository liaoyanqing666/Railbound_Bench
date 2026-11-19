"""Direct, tool-free model providers."""
import base64
import copy
import json
import mimetypes
import os
import socket
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote, urlparse

import httpx


@dataclass
class ModelConfig:
    provider: str
    model_name: str
    api_model: str
    version_suffix: str = ''
    base_url: str = ''
    api_key_env: str = ''
    supports_images: bool = True
    timeout: float = 7200.0
    stream: bool = False
    output_budget: str = 'model_max'
    model_limits: Dict[str, Any] = field(default_factory=dict)
    output_token_parameter: str = ''
    parameters: Dict[str, Any] = field(default_factory=dict)
    oauth_profile: str = 'default'
    oauth_dir: str = '.auth/openai'
    api_version: str = ''
    max_text_continuations: int = 0

    def __post_init__(self):
        allowed = {'temperature', 'top_p', 'max_tokens', 'max_completion_tokens',
                   'max_output_tokens', 'seed', 'reasoning_effort', 'reasoning',
                   'thinking', 'response_format', 'text', 'frequency_penalty',
                   'presence_penalty', 'stop'}
        unknown = set(self.parameters) - allowed
        if unknown:
            raise ValueError('Unsupported model parameters: ' + ', '.join(sorted(unknown)))
        if not self.model_name or not self.api_model:
            raise ValueError('model_name and api_model must be explicit, nonempty strings')
        if self.timeout <= 0:
            raise ValueError('timeout must be positive')
        if not isinstance(self.stream, bool):
            raise ValueError('stream must be a boolean')
        if self.output_budget not in ('model_max', 'explicit', 'server_managed'):
            raise ValueError('output_budget must be model_max, explicit or server_managed')
        if not isinstance(self.model_limits, dict):
            raise ValueError('model_limits must be an object')
        if self.output_token_parameter not in ('', 'max_tokens', 'max_completion_tokens', 'max_output_tokens'):
            raise ValueError('Unsupported output_token_parameter')
        defaults = {
            'deepseek': ('https://api.deepseek.com', 'DEEPSEEK_API'),
            'aliyun': ('https://dashscope.aliyuncs.com/compatible-mode/v1', 'ALIYUN_API'),
            'openai_compatible': ('https://api.openai.com/v1', 'OPENAI_API_KEY'),
            'azure_openai': (os.environ.get('AZURE_OPENAI_ENDPOINT', ''), 'AZURE_OPENAI_API_KEY'),
            'openai_chatgpt': ('https://api.openai.com/v1', ''),
            'anthropic': ('https://api.anthropic.com/v1', 'CLAUDE_API'),
        }
        if self.provider not in defaults:
            raise ValueError('Unknown provider: ' + self.provider)
        default_url, default_env = defaults[self.provider]
        self.base_url = (self.base_url or default_url).rstrip('/')
        self.api_key_env = self.api_key_env or default_env
        if self.provider == 'anthropic':
            self.api_version = self.api_version or '2026-10-21'
        if type(self.max_text_continuations) is not int or self.max_text_continuations < 0:
            raise ValueError('max_text_continuations must be a nonnegative integer')
        if self.max_text_continuations and self.provider != 'anthropic':
            raise ValueError('Text continuations are supported only by the native Anthropic provider')
        if self.provider == 'azure_openai':
            if not self.base_url or not self.api_version:
                raise ValueError('Azure requires a resource base_url (or AZURE_OPENAI_ENDPOINT) and api_version')
            if not self.model_limits:
                raise ValueError('Azure requires documented model_limits for the selected deployment')
        if self.base_url:
            parsed = urlparse(self.base_url)
            if parsed.scheme != 'https' and not (parsed.scheme == 'http' and parsed.hostname in ('127.0.0.1', 'localhost', '::1')):
                raise ValueError('Provider URL must use HTTPS, except loopback test servers')
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError('Provider URL must not contain credentials, query, or fragment')
        if self.provider == 'openai_chatgpt' and self.base_url != 'https://api.openai.com/v1':
            raise ValueError('ChatGPT OAuth uses only the documented public OpenAI endpoint')

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8-sig')))

    def to_dict(self):
        result = asdict(self)
        if self.provider not in ('azure_openai', 'anthropic'):
            result.pop('api_version')
        if not self.max_text_continuations:
            result.pop('max_text_continuations')
        return result


class ProviderError(RuntimeError):
    def __init__(self, kind, message, *, status_code=None, raw=None, request=None):
        super().__init__(message)
        self.kind, self.status_code, self.raw, self.request = kind, status_code, raw, request
        self.provider_message = None

    def to_dict(self):
        return {'kind': self.kind, 'message': str(self), 'status_code': self.status_code,
                'raw': self.raw, 'request': self.request}


@dataclass
class Completion:
    text: str
    raw: Dict[str, Any]
    usage: Dict[str, Any]
    request: Dict[str, Any]
    elapsed_seconds: float = 0.0
    assistant_message: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


def image_part(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError('Image does not exist: ' + str(path))
    return {'type': 'image', 'path': str(path),
            'mime_type': mimetypes.guess_type(str(path))[0] or 'image/png'}


def materialize_messages(messages, supports_images=True, responses=False):
    result = []
    for message in messages:
        role = message.get('role')
        if role not in ('system', 'developer', 'user', 'assistant'):
            raise ValueError('Only text/image conversation roles are permitted')
        content = message.get('content')
        if isinstance(content, str):
            output_content = content
        elif isinstance(content, list):
            output_content = []
            for part in content:
                if part.get('type') == 'text':
                    output_content.append({'type': 'input_text' if responses else 'text', 'text': part['text']})
                elif part.get('type') == 'image':
                    if not supports_images:
                        raise ProviderError('capability_error', 'Configured model does not support images')
                    if role != 'user':
                        raise ValueError('Images are allowed only in user messages')
                    data = Path(part['path']).read_bytes()
                    url = 'data:' + part['mime_type'] + ';base64,' + base64.b64encode(data).decode('ascii')
                    output_content.append({'type': 'input_image', 'image_url': url, 'detail': 'high'} if responses else
                                          {'type': 'image_url', 'image_url': {'url': url, 'detail': 'high'}})
                else:
                    raise ValueError('Only text and local image parts are allowed')
        else:
            raise ValueError('Message content must be a string or text/image array')
        output = {'role': 'developer' if responses and role == 'system' else role, 'content': output_content}
        # Preserve provider-returned reasoning in conversational memory when required.
        if not responses and role == 'assistant' and message.get('reasoning_content') is not None:
            output['reasoning_content'] = message['reasoning_content']
        result.append(output)
    return result


def _http_error(response, request):
    try:
        raw = response.json()
    except ValueError:
        raw = {'body': response.text}
    status = response.status_code
    kind = ('authentication_error' if status in (401, 403) else 'rate_limit' if status == 429
            else 'provider_error' if status >= 500 else 'request_error')
    error = ProviderError(kind, 'Provider returned HTTP %d' % status, status_code=status, raw=raw, request=request)
    detail = raw.get('error', raw) if isinstance(raw, dict) else {}
    if isinstance(detail, dict) and isinstance(detail.get('message'), str):
        message = detail['message']
        for name in ('api-key', 'Authorization'):
            credential = response.request.headers.get(name)
            if credential:
                message = message.replace(credential, '[redacted]')
                message = message.replace(credential.split(' ', 1)[-1], '[redacted]')
        error.provider_message = message[:2000]
        detail['message'] = message
    raise error


class OpenAICompatible:
    def __init__(self, config, client=None, capabilities=None):
        self.config = config
        self.capabilities = copy.deepcopy(capabilities)
        transport = None
        if client is None and config.provider == 'azure_openai':
            # Azure may buffer the response throughout hidden reasoning.
            # Keep an otherwise idle TCP flow alive until that response arrives.
            options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
            idle = getattr(socket, 'TCP_KEEPIDLE', getattr(socket, 'TCP_KEEPALIVE', None))
            if idle is not None:
                options.append((socket.IPPROTO_TCP, idle, 60))
            if hasattr(socket, 'TCP_KEEPINTVL'):
                options.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 30))
            transport = httpx.HTTPTransport(socket_options=options)
        self.client = client or httpx.Client(timeout=httpx.Timeout(config.timeout, connect=60.0, write=600.0, pool=60.0),
                                            follow_redirects=False, transport=transport)

    def close(self):
        self.client.close()

    def build_request(self, messages):
        content = materialize_messages(messages, self.config.supports_images)
        request = dict(self._budget_parameters(), model=self.config.api_model,
                       messages=content, stream=self.config.stream)
        if self.config.stream:
            request['stream_options'] = {'include_usage': True}
        return request

    def resolve_capabilities(self):
        """Resolve model-specific limits independently; never start inference here."""
        from .capabilities import resolve_model_limits, validate_budget_policy
        validate_budget_policy(self.config)
        if self.capabilities is None:
            token = '' if self.config.model_limits else self._token()
            self.capabilities = resolve_model_limits(self.config, self.client, token)
        if self.capabilities.get('api_model') != self.config.api_model:
            raise ProviderError('capability_error', 'Resolved model limits belong to a different API model')
        for key in ('max_output_tokens', 'requested_output_tokens'):
            if type(self.capabilities.get(key)) is not int or self.capabilities[key] <= 0:
                raise ProviderError('capability_error', 'Resolved model limits must contain positive integer budgets')
        if self.capabilities['requested_output_tokens'] > self.capabilities['max_output_tokens']:
            raise ProviderError('capability_error', 'Requested output budget exceeds the model maximum')
        if self.config.output_budget == 'model_max' and self.capabilities['requested_output_tokens'] != self.capabilities['max_output_tokens']:
            raise ProviderError('capability_error', 'model_max requires the selected model maximum')
        return copy.deepcopy(self.capabilities)

    def _budget_parameters(self):
        resolved = self.resolve_capabilities()
        params = copy.deepcopy(self.config.parameters)
        previous_keys = [key for key in ('max_tokens', 'max_completion_tokens', 'max_output_tokens') if key in params]
        if len(previous_keys) > 1:
            raise ValueError('Configure only one output-token parameter')
        key = self.config.output_token_parameter or (previous_keys[0] if previous_keys else 'max_tokens')
        for previous in previous_keys:
            params.pop(previous)
        params[key] = resolved['requested_output_tokens']
        return params

    def _token(self):
        token = os.environ.get(self.config.api_key_env, '').strip()
        if not token:
            raise ProviderError('authentication_error', 'Missing environment variable ' + self.config.api_key_env)
        return token

    def _chat_transport(self):
        return (self.config.base_url + '/chat/completions',
                {'headers': {'Authorization': 'Bearer ' + self._token()}})

    def complete(self, messages):
        request = self.build_request(messages)
        start = time.monotonic()
        try:
            url, transport = self._chat_transport()
            if self.config.stream:
                with self.client.stream('POST', url, json=request, **transport) as response:
                    if response.is_error:
                        response.read()
                        _http_error(response, request)
                    raw = parse_chat_stream(response.iter_lines())
            else:
                response = self.client.post(url, json=request, **transport)
                if response.is_error:
                    _http_error(response, request)
                try:
                    raw = response.json()
                except ValueError as exc:
                    raise ProviderError('incomplete_response', 'Invalid provider JSON', raw=response.text, request=request) from exc
        except ProviderError as exc:
            exc.request = request
            raise
        except httpx.HTTPError as exc:
            raise ProviderError('transport_error', type(exc).__name__, request=request) from exc
        try:
            if not isinstance(raw, dict) or not isinstance(raw.get('choices'), list) or not raw['choices']:
                raise ValueError('Response must have a nonempty choices array')
            choice = raw['choices'][0]
            if not isinstance(choice, dict) or not isinstance(choice.get('message'), dict):
                raise ValueError('Response choice must contain a message object')
            answer = choice['message']
            finish_reason = choice.get('finish_reason')
            if finish_reason is None:
                raise ProviderError('incomplete_response', 'Missing completion finish reason', raw=raw, request=request)
            if finish_reason is not None and not isinstance(finish_reason, str):
                raise ValueError('Invalid finish reason')
            if finish_reason in ('tool_calls', 'function_call'):
                raise ProviderError('protocol_error', 'Unexpected tool completion', raw=raw, request=request)
            if finish_reason in ('insufficient_system_resource', 'aborted'):
                raise ProviderError('provider_error', 'Generation interrupted by provider: ' + finish_reason,
                                    raw=raw, request=request)
            if raw.get('usage') is not None and not isinstance(raw['usage'], dict):
                raise ValueError('Usage must be an object')
            if answer.get('tool_calls') or answer.get('function_call'):
                raise ProviderError('protocol_error', 'Unexpected tool call; no tools are available', raw=raw, request=request)
            text = answer.get('content')
            if text is None and finish_reason in ('length', 'content_filter'):
                text = ''
            if not isinstance(text, str):
                raise ValueError('Response has no text content')
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError('protocol_error', 'Malformed provider response', raw=raw, request=request) from exc
        assistant = {'role': 'assistant', 'content': text}
        if answer.get('reasoning_content') is not None:
            assistant['reasoning_content'] = answer['reasoning_content']
        return Completion(text, raw, raw.get('usage') or {}, request, time.monotonic() - start, assistant, finish_reason)


class AzureOpenAICompatible(OpenAICompatible):
    """Azure deployment inference with the shared tool-free chat parser."""
    def build_request(self, messages):
        messages = [{key: message[key] for key in ('role', 'content')}
                    if message.get('role') == 'assistant' else message for message in messages]
        return super().build_request(messages)

    def _chat_transport(self):
        deployment = quote(self.config.api_model, safe='')
        url = self.config.base_url + '/openai/deployments/' + deployment + '/chat/completions'
        return url, {'headers': {'api-key': self._token()},
                     'params': {'api-version': self.config.api_version}}


def parse_chat_stream(lines):
    """Aggregate complete text while retaining only the last 16 raw SSE chunks."""
    chunks, text_parts, reasoning_parts, pending = [], [], [], []
    metadata, usage, finish, done, chunk_count = {}, {}, None, False, 0

    def snapshot():
        message = {'role': 'assistant', 'content': ''.join(text_parts)}
        if reasoning_parts:
            message['reasoning_content'] = ''.join(reasoning_parts)
        return dict(metadata, object='chat.completion', choices=[{'index': 0, 'message': message,
                    'finish_reason': finish}], usage=usage, chunks=chunks, chunk_count=chunk_count, stream_done=done)

    def consume(data):
        nonlocal finish, usage, done, chunk_count
        if not data:
            return
        if data == '[DONE]':
            done = True
            return
        try:
            chunk = json.loads(data)
        except ValueError as exc:
            raw = snapshot()
            raw['fragment'] = data
            raise ProviderError('protocol_error', 'Invalid chat stream JSON', raw=raw) from exc
        chunks.append(chunk)
        chunk_count += 1
        if len(chunks) > 16:
            del chunks[0]
        if not isinstance(chunk, dict):
            raise ProviderError('protocol_error', 'Chat stream chunk must be an object', raw=snapshot())
        if chunk.get('error'):
            raw = snapshot()
            raw['error'] = chunk['error']
            raise ProviderError('provider_error', 'Chat stream reported an error', raw=raw)
        choices = chunk.get('choices')
        if not isinstance(choices, list):
            raise ProviderError('protocol_error', 'Chat stream choices must be an array', raw=snapshot())
        for key in ('id', 'created', 'model', 'system_fingerprint', 'service_tier'):
            if key in chunk:
                metadata[key] = chunk[key]
        if chunk.get('usage') is not None:
            if not isinstance(chunk['usage'], dict):
                raise ProviderError('protocol_error', 'Chat stream usage must be an object', raw=snapshot())
            usage = chunk['usage']
        for choice in choices:
            if not isinstance(choice, dict) or choice.get('index', 0) != 0 or not isinstance(choice.get('delta'), dict):
                raise ProviderError('protocol_error', 'Malformed chat stream choice', raw=snapshot())
            delta = choice['delta']
            if delta.get('tool_calls') or delta.get('function_call'):
                raise ProviderError('protocol_error', 'Unexpected streamed tool call', raw=snapshot())
            for key, destination in (('content', text_parts), ('reasoning_content', reasoning_parts)):
                value = delta.get(key)
                if value is not None:
                    if not isinstance(value, str):
                        raise ProviderError('protocol_error', 'Streamed text must be a string', raw=snapshot())
                    if finish is not None and value:
                        raise ProviderError('protocol_error', 'Text arrived after finish reason', raw=snapshot())
                    destination.append(value)
            reason = choice.get('finish_reason')
            if reason is not None:
                if not isinstance(reason, str) or (finish is not None and reason != finish):
                    raise ProviderError('protocol_error', 'Invalid or conflicting finish reason', raw=snapshot())
                finish = reason
                if reason in ('tool_calls', 'function_call'):
                    raise ProviderError('protocol_error', 'Unexpected streamed tool completion', raw=snapshot())

    try:
        for line in lines:
            if not line:
                consume('\n'.join(pending))
                pending = []
                if done:
                    break
            elif line.startswith('data:'):
                pending.append(line[5:].lstrip())
        if pending and not done:
            consume('\n'.join(pending))
    except httpx.HTTPError as exc:
        raw = snapshot()
        if pending:
            raw['fragment'] = '\n'.join(pending)
        raise ProviderError('transport_error', 'Interrupted chat response stream', raw=raw) from exc
    if not done or finish is None:
        raise ProviderError('incomplete_response', 'Chat stream ended without a finish reason and [DONE]', raw=snapshot())
    return snapshot()


def response_finish_reason(response):
    """Map explicit Responses outcomes to the common completion finish reason."""
    if response.get('status') == 'incomplete':
        details = response.get('incomplete_details')
        reason = details.get('reason') if isinstance(details, dict) else None
        return {'max_output_tokens': 'length', 'content_filter': 'content_filter'}.get(reason)
    for item in response.get('output', []):
        if isinstance(item, dict) and item.get('type') == 'message':
            if any(isinstance(part, dict) and part.get('type') == 'refusal' for part in item.get('content', [])):
                return 'content_filter'
    return 'stop'


def parse_response_stream(lines):
    """Preserve SSE terminal results; explicit budget exhaustion is a model outcome."""
    events, chunk, completed = [], [], None
    output_items, text_parts_by_index, finished_text, text_kinds = {}, {}, {}, {}
    def consume(data):
        nonlocal completed
        if not data or data == '[DONE]':
            return
        try:
            event = json.loads(data)
        except ValueError as exc:
            raise ProviderError('protocol_error', 'Invalid SSE JSON', raw={'events': events, 'fragment': data}) from exc
        events.append(event)
        if not isinstance(event, dict) or not isinstance(event.get('type'), str):
            raise ProviderError('protocol_error', 'SSE event must be an object with a type', raw={'events': events})
        kind = event['type']
        if kind in ('response.output_item.added', 'response.output_item.done'):
            item = event.get('item')
            if not isinstance(item, dict) or item.get('type') not in ('message', 'reasoning'):
                raise ProviderError('protocol_error', 'Unexpected or malformed streamed output item', raw={'events': events})
            if any(key in item for key in ('tool_calls', 'function_call', 'call_id')):
                raise ProviderError('protocol_error', 'Unexpected streamed tool call', raw={'events': events})
            if kind == 'response.output_item.done':
                output_items[event.get('output_index', len(output_items))] = item
        if kind in ('response.output_text.delta', 'response.output_text.done', 'response.refusal.delta', 'response.refusal.done'):
            key = (event.get('output_index', 0), event.get('content_index', 0))
            text_kinds[key] = 'refusal' if 'refusal' in kind else 'output_text'
            field = 'delta' if kind.endswith('.delta') else 'text' if 'output_text' in kind else 'refusal'
            value = event.get(field)
            if not isinstance(value, str):
                raise ProviderError('protocol_error', 'Streamed output text must be a string', raw={'events': events})
            if kind.endswith('.delta'): text_parts_by_index.setdefault(key, []).append(value)
            else: finished_text[key] = value
        if kind in ('response.failed', 'error'):
            raise ProviderError('provider_error', 'Responses stream ended with ' + kind, raw={'events': events})
        if kind in ('response.completed', 'response.incomplete'):
            terminal = event.get('response')
            if not isinstance(terminal, dict):
                raise ProviderError('protocol_error', 'SSE terminal response must be an object', raw={'events': events})
            expected_status = 'completed' if kind == 'response.completed' else 'incomplete'
            if terminal.get('status') != expected_status:
                raise ProviderError('protocol_error', 'SSE event and terminal status disagree', raw={'events': events})
            if kind == 'response.incomplete':
                details = terminal.get('incomplete_details')
                if not isinstance(details, dict):
                    raise ProviderError('protocol_error', 'Incomplete response requires detail object', raw={'events': events})
                if details.get('reason') not in ('max_output_tokens', 'content_filter'):
                    raise ProviderError('incomplete_response', 'Response ended incompletely', raw={'events': events})
            completed = terminal
    try:
        for line in lines:
            if line == '':
                consume('\n'.join(chunk))
                chunk = []
                if completed is not None:
                    break
            elif line.startswith('data:'):
                chunk.append(line[5:].lstrip())
        if chunk:
            consume('\n'.join(chunk))
    except httpx.HTTPError as exc:
        raise ProviderError('transport_error', 'Interrupted response stream', raw={'events': events}) from exc
    if completed is None:
        raise ProviderError('incomplete_response', 'Stream ended without response.completed', raw={'events': events})
    output = completed.get('output', [])
    if not isinstance(output, list) or (completed.get('usage') is not None and not isinstance(completed['usage'], dict)):
        raise ProviderError('protocol_error', 'Malformed terminal output or usage', raw={'events': events})
    if not output:
        # Some plan streams omit repeated output from the terminal envelope.
        # Completed items are authoritative; text deltas must not be appended twice.
        output = [item for _, item in sorted(output_items.items())]
        if not any(item.get('type') == 'message' for item in output) and (finished_text or text_parts_by_index):
            parts = []
            for key in sorted(set(finished_text) | set(text_parts_by_index)):
                kind = text_kinds[key]
                field = 'refusal' if kind == 'refusal' else 'text'
                parts.append({'type':kind, field:finished_text.get(key, ''.join(text_parts_by_index.get(key, [])))})
            output.append({'type':'message', 'role':'assistant', 'content':parts})
        completed = dict(completed, output=output)
    text_parts = []
    for item in output:
        if not isinstance(item, dict) or item.get('type') not in ('message', 'reasoning'):
            raise ProviderError('protocol_error', 'Unexpected or malformed output item', raw={'events': events})
        if any(key in item for key in ('tool_calls', 'function_call', 'call_id')):
            raise ProviderError('protocol_error', 'Unexpected tool call; no tools are available', raw={'events': events})
        if item['type'] != 'message':
            continue
        content = item.get('content')
        if item.get('role') != 'assistant' or not isinstance(content, list):
            raise ProviderError('protocol_error', 'Message output must contain assistant content', raw={'events': events})
        for part in content:
            if not isinstance(part, dict) or part.get('type') not in ('output_text', 'refusal'):
                raise ProviderError('protocol_error', 'Malformed message content part', raw={'events': events})
            field = 'text' if part['type'] == 'output_text' else 'refusal'
            if not isinstance(part.get(field), str):
                raise ProviderError('protocol_error', 'Message text must be a string', raw={'events': events})
            text_parts.append(part[field])
    text = ''.join(text_parts)
    if not text and response_finish_reason(completed) not in ('length', 'content_filter'):
        raise ProviderError('protocol_error', 'Completed response has no text', raw={'events': events})
    return text, completed, events


def response_reasoning_summary(response, events):
    """Collect observable summary text only, never encrypted or private reasoning."""
    item_ids, item_order, summaries, deltas = {}, {}, {}, {}
    for event in events:
        item = event.get('item')
        item_id = event.get('item_id') or (item.get('id') if isinstance(item, dict) else None)
        index = event.get('output_index')
        if isinstance(item_id, str) and type(index) is int and index >= 0:
            item_ids[index] = item_id
    known_ids = set(item_ids.values())
    for index, item in enumerate(response.get('output', [])):
        item_id = item.get('id') if isinstance(item, dict) else None
        if isinstance(item_id, str) and item_id not in known_ids:
            item_ids.setdefault(index, item_id)
            known_ids.add(item_id)

    def item_key(item_id, index):
        item_id = item_id if isinstance(item_id, str) else item_ids.get(index)
        key = ('id', item_id) if item_id is not None else ('index', index)
        item_order.setdefault(key, index)
        return key

    def remember(key, part_index, text, priority):
        if isinstance(text, str) and text:
            previous = summaries.get((key, part_index))
            if previous is None or priority >= previous[0]:
                summaries[(key, part_index)] = (priority, text)

    def remember_item(item, index, priority):
        key = item_key(item.get('id'), index)
        summary = item.get('summary')
        if isinstance(summary, list):
            for part_index, part in enumerate(summary):
                if isinstance(part, dict) and part.get('type') == 'summary_text':
                    remember(key, part_index, part.get('text'), priority)

    for event in events:
        kind = event['type']
        index = event.get('output_index', 0)
        if type(index) is not int or index < 0:
            continue
        if kind == 'response.output_item.done':
            item = event.get('item')
            if isinstance(item, dict) and item.get('type') == 'reasoning':
                remember_item(item, index, 3)
        elif kind in ('response.reasoning_summary_part.done',
                      'response.reasoning_summary_text.done', 'response.reasoning_summary_text.delta'):
            part_index = event.get('summary_index', 0)
            if type(part_index) is not int or part_index < 0:
                continue
            key = item_key(event.get('item_id'), index)
            if kind == 'response.reasoning_summary_part.done':
                part = event.get('part')
                if isinstance(part, dict) and part.get('type') == 'summary_text':
                    remember(key, part_index, part.get('text'), 2)
            elif kind == 'response.reasoning_summary_text.done':
                remember(key, part_index, event.get('text'), 1)
            elif isinstance(event.get('delta'), str):
                deltas.setdefault((key, part_index), []).append(event['delta'])
    for index, item in enumerate(response.get('output', [])):
        if isinstance(item, dict) and item.get('type') == 'reasoning':
            remember_item(item, index, 4)
    parts = []
    for key, part_index in sorted(set(summaries) | set(deltas),
                                  key=lambda value: (item_order[value[0]], value[1])):
        text = summaries[(key, part_index)][1] if (key, part_index) in summaries else ''.join(deltas[(key, part_index)])
        if text:
            parts.append(text)
    return '\n\n'.join(parts)


class ChatGPTOAuth(OpenAICompatible):
    def resolve_capabilities(self):
        from .capabilities import resolve_model_limits, validate_budget_policy
        validate_budget_policy(self.config)
        if self.capabilities is None:
            token = '' if self.config.model_limits else self._token()
            self.capabilities = resolve_model_limits(self.config, self.client, token)
        resolved = self.capabilities
        if (resolved.get('api_model') != self.config.api_model or
                resolved.get('output_budget') != 'server_managed' or
                resolved.get('budget_control') != 'server_managed' or
                resolved.get('maximum_enforced') is not False or
                resolved.get('requested_output_tokens') is not None):
            raise ProviderError('capability_error', 'Saved ChatGPT plan capabilities must record server_managed output budgeting')
        for key in ('max_output_tokens', 'context_window'):
            value = resolved.get(key)
            if value is not None and (type(value) is not int or value <= 0):
                raise ProviderError('capability_error', 'Reported ChatGPT model capacities must be positive integers or unknown')
        return copy.deepcopy(resolved)

    def build_request(self, messages):
        self.resolve_capabilities()
        content = materialize_messages(messages, self.config.supports_images, responses=True)
        return dict(copy.deepcopy(self.config.parameters), model=self.config.api_model,
                    input=content,
                    store=False, stream=True)

    def _token(self):
        from .openai_login import access_token
        return access_token(Path(self.config.oauth_dir), self.config.oauth_profile, self.client)

    def complete(self, messages):
        request = self.build_request(messages)
        start = time.monotonic()
        try:
            with self.client.stream('POST', self.config.base_url + '/responses', json=request,
                                    headers={'Authorization': 'Bearer ' + self._token()}) as response:
                if response.is_error:
                    response.read()
                    _http_error(response, request)
                text, raw, events = parse_response_stream(response.iter_lines())
        except ProviderError as exc:
            exc.request = request
            raise
        except httpx.HTTPError as exc:
            raise ProviderError('transport_error', type(exc).__name__, request=request) from exc
        assistant = {'role': 'assistant', 'content': text}
        summary = response_reasoning_summary(raw, events)
        if summary:
            assistant['reasoning_summary'] = summary
        return Completion(text, {'response': raw, 'events': events}, raw.get('usage') or {}, request,
                          time.monotonic() - start, assistant, response_finish_reason(raw))


def create_provider(config, client=None, capabilities=None):
    if isinstance(config, (str, Path)):
        config = ModelConfig.load(config)
    if config.provider == 'anthropic':
        from .anthropic_api import AnthropicMessages
        return AnthropicMessages(config, client=client, capabilities=capabilities)
    if config.provider == 'azure_openai':
        return AzureOpenAICompatible(config, client, capabilities)
    return (ChatGPTOAuth if config.provider == 'openai_chatgpt' else OpenAICompatible)(config, client, capabilities)
