"""Resolve an exact selected model's declared maximum before generation.

There are no provider-wide guesses: use per-model catalog metadata, or an
explicit ``model_limits`` declaration for providers without that metadata.
ChatGPT plan calls explicitly opt into server-managed budgets because their
documented API does not accept client output-token limits.
"""
import httpx


_OUTPUT_FIELDS = ('max_output_tokens', 'max_completion_tokens', 'output_token_limit')
_CONTEXT_FIELDS = ('context_window', 'context_length')
_REQUEST_FIELDS = ('max_tokens', 'max_completion_tokens', 'max_output_tokens')
SIWC_LIMITS_URL = 'https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations'


def validate_budget_policy(config):
    """Reject unsupported policies before authentication or catalog requests."""
    from .providers import ProviderError
    mode = getattr(config, 'output_budget', 'model_max')
    if getattr(config, 'provider', '') == 'openai_chatgpt':
        if mode != 'server_managed':
            raise ProviderError('capability_error',
                                'ChatGPT plan calls do not support a client output-token limit; '
                                'model_max and explicit cannot be enforced. Select output_budget=server_managed '
                                'only if accepting server-controlled generation length. See ' + SIWC_LIMITS_URL)
        unsupported = set(getattr(config, 'parameters', {})) - {'reasoning', 'text'}
        if unsupported or getattr(config, 'output_token_parameter', ''):
            raise ProviderError('capability_error',
                                'ChatGPT plan parameters support only reasoning and text in this adapter; '
                                'output token limits and other configured parameters must be removed. See ' + SIWC_LIMITS_URL)
    elif mode not in ('model_max', 'explicit'):
        raise ProviderError('capability_error',
                            'output_budget=server_managed is reserved for the ChatGPT plan API; '
                            'this provider requires model_max or explicit')
    return mode


def _fail(message, status_code=None):
    # Lazy import avoids a cycle when providers imports this resolver.
    from .providers import ProviderError
    raise ProviderError('capability_error', message +
                        '; configure model_limits.max_output_tokens with the selected model\'s documented maximum',
                        status_code=status_code) from None


def _positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(label + ' must be a positive integer')
    return value


def _capacity(metadata, names, label):
    """Accept only named capacities at the model level or in its limits object."""
    containers = [metadata]
    if 'limits' in metadata:
        if not isinstance(metadata['limits'], dict):
            _fail('Model limits metadata must be an object')
        containers.append(metadata['limits'])
    values = [_positive_integer(container[name], label)
              for container in containers for name in names if name in container]
    if len(set(values)) > 1:
        _fail('Model catalog has conflicting ' + label + ' capacities')
    return values[0] if values else None


def resolve_model_limits(config, client, token):
    """Return serializable resolved capacities and the effective request budget.

    ``model_max`` uses the established model maximum even if a request parameter
    contains a lower value. ``explicit`` uses one configured token parameter,
    bounded by that same established maximum. Parameters alone never establish
    model capacity. ChatGPT-only ``server_managed`` records metadata without
    claiming to request or enforce the maximum. Configured ``model_limits``
    avoids catalog requests.
    """
    mode = validate_budget_policy(config)
    parameters = getattr(config, 'parameters', {})
    if not isinstance(parameters, dict):
        _fail('Model parameters must be an object')
    configured = [(key, parameters[key]) for key in _REQUEST_FIELDS if key in parameters]
    if len(configured) > 1:
        _fail('Configure only one output token request parameter')
    requested = _positive_integer(configured[0][1], configured[0][0]) if configured else None
    if mode == 'explicit' and requested is None:
        _fail('Explicit output budget requires one output token request parameter')

    declared = getattr(config, 'model_limits', {})
    if not isinstance(declared, dict):
        _fail('model_limits must be an object')
    if declared:
        if declared.get('api_model') != config.api_model:
            _fail('model_limits.api_model must exactly match the selected api_model')
        description = declared.get('source')
        if not isinstance(description, str) or not description.strip():
            _fail('model_limits.source must name a nonempty documentation source')
        if 'max_output_tokens' not in declared:
            _fail('Configured model_limits has no output maximum')
        maximum = _positive_integer(declared['max_output_tokens'], 'model_limits.max_output_tokens')
        context = (_positive_integer(declared['context_window'], 'model_limits.context_window')
                   if 'context_window' in declared else None)
        source = 'config:model_limits: ' + description.strip()
    else:
        try:
            response = client.get(config.base_url.rstrip('/') + '/models',
                                  headers={'Authorization': 'Bearer ' + token})
        except httpx.HTTPError:
            # Do not retain exception text or request headers: they may contain credentials.
            _fail('Unable to read the model catalog due to a transport error')
        if response.status_code != 200:
            _fail('Model catalog returned HTTP %d' % response.status_code, response.status_code)
        try:
            payload = response.json()
        except ValueError:
            _fail('Model catalog did not return valid JSON')
        if not isinstance(payload, dict):
            _fail('Model catalog must contain a data or models array')
        records = payload.get('data') if 'data' in payload else payload.get('models')
        if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
            _fail('Model catalog must contain a data or models array of objects')
        matches = [record for record in records
                   if record.get('id') == config.api_model or record.get('slug') == config.api_model]
        if not matches:
            _fail('The exact selected api_model is absent from the model catalog')
        if len(matches) != 1:
            _fail('The model catalog contains duplicate records for the selected api_model')
        maximum = _capacity(matches[0], _OUTPUT_FIELDS, 'maximum output token')
        context = _capacity(matches[0], _CONTEXT_FIELDS, 'context window')
        if maximum is None and mode != 'server_managed':
            _fail('The selected model catalog entry has no documented output maximum')
        source = 'model_catalog:' + config.base_url.rstrip('/') + '/models'

    if mode == 'explicit' and requested > maximum:
        _fail('The explicit output budget exceeds the selected model maximum')
    # Source is user-supplied metadata in the config path; never echo a credential.
    if token:
        source = source.replace(token, '[redacted]')
    result = {'api_model': config.api_model,
            'max_output_tokens': maximum,
            'context_window': context,
            'source': source,
            'output_budget': mode,
            'requested_output_tokens': maximum if mode == 'model_max' else requested}
    if mode == 'server_managed':
        result.update(requested_output_tokens=None, budget_control='server_managed', maximum_enforced=False)
    return result
