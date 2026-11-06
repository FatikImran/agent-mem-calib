"""Shared prompt construction and explicit capacity checks for memory experiments."""
import json

JUDGE_PROMPT = (
    'For each memory ID and each parameter, decide if the memory supplies evidence '
    'for that parameter in this specific request. Use applicable, inapplicable, or uncertain. '
    'Return a JSON object mapping every memory ID to an object mapping every parameter '
    'name to its verdict. /no_think'
)
FILTER_PROMPT = ('You filter retrieved memories for a conversational answer.\n'
    'Keep only the original sentences that are directly useful for the question.\n'
    'Do not answer the question, rewrite the memories, or invent information.\n'
    'If nothing is useful, return exactly: NO RELEVANT CONTEXT')


def validate_memories(memories):
    ids = [m['id'] for m in memories]
    if len(set(ids)) != len(ids) or not all(isinstance(i, str) and i for i in ids):
        raise ValueError('Memory IDs must be unique nonempty strings')
    if not all(isinstance(m['text'], str) for m in memories):
        raise ValueError('Memory texts must be strings')


def generator_input(query, schema, memories, admitted_fields=None):
    validate_memories(memories)
    fields = schema['parameters'].get('properties', {})
    if admitted_fields is not None and set(admitted_fields) - {m['id'] for m in memories}:
        raise ValueError('Admission refers to an unknown memory')
    lines = []
    for memory in memories:
        text = memory['text']
        if admitted_fields is not None:
            admitted = admitted_fields.get(memory['id'], [])
            if set(admitted) - fields.keys():
                raise ValueError('Admission refers to an unknown parameter')
            if not admitted:
                continue
            ordered = [field for field in fields if field in admitted]
            text = 'Use only for these parameters: ' + ', '.join(ordered) + '. ' + text
        lines.append('[memory]: ' + text + '\n')
    system = "You are a helpful assistant. Here is the user's background context:\n" + ''.join(lines) + '\n\nUse this context to infer missing parameters if needed.'
    function = {k: schema.get(k, {} if k == 'parameters' else '') for k in ('name', 'description', 'parameters')}
    return {'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': query}],
            'tools': [{'type': 'function', 'function': function}], 'tool_choice': 'auto'}


def judge_input(query, schema, memories):
    validate_memories(memories)
    payload = {'request': query, 'tool': schema,
               'memories': [{'id': m['id'], 'text': m['text']} for m in memories]}
    return {'messages': [{'role': 'system', 'content': JUDGE_PROMPT},
                         {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]}


def judge_response_schema(schema, memories):
    validate_memories(memories)
    fields = list(schema['parameters'].get('properties', {}))
    verdict = {'type': 'string', 'enum': ['applicable', 'inapplicable', 'uncertain']}
    row = {'type': 'object', 'properties': {field: verdict for field in fields},
           'required': fields, 'additionalProperties': False}
    return {'type': 'object', 'properties': {m['id']: row for m in memories},
            'required': [m['id'] for m in memories], 'additionalProperties': False}


def require_capacity(prompt_tokens, output_tokens, context_tokens):
    if prompt_tokens < 0 or output_tokens <= 0 or context_tokens <= 0:
        raise ValueError('Invalid token budget')
    if prompt_tokens + output_tokens > context_tokens:
        raise ValueError(f'Prompt/output budget exceeds context: {prompt_tokens}+{output_tokens}>{context_tokens}')
