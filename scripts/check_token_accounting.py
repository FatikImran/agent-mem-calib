"""Compare offline prompt-token counts with saved llama.cpp inference usage."""
import hashlib
import json
from pathlib import Path

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Jinja2ChatFormatter


def main():
    llm = Llama(model_path='models/Qwen_Qwen3-4B-Q4_K_M.gguf', vocab_only=True, verbose=False)
    formatter = Jinja2ChatFormatter(llm.metadata['tokenizer.chat_template'], '<|im_end|>', '')
    rows, hashes = [], {}
    for name in ('gap3-cap-replay.json', 'gap3-cap-remaining.json', 'gap3-dev-clean-1024.json'):
        path = Path('results') / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        data = json.loads(path.read_text(encoding='utf8'))
        for row in data['rows']:
            state = row['input']
            schema = state['tool_schema']
            function = {k: schema.get(k, {} if k == 'parameters' else '') for k in ('name', 'description', 'parameters')}
            messages = state['messages'] + [{'role': 'user', 'content': state['query']}]
            prompt = formatter(messages=messages, tools=[{'type': 'function', 'function': function}], tool_choice='auto').prompt
            predicted = len(llm.tokenize(prompt.encode('utf8'), add_bos=False, special=True))
            recorded = row['usage']['prompt_tokens']
            rows.append({'source': name, 'task_id': row['task_id'], 'arm': row['source_arm'],
                'kind': 'generator',
                'cap': row['cap'], 'measured': predicted, 'recorded': recorded,
                'match': predicted == recorded})
    judge_system = (
        'For each parameter, decide if the memory supplies evidence for its value '
        'in this specific request. Use applicable, inapplicable, or uncertain. '
        'Return a JSON object mapping every parameter name to its verdict. /no_think'
    )
    for name in ('gap3-dev-judge.json', 'gap3-dev-judge-required.json',
                 'gap3-screen-judge.json', 'gap3-screen-judge-required.json'):
        path = Path('results') / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        data = json.loads(path.read_text(encoding='utf8'))
        for row in data:
            messages = [{'role': 'system', 'content': judge_system},
                        {'role': 'user', 'content': json.dumps(row['input'], ensure_ascii=False)}]
            prompt = formatter(messages=messages).prompt
            predicted = len(llm.tokenize(prompt.encode('utf8'), add_bos=False, special=True))
            recorded = row['usage']['prompt_tokens']
            rows.append({'source': name, 'task_id': row['task_id'], 'arm': row['condition'],
                'kind': 'judge', 'cap': 200, 'measured': predicted, 'recorded': recorded,
                'match': predicted == recorded})
    result = {'status': 'saved-inference token-accounting check', 'source_hashes': hashes,
        'rows': rows, 'checked': len(rows), 'matched': sum(r['match'] for r in rows),
        'chat_format': llm.chat_format,
        'template_sha256': hashlib.sha256(llm.metadata['tokenizer.chat_template'].encode()).hexdigest()}
    Path('results/token-accounting-check.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(f"{result['matched']}/{result['checked']} saved prompt-token counts matched.")
    if result['matched'] != result['checked']:
        raise ValueError('Offline token accounting differs from recorded runtime usage')


if __name__ == '__main__':
    main()
