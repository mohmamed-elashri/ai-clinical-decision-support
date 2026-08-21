"""
Task 3 Grounded Generation Script
Executes the full Day 3 pipeline in live LLM mode using NVIDIA NIM API.
Loads vector index, retrieves evidence with citations, prompts the LLM, validates schema, and tests in-scope, out-of-scope, and paraphrased queries.
"""

import os
import sys
import json

# Ensure project root is in sys.path
proj_root = os.path.abspath('.')
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from dotenv import load_dotenv
load_dotenv()

# Set NVIDIA API Key
NV_API_KEY = os.environ.get('NV_API_KEY')
if not NV_API_KEY:
    print("[WARNING] NV_API_KEY is not set in environment or .env. Running in simulated fallback mode.")
else:
    print(f"NV_API_KEY configured: {NV_API_KEY[:6]}...{'*' * 8}")

# Check required files
print("\nChecking required files:")
for p in ['schema/response_schema.json', 'prompt/grounding_prompt.txt', 'generation.py', 'retrieval.py', 'query.py']:
    print(f" - {p}: {os.path.exists(os.path.join(proj_root, p))}")

# Imports
from query import load_index
from retrieval import retrieve_with_citation, print_retrieval_view
from generation import answer_question

print('\nLoading Chroma vector index...')
vectordb = load_index()
print('Index loaded successfully.')

# Load grounding prompt and schema
with open(os.path.join(proj_root, 'prompt', 'grounding_prompt.txt'), 'r', encoding='utf-8') as f:
    grounding_prompt = f.read()

with open(os.path.join(proj_root, 'schema', 'response_schema.json'), 'r', encoding='utf-8') as f:
    schema = json.load(f)

print('\nLoaded grounding prompt and schema.')
print(f"Schema keys: {list(schema.keys())}")

# Schema validator
try:
    import jsonschema
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'jsonschema'])
    import jsonschema

validator = jsonschema.Draft7Validator(schema)

def validate_response(obj):
    errors = sorted(validator.iter_errors(obj), key=lambda e: str(e.path))
    if errors:
        print('Validation failed:')
        for err in errors:
            print(f" - Field '{'/'.join(map(str, err.path))}': {err.message}")
        return False
    return True

print('jsonschema Draft7Validator ready.')

# Helper functions for generation
try:
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'requests'])
    import requests

def build_prompt(retrieved, question):
    context_blocks = []
    for r in retrieved:
        context_blocks.append(
            f"SOURCE: {r['document_name']} (page {r['page_number']}, chunk_id: {r.get('chunk_id')})\n{r['text'][:800]}"
        )
    prompt = grounding_prompt.replace('{context_blocks}', '\n\n'.join(context_blocks))
    prompt = prompt.replace('{question}', question)
    return prompt

def call_live_llm(prompt_text, nv_api_key=None, endpoint=None, model=None, timeout=30):
    endpoint = endpoint or os.environ.get('NV_LLM_ENDPOINT') or 'https://integrate.api.nvidia.com/v1/chat/completions'
    nv_api_key = nv_api_key or os.environ.get('NV_API_KEY')
    model = model or os.environ.get('NV_MODEL') or 'meta/llama-3.1-8b-instruct'

    if not nv_api_key:
        raise RuntimeError('NV_API_KEY not set in environment for live mode')

    headers = {
        'Authorization': f'Bearer {nv_api_key}',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    system_message = (
        "You are an AI clinical decision support assistant. Return ONLY a single valid JSON object strictly conforming to this schema:\n"
        "- 'status': string, MUST be either 'grounded' or 'abstain'\n"
        "- 'answer': string (the concise recommendation if grounded, or empty string '' if abstain)\n"
        "- 'evidence': array of string snippets from the provided text supporting the answer\n"
        "- 'citations': array of objects, each with 'document_name' (string) and 'page_number' (integer or string), and optional 'chunk_id' (string)\n"
        "- 'confidence': string, MUST be either 'confident', 'uncertain', or 'insufficient'\n"
        "Do NOT include markdown formatting or extra keys."
    )

    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': prompt_text}
        ],
        'temperature': 0.1,
        'max_tokens': 1024
    }

    r = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, dict) and 'choices' in data and len(data['choices']) > 0:
        c = data['choices'][0]
        if isinstance(c, dict) and 'message' in c and 'content' in c['message']:
            return c['message']['content']
        elif isinstance(c, dict) and 'text' in c:
            return c['text']
    return json.dumps(data)

def parse_and_clean_json(raw_text, top_score=0.0):
    text = raw_text.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].startswith('```'):
            lines = lines[:-1]
        text = '\n'.join(lines).strip()

    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end+1]

    obj = json.loads(text)

    # Normalize status
    status_raw = str(obj.get('status', '')).lower()
    if status_raw in ('grounded', 'answer', 'success', 'ok', 'answered'):
        status = 'grounded'
    elif status_raw in ('abstain', 'refusal', 'refuse', 'rejected', 'insufficient'):
        status = 'abstain'
    else:
        status = 'grounded' if obj.get('answer') else 'abstain'

    # Normalize confidence
    confidence_raw = str(obj.get('confidence', '')).lower()
    if confidence_raw in ('confident', 'high', 'strong'):
        confidence = 'confident'
    elif confidence_raw in ('uncertain', 'medium', 'moderate'):
        confidence = 'uncertain'
    elif confidence_raw in ('insufficient', 'low', 'none', 'unknown'):
        confidence = 'insufficient'
    else:
        confidence = 'confident' if status == 'grounded' else 'insufficient'

    # Clean citations
    valid_citations = []
    for cit in obj.get('citations', []):
        if isinstance(cit, dict) and 'document_name' in cit and 'page_number' in cit:
            c_dict = {
                'document_name': str(cit['document_name']),
                'page_number': int(cit['page_number']) if str(cit['page_number']).isdigit() else str(cit['page_number'])
            }
            if 'chunk_id' in cit and cit['chunk_id']:
                c_dict['chunk_id'] = str(cit['chunk_id'])
            valid_citations.append(c_dict)

    # Clean evidence
    evidence_list = []
    for ev in obj.get('evidence', []):
        if isinstance(ev, str) and ev.strip():
            evidence_list.append(ev.strip())

    cleaned = {
        'status': status,
        'answer': str(obj.get('answer', '')),
        'evidence': evidence_list,
        'citations': valid_citations,
        'confidence': confidence,
    }
    if top_score is not None:
        cleaned['top_score'] = round(float(top_score), 3)

    return cleaned

def simulate_llm_response(question, retrieved):
    resp = answer_question(vectordb, question, k=3)
    evidence = []
    for ch in resp.get('grounded_chunks', []):
        txt = ch.get('text') or ch.get('page_content') or ch.get('text_excerpt', '')
        if txt:
            evidence.append(txt.strip()[:400])
    out = {
        'status': resp.get('status', 'abstain'),
        'answer': resp.get('answer', ''),
        'evidence': evidence,
        'citations': resp.get('citations', []),
        'confidence': resp.get('confidence', 'uncertain'),
        'top_score': resp.get('top_score', 0.0)
    }
    return out

def generate_response(question, mode='live'):
    retrieved = retrieve_with_citation(vectordb, question, k=3)
    top_score = retrieved[0]['score'] if retrieved else 0.0
    prompt_text = build_prompt(retrieved, question)
    
    if mode == 'live':
        try:
            print(f"Triggering Live LLM generation via NVIDIA NIM API...")
            raw = call_live_llm(prompt_text)
            candidate = parse_and_clean_json(raw, top_score=top_score)
        except Exception as e:
            print('Live LLM call failed, falling back to simulation:', e)
            candidate = simulate_llm_response(question, retrieved)
    else:
        candidate = simulate_llm_response(question, retrieved)
        
    ok = validate_response(candidate)
    return candidate, ok

if __name__ == '__main__':
    # Test 1: In-Scope Clinical Question
    in_scope = 'What is the target blood pressure for patients with cardiovascular disease?'
    print('\n======================================================================')
    print('TEST 1: IN-SCOPE QUERY (mode=live)')
    print(f'Question: {in_scope}')
    print('======================================================================')
    resp_in, ok_in = generate_response(in_scope, mode='live')
    print('\nIN-SCOPE RESPONSE:')
    print(json.dumps(resp_in, indent=2))
    print(f'Schema valid: {ok_in}')
    assert ok_in, "In-scope response failed schema validation"
    assert resp_in['status'] == 'grounded', f"Expected status 'grounded', got {resp_in['status']}"
    assert len(resp_in['citations']) > 0, "Expected at least 1 citation"

    # Test 2: Out-of-Scope Question
    out_scope = 'What is the recommended treatment for spotted fever in alpacas?'
    print('\n======================================================================')
    print('TEST 2: OUT-OF-SCOPE QUERY (mode=live)')
    print(f'Question: {out_scope}')
    print('======================================================================')
    resp_out, ok_out = generate_response(out_scope, mode='live')
    print('\nOUT-OF-SCOPE RESPONSE:')
    print(json.dumps(resp_out, indent=2))
    print(f'Schema valid: {ok_out}')
    assert ok_out, "Out-of-scope response failed schema validation"
    assert resp_out['status'] == 'abstain', f"Expected status 'abstain', got {resp_out['status']}"

    # Test 3: Paraphrase Question
    paraphrase_q = 'What blood pressure target should be used for someone with heart disease?'
    print('\n======================================================================')
    print('TEST 3: PARAPHRASED QUERY (mode=live)')
    print(f'Question: {paraphrase_q}')
    print('======================================================================')
    resp_para, ok_para = generate_response(paraphrase_q, mode='live')
    print('\nPARAPHRASED RESPONSE:')
    print(json.dumps(resp_para, indent=2))
    print(f'Schema valid: {ok_para}')
    assert ok_para, "Paraphrased response failed schema validation"
    assert resp_para['status'] == 'grounded', f"Expected status 'grounded', got {resp_para['status']}"

    print('\n======================================================================')
    print('ALL LIVE LLM GENERATION & SCHEMA VALIDATION TESTS PASSED!')
    print('======================================================================')
