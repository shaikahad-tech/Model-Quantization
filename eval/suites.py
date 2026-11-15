"""Frozen licensed JSONL suites; no silent sampling or dropped rows."""
import hashlib
import json
from pathlib import Path


def load_suite(directory,name):
    directory=Path(directory);path=directory/(name+'.jsonl');raw=path.read_bytes()
    manifest=json.loads((directory/'manifest.json').read_text());entry=manifest['suites'][name]
    digest=hashlib.sha256(raw).hexdigest()
    if digest!=entry['sha256']:raise ValueError('Suite hash mismatch')
    if not entry.get('license') or entry.get('license') in ['unknown','NOT VERIFIED'] or entry.get('gated') or 'nc' in entry['license'].lower():raise ValueError('Suite license/gating not cleared')
    items=[json.loads(x) for x in raw.decode().split('\n') if x]
    if len(items)!=entry['count'] or len({r['id'] for r in items})!=len(items):raise ValueError('Suite count/ID mismatch')
    for item in items:
        if not isinstance(item['prompt'],str) or not item['prompt'].strip() or not item.get('scorer'):raise ValueError('Invalid suite item')
    return items,dict(entry,content_sha256=digest)
