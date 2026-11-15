"""Small shared helpers for recorded, resumable stages."""
import datetime
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as out:
        json.dump(value,out,indent=2,sort_keys=True,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
    os.replace(tmp,path)


def load_config(path):
    # JSON is valid YAML 1.2 and avoids an unrecorded parser dependency.
    return json.loads(Path(path).read_text())


def inventory(directory):
    directory=Path(directory)
    files=[]
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():raise ValueError('Artifact symlink is not allowed: '+str(path))
        if path.is_file():files.append({'path':path.relative_to(directory).as_posix(),'bytes':path.stat().st_size,'sha256':sha256(path)})
    if not files:raise ValueError('Empty artifact: '+str(directory))
    return files


def check_files(root,files):
    root=Path(root).resolve()
    if not files:return False
    for item in files:
        path=(root/item['path']).resolve()
        if root not in path.parents or not path.is_file() or path.stat().st_size!=item['bytes'] or sha256(path)!=item['sha256']:return False
    return True


def valid_stage(run,stage):
    run=Path(run);record=run/f'stage_{stage}.json';marker=run/f'{stage}.complete'
    if not record.exists() or not marker.exists():return False
    if marker.read_text().strip()!=sha256(record):return False
    data=json.loads(record.read_text())
    return data.get('success') is True and check_files(run,data['files'])


def complete_stage(run,stage,files,metrics):
    run=Path(run)
    entries=[{'path':Path(p).relative_to(run).as_posix(),'bytes':Path(p).stat().st_size,'sha256':sha256(p)} for p in sorted(files)]
    if not entries or not check_files(run,entries):raise ValueError('Stage outputs missing')
    record=run/f'stage_{stage}.json'
    atomic_json(record,dict(metrics,success=True,stage=stage,completed_utc=utc(),files=entries))
    marker=run/f'{stage}.complete';temp=marker.with_suffix('.tmp');temp.write_text(sha256(record)+'\n');temp.replace(marker)
