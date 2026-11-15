"""Read-only audit: computed metrics, frozen history, generated docs, archive and hashes."""
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from metrics import compute,render
from gen_status import render as render_status
from make_manifest import content as manifest_content
from redact_check import scan
from repo_files import files


def audit():
    facts=compute();errors=[]
    status=json.loads((ROOT/'status.json').read_text())
    if (ROOT/'status.md').read_text()!=render_status(status):errors.append('Generated status.md differs from status.json')
    for source in (ROOT/'docs/templates').glob('*.md'):
        target={'memory.md':ROOT/'docs/memory.md'}.get(source.name,ROOT/source.name)
        if target.read_text()!=render(source.read_text(),facts):errors.append('Generated numbers/content drift: '+str(target.relative_to(ROOT)))
    if (ROOT/'evidence/public_manifest.sha256').read_text()!=manifest_content():errors.append('Manifest differs from current distributable file bytes')
    expected=set(files())-{'experiment-evidence.zip','RELEASE_HASHES.txt'}
    with zipfile.ZipFile(ROOT/'experiment-evidence.zip') as z:
        if set(z.namelist())!=expected:errors.append('ZIP file list differs from distribution tree')
        for name in sorted(set(z.namelist())&expected):
            if z.read(name)!=(ROOT/name).read_bytes():errors.append('ZIP byte mismatch: '+name)
        if z.testzip() is not None:errors.append('ZIP CRC error')
    for line in (ROOT/'RELEASE_HASHES.txt').read_text().split('\n'):
        if not line:continue
        digest,name=line.split('  ',1)
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:errors.append('Release hash mismatch: '+name)
    problems=scan()
    if problems:errors.append('Redaction findings: '+json.dumps(problems))
    for name in ['LICENSE','NOTICE.md','DATA_LICENSES.md']:
        if not (ROOT/name).is_file():errors.append('Missing '+name)
    from pypdf import PdfReader
    reader=PdfReader(ROOT/'report.pdf');pages=len(reader.pages)
    if reader.metadata.subject!='report.md SHA-256: '+hashlib.sha256((ROOT/'report.md').read_bytes()).hexdigest():errors.append('PDF source hash differs from report.md')
    if pages>2:errors.append('Report exceeds two pages')
    original='1e439acc6ac15a280b802adbc90a1e2b407068b9'
    old_names=subprocess.check_output(['git','ls-tree','-r','--name-only',original],cwd=ROOT,text=True).split('\n')
    frozen=[n for n in old_names if n.startswith('evidence/') and n!='evidence/public_manifest.sha256']+['code/verify_artifacts.py','code/trainer.yaml','data/sampled_dataset.zip','data/ship_tasks.jsonl','code/experiment_notebook_redacted.ipynb','code/verify_artifacts_100.py']
    frozen_checked=0
    for name in frozen:
        old=subprocess.check_output(['git','show',original+':'+name],cwd=ROOT)
        current=ROOT/('code/historical/verify_artifacts_100.py' if name=='code/verify_artifacts_100.py' else name)
        if name in ['code/experiment_notebook_redacted.ipynb','evidence/initial_failed_train_excerpt_redacted.txt']:
            old=old.replace(b'trainer'+b'-cli[train]',b'[vendor-package-redacted][train]').replace(b'https://github.com/'+b'MakazhanAlpamys/Trainer',b'[vendor-repo-redacted]')
        if current.read_bytes()!=old:errors.append('Unexpected historical modification: '+name)
        frozen_checked+=1
    return {'passed':not errors,'computed':facts,'pdf_pages':pages,'frozen_files_checked':frozen_checked,
            'missing_work':{k:v for k,v in status['work'].items() if v['status'] in ['NOT RUN','NOT BUILT','UNVERIFIED']},'errors':errors}


if __name__=='__main__':
    try:result=audit()
    except Exception as error:result={'passed':False,'errors':[str(error)]}
    print(json.dumps(result,indent=2));raise SystemExit(not result['passed'])
