"""Scan distributable text and extracted PDF/ZIP text; exceptions are exact-match scoped."""
import hashlib
import io
import json
import re
import zipfile
from repo_files import ROOT,files

PATTERNS={
    'vendor_package':re.compile(r'trainer'+r'-cli',re.I),
    'vendor_repository':re.compile(r'github\.com/'+r'MakazhanAlpamys/[^\s"\\]+',re.I),
    'token':re.compile(r'hf_[A-Za-z0-9]{20,}'),
    'email':re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
    'home_path':re.compile(r'/(?:Users|home)/[^\s"\\]+'),
}


def text_parts(name,raw):
    if name.endswith('.pdf'):
        from pypdf import PdfReader
        yield name,'\n'.join(p.extract_text() for p in PdfReader(io.BytesIO(raw)).pages)
    elif name.endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for item in z.namelist():
                for result in text_parts(name+'!'+item,z.read(item)):yield result
    else:
        try:yield name,raw.decode('utf-8')
        except UnicodeDecodeError:return


def scan(include_archive=False):
    allow=json.loads((ROOT/'tools/redaction_allowlist.json').read_text());failures=[]
    for name in files():
        if name=='experiment-evidence.zip' and not include_archive:continue
        for location,text in text_parts(name,(ROOT/name).read_bytes()):
            for kind,pattern in PATTERNS.items():
                for match in pattern.finditer(text):
                    rule={'path':location,'kind':kind,'match_sha256':hashlib.sha256(match.group().encode()).hexdigest()}
                    if not any(all(item.get(k)==v for k,v in rule.items()) for item in allow):
                        failures.append(dict(rule,line=text.count('\n',0,match.start())+1))
    return failures


if __name__=='__main__':
    problems=scan();print(json.dumps({'passed':not problems,'failures':problems},indent=2))
    raise SystemExit(bool(problems))
