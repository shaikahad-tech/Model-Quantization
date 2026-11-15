import hashlib
from repo_files import ROOT,GENERATED,files


def content():
    return ''.join(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()+'  '+p+'\n' for p in files() if p not in GENERATED)


if __name__=='__main__':
    (ROOT/'evidence/public_manifest.sha256').write_text(content())
