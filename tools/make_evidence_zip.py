import hashlib
import zipfile
from repo_files import ROOT,files
from make_manifest import content


def build():
    (ROOT/'evidence/public_manifest.sha256').write_text(content())
    with zipfile.ZipFile(ROOT/'experiment-evidence.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name in files():
            if name in ['experiment-evidence.zip','RELEASE_HASHES.txt']:continue
            info=zipfile.ZipInfo(name,date_time=(2000,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.create_system=3
            info.external_attr=((0o100755 if name.endswith('.sh') else 0o100644)<<16)
            archive.writestr(info,(ROOT/name).read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    hashes=''.join(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()+'  '+name+'\n' for name in ['evidence/public_manifest.sha256','experiment-evidence.zip'])
    (ROOT/'RELEASE_HASHES.txt').write_text(hashes)
    print(hashes,end='')


if __name__=='__main__':build()
