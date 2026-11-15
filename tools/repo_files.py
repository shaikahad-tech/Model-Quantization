"""Distribution inputs; avoid recursive hashes and generated/runtime caches."""
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
GENERATED={'experiment-evidence.zip','RELEASE_HASHES.txt','evidence/public_manifest.sha256'}


def files():
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    return sorted({name for name in names if name and (ROOT/name).is_file() and not name.startswith('runs/')})
