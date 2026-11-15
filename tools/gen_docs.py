from pathlib import Path
from metrics import ROOT,compute,render
from gen_status import render as render_status
import json


def generate():
    metrics=compute()
    for source in (ROOT/'docs/templates').glob('*.md'):
        target={'memory.md':ROOT/'docs/memory.md'}.get(source.name,ROOT/source.name)
        target.write_text(render(source.read_text(),metrics))
    (ROOT/'status.md').write_text(render_status(json.loads((ROOT/'status.json').read_text())))


if __name__=='__main__':generate()
