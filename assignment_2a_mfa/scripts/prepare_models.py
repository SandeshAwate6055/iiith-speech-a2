"""Download and package the pinned official ARPA snapshot; no raw audio involved."""
import hashlib
import json
import zipfile
from pathlib import Path
from huggingface_hub import snapshot_download
import huggingface_hub.file_download as file_download

REPO='MontrealCorpusTools/english_us_arpa'
REVISION='9c8d5d8e1dd5d9cf188a5571203de0e09ca6bede'


def main():
    # Standard file copies work without Windows symlink privileges.
    file_download.are_symlinks_supported=lambda *args,**kwargs:False
    root=Path(__file__).resolve().parents[1]
    snap=Path(snapshot_download(REPO,revision=REVISION,allow_patterns=['README.md','acoustic/*','dictionary/*','g2p/english_us_arpa/*']))
    out=root/'work/models';out.mkdir(parents=True,exist_ok=True)
    for name,folder in [('hf_arpa','acoustic'),('hf_arpa_g2p','g2p/english_us_arpa')]:
        dest=out/(name+'.zip')
        if not dest.exists():
            with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as archive:
                for path in sorted((snap/folder).iterdir()):archive.write(path,name+'/'+path.name)
    (out/'hf_arpa.dict').write_bytes((snap/'dictionary/english_us_arpa.dict').read_bytes())
    record=dict(repo=REPO,revision=REVISION,acoustic_metadata=json.loads((snap/'acoustic/meta.json').read_text()),files={p.relative_to(snap).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(snap.rglob('*')) if p.is_file()})
    (out/'hf_source.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    evidence=root/'outputs/evaluation';evidence.mkdir(parents=True,exist_ok=True)
    (evidence/'model_provenance.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(f'Prepared {REPO}@{REVISION}; byte hashes recorded')

if __name__=='__main__':main()
