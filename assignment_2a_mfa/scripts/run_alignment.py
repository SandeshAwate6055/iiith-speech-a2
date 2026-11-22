"""Reproducible P0 MFA pipeline; reuse completed corpus/models/alignments.
Run from assignment_2a_mfa with the ML Python. MFA must be on PATH, or pass
--mfa-env and --micromamba-root to use the existing activated micromamba env.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path)
    p.add_argument('--mfa-env',type=Path)
    p.add_argument('--micromamba-root',type=Path)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    def run(args):subprocess.run([str(x) for x in args],cwd=root,check=True)
    if not list((root/'corpus').rglob('*.lab')):
        if a.raw is None:raise ValueError('Prepared corpus absent; provide --raw with the six supplied pairs')
        run([sys.executable,'scripts/prepare_corpus.py','--raw',a.raw.resolve(),'--out','corpus'])
    if len(list((root/'corpus').rglob('*.lab')))!=6:raise ValueError('Expected exactly six supplied corpus pairs')
    run([sys.executable,'scripts/prepare_models.py'])
    mfa=['mfa']
    if a.mfa_env:
        if a.micromamba_root is None:raise ValueError('--micromamba-root required')
        mm=a.micromamba_root.parent/'Library/bin/micromamba.exe'
        mfa=[mm,'run','-r',a.micromamba_root,'-p',a.mfa_env,'mfa']
    from audit_lexicon import read_lexicon
    base,_=read_lexicon(root/'work/models/hf_arpa.dict')
    words=set(' '.join(f.read_text() for f in (root/'corpus').rglob('*.lab')).lower().split())
    (root/'work/oov_before.txt').write_text('\n'.join(sorted(words-base.keys()))+'\n')
    if not (root/'dictionaries/g2p_oov.dict').exists():
        run(mfa+['g2p','work/oov_before.txt','work/models/hf_arpa_g2p.zip','dictionaries/g2p_oov.dict','--no_use_mp','--temporary_directory','work/g2p'])
    run([sys.executable,'scripts/prepare_lexicon.py'])
    dest=root/'outputs/textgrids/english_us_arpa'
    grids=list(dest.rglob('*.TextGrid'))
    evidence = root/'outputs/evaluation/alignment_run_provenance.json'
    if grids and evidence.exists():
        provenance = json.loads(evidence.read_text())
        for relative, expected in provenance['input_sha256'].items():
            if hashlib.sha256((root/relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Existing alignment inputs changed: {relative}; choose a separate output directory')
    if grids and len(grids)!=6:raise RuntimeError('Partial existing alignment; inspect before choosing a new output directory')
    if not grids:
        run(mfa+['align','corpus','dictionaries/custom_lexicon.dict','work/models/hf_arpa.zip',dest,'--config_path','configs/pipeline.yaml','--temporary_directory','work/align_arpa','--num_jobs','2','--no_use_mp','--no_use_postgres','--no_tokenization'])
    run([sys.executable,'scripts/audit_lexicon.py','--dictionary','dictionaries/custom_lexicon.dict'])
    run([sys.executable,'scripts/evaluate_alignments.py'])
    run([sys.executable,'scripts/visualize.py'])
    print('Computational pipeline complete. Praat inspection and evidence must be completed separately.')

if __name__=='__main__':main()
