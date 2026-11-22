"""Re-evaluate final checkpoints into distinct outputs after concurrent writers stop.
Does not train, alter legacy run files, or adopt invalidated P0 markers.
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
import torch
from mer.training.runner import run_test
from mer.utils.common import load_config,load_split
from mer.evaluation.validate_evaluation import validate_run


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def require_idle(queue_pid):
    if os.name!='nt':raise RuntimeError('This ownership guard is Windows-specific; do not bypass on another host')
    command=f"$q=Get-CimInstance Win32_Process -Filter 'ProcessId={queue_pid}'; if($q.CommandLine -notmatch 'models/(text|fusion)_pipeline|continue_p0.ps1'){{$q=$null}}; $p=Get-CimInstance Win32_Process | Where-Object {{ $_.Name -eq 'python.exe' -and $_.CommandLine -match '(models[/\\\\](speech|text|fusion)_pipeline[/\\\\](train|test).py|mer.training.p0_train)' }}; if($q -or $p){{exit 17}}"
    result=subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True,text=True)
    if result.returncode:raise RuntimeError('Legacy queue/training still active, or ownership check failed; no evaluation started')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-pid',type=int,default=10420)
    parser.add_argument('--out',type=Path,default=Path('Results/verified_p0'))
    args=parser.parse_args()
    require_idle(args.queue_pid)
    root=Path.cwd().parent
    incident=Path('Results/audits/process_collision/incident.json')
    if not incident.is_file():raise FileNotFoundError(incident)
    jobs=[]
    for mode,config in [('speech','configs/speech.yaml'),('text','configs/text.yaml'),('fusion','configs/fusion_concat.yaml'),('fusion','configs/fusion.yaml')]:
        cfg=load_config(config)
        for protocol in 'ABC':
            name=f"{cfg['name']}_proto{protocol}"
            files=[Path(cfg.get('ckpt_dir','checkpoints'))/name/f"fold_{f['fold']}.pt" for f in load_split(protocol)['folds']]
            if not all(p.is_file() for p in files):raise FileNotFoundError(f'Incomplete final checkpoint set: {name}; finish missing P0 training first')
            jobs.append((mode,config,protocol,name,files))
    torch.set_num_threads(4)
    args.out.mkdir(parents=True,exist_ok=True)
    source=list(Path('src/mer/models').glob('*.py'))+[Path('src/mer/training/trainer.py'),Path('src/mer/training/runner.py'),Path('src/mer/data/datasets.py')]
    for mode,config,protocol,name,files in jobs:
        require_idle(args.queue_pid)
        stable=[*files,*source,Path(config),Path('data/metadata.csv'),Path(f'splits/protocol_{protocol}.json')]
        before={p.as_posix():sha(p) for p in stable}
        out=(args.out/name).resolve()
        marker=out/'verification.json'
        if marker.exists():
            previous=json.loads(marker.read_text())
            if previous.get('input_sha256')!=before:raise RuntimeError(f'Verified inputs changed for {name}; choose a fresh verification root')
            validate_run(out)
            print(f'{name}: existing verified evaluation preserved',flush=True)
            continue
        if out.exists() and any(out.iterdir()):raise RuntimeError(f'Incomplete prior verification at {out}; inspect before choosing a fresh root')
        run_test(mode,config,protocol,save_tag=str(out))
        require_idle(args.queue_pid)
        after={p.as_posix():sha(p) for p in stable}
        if before!=after:raise RuntimeError('Checkpoint/code/input changed during evaluation. Outputs remain unverified.')
        import pandas as pd
        predictions=pd.read_csv(out/'predictions.csv')
        predictions[['fold','utt_id']].to_csv(out/'embedding_ids.csv',index=False)
        validate_run(out)
        record=dict(queue_ownership_cleared=True,reconciled_after_collision=True,incident_sha256=sha(incident),prediction_sha256=sha(out/'predictions.csv'),summary_sha256=sha(out/'summary.json'),embeddings_sha256=sha(out/'embeddings.npy'),checkpoint_sha256={p.resolve().relative_to(root).as_posix():sha(p) for p in files},input_sha256=before,note='run_test executed from these stable final checkpoint bytes; hashes checked before and after evaluation. Original training lineage cannot be recovered from invalidated P0 markers.')
        marker.write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(f'{name}: checkpoint-linked re-evaluation verified',flush=True)
    print('All12 checkpoint-linked evaluations complete; regenerate tables/plots/failure cases within the verification root before final report build.')

if __name__=='__main__':main()
