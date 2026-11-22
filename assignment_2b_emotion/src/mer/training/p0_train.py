"""Resume-safe P0 fold training using the approved shared implementations.
Existing legacy runs are not adopted. Only completed folds with matching provenance
are skipped; interrupted folds restart deterministically from their initial seed.
"""
import argparse
import contextlib
import gc
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from mer.data.datasets import TESSDataset, collate_tess
from mer.training.runner import build_model, run_test, _ssl_dir_from_cfg, _tok_from_cfg
from mer.training.trainer import train_fold
from mer.utils.common import load_config, load_split, set_seed


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextlib.contextmanager
def run_lock(path):
    with path.open('a+b') as handle:
        handle.seek(0)
        handle.write(b'0')
        handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def run(mode, config, protocol):
    torch.set_num_threads(4)
    cfg = load_config(config)
    name = f"{cfg['name']}_proto{protocol}"
    incident = Path('Results/audits/process_collision/incident.json')
    if incident.exists():
        event = json.loads(incident.read_text(encoding='utf-8-sig'))
        if name in event.get('affected_runs', []) and event.get('status') != 'RECONCILED':
            raise RuntimeError('This run has invalidated concurrent-writer provenance; reconcile final checkpoints first.')
    out = Path(cfg.get('out_dir', 'Results/per_fold')) / name
    checkpoints = Path(cfg.get('ckpt_dir', 'checkpoints')) / name
    checkpoints.mkdir(parents=True, exist_ok=True)
    with run_lock(checkpoints / 'p0.lock'):
        out.mkdir(parents=True, exist_ok=True)
        source = Path(__file__).resolve().parents[1]
        source_files = [*sorted((source / 'models').glob('*.py')), source / 'training/trainer.py', source / 'data/datasets.py', Path(__file__).resolve()]
        identity = dict(mode=mode, protocol=protocol, cfg=cfg, metadata_sha256=sha(cfg['data'].get('meta', 'data/metadata.csv')), split_sha256=sha(Path(cfg['data'].get('splits_dir', 'splits')) / f'protocol_{protocol}.json'), source_sha256={str(p.relative_to(source)): sha(p) for p in source_files})
        provenance = out / 'p0_provenance.json'
        if provenance.exists():
            old = json.loads(provenance.read_text())
            if old['identity'] != identity:
                raise RuntimeError('Inputs/code/config changed: use a distinct experiment name.')
        else:
            if list(checkpoints.glob('fold_*.pt')) or (out / 'summary.json').exists():
                raise RuntimeError('Unowned existing run: refusing to overwrite or adopt checkpoints.')
            git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
            atomic_json(provenance, dict(identity=identity, git_head=git.stdout.strip(), python=platform.python_version(), torch=torch.__version__, device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', note='Source hashes include uncommitted implementation; hashes are authoritative.'))
        atomic_json(out / 'config_snapshot.json', dict(mode=mode, protocol=protocol, cfg=cfg))
        df = pd.read_csv(cfg['data'].get('meta', 'data/metadata.csv'))
        split = load_split(protocol, cfg['data'].get('splits_dir', 'splits'))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        history_path = out / 'train_history.json'
        histories = json.loads(history_path.read_text()) if history_path.exists() else {}
        for fold in split['folds']:
            k = fold['fold']
            checkpoint = checkpoints / f'fold_{k}.pt'
            marker = out / f'fold_{k}_complete.json'
            if marker.exists():
                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)
                print(f'[{name}] fold {k} already complete', flush=True)
                continue
            seed = int(cfg['train'].get('seed', 42)) + k
            set_seed(seed)
            kwargs = dict(mode=mode, ssl_dir=_ssl_dir_from_cfg(mode, cfg), tokenizer_name=_tok_from_cfg(mode, cfg))
            train = TESSDataset(df, fold['train'], **kwargs)
            val = TESSDataset(df, fold['val'], **kwargs)
            bs = int(cfg['train'].get('batch_size', 32))
            tl = DataLoader(train, batch_size=bs, shuffle=True, collate_fn=collate_tess, num_workers=0)
            vl = DataLoader(val, batch_size=bs, shuffle=False, collate_fn=collate_tess, num_workers=0)
            model = build_model(mode, cfg)
            train_cfg = dict(cfg['train'], seed=seed)
            model, history, epoch, f1 = train_fold(model, tl, vl, mode, train_cfg, checkpoint, device)
            histories[f'fold_{k}'] = history
            atomic_json(history_path, histories)
            atomic_json(marker, dict(fold=k, seed=seed, best_epoch=int(epoch), validation_macro_f1=float(f1), completed_epochs=len(history)))
            print(f'[{name}] fold {k}: epoch={epoch}, val_F1={f1:.5f}', flush=True)
            del model, train, val, tl, vl
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        if not (out / 'summary.json').exists():
            run_test(mode, config, protocol)
        predictions = pd.read_csv(out / 'predictions.csv')
        predictions[['fold', 'utt_id']].to_csv(out / 'embedding_ids.csv', index=False)
        print(f'[{name}] complete', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['text', 'fusion'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--protocol', required=True, choices=list('ABC'))
    args = parser.parse_args()
    run(args.mode, args.config, args.protocol)

if __name__ == '__main__':
    main()
