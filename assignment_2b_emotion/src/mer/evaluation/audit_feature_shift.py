"""Recheck sampled cached features, baseline aggregation and LOSO RBF shift.
Run from assignment_2b_emotion: python -m mer.evaluation.audit_feature_shift
"""
import json
from pathlib import Path
import numpy as np
import opensmile
import pandas as pd
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


def main():
    threadpool_limits(1)
    d = pd.read_csv('data/metadata.csv').set_index('utt_id')
    e = pd.read_csv('data/features/egemaps.csv').set_index('utt_id')
    result = dict(feature_rechecks=[], speaker_shift=[], aggregation_checks=[])
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
    for _, group in d.groupby('speaker'):
        for uid, row in group.iloc[[0, -1]].iterrows():
            path = Path(row.proc_path)
            values = smile.process_file(str(path)).iloc[0].values
            reference = e.loc[uid].values
            assert np.allclose(values, reference, rtol=1e-4, atol=1e-5)
            audio, sr = sf.read(path)
            assert sr == 16000 and audio.ndim == 1 and np.isfinite(audio).all()
            result['feature_rechecks'].append(dict(utt_id=uid, sample_rate=sr, max_abs_feature_error=float(np.max(np.abs(values-reference))), peak=float(np.max(np.abs(audio)))))
    for fold in json.loads(Path('splits/protocol_B.json').read_text())['folds']:
        tr, te = fold['train'], fold['test']
        scaler = StandardScaler().fit(e.loc[tr])
        a, b = scaler.transform(e.loc[tr]), scaler.transform(e.loc[te])
        model = SVC(C=10, probability=False, random_state=42).fit(a, d.loc[tr, 'emotion_id'])
        nearest = ((b[:, None, :] - a[None, :, :])**2).sum(2).min(1)
        similarity = np.exp(-model._gamma * nearest)
        result['speaker_shift'].append(dict(fold=fold['fold'], train_speaker=fold['train_speaker'], test_speaker=fold['test_speaker'], gamma=float(model._gamma), median_nearest_train_squared_distance=float(np.median(nearest)), median_max_rbf_similarity=float(np.median(similarity)), fraction_max_rbf_similarity_below_001=float(np.mean(similarity < .01)), fraction_test_feature_values_outside_train_range=float(np.mean((b < a.min(0)) | (b > a.max(0))))))
    for protocol in 'ABC':
        for modality in ['speech', 'text']:
            summary = json.loads(Path(f'Results/per_fold/classical_{modality}_proto{protocol}/summary.json').read_text())
            for metric in ['accuracy', 'uar', 'macro_f1']:
                values = [f[metric] for f in summary['folds']]
                assert abs(np.mean(values) - summary[metric + '_mean']) <= .000051
                assert abs(np.std(values, ddof=0) - summary[metric + '_std']) <= .000051
            result['aggregation_checks'].append(dict(protocol=protocol, modality=modality, mean_and_population_std_valid=True))
    out = Path('Results/audits/baseline_recovery/feature_shift_checks.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f'Passed 4 feature rechecks and 6 aggregation checks; wrote {out}')

if __name__ == '__main__':
    main()
