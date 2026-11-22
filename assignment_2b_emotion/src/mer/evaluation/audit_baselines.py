"""Independent CPU-only baseline audit; never replaces original runs.
Run from assignment_2b_emotion: python -m mer.evaluation.audit_baselines
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


def metrics(y, p):
    return dict(accuracy=float(accuracy_score(y, p)), macro_f1=float(f1_score(y, p, labels=range(7), average='macro', zero_division=0)), uar=float(recall_score(y, p, labels=range(7), average='macro', zero_division=0)))


def main():
    threadpool_limits(1)
    out = Path('Results/audits/baseline_recovery')
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv('data/metadata.csv').set_index('utt_id')
    e = pd.read_csv('data/features/egemaps.csv').set_index('utt_id')
    assert d.index.is_unique and e.index.is_unique
    assert set(d.index) == set(e.index) and e.shape == (2800, 88)
    assert np.isfinite(e.values).all()
    expected = dict(enumerate(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']))
    assert all(d.emotion == d.emotion_id.map(expected))
    counts = pd.crosstab(d.transcript, d.emotion_id)
    assert counts.shape == (200, 7) and (counts.values == 2).all()
    rows, checks, hashes = [], [], {}
    for path in [Path('data/metadata.csv'), Path('data/features/egemaps.csv'), *sorted(Path('splits').glob('*.json'))]:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for protocol in 'ABC':
        folds = json.loads(Path(f'splits/protocol_{protocol}.json').read_text())['folds']
        for f in folds:
            tr, va, te = f['train'], f['val'], f['test']
            assert len(set(tr + va + te)) == len(tr + va + te)
            if protocol in 'AB':
                assert not set(d.loc[tr + va, 'word']) & set(d.loc[te, 'word'])
            if protocol == 'B':
                assert not set(d.loc[tr + va, 'speaker']) & set(d.loc[te, 'speaker'])
            for modality in ['speech', 'text']:
                saved = Path(f'Results/per_fold/classical_{modality}_proto{protocol}')
                pdf = pd.read_csv(saved / 'predictions.csv')
                pf = pdf[pdf.fold == f['fold']].set_index('utt_id').loc[te]
                assert list(pf.y_true) == list(d.loc[te, 'emotion_id'])
                sm = json.loads((saved / 'summary.json').read_text())
                fm = next(x for x in sm['folds'] if x['fold'] == f['fold'])
                measured = metrics(pf.y_true, pf.y_pred)
                assert all(abs(measured[k] - fm[k]) < 0.000011 for k in measured)
                checks.append(dict(protocol=protocol, fold=f['fold'], modality=modality, saved_metrics_valid=True))
                # Preserve train+val fit for exact reproduction; also audit train-only fairness.
                for fit_on, ids in [('train_plus_val', tr + va), ('train_only', tr)]:
                    y = d.loc[te, 'emotion_id']
                    if modality == 'speech':
                        model = make_pipeline(StandardScaler(), SVC(C=10, probability=True, random_state=42))
                        model.fit(e.loc[ids], d.loc[ids, 'emotion_id'])
                        predictions = {'decision': model.predict(e.loc[te]), 'probability_argmax': model.classes_[model.predict_proba(e.loc[te]).argmax(1)]}
                    else:
                        model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000, random_state=42))
                        model.fit(d.loc[ids, 'transcript'], d.loc[ids, 'emotion_id'])
                        predictions = {'decision': model.predict(d.loc[te, 'transcript'])}
                    for rule, pred in predictions.items():
                        row = dict(protocol=protocol, fold=f['fold'], modality=modality, fit_on=fit_on, prediction_rule=rule, **metrics(y, pred))
                        if modality == 'text' and protocol == 'C':
                            c = pd.crosstab(d.loc[ids, 'transcript'], d.loc[ids, 'emotion_id']).reindex(columns=range(7), fill_value=0)
                            row['true_label_below_train_text_majority'] = float(np.mean([c.loc[d.loc[i, 'transcript'], d.loc[i, 'emotion_id']] < c.loc[d.loc[i, 'transcript']].max() for i in te]))
                        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / 'fold_metrics.csv', index=False)
    group = table.groupby(['protocol', 'modality', 'fit_on', 'prediction_rule'])[['accuracy', 'macro_f1', 'uar']].agg(['mean', lambda x: np.std(x, ddof=0)])
    group.to_csv(out / 'summary.csv')
    (out / 'validation.json').write_text(json.dumps(dict(input_sha256=hashes, metadata_rows=len(d), feature_dimensions=88, saved_fold_checks=checks), indent=2))
    (out / 'interpretation.md').write_text('''# Independent baseline recovery audit

Original predictions and summaries are preserved. See fold_metrics.csv and summary.csv.

All 2,800 IDs are unique and match the feature table; all 88 features are finite. Label mapping matches the shared emotion order. Each of 200 transcripts occurs twice in each of seven classes. Saved per-fold accuracy, macro-F1 and UAR are independently recomputed and checked. Existing split manifests are used unchanged.

## Text random-split result

For a transcript with two recordings per emotion, removing one test item reduces its true-label count in the training complement. A lexical classifier can learn this complementary imbalance and systematically predict another emotion. The audit measures this for each fold. Thus population P(label|text)=1/7 does NOT imply chance test accuracy under every finite, dependent partition. Zero percent is reproducible and is not by itself a label-mapping failure. Protocol A and B retain word-disjoint evaluation. Do not change the frozen random splits to obtain chance performance.

## Speaker-independent SVM

Both the direct SVM class decision and calibrated-probability argmax are evaluated. They need not agree. The legacy implementation uses probability argmax and assumes class indices equal labels (the assumption holds for the present seven-class folds). Direct predictions should use classifier.predict; any revised baseline must be separately named and rerun across all protocols. Poor cross-speaker performance remains after changing the decision rule. This audit does not establish that domain shift is the sole cause; audio preprocessing sensitivity and acoustic feature distributions still require investigation.

## Training-set comparability

Legacy classical baselines merge train and validation, whereas neural models fit only train and select checkpoints on validation. Train-only rows quantify this discrepancy without replacing results. Report fit policy explicitly; use train-only baselines for matched-data comparisons.

No test-based tuning was performed. CPU threads are limited to one; existing training, feature caches, checkpoints and original results are untouched.
''', encoding='utf-8')
    print(group.to_string())
    print(f'Validated {len(checks)} saved folds; wrote {out}')

if __name__ == '__main__':
    main()
