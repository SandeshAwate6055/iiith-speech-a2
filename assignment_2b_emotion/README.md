# 2B: TESS speech, text and fusion

Run from `assignment_2b_emotion` with the root Python 3.10 environment and matching torch/torchaudio 2.5.1. PowerShell: `$env:PYTHONPATH = (Resolve-Path src).Path`; bash: `export PYTHONPATH="$PWD/src"`. Check GPU with `python -c "import torch; print(torch.cuda.is_available())"`. Keep this repository PRIVATE; no audio, features, caches, checkpoints or secrets in Git.

## Local preparation

The following is for an authorized new local preparation, not required for cached-feature Colab:

```powershell
python -m mer.data.download_tess --out data/raw/TESS
python -m mer.data.prepare_tess --raw data/raw/TESS --out data/metadata.csv
python -m pytest tests/test_splits.py -q
python -m mer.features.preprocess_audio --meta data/metadata.csv --out data/processed_16k
python -m mer.features.extract_egemaps --meta data/metadata.csv --out data/features/egemaps.csv
python -m mer.features.extract_ssl --meta data/metadata.csv --model microsoft/wavlm-base-plus --tag wavlm_base_plus --out data/features --batch_size 8
```

Preprocessing updates processed paths in metadata; local extraction requires the waveforms. Cached text/fusion training only requires metadata columns `utt_id, emotion_id, transcript, speaker, word`, frozen splits and (fusion) trusted tensors `data/features/wavlm_base_plus/<utt_id>.pt` shaped `[13, time, 768]`. Audit 2800 unique clips, 400 per class, 200 words, two speakers and finite exact-ID cache coverage. Preparation handles filename `OA_bite_neutral`. Never load untrusted tensors/checkpoints.

Use frozen `splits/protocol_{A,B,C}.json` unchanged. For a genuinely new dataset only: `python -m mer.data.make_splits --meta data/metadata.csv --out <new-splits-directory> --seed 42`. Do not replace benchmark splits.

| Protocol | Evaluation |
|---|---|
| A | Primary: five word-disjoint folds, 128/32/40 train/validation/test words, both speakers |
| B | Two speaker-transfer folds, also word-disjoint; test uses 40 held-out words on the other speaker |
| C | Optimistic random stratified five-fold reference, about 70/10/20% train/validation/test, shared words/speakers |

Two speakers cannot establish broad speaker generalization.

## Training and testing

Speech: learned WavLM layer sum, temporal BiGRU, attentive statistics pooling. Text uses canonical pretrained ID `FacebookAI/roberta-base`; the canonical snapshot is cached locally. Fusion provides concat and gated cross attention. Hugging Face weights/tokenizers must be available from an authorized cache or downloaded; local caches are not automatically present in Colab.

The P0 runner uses matching provenance and completed-fold markers. Its interface is:

```powershell
python -m mer.training.p0_train --mode text --config configs/text.yaml --protocol A
python -m mer.training.p0_train --mode fusion --config configs/fusion_concat.yaml --protocol A
python -m mer.training.p0_train --mode fusion --config configs/fusion.yaml --protocol A
```

Repeat for B/C only when GPU ownership permits. Check `python -m mer.training.p0_train --help`. Completed folds with matching provenance are skipped. Interrupted folds restart deterministically from their initial seed; there is no optimizer/epoch resume. Existing legacy checkpoints are not adopted. The runner locks each run and refuses changed inputs/code/config. The notebook additionally hashes every feature cache. Partial evaluation artifacts require a new attempt ID. The older shared trainer is not resume-safe.

The six assignment entry points are also available:

```powershell
python models/speech_pipeline/train.py --config configs/speech.yaml --protocol A
python models/speech_pipeline/test.py --config configs/speech.yaml --protocol A
python models/text_pipeline/train.py --config configs/text.yaml --protocol A
python models/text_pipeline/test.py --config configs/text.yaml --protocol A
python models/fusion_pipeline/train.py --config configs/fusion.yaml --protocol A
python models/fusion_pipeline/test.py --config configs/fusion.yaml --protocol A
```

Legacy training overwrites existing fold checkpoints. Use unique `--name` values matched between train/test, or copied configs with external private `ckpt_dir`/`out_dir`; preserve active/original runs. Legacy test requires every protocol fold checkpoint. Default run name: `<config name>_proto<protocol>`. Test writes `summary.json`, `predictions.csv`, `embeddings.npy` under `Results/per_fold/<run>`; training saves best checkpoints and validation history. Embeddings represent the temporal/contextual/fusion block. JSON metrics are fractions; tables show percentages and population standard deviation across folds.

Classical commands: `python -m mer.models.classical --modality speech --protocol A` and similarly `--modality text`. These overwrite default outputs; preserve originals. After completed tests: `python -m mer.evaluation.aggregate_results` and `python -m mer.evaluation.plots`. Failure-case plots need local audio. The final checkpoint-linked neural results, aggregate comparison and reports are complete; use the verified final recovery workflow below. Optional interrupted runs are excluded, not assigned zero scores.

## Audit interpretation

`python -m mer.evaluation.audit_baselines` and `python -m mer.evaluation.audit_feature_shift` write independent recovery audits under `Results/audits/baseline_recovery/`. They require existing metadata, eGeMAPS and saved baseline artifacts; feature reproduction also needs processed audio. Preserve original baseline rows.

Classical baselines fit **train+validation**, while neural models fit train and select checkpoints with validation. They have different fitting budgets. Train-only/direct-SVM audit comparisons are separate experiments, not replacements for original E1 rows. Legacy SVM uses probability argmax, which can differ from direct SVM prediction. Low cross-speaker RBF similarity is consistent with feature shift, not proof of its sole cause; do not tune on test data.

Every full-corpus transcript occurs with seven emotions, so text has no intrinsic label information and balanced population chance is 1/7. This does **not** imply chance for every dependent finite partition. Saved classical text Protocol C is 0% in every fold: each held-out true label is below its transcript's training-complement majority under train+validation fitting. This complementary-imbalance artifact can induce systematic below-chance prediction. It is not evidence of emotion understanding or grounds to change frozen splits. The recovery log records a near-zero train-only check too. This audit interpretation supersedes PLAN.md's assertion of chance under any split; neural performance must be measured separately.

## Private Colab

Open [notebooks/P0_colab.ipynb](notebooks/P0_colab.ipynb), select a GPU, provide a private-repo read token through hidden input and configure private Drive paths. No embedded secrets, public repository or raw/2A audio is needed. Use a revision containing the final P0 runner and canonical configs. Jobs are text, concat fusion and gated fusion across A/B/C. Outputs go to Drive outside Git, with provenance and input/config fingerprints. Completed jobs are skipped; interrupted jobs rely on the P0 runner's resume contract. Do not use two sessions for the same RUN_ID. Clear outputs before sharing. Live Colab execution remains unvalidated.


## Verified final recovery (22 November)

The canonical submission artifacts are `Results/accuracy_tables.{csv,md}`, `Results/plots/`, `Results/failure_cases.md`, `Results/comparative_analysis.md`, and `report/report_2b.pdf`. They are generated from the separate `Results/verified_p0/` outputs. Original `Results/per_fold/` files and all checkpoints are preserved. Verification JSON files bind final checkpoint bytes to re-evaluated predictions; historical text-training provenance was not reconstructed.

From the repository root, use `python tools/complete_model_verification.py` to verify existing weights and `python tools/finalize_evaluation.py` to rebuild tables/figures. Build the report with a Python containing the report dependencies:

```powershell
python tools/build_reports.py --final --verification-root assignment_2b_emotion/Results/verified_p0
```

All 12 speech/text/concat/gated headline combinations across A/B/C are available. Interrupted late fusion is optional and excluded. Gated C lacks historical epoch curves; final checkpoint inference is verified and no curve was invented. Per-class confusion/recall, five actual failures, paired help/harm counts and fold-0 t-SNE/UMAP for temporal/contextual/fused blocks are included. Do not interpret projected separation as independent predictive evidence.

The component requirements now match the actual installed versions; `environment_versions.json` and `requirements-lock-windows.txt` record the evaluated Windows environment. The latter is platform-specific, not a Colab lockfile. No dependencies were upgraded during recovery.

For a fresh training reproduction, prepare data/features first, then use a NEW experiment name to avoid the intentionally blocked historical text markers, for example:

```powershell
python models/text_pipeline/train.py --config configs/text.yaml --protocol A --name reproduce_text_A
python models/text_pipeline/test.py --config configs/text.yaml --protocol A --name reproduce_text_A
```

Use the analogous speech/fusion entry points and frozen split manifests. These commands were not rerun during recovery: the existing completed models were reused. Full training reproduction requires the documented datasets/features and suitable compute; checkpoint inference was reproduced locally.
