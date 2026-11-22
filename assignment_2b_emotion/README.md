# Assignment 2B: Multimodal Emotion Recognition (Speech, Text & Fusion)

This directory contains the speech, text, and multimodal fusion pipelines, frozen evaluation splits, verified results, visualizations, and final report for **Assignment 2B** on the Toronto Emotional Speech Set (TESS).

Run commands from `assignment_2b_emotion/` using the root Python 3.10 virtual environment (`torch==2.5.1`, `torchaudio==2.5.1`). Set `PYTHONPATH` to include `src`:

```powershell
# PowerShell
$env:PYTHONPATH = (Resolve-Path src).Path

# Bash
export PYTHONPATH="$PWD/src"
```

## 1. Dataset & Split Protocols

Download and prepare TESS (2,800 clips, 7 emotions × 400 clips, 200 target words, 2 female speakers OAF and YAF; Borealis DOI `10.5683/SP2/E8H2MF`):

```powershell
python -m mer.data.download_tess --out data/raw/TESS
python -m mer.data.prepare_tess --raw data/raw/TESS --out data/metadata.csv
python -m pytest tests/test_splits.py -q
python -m mer.features.preprocess_audio --meta data/metadata.csv --out data/processed_16k
python -m mer.features.extract_egemaps --meta data/metadata.csv --out data/features/egemaps.csv
python -m mer.features.extract_ssl --meta data/metadata.csv --model microsoft/wavlm-base-plus --tag wavlm_base_plus --out data/features --batch_size 8
```

All models are evaluated on three frozen, leakage-tested split protocols in [`splits/`](splits/):

| Protocol | Split Design | Purpose |
|---|---|---|
| **A** | 5-fold word-disjoint CV (128 train / 32 val / 40 test words; both speakers) | Primary lexical-generalization benchmark |
| **B** | 2-fold speaker-transfer (LOSO) + word-disjoint (40 held-out test words on unseen speaker) | Strict cross-speaker & lexical stress test |
| **C** | 5-fold random stratified CV (~70% train / 10% val / 20% test; shared words & speakers) | Optimistic in-distribution reference |

## 2. Pipeline Architectures

1. **Speech Pipeline (`configs/speech.yaml`)**:
   - **Preprocessing**: 16 kHz mono conversion, 30 dB silence trimming with 50 ms padding, -1 dBFS peak normalization.
   - **Features**: Frozen `microsoft/wavlm-base-plus` (`[13, T, 768]`) with a learnable softmax layer-importance weighted sum.
   - **Temporal Modelling & Classifier**: 256-d projection → 2-layer BiGRU (128 hidden/dir) → Attentive Statistics Pooling (mean + std) → MLP classifier.
2. **Text Pipeline (`configs/text.yaml`)**:
   - **Preprocessing & Features**: Standardized carrier transcript (`"Say the word <target>"`, emotion/speaker suffixes stripped) tokenized via `FacebookAI/roberta-base` (max length 32).
   - **Contextual Modelling & Classifier**: Top-2 RoBERTa transformer layers fine-tuned → 256-d projection → masked attention pooling → MLP classifier.
3. **Multimodal Fusion Pipelines (`configs/fusion_concat.yaml`, `configs/fusion.yaml`)**:
   - **Concat Fusion**: Concatenates pooled 256-d speech and 256-d text representations → MLP classifier.
   - **Gated Cross-Attention Fusion**: Bidirectional 4-head cross-modal attention between temporal speech and token-level text streams + learned sigmoid vector gate (`256-d`) → MLP classifier.

## 3. Training & Testing Entry Points

The six required deliverable scripts under [`models/`](models/) run training and evaluation across Protocols `A`, `B`, and `C`:

```powershell
# 1. Speech Pipeline
python models/speech_pipeline/train.py --config configs/speech.yaml --protocol A
python models/speech_pipeline/test.py --config configs/speech.yaml --protocol A

# 2. Text Pipeline
python models/text_pipeline/train.py --config configs/text.yaml --protocol A
python models/text_pipeline/test.py --config configs/text.yaml --protocol A

# 3. Multimodal Fusion Pipeline (Gated Cross-Attention or Concat)
python models/fusion_pipeline/train.py --config configs/fusion.yaml --protocol A
python models/fusion_pipeline/test.py --config configs/fusion.yaml --protocol A
python models/fusion_pipeline/train.py --config configs/fusion_concat.yaml --protocol A
python models/fusion_pipeline/test.py --config configs/fusion_concat.yaml --protocol A
```

*(Note: When running fresh training experiments, pass `--name <new_run_name>` to avoid overwriting existing benchmark runs. Deterministic lock-safe training is also available via `python -m mer.training.p0_train --mode <text|fusion> --config <cfg> --protocol <A|B|C>`.)*

Classical baselines (`eGeMAPS+SVM` and `TF-IDF+LR`):

```powershell
python -m mer.models.classical --modality speech --protocol A
python -m mer.models.classical --modality text --protocol A
```

## 4. Verified Results & Deliverables

All 12 neural headline configurations were independently re-evaluated from saved checkpoints into [`Results/verified_p0/`](Results/verified_p0/) with SHA256 checkpoint-to-prediction verification records (`verification.json`).

- **Consolidated Accuracy Tables**: [`Results/accuracy_tables.md`](Results/accuracy_tables.md) and [`Results/accuracy_tables.csv`](Results/accuracy_tables.csv)
- **Per-Emotion & Help/Harm Comparative Analysis**: [`Results/comparative_analysis.md`](Results/comparative_analysis.md) and [`Results/per_emotion_validated.csv`](Results/per_emotion_validated.csv)
- **Failure Case Analysis (5 Prediction-Backed Cases)**: [`Results/failure_cases.md`](Results/failure_cases.md), [`Results/failure_cases.csv`](Results/failure_cases.csv), and [`Results/plots/failure_cases.png`](Results/plots/failure_cases.png)
- **Embedding Separability (Temporal, Contextual, Fusion)**: Within-fold UMAP ([`Results/plots/embeddings_umap_protoA.png`](Results/plots/embeddings_umap_protoA.png)) and t-SNE ([`Results/plots/embeddings_tsne_protoA.png`](Results/plots/embeddings_tsne_protoA.png))
- **Final PDF Report**: [`report/report_2b.pdf`](report/report_2b.pdf) (6 pages)

### Key Scientific Notes & Limitations
1. **Uninformative Text & Complementary Imbalance**: Every TESS target word occurs exactly once per emotion per speaker (full-corpus chance = $1/7 \approx 14.29\%$). Under random stratified splits without word disjointness (Protocol C), each held-out sample's true emotion appears one fewer time for that word in the training complement, causing classical train+val `TF-IDF+LR` to predict the complement majority and achieve `0.00%` accuracy.
2. **Classical vs. Neural Fitting Budget**: Classical baselines fit `train+val`, whereas neural models fit `train` only and use `val` for early stopping.
3. **Cross-Speaker Generalization (Protocol B)**: With only two female speakers, Protocol B is a 2-speaker transfer stress test rather than a population estimate. Speech achieves `60.71%` accuracy (`56.23%` Macro-F1), with `100%` fear recall but `0%` happiness recall due to cross-speaker pitch/arousal shift.
4. **Provenance Disclosure**: Historical text-training lineage experienced a multi-process file collision; invalidated P0 completion markers are not used. All reported metrics in [`Results/`](Results/) and [`report/report_2b.pdf`](report/report_2b.pdf) are derived from independent checkpoint re-evaluation in [`Results/verified_p0/`](Results/verified_p0/).
