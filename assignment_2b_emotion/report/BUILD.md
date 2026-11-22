# Report build contract

From the repo root, use Python with `reportlab`, `pymupdf`, `numpy`, `PyYAML` installed. For this machine the isolated interpreter is `assignment_2b_emotion/report/.build-env/Scripts/python.exe` (ignored; never commit it).

```powershell
python tools/build_reports.py --self-test
python tools/build_reports.py
python tools/build_reports.py --final --allow-provisional-2a --verification-root assignment_2b_emotion/Results/verified_p0
```

The default produces DRAFT PDF+MD in both component report directories. PyMuPDF renders every page into `report/qa/`; build manifests record source SHA256 hashes, pending requirements and PDF hashes. Reports are generated only from read-only evidence. No training or evaluation output is altered. The final gate runs before any report write. Without the provisional flag, missing Praat evidence blocks finalization; with it, 2A retains DRAFT labeling. There is no option to bypass missing 2B runs or the collision gate.

## Collision reconciliation

The original text checkpoint/P0-marker correspondence was invalidated by overlapping writers. Default report builds exclude affected text rows from completed results and remain provisional. Numerical consistency does not prove checkpoint provenance. Do not resume or adopt the invalidated P0 markers.

After ownership clears, main must re-evaluate final checkpoints into a distinct verification root. This root contains `accuracy_tables.csv`, `failure_cases.csv`, `plots/`, and all twelve canonical run directories (`<speech_wavlm_bigru_asp|text_roberta_top2ft|fusion_concat|fusion_gated_xattn>_proto<A|B|C>`). Each run contains summary.json, predictions.csv, embeddings.npy and verification.json. Rebuild plots from verified predictions; copy nothing merely to satisfy existence gates. Projection metrics JSON files must contain all three nonempty fold-0 blocks. The generator independently checks metadata labels, exact frozen fold membership, no split/word/speaker leakage, metrics, finite embeddings/order and five distinct real errors.

Each verification.json is written by the re-evaluation owner, not this report generator:

```json
{
  "queue_ownership_cleared": true,
  "reconciled_after_collision": true,
  "incident_sha256": "<SHA256 of Results/audits/process_collision/incident.json>",
  "prediction_sha256": "<predictions.csv SHA256>",
  "summary_sha256": "<summary.json SHA256>",
  "embeddings_sha256": "<embeddings.npy SHA256>",
  "checkpoint_sha256": {
    "assignment_2b_emotion/checkpoints/<run>/fold_0.pt": "<SHA256>",
    "assignment_2b_emotion/checkpoints/<run>/fold_1.pt": "<SHA256>"
  }
}
```

Include all five fold files for A/C and both for B. Paths are repository-root-relative. Hashes are checked against current bytes; this is an explicit owner attestation of re-evaluation correspondence, not proof of execution from hashes alone. Evidence must be within the repository workspace; audio/checkpoints remain ignored and private. Accuracy CSV uses numeric `accuracy_mean`, `accuracy_std`, `uar_mean`, `uar_std`, `macro_f1_mean`, `macro_f1_std`, exactly as the existing aggregate module emits. Failures match the existing failure_cases.csv schema, including fold/utt_id/y_true/y_pred and source run where present.

All mandatory figures are listed in `PLOTS` in the generator. Missing/corrupt inputs stay pending. Re-run final build after inspection/evaluation is reconciled. Current PDF drafts are not submission-ready.


Final recovery: both reports passed `--final --verification-root assignment_2b_emotion/Results/verified_p0` on22 November, with zero pending requirements. The former Praat interruption was resolved by scoped visual inspection and genuine captures. Historical collision records remain unchanged; final results use newly re-evaluated, hash-linked checkpoints. The final reports are3 pages (2A) and6 pages (2B). No automatic submission/public release is performed.
