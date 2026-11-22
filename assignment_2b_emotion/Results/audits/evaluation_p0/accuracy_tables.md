# Prediction-derived evaluation

Percentages: unweighted fold mean +/- population fold SD (ddof=0), not confidence intervals. Only complete validated runs are included. No universal macro-F1 chance threshold is assumed.

| run | modality | protocol | n_folds | accuracy | uar | macro_f1 |
|---|---|---|---|---|---|---|
| classical_speech_protoA | speech | A | 5 | 98.43 +/- 0.58 | 98.43 +/- 0.58 | 98.43 +/- 0.58 |
| classical_speech_protoB | speech | B | 2 | 18.04 +/- 1.61 | 18.04 +/- 1.61 | 12.45 +/- 0.71 |
| classical_speech_protoC | speech | C | 5 | 98.54 +/- 0.50 | 98.54 +/- 0.50 | 98.55 +/- 0.49 |
| classical_text_protoA | text | A | 5 | 14.29 +/- 0.00 | 14.29 +/- 0.00 | 3.57 +/- 0.00 |
| classical_text_protoB | text | B | 2 | 14.29 +/- 0.00 | 14.29 +/- 0.00 | 3.57 +/- 0.00 |
| classical_text_protoC | text | C | 5 | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0.00 +/- 0.00 |
| speech_wavlm_bigru_asp_protoA | speech | A | 5 | 99.96 +/- 0.07 | 99.96 +/- 0.07 | 99.96 +/- 0.07 |
| speech_wavlm_bigru_asp_protoB | speech | B | 2 | 60.71 +/- 8.21 | 60.71 +/- 8.21 | 56.23 +/- 7.53 |
