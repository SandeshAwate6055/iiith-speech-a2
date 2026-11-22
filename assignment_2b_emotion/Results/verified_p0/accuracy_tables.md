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
| fusion_concat_protoA | fusion | A | 5 | 99.82 +/- 0.11 | 99.82 +/- 0.11 | 99.82 +/- 0.11 |
| fusion_concat_protoB | fusion | B | 2 | 43.75 +/- 5.89 | 43.75 +/- 5.89 | 34.44 +/- 4.66 |
| fusion_concat_protoC | fusion | C | 5 | 99.71 +/- 0.27 | 99.71 +/- 0.27 | 99.72 +/- 0.27 |
| fusion_gated_xattn_protoA | fusion | A | 5 | 99.79 +/- 0.13 | 99.79 +/- 0.13 | 99.79 +/- 0.13 |
| fusion_gated_xattn_protoA_zerotext | fusion | A | 5 | 91.96 +/- 8.82 | 91.96 +/- 8.82 | 90.79 +/- 10.26 |
| fusion_gated_xattn_protoB | fusion | B | 2 | 52.14 +/- 5.36 | 52.14 +/- 5.36 | 46.40 +/- 1.44 |
| fusion_gated_xattn_protoB_zerotext | fusion | B | 2 | 54.64 +/- 7.14 | 54.64 +/- 7.14 | 48.94 +/- 5.68 |
| fusion_gated_xattn_protoC | fusion | C | 5 | 99.82 +/- 0.28 | 99.82 +/- 0.28 | 99.82 +/- 0.28 |
| speech_wavlm_bigru_asp_protoA | speech | A | 5 | 99.96 +/- 0.07 | 99.96 +/- 0.07 | 99.96 +/- 0.07 |
| speech_wavlm_bigru_asp_protoB | speech | B | 2 | 60.71 +/- 8.21 | 60.71 +/- 8.21 | 56.23 +/- 7.53 |
| speech_wavlm_bigru_asp_protoC | speech | C | 5 | 99.86 +/- 0.13 | 99.86 +/- 0.13 | 99.86 +/- 0.13 |
| text_roberta_top2ft_protoA | text | A | 5 | 14.29 +/- 0.00 | 14.29 +/- 0.00 | 3.57 +/- 0.00 |
| text_roberta_top2ft_protoB | text | B | 2 | 14.29 +/- 0.00 | 14.29 +/- 0.00 | 6.31 +/- 0.22 |
| text_roberta_top2ft_protoC | text | C | 5 | 14.25 +/- 0.62 | 14.25 +/- 0.62 | 4.34 +/- 0.97 |
