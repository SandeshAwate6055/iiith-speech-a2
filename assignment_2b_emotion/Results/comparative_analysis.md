# Prediction-backed comparative analysis

All scores use frozen test IDs and final checkpoint-linked re-evaluation. Changes are descriptive; with only two speakers no population-generalization claim is justified.

## Per-emotion recall and fusion effects
- Protocol A, fusion_concat: helped 0, harmed 4 utterances versus speech. Exploratory paired statistics ignore within-word/speaker and CV dependence.
  Per-emotion accuracy changes (percentage points): angry +0.00, disgust +0.00, neutral +0.00, fear -0.25, happy -0.25, sad -0.25, surprise -0.25
- Protocol A, fusion_gated_xattn: helped 1, harmed 6 utterances versus speech. Exploratory paired statistics ignore within-word/speaker and CV dependence.
  Per-emotion accuracy changes (percentage points): surprise +0.25, neutral +0.00, angry +0.00, sad +0.00, disgust -0.25, happy -0.25, fear -1.00
- Protocol A, speech_wavlm_bigru_asp: lowest recall surprise=99.75%; highest sad=100.00%. Ties are not evidence of a unique easiest emotion.
- Protocol A, fusion_concat: lowest recall surprise=99.50%; highest neutral=100.00%. Ties are not evidence of a unique easiest emotion.
- Protocol A, fusion_gated_xattn: lowest recall fear=99.00%; highest surprise=100.00%. Ties are not evidence of a unique easiest emotion.
- Protocol B, fusion_concat: helped 24, harmed 119 utterances versus speech. Exploratory paired statistics ignore within-word/speaker and CV dependence.
  Per-emotion accuracy changes (percentage points): happy +1.25, angry -2.50, fear -7.50, surprise -12.50, neutral -15.00, disgust -17.50, sad -65.00
- Protocol B, fusion_gated_xattn: helped 45, harmed 93 utterances versus speech. Exploratory paired statistics ignore within-word/speaker and CV dependence.
  Per-emotion accuracy changes (percentage points): neutral +26.25, surprise +12.50, happy +2.50, disgust +0.00, fear -27.50, sad -35.00, angry -38.75
- Protocol B, speech_wavlm_bigru_asp: lowest recall happy=0.00%; highest fear=100.00%. Ties are not evidence of a unique easiest emotion.
- Protocol B, fusion_concat: lowest recall happy=1.25%; highest fear=92.50%. Ties are not evidence of a unique easiest emotion.
- Protocol B, fusion_gated_xattn: lowest recall happy=2.50%; highest neutral=91.25%. Ties are not evidence of a unique easiest emotion.
- Protocol C, fusion_concat: helped 4, harmed 8 utterances versus speech. Exploratory paired statistics ignore within-word/speaker and CV dependence.
  Per-emotion accuracy changes (percentage points): happy +0.25, sad +0.25, disgust +0.00, surprise +0.00, neutral -0.25, fear -0.50, angry -0.75
- Protocol C, fusion_gated_xattn: helped 4, harmed 5 utterances versus speech. Exploratory paired statistics ignore within-word/speaker and CV dependence.
  Per-emotion accuracy changes (percentage points): happy +0.25, sad +0.25, angry +0.00, fear +0.00, disgust +0.00, neutral +0.00, surprise -0.75
- Protocol C, speech_wavlm_bigru_asp: lowest recall happy=99.50%; highest neutral=100.00%. Ties are not evidence of a unique easiest emotion.
- Protocol C, fusion_concat: lowest recall angry=99.25%; highest sad=100.00%. Ties are not evidence of a unique easiest emotion.
- Protocol C, fusion_gated_xattn: lowest recall surprise=99.00%; highest sad=100.00%. Ties are not evidence of a unique easiest emotion.

## Interpretation
Speech alone exceeds gated fusion in aggregate A/B/C. Text is label-independent in the complete balanced corpus, so fusion improvement is not presumed. The B drop is a two-speaker transfer stress test, not a reliable population estimate. Measured F0/duration in the five failures are descriptive; they do not establish causal explanations without controlled tests or listening-based reference judgments.

The classical baselines fit train+validation, whereas neural models fit train only. Matched-data baseline diagnostics are retained under Results/audits/baseline_recovery and should be used when discussing fairness. Zero-percent random-split TF-IDF accuracy is reproduced by complementary per-transcript class imbalance, not evidence of swapped labels. A 14.29% accuracy reference is not a universal macro-F1 threshold.

Gate values summarize the mixture of two already cross-attended streams; they are not causal modality contributions. Zero-text results test an intermediate intervention, not a retrained speech-only model. The performance drop under that intervention must be reported, not asserted to match speech-only.

Gated Protocol C has no saved training history. Its final checkpoints and test inference are verified; no missing epoch curve was fabricated. Interrupted late fusion is excluded from headline results and was not restarted. Historical text-training lineage remains uncertain, but the final checkpoint-to-prediction mapping is now reproducible and hash-linked.
