# Error & Failure Case Analysis

Selection criteria: highest-confidence misclassifications with diverse `(true, predicted)` pairs  -  up to 3 from Protocol A (word-disjoint) and enough from Protocol B (LOSO cross-speaker) to reach five when available.

| # | Source | Utt ID | Speaker | Word | True | Predicted | Confidence | Duration (s) | Mean F0 (Hz) |
|---|---|---|---|---|---|---|---:|---:|---:|
| 1 | Protocol A (speech_wavlm_bigru_asp_protoA) | `OAF_make_surprise` | OAF | make | **surprise** | **happy** | 0.568 | 2.08 | 226.0 |
| 2 | Protocol B LOSO (speech_wavlm_bigru_asp_protoB) | `YAF_fall_angry` | YAF | fall | **angry** | **fear** | 0.825 | 1.78 | 237.2 |
| 3 | Protocol B LOSO (speech_wavlm_bigru_asp_protoB) | `OAF_cool_happy` | OAF | cool | **happy** | **surprise** | 0.763 | 2.07 | 252.8 |
| 4 | Protocol B LOSO (speech_wavlm_bigru_asp_protoB) | `OAF_cool_neutral` | OAF | cool | **neutral** | **angry** | 0.747 | 1.98 | 206.0 |
| 5 | Protocol B LOSO (speech_wavlm_bigru_asp_protoB) | `YAF_week_happy` | YAF | week | **happy** | **angry** | 0.710 | 1.54 | 286.7 |

These are observed prediction errors. F0 and duration are descriptive; no causal explanation or listening judgment is inferred.
