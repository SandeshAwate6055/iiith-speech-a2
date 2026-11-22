# Error & Failure Case Analysis

Selection criteria: highest-confidence misclassifications with diverse `(true, predicted)` pairs — up to 3 from Protocol A (word-disjoint) and 2 from Protocol B (LOSO cross-speaker).

| # | Source | Utt ID | Speaker | Word | True | Predicted | Confidence | Duration (s) | Mean F0 (Hz) |
|---|---|---|---|---|---|---|---:|---:|---:|
| 1 | Protocol A (classical_speech_protoA) | `OAF_chain_happy` | OAF | chain | **happy** | **surprise** | 0.980 | 2.02 | 234.5 |
| 2 | Protocol A (classical_speech_protoA) | `OAF_keen_disgust` | OAF | keen | **disgust** | **sad** | 0.935 | 2.60 | 194.6 |
| 3 | Protocol A (classical_speech_protoA) | `OAF_sail_surprise` | OAF | sail | **surprise** | **happy** | 0.873 | 2.16 | 233.8 |
| 4 | Protocol B LOSO (classical_speech_protoB) | `YAF_fail_fear` | YAF | fail | **fear** | **angry** | 0.921 | 1.70 | 317.9 |
| 5 | Protocol B LOSO (classical_speech_protoB) | `YAF_said_happy` | YAF | said | **happy** | **angry** | 0.917 | 1.75 | 272.2 |

These are observed prediction errors. F0 and duration are descriptive; no causal explanation or listening judgment is inferred.
