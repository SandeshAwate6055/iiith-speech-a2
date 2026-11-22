# TESS Dataset Audit

- **Total clips:** 2800 (expected 2800)
- **Speakers:** 2 ({'OAF': 1400, 'YAF': 1400})
- **Target words:** 200 (expected 200)
- **Emotions:** 7 ({'angry': 400, 'disgust': 400, 'fear': 400, 'happy': 400, 'neutral': 400, 'sad': 400, 'surprise': 400})
- **Sample rates:** {24414: 2799, 96000: 1}
- **Duplicate MD5 hashes:** 0 | **Duplicate utt_ids:** 0 | **Incomplete (spk,word) pairs:** 0
- **Duration (s):** min=1.254, median=2.048, max=2.985

## Per-Emotion Acoustic Summary (Raw Audio)
| Emotion | Duration mean ± std (s) | RMS mean ± std (dBFS) | Peak mean ± std |
|---|---:|---:|---:|
| angry | 1.841 ± 0.300 | -25.66 ± 2.99 | 0.406 ± 0.211 |
| disgust | 2.449 ± 0.155 | -32.37 ± 2.18 | 0.179 ± 0.051 |
| fear | 1.659 ± 0.151 | -27.17 ± 3.93 | 0.289 ± 0.151 |
| happy | 1.973 ± 0.117 | -27.37 ± 6.32 | 0.353 ± 0.239 |
| neutral | 2.055 ± 0.124 | -33.98 ± 4.24 | 0.103 ± 0.060 |
| sad | 2.402 ± 0.190 | -32.45 ± 5.16 | 0.151 ± 0.099 |
| surprise | 2.008 ± 0.179 | -30.27 ± 2.45 | 0.205 ± 0.083 |

## Per-Speaker Summary
| Speaker | Duration mean ± std (s) | RMS mean ± std (dBFS) |
|---|---:|---:|
| OAF | 2.020 ± 0.354 | -33.14 ± 3.80 |
| YAF | 2.090 ± 0.279 | -26.65 ± 4.01 |

## Key Scientific Implication
- Every transcript (`say the word <w>`) occurs identically across all 7 emotions and both speakers.
- Conditional label distribution `P(emotion | text)` is strictly uniform (`1/7 = 14.29%`) for all 200 transcripts.
- Noticeable raw RMS and duration differences exist across emotions and speakers -> peak normalization and silence-trimming ablations are essential.
