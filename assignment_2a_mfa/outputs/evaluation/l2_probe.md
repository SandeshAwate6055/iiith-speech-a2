# ISLE L2 Minimal-Pair Pronunciation Probe

MFA forced alignment with a multi-variant probe dictionary allowing canonical vs. L1-confusable realizations:

| File | Word | Span (s) | Canonical (ARPAbet) | MFA Selected Phones | Matches Canonical? |
|---|---|---|---|---|---|
| `ISLE_SESS0131_BLOCKD02_01_sprt1` | **white** | 0.920–1.330 | `W AY1 T` | `W AY1 T` | Yes |
| `ISLE_SESS0131_BLOCKD02_01_sprt1` | **bait** | 1.800–2.250 | `B EY1 T` | `B EY1 T` | Yes |
| `ISLE_SESS0131_BLOCKD02_02_sprt1` | **new** | 1.010–1.420 | `N UW1` | `N Y UW1` | **No (variant selected)** |
| `ISLE_SESS0131_BLOCKD02_02_sprt1` | **no** | 1.750–2.220 | `N OW1` | `N OW1` | Yes |
| `ISLE_SESS0131_BLOCKD02_03_sprt1` | **bad** | 0.660–1.170 | `B AE1 D` | `B EH1 T` | **No (variant selected)** |
| `ISLE_SESS0131_BLOCKD02_03_sprt1` | **bed** | 1.490–2.040 | `B EH1 D` | `B EH1 D` | Yes |

*Caveat: GMM-HMM acoustic likelihoods provide an unsupervised pronunciation cue rather than a clinical/human phonetic transcription; all findings should be verified against the spectrogram in Praat.*
