# Custom Pronunciation Dictionary & OOV Resolution Report

- **Corpus unique word types:** 145
- **OOV before custom lexicon (base `english_us_arpa`):** 6 (`de-politicize, dukakis, maffy, melnicove, sjc's, wbur's`)
- **OOV after custom lexicon (G2P + manual overrides):** 0

| OOV Word | G2P Prediction | Final Custom Dictionary Pronunciation(s) | Source |
|---|---|---|---|
| `de-politicize` | `—` | `D IY2 P AH0 L IH1 T AH0 S AY2 Z; D IH0 P AH0 L IH1 T AH0 S AY2 Z` | Manual override (verified) |
| `dukakis` | `D AH0 K AA1 K IH0 S; D AH1 K AA1 K IH0 S; D UW0 K AA1 K IH0 S; D UW1 K AA1 K IH0 S; D UW2 K AA1 K IH0 S` | `D UW0 K AA1 K IH0 S; D AH0 K AA1 K IH0 S` | Manual override (verified) |
| `maffy` | `M AE1 F IY0` | `M AE1 F IY0` | Manual override (verified) |
| `melnicove` | `M EH1 L N IH0 K OW2 V` | `M EH1 L N IH0 K OW2 V; M EH1 L N AH0 K OW2 V` | Manual override (verified) |
| `sjc's` | `S JH AH0 K S; S JH K S; SH K S` | `EH1 S JH EY1 S IY1 Z` | Manual override (verified) |
| `wbur's` | `B ER0 Z; V B ER0 Z; W AY1 B ER0 Z` | `D AH1 B AH0 L Y UW0 B IY1 Y UW1 AA1 R Z` | Manual override (verified) |
