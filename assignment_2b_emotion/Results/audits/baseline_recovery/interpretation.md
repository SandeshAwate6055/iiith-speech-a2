# Independent baseline recovery audit

Original predictions and summaries are preserved. See fold_metrics.csv and summary.csv.

All 2,800 IDs are unique and match the feature table; all 88 features are finite. Label mapping matches the shared emotion order. Each of 200 transcripts occurs twice in each of seven classes. Saved per-fold accuracy, macro-F1 and UAR are independently recomputed and checked. Existing split manifests are used unchanged.

## Text random-split result

For a transcript with two recordings per emotion, removing one test item reduces its true-label count in the training complement. A lexical classifier can learn this complementary imbalance and systematically predict another emotion. The audit measures this for each fold. Thus population P(label|text)=1/7 does NOT imply chance test accuracy under every finite, dependent partition. Zero percent is reproducible and is not by itself a label-mapping failure. Protocol A and B retain word-disjoint evaluation. Do not change the frozen random splits to obtain chance performance.

## Speaker-independent SVM

Both the direct SVM class decision and calibrated-probability argmax are evaluated. They need not agree. The legacy implementation uses probability argmax and assumes class indices equal labels (the assumption holds for the present seven-class folds). Direct predictions should use classifier.predict; any revised baseline must be separately named and rerun across all protocols. Poor cross-speaker performance remains after changing the decision rule. This audit does not establish that domain shift is the sole cause; audio preprocessing sensitivity and acoustic feature distributions still require investigation.

## Training-set comparability

Legacy classical baselines merge train and validation, whereas neural models fit only train and select checkpoints on validation. Train-only rows quantify this discrepancy without replacing results. Report fit policy explicitly; use train-only baselines for matched-data comparisons.

No test-based tuning was performed. CPU threads are limited to one; existing training, feature caches, checkpoints and original results are untouched.
