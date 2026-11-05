# Verification

Run from the repository root with the supplied environment:

```bash
python tools/verify_export.py
```

The verifier writes its report and temporary outputs only beneath `verification/`. It uses synthetic inputs for short model training and no API requests. Optional source comparison is available with:

```bash
python tools/verify_export.py --source-root /path/to/original/LLM
```

The checks cover copied-file hashes, dataset schema and finite values, abundance/label reconstruction, historical prompt counts and hashes, offline prompt orchestration using stubbed responses, executable notebook syntax, command help, and one-epoch training for all four variants. When sources are provided, their SHA-256 fingerprints are checked and original graph/model definitions are used for numerical comparisons. CUDA comparisons are performed only when CUDA is available; any skipped comparisons are reported.

The preparation-time results are recorded in `verification/report.json`. Full 3,000-epoch training, fresh paid DeepSeek searches, and fresh large-scale FCG generation are not part of verification. A successful smoke run establishes executable paths and tensor compatibility; it does not establish reproduced model performance.

## Preparation-time results

- All 68 copied-file mappings matched their source SHA-256 values.
- All 305 fingerprinted original source files remained unchanged.
- Abundance features, labels, sample IDs and OTU names reconstructed with identical arrays and dtypes; all 4,012 text rows matched a full merge from the original embedding intermediate.
- Fresh BiomedBERT inference on three representative samples matched the archived vectors within a maximum absolute error of 0.000002862.
- Main and prompt kNN graphs matched the original implementations exactly. All four requested networks and the prompt scorer had zero maximum difference in evaluation output, seeded training output, and parameter gradients on the comparison inputs.
- All five model profiles completed one-epoch, 40-sample CPU runs with checkpoints and predictions. Offline prompt tests covered duplicate generation, partial targets, missing parents, the 200-sample scoring integration, and caching using stubs.
- The 15-cell notebook executed offline without errors. Command interfaces, shell syntax, credential scans and machine-path scans passed. No paid API request was made.

The packaged final NPZ is an unchanged source copy. Its abundance, labels, IDs and feature names can be reconstructed from the bundled CSV without text-model inference. Matching the text embeddings additionally requires the original BiomedBERT files; representative numerical checks and any full merge checks are recorded separately in the report.
