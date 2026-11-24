# Math reliability gate

The equation pipeline treats transcription as a separate, cached perception step. It compares
SymPy and PDF-geometry candidates first. Display equations are sent to the bounded PDF-crop vision
reader only when those candidates are missing, disagree, or fall below the configured threshold.

If final transcription confidence is below `0.6`, the pipeline fails closed: it stores the candidate
set and diagnostics, shows the source crop beside the reconstruction, and does not call the
explanation model or LLM judge. Paper-specific explanation starts only after this gate passes.

Transcriptions are cached under `math/transcriptions/` independently of explanations. The cache
records candidate agreement, whether vision ran, its routing reason, bounded-crop provenance, and
whether the result was verified, usable-but-unverified, or requires review.

Paper-defined symbol meanings retain their supporting excerpt and one-based PDF page. The
deterministic judge rejects a symbol labeled `paper` when that evidence is absent.

## Benchmark

`llmops/math_reliability_benchmark.json` contains 25 manually labeled equations from the local DPO
and FlashMemory PDFs. Each case includes exact LaTeX, display-region boundaries, symbol inventory,
nearby definitions, role, and a label indicating whether crop vision is expected.

Evaluate a prediction file with:

```powershell
python scripts/evaluate_math_reliability.py --predictions path/to/predictions.json
```

Without `--predictions`, the script scores the checked-in PDF-geometry fixtures as a dataset and
scorer smoke test. Reports include exact and structural transcription accuracy, symbol precision,
symbol recall and evidence-backed grounding recall, boundary and role accuracy, unsupported-claim
rate, latency percentiles, and vision-routing recall.
