# Evidence retrieval and reading tools

## Local embeddings

The reader defaults to Hugging Face `BAAI/bge-small-en-v1.5` through Sentence Transformers.
It runs on CPU by default so it can coexist with the local generation models. The first use
downloads the model; subsequent uses reuse the Hugging Face model cache. Document vectors are
normalized and cached per paper, content, model, provider and optional model revision. BGE's
retrieval instruction is applied only to queries.

BM25 and semantic rankings are combined with reciprocal rank fusion. Exact numbered artifact
references take precedence. Optional `PAPER_RETRIEVAL_RERANKER` enables a Sentence Transformers
cross-encoder on at most 30 candidates. All tables and figures are candidates. References are
excluded; appendices are downweighted for ordinary questions and available for proof/derivation
questions. Nearby text supplies a bounded amount of definition and assumption context.

`PAPER_RETRIEVAL_MIN_SIMILARITY` is a configurable heuristic, not a calibrated probability.
Weak semantic matches with little lexical support can abstain. Provider failures fall back to
BM25 and are recorded in `retrieval/latest.json`. Set the provider to `off` for offline lexical-only
operation, or `ollama` with an Ollama embedding model. Existing `.env` files need no change to use
the Hugging Face default.

## Evidence and interaction

`parsed/evidence_index.json` stores stable document-bound paragraphs and artifacts. Native PDF
text blocks retain PDF-point coordinates. Extracted/OCR-only text without verified coordinates
does not receive invented boxes. Figure/table coordinates may locate only a caption, explicitly
recorded as such. Citation popovers show source crops and highlight known regions on the PDF.

Short follow-ups incorporate the preceding user question. Deep/Explore can perform one additional
retrieval pass for unresolved claims. Source previews appear while generation runs; draft answers
are explicitly marked as awaiting claim checks. Specialist results have document/evidence/prompt/
model-dependent caches, including confirmed correction context.

Equation panels allow paper-local notation corrections, symbol questions, step follow-ups,
intuition, derivation and implementation questions. Symbol retrieval prefers the current page
and section. Nearby equations are labeled as proximity, never asserted dependencies. Corrected
notation invalidates dependent explanation keys. Shared math artifacts let chat reuse the same
transcription and symbol evidence. Printed equation labels split adjacent numbered display regions.

Experiments questions can attach at most two original PDF pages containing retrieved figures or
tables to `PAPER_READER_VISUAL_MODEL`. This is bounded page vision, not visual-embedding search.
The prompt requires inspecting axes, units, conditions and footnotes and withholding unreadable
values. Caption locations are not assumed to be figure boundaries.

## Scholarly full text

After resolving matching OpenAlex metadata, **Load OpenAlex full text** fetches bounded GROBID TEI
XML using the configured key. The external copy is cached separately and labeled as version
unverified. It never silently replaces uploaded-PDF evidence. No credentialed URLs are persisted;
redirects and XML entity declarations are rejected. Semantic Scholar snippet search is available
on Web/Explore when its key is configured. Neither integration turns citation edges into proof
of method inheritance.

## Evaluation commands

```powershell
python scripts/evaluate_paper_retrieval.py --pdf-dir uploaded_papers
python scripts/evaluate_reader_regression.py --require-pass
python scripts/evaluate_math_reliability.py --pdf-dir uploaded_papers --output llmops/reports/math-live.json
python scripts/evaluate_math_reliability.py --pdf-dir uploaded_papers --case lsa-eq-01 --explain
python scripts/smoke_reader_ui.py --url http://localhost:8517
```

The retrieval development set has 15 cases over DPO and FlashMemory, including paraphrases,
follow-ups, numbered equations and two unanswerable controls. It is a small development set, not
evidence of generalization to other fields or scans. A broader independent held-out corpus remains
necessary before promoting model/parser changes based on these scores.

The live math runner reads actual PDFs and locates equations using printed labels or an explicit
input crop. It never selects a prediction by comparing it to the gold LaTeX. Add `--explain` for
actual vision, reasoning and judge calls; without it, results measure geometry extraction only.
The default run without PDFs is explicitly labeled `fixture_smoke_test`. Exact matching preserves
case and brace scope. Missing claim assessments/latency are null, not successful zeroes. Four
manually reviewed region boxes supply initial IoU labels; remaining regions need independent
annotation. Grounding recall measures evidence-bearing definition coverage, not semantic proof
that every definition is correct. Provenance checks likewise do not replace claim entailment tests.

## P0 source-integrity regression gate

The separate 30-case authored corpus generates six small, redistributable PDFs covering
single-column Bayesian calibration, two-column materials results, graph routing, scanned robotics,
a scanned body with a native header, and section-scoped symbols. Expected passages are checked,
rather than counting any text from the correct page as a retrieval success. It is a software
regression corpus, not a held-out collection of real scientific papers.

The default local model runs this same corpus through hybrid BGE retrieval; set
`PAPER_RETRIEVAL_EMBEDDING_PROVIDER=off` to isolate the lexical path. CI installs Tesseract and
runs the lexical regression gate without downloading model weights. Missing required OCR cases
reduce coverage and fail `--require-pass`; they never count as successful abstentions.

Evidence identity includes the source PDF bytes as well as the parsed-text/structure cache key.
OCR text and its stored coordinates are retained even when a page has a small native header.
Section boundaries are applied per text block, preserving results above a bibliography heading.
Reference and appendix filtering also applies to equations and neighbor expansion.

Embedding caches reject invalid dimensions, NaNs and zero vectors, then rebuild atomically.
Shared equation artifacts must match document identity, page, raw notation and coordinates before
their symbol definitions can enter chat. A newer reader correction suppresses stale interpretations.
Claim verification preserves symbol case and checks signed numbers against the actual supporting
quotation, rather than unrelated numbers elsewhere in the passage.
