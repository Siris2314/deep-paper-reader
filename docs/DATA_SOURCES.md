# Data sources for paper understanding

Research checked 2026-09-27. Exact-version arXiv HTML plus exact-identifier Crossref,
OpenAlex, and Semantic Scholar metadata are implemented as explicit reader actions. The
remaining entries are integration recommendations. Priorities below are engineering judgments.

## Recommended order

| Priority | Source | Benefit here | Access and limitations |
| --- | --- | --- | --- |
| 1 | arXiv HTML | Implemented: supplements PDF evidence with structured paragraphs and equation LaTeX | Requires one unambiguous version stamp on PDF page 1 and an explicit fetch. HTML coverage/conversion is imperfect. |
| 2 | Semantic Scholar | Implemented: exact DOI/arXiv lookup and reference enrichment when configured | Citation edges are discovery signals; method-inheritance claims still require citation context. |
| 3 | Crossref | Implemented: exact DOI metadata lookup | Crossref is metadata, not a universal full-text service. |
| 4 | OpenAlex | Implemented: exact DOI metadata, open-access location, and bounded related-work discovery | Related-work edges do not establish method lineage. Casual use works anonymously; a key raises the available budget. |
| 5 | Mathpix image/document OCR | A specialist fallback for equation crops that local transcription cannot resolve | Credentialed external processing. Verify current pricing and retention before enabling. The old equation endpoint is deprecated; use the documented image/document APIs. |

arXiv documents its HTML rollout and conversion limitations in
[HTML as an accessible format](https://info.arxiv.org/about/accessible_HTML.html).
Its [API overview](https://info.arxiv.org/help/api/index.html) separates metadata access
from bulk full-text access. Respect the [API terms](https://info.arxiv.org/help/api/tou.html)
and cache versioned artifacts instead of repeatedly fetching the same paper.

The reader's **Load matching arXiv HTML** action requests only
`https://arxiv.org/html/<id>v<version>`, rejects redirects and identity mismatches, and
caches bounded parsed content against the uploaded document hash. It does not fetch or
execute TeX archives. The uploaded PDF remains the primary source, while matching HTML
passages are labeled as supplementary structured evidence.

Semantic Scholar offers graph data, recommendations, and datasets in its
[API overview](https://webflow.semanticscholar.org/product/api). Current introductory API-key
capacity is listed as one request per second across endpoints; confirm allocated limits.
The existing integration in `research_lineage.py` resolves arXiv references but does not
yet provide general search or recommendation retrieval. S2ORC is an additional bulk
full-text resource, now distributed through the API datasets offering; it is more suitable
for offline indexing than a latency-sensitive hover request.
[S2ORC repository](https://github.com/allenai/s2orc).

Crossref's [REST documentation](https://github.com/CrossRef/rest-api-doc) describes DOI
lookup and bibliography queries; its [access guidance](https://www.crossref.org/documentation/retrieve-metadata/rest-api/access-and-authentication/)
recommends identifiable requests with a contact email. Unpaywall retired its search endpoint
on 2026-09-18 and directs search users to OpenAlex; this project therefore uses OpenAlex for
open-access and related-work discovery instead of adding a new dependency on retired search.

OpenAlex's current [authentication documentation](https://help.openalex.org/api/authentication/)
describes keys and budgets. Query usage and limits should be read from current documentation
and responses, not hard-coded from older descriptions of the free API.

The metadata action accepts only an unambiguous DOI or versioned arXiv stamp extracted from
page 1. It does not guess identity from a filename or bibliography. Provider responses are
bounded, host-allowlisted, cached against the uploaded document hash, title-checked, and escaped
in the UI. A title mismatch prevents related-graph expansion.

Mathpix's [OCR documentation](https://docs.mathpix.com/) covers STEM image/document
recognition and mathematical Markdown output. Use only an explicitly configured provider;
send a bounded crop where possible, keep provenance, and run the existing transcription
comparison and paper-alignment checks on its output.

## Integration shape

1. Resolve paper identity once at ingestion: DOI, arXiv ID and version, title, authors.
2. Fetch and cache structured text only after an explicit identity/version match. Keep
   original source URLs, retrieval times, content hashes, and license/access metadata.
3. Retrieve an equation and its defining paragraph by section/equation ID. Add source
   LaTeX/MathML as another transcription candidate; compare it to the uploaded PDF.
4. Use local crop vision when structured text is unavailable or disagrees. Escalate to an
   optional OCR service only for unresolved crops, with a separate request/cost budget.
5. Ground explanations in the uploaded paper. Mark version differences, external context,
   and inferred meanings explicitly; never replace the uploaded evidence silently.

For all adapters: bounded response sizes, provider timeouts, per-provider rate limiting,
`Retry-After` handling, bounded transient retries, short negative-cache TTLs, and no API
keys in URLs written to logs. These controls belong in the data adapter, independently
of model-call budgets. Do not auto-download or execute TeX macros or archive contents.

## Local alternatives

Docling's [enrichment pipeline](https://docling-project.github.io/docling/usage/enrichments/)
is worth evaluating for optional formula/code enrichment. GROBID's
[service API](https://grobid.readthedocs.io/en/latest/Grobid-service/) targets scholarly
document structure and references. These are local pipeline candidates with deployment
and model/runtime costs; neither should replace the current parser without a corpus test.

Before choosing an OCR provider, build a held-out set spanning at least two-column ML,
single-column mathematics, scanned pages, and papers without arXiv source. Track equation
region precision/recall, exact structural transcription errors, unresolved symbols,
explanation alignment, median/p95 latency, calls, and tokens. Provider selection should
follow those measurements, not metadata coverage or marketing accuracy figures.
