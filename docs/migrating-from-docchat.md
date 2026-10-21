# Migrating from DocChat to FactLens

FactLens is DocChat with a new name, a `src/` package layout, an OpenAI backend and a tested
pipeline. This page lists every rename and every deliberate behaviour change.

## Renames

### Files and classes

| DocChat | FactLens |
|---|---|
| `app.py` | `src/factlens/ui/app.py` (`python -m factlens` / `factlens`) |
| `config/constants.py` | `core/constants.py` |
| `config/settings.py` (`settings` global) | `core/settings.py` (`get_settings()`) |
| `utils/logging.py` | `core/logging.py` (`configure_logging()`) |
| `document_processor/file_handler.py` (`DocumentProcessor`) | `ingestion/ingestor.py` (`DocumentIngestor`) |
| cache methods inside `DocumentProcessor` | `storage/chunk_cache.py` (`ChunkCache`) |
| hashing inside `DocumentProcessor`/`app.py` | `ingestion/hashing.py` |
| `retriever/builder.py` (`RetrieverBuilder`) | `retrieval/hybrid.py` (`build_hybrid_retriever`) |
| `agents/relevance_checker.py` (`RelevanceChecker`) | `agents/scope_classifier.py` (`ScopeClassifier`) |
| `agents/research_agent.py` (`ResearchAgent`) | `agents/answer_synthesizer.py` (`AnswerSynthesizer`) |
| `agents/verification_agent.py` (`VerificationAgent`) | `agents/claim_verifier.py` (`ClaimVerifier`) |
| `agents/workflow.py` (`AgentWorkflow`, `AgentState`) | `pipeline/graph.py`, `pipeline/runner.py` (`AnswerPipeline`, `PipelineState`) |
| `test/test1.py` | `scripts/inspect_parsing.py` (it was a script, not a test) |
| `requirements.txt` | `pyproject.toml` + `uv.lock` |
| `document_cache/`, `chroma_db/` | `.factlens/cache/`, `.factlens/vectors/` (git-ignored) |

### Functions and methods

| DocChat | FactLens |
|---|---|
| `process_question` | `handle_question_submission` |
| `load_example` | `load_example_into_inputs` |
| `_get_file_hashes` | `compute_file_hashes` |
| `DocumentProcessor.process` | `DocumentIngestor.ingest_files` |
| `validate_files` | `validate_upload_sizes` |
| `_process_file` | `_parse_and_chunk_file` |
| `_generate_hash` | `hash_bytes` |
| `_save_to_cache` / `_load_from_cache` / `_is_cache_valid` | `ChunkCache.save` / `.load` / `.is_fresh` |
| `RelevanceChecker.check` | `ScopeClassifier.classify` |
| `ResearchAgent.generate` | `AnswerSynthesizer.synthesize` |
| `VerificationAgent.check` | `ClaimVerifier.verify` |
| `AgentWorkflow.full_pipeline` | `AnswerPipeline.run` |
| `_check_relevance_step` / `_research_step` / `_verification_step` | `scope_check_node` / `synthesis_node` / `verification_node` |
| `_decide_after_relevance_check` / `_decide_next_step` | `route_after_scope_check` / `route_after_verification` |
| `generate_prompt` (in two agents) | `_PROMPT_TEMPLATE` constants |
| `parse_verification_response`, `format_verification_report` | `ClaimVerifier._parse_report`, `ui.rendering.render_verification_report` |

### Settings and labels

- Settings are lower-case fields with a `FACTLENS_` prefix (`VECTOR_SEARCH_K` becomes
  `FACTLENS_VECTOR_SEARCH_K`, `HYBRID_RETRIEVER_WEIGHTS` becomes `FACTLENS_HYBRID_WEIGHTS`,
  `CHROMA_DB_PATH` and `CACHE_DIR` are replaced by `FACTLENS_DATA_DIR`). `OPENAI_API_KEY` is unchanged.
- `CAN_ANSWER` / `PARTIAL` / `NO_MATCH` strings are now `ScopeVerdict.ANSWERABLE` /
  `PARTIAL` / `OUT_OF_SCOPE`. The model-facing labels in the prompt are unchanged.
- The verifier's `Supported: YES/NO` text is now a typed `VerificationReport`.

## Behaviour changes

**Required by the move to OpenAI**
- All chat and embedding calls use OpenAI. The hard-coded IBM watsonx project and URL are gone, so
  the app runs outside the IBM lab. Models are configurable; the default is `gpt-6-luna`.
- The 10/200/300-token output caps are removed. Reasoning-style models spend completion tokens
  on thinking, so a 10-token cap would return empty labels, and the 300-token cap truncated answers.
- Chunks are now capped at 2,000 characters. OpenAI embeddings reject very long inputs, and the
  old header-only splitting could produce arbitrarily large chunks.

**Bug fixes**
- Uploads no longer pile up in one shared Chroma store, which could let a previous upload's
  chunks appear in later answers. Each set of chunks gets its own content-addressed collection,
  reused when the same documents and embedding model come back.
- The "self-correction" loop now works: a failed verification sends the verifier's feedback to the
  synthesizer, and revisions are capped (`FACTLENS_MAX_REVISIONS`, default 2). Before, it re-ran
  identical inputs with no limit until LangGraph's recursion error.
- The verifier now receives the question. Its prompt asked about relevance to the question but the
  question was never passed in.
- Model or API failures surface as errors instead of being reported as "NO_MATCH".
- Retrieval ran twice per question (once in the pipeline, once in the relevance check); it now
  runs once.
- The parse cache is JSON, not pickle, and expired or unreadable entries are ignored.
- The cache key includes the chunking settings, so changing them re-parses.
- A per-file upload limit is now enforced (`MAX_FILE_SIZE` existed but was never used).

**Safer defaults**
- The app binds to `127.0.0.1:7860` and does not create a public Gradio link. DocChat launched
  with `share=True` on port 5000.
- No debug `print` calls; the full document context is no longer dumped to the console.

**Small retrieval and parsing changes**
- BM25 lower-cases text and ignores punctuation, and returns only chunks that match a query term.
- Chunks carry `source` and `file_hash` metadata (needed for citations).
- `.txt` and `.md` files are read directly instead of through Docling.
- The Docling converter is created once and reused.
- PDFs are parsed without OCR first and retried with OCR only if almost no text comes out
  (`FACTLENS_OCR_MODE`, default `auto`). DocChat always ran Docling's default OCR, which dominates
  parse time on image-heavy PDFs.
- The UI streams status messages (reading, indexing, drafting) instead of waiting silently, and
  the console logs per-file conversion time.

## Data

Old `document_cache/*.pkl` files and the `chroma_db/` folder are not read by FactLens. Delete them.
Embeddings from watsonx are not compatible with OpenAI embeddings.

## Applying this to your repository in place

```bash
git checkout -b refactor/factlens
git rm -r agents config document_cache document_processor retriever test utils app.py \
          requirements.txt screenshots examples
cp -R /path/to/factlens/. .          # includes hidden files such as .github and .gitignore
git add -A && git commit -m "Rebrand DocChat to FactLens and restructure"
```

Then rename the repository under **Settings → General** on GitHub (old URLs redirect), and update
the description to the FactLens tagline.
