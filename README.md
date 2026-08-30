# TechJam Conversational Product Search

## Project overview

This project is a deterministic, offline shopping agent built for the TechJam
Conversational E-Commerce Search Challenge. It searches a frozen catalog of
50,000 Amazon products and tries to identify the customer's target product as
early and as highly ranked as possible within a ten-turn conversation.

On each turn, the agent:

- extracts structured preferences such as category, material, color, size,
  style, brand, budget, features, and use case;
- retrieves candidates using SQLite FTS5 BM25, Porter stemming, exact-phrase
  matching, and a local MiniLM semantic model;
- reranks products using the customer's current constraints and aggregate
  profile preferences;
- asks an adaptive clarification question when the request is underspecified;
  and
- avoids repeating recommendations unless the customer changes intent.

The scoring path makes no external API calls, is deterministic after setup, and
reports zero prompt and completion tokens.

## Setup and installation

### Prerequisites

- Git
- Python 3.10 or later
- A POSIX-compatible shell
- Approximately 484 MiB of free space for generated artifacts
- Internet access for the initial catalog, model, and runtime downloads

### Installation

1. Clone the repository and enter its root directory:

   ```bash
   git clone https://github.com/zekaistic/techjam-conversational-search.git
   cd techjam-conversational-search
   ```

2. Download `catalog.jsonl.gz` and `SHA256SUMS` from the repository's
   [GitHub Releases](https://github.com/zekaistic/techjam-conversational-search/releases),
   verify the download, and place the decompressed catalog at
   `data/catalog.jsonl`:

   ```bash
   shasum -a 256 -c SHA256SUMS
   gzip -dk catalog.jsonl.gz
   mv catalog.jsonl data/catalog.jsonl
   ```

3. Build the attribute table, SQLite search index, pinned MiniLM model, and
   dense vectors:

   ```bash
   python3 -m tools.build_index
   ```

   On the first run, this command downloads the pinned
   `sentence-transformers/all-MiniLM-L6-v2` ONNX model and installs
   `onnxruntime==1.22.1` and NumPy under the gitignored `artifacts/_vendor`
   directory. It does not modify the global Python environment. Later builds
   reuse these files.

No API key is required.

## Steps to reproduce the results

Complete the full setup above, run the commands from the repository root, and
leave all `TJ_*` tuning environment variables at their defaults.

1. Run the official local evaluator over all 200 released sessions. The
   command writes detailed session results to `results.json`:

   ```bash
   python3 -m evaluator.local_evaluator --output results.json
   ```

2. Generate the aggregate, per-scenario, and first-hit benchmark report:

   ```bash
   python3 -m tools.bench --output benchmark-results.json
   ```

3. Confirm that the diagnostic replay loop produces the same metrics as the
   official evaluator:

   ```bash
   python3 -m tools.bench --verify
   ```

   The verification should report `OK` for all four metrics and finish with
   `replay is faithful`.

Using the full fused index, the expected aggregate result is:

| Metric | Expected result |
|---|---:|
| TechnicalScore | **0.8578** |
| Hit Rate@10 | **0.9900** |
| MRR | **0.6530** |
| MTTC | **2.655** |
| Reported tokens | **0** |

The evaluation is deterministic, so hardware should affect build and runtime
latency but not these metrics.

## Limitations and future improvements

The deterministic, offline approach was chosen for reproducibility, low
latency, and zero API cost, but it limits how well the agent generalizes beyond
the development data. Rule-based extraction can miss unseen paraphrases,
brands, fashion subcultures, and non-US sizing, while canonicalization merges
some nearby concepts and loses nuance. Most catalog products also lack price
data, so a missing price must be treated as unknown rather than over budget.

The public simulator produces more structured replies than real customers, so
the measured result may overstate real-world conversational performance. Given
more time, we would evaluate on held-out, human-written conversations; add a
learned semantic extractor behind the deterministic fallback; improve
locale-aware size and currency handling; estimate missing prices with explicit
confidence; and calibrate retrieval and clarification decisions on data not
used for tuning. If a privacy-approved identity and write-back interface became
available, we would also evaluate cross-session personalization.

## Team member contributions

| Team member | Main contributions |
|---|---|
| Esther | Catalog attribute modeling and extraction; adaptive intent orchestration; override handling and recommendation deduplication; integration tests. |
| Timothy | Hybrid BM25/MiniLM retrieval, reciprocal-rank fusion, reranking, artifact builds, performance measurement, and open-ended question-policy improvements. |
| Ze Kai | Initial agent and dialogue-policy foundation; benchmark and interactive tooling; adversarial robustness suite and natural-language extraction fixes. |
