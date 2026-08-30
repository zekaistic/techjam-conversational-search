# Team TokTik's Shopping Copilot

## Project overview

Our Shopping Copilot is a deterministic, offline shopping agent built
for the TechJam Conversational E-Commerce Search Challenge. It searches a
frozen catalog of 50,000 Amazon products and tries to identify the customer's
target product as early and as highly ranked as possible within a ten-turn
conversation.

On each turn, the agent:

- extracts structured preferences into nine slots — category, material, color,
  size, style, brand, budget, feature, and use case;
- compiles a Buying or Browsing context program from the live state and the
  aggregate user profile;
- retrieves candidates using SQLite FTS5 BM25, a second Porter-stemmed FTS
  table, exact-phrase matching, and a local MiniLM ONNX semantic model, fused
  by reciprocal rank;
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
- Python 3.10 or later (measured on 3.12.1; no third-party packages are
  required at import time)
- A POSIX-compatible shell
- Approximately 560 MiB of free space for generated artifacts
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

   For a faster, dependency-light development build, use
   `python3 -m tools.build_index --skip-dense`. The agent falls back to BM25
   automatically when the dense artifacts or their dependencies are absent,
   though the fused route scores better.

No API key is required.

### Generated artifacts

| Artifact | Size |
|---|---:|
| `artifacts/attributes.json` | 9.4 MiB |
| `artifacts/bm25.sqlite3` (exact + Porter tables) | 186.8 MiB |
| `artifacts/all-MiniLM-L6-v2/` (`model.onnx` + vocabulary) | 86.4 MiB |
| `artifacts/dense_vectors.npy` (float16, 50,000 × 384) | 36.6 MiB |
| `artifacts/dense_asins.npy` | 3.1 MiB |
| `artifacts/_vendor/` (local ONNX Runtime + NumPy) | 233.1 MiB |
| **Total** | **≈ 555 MiB** |

## Steps to reproduce the results

Complete the full setup above, run the commands from the repository root, and
leave all `TJ_*` tuning environment variables at their defaults.

1. Run the official local evaluator over all 200 released sessions. The
   command writes detailed session results to `results.json`:

   ```bash
   python3 -m evaluator.local_evaluator --output results.json
   ```

2. Generate the aggregate, per-scenario, first-hit, and latency benchmark
   report:

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
| TechnicalScore | **0.8601** |
| Hit Rate@10 | **0.9900** |
| MRR | **0.6618** |
| MTTC | **2.670** |
| Reported tokens | **0** |

Per scenario:

| Scenario | n | Hit Rate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Buying | 80 | 0.9875 | 0.6667 | 2.238 |
| Browsing | 80 | 0.9875 | 0.6522 | 2.438 |
| Intent override | 30 | 1.0000 | 0.6492 | 4.167 |
| Boundary | 10 | 1.0000 | 0.7375 | 3.500 |

Two of the 200 sessions miss and are assigned turn 11 by the evaluator.
Intent-override hits cannot be counted before the override arrives on turn 3
or 4, which is why that scenario's MTTC is the highest despite a perfect hit
rate. The evaluation is deterministic, so hardware should affect build and
runtime latency but not these metrics.

The weak BM25 starter baseline scores TechnicalScore `0.10671` on the same set
(`docs/baseline_results.json`), so these numbers are roughly eight times the
provided baseline.

## Cost, token usage, and latency disclosure

- **Model cost:** $0 per session. The scoring path performs no external API
  calls, uses no LLM, and reports `prompt_tokens: 0` and `completion_tokens: 0`.
- **Network access:** required once, during setup, to download the catalog,
  the pinned MiniLM ONNX model, and ONNX Runtime. After
  `tools.build_index` completes, evaluation runs entirely offline. If the
  dense artifacts are unavailable the agent degrades to the BM25 route rather
  than failing.
- **Latency:** measured end to end around `Agent.respond` by
  `tools.bench` over the 532 turns of the public set, on an Apple M3 Pro with
  Python 3.12.1:

  | Measurement | Value |
  |---|---:|
  | Cold start (`Agent` construction, once per run) | 1.2 s |
  | First turn, including index warm-up | 388.2 ms |
  | Median (p50) per turn | 46.4 ms |
  | Mean per turn | 50.7 ms |
  | p95 per turn | 110.5 ms |
  | p99 per turn | 154.2 ms |
  | Maximum per turn | 168.5 ms |

  Percentiles exclude the first turn, which pays a one-off warm-up no later
  turn repeats. Retrieval-only microbenchmarks and the measurement methodology
  are in [`docs/lane_b_notes.md`](docs/lane_b_notes.md).

## Agent interface

The evaluator imports `Agent` from `starter/agent.py`:

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        return {
            "message": "Here are my closest matches. Do you prefer a material?",
            "ask_attribute": "material",
            "recommendations": [{"parent_asin": "B000..."}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
```

`ask_attribute` may be one of the nine modeled slots, `other`, or `null`.
`respond` never raises: an internal failure is caught and answered from the
popularity prior so the session stays alive. The complete contract is in
[`docs/agent_api_contract.json`](docs/agent_api_contract.json).

## Development commands

```bash
python3 -m unittest discover -s tests -t . -v   # 98 unit/integration tests
python3 -m tools.robustness                     # adversarial input corpus
python3 -m tools.chat                           # interactive local session
python3 -m tools.bench --failures 5             # failed session transcripts
python3 -m tools.bench --depth                  # where the target really ranks
python3 -m tools.bench --compare a.json b.json  # diff two benchmark runs
python3 -m tools.bench --sweep TJ_OPEN_QUESTION_BASELINE=3,4,5
```

Non-obvious runtime environment variables:

- `TECHJAM_ARTIFACTS`: artifact directory; defaults to `artifacts`.
- `TECHJAM_DEBUG=1`: re-raise response failures instead of serving the safe
  fallback.
- `TJ_RETRIEVAL_MODE`: `fused` (default), `bm25`, or `dense`.
- `TJ_BM25_WEIGHT`, `TJ_STEM_WEIGHT`, `TJ_STEM_EXACT_WEIGHT`,
  `TJ_DENSE_WEIGHT`, `TJ_EXACT_WEIGHT`, `TJ_SLOT_WEIGHT`, `TJ_BUDGET_WEIGHT`,
  `TJ_PROFILE_WEIGHT`, `TJ_CATALOG_CONFIDENCE_POWER`: retrieval ablation and
  tuning controls.
- `TJ_OPEN_QUESTION_BASELINE`, `TJ_OPEN_QUESTION_DECAY`,
  `TJ_OPEN_QUESTION_EXPECTED_YIELD`, `TJ_OPEN_QUESTION_MAX_CONSECUTIVE`,
  `TJ_OPEN_QUESTION_ZERO_YIELD_PATIENCE`, `TJ_OPEN_QUESTION_DECLINE_PATIENCE`:
  value, decay, and guardrails for the open-ended "anything else?" question.
- `TJ_ONNX_THREADS`, `TJ_ONNX_BUILD_THREADS`: ONNX Runtime thread counts at
  query and build time.

Question-policy settings are read once, when `src.policy.question` is imported.
Set them inline for a single run, or copy
`config/question-policy.env.example` to the gitignored `.env` and source it.

## Repository map

```text
starter/agent.py              evaluator entry point and turn orchestration
src/contracts.py              shared state and candidate types
src/extract.py                customer constraint extraction
src/attributes.py             catalog attribute table
src/lexicons/                 canonical synonym and no-preference vocabularies
src/orchestration.py          intent routing and turn-scoped context programs
src/retrieval/                BM25, MiniLM dense retrieval, fusion, reranking
src/policy/                   state transitions, question policy, response text
tools/build_index.py          reproducible artifact builder
tools/bench.py                metrics, replay, transcripts, sweeps, verification
tools/chat.py                 interactive local conversation
tools/robustness.py           adversarial input corpus runner
tests/                        unit, integration, evaluator, and robustness tests
data/public_set.jsonl         200 labeled development sessions
evaluator/local_evaluator.py  frozen public simulator and scorer
docs/                         competition spec, submission rules, lane notes
```

## Limitations and future improvements

The deterministic, offline approach was chosen for reproducibility, low
latency, and zero API cost, but it limits how well the agent generalizes beyond
the development data. Rule-based extraction can miss unseen paraphrases,
brands, fashion subcultures, and non-US sizing, while canonicalization merges
some nearby concepts — navy into blue, water-resistant into waterproof — and
loses nuance. Only 21% of catalog products carry price data, so a missing price
must be treated as unknown rather than over budget.

The public simulator copies catalog clauses into its replies and so produces
more structured language than real customers, meaning the measured result may
overstate real-world conversational performance. `tools/robustness.py` exists
because that gap is invisible to the TechnicalScore: it exercises an
adversarial corpus of sentences a person would actually type, and a failure
there is a bug the scored loop cannot see.

The evaluator also supplies an aggregate profile but no stable user identifier
or write-back API, so the agent personalizes within a session and deliberately
does not invent cross-session identity or durable profile mutations.

Given more time, we would evaluate on held-out, human-written conversations;
add a learned semantic extractor behind the deterministic fallback; improve
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

Competition rules and submission requirements are in
[`docs/competition_specification.md`](docs/competition_specification.md) and
[`docs/submission_rules.md`](docs/submission_rules.md). Data is derived from
Amazon Reviews 2023 by McAuley Lab, UCSD; read
[`DATA_ATTRIBUTION.md`](DATA_ATTRIBUTION.md) before redistribution.
