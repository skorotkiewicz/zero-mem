# pi-zero-mem

A Pi adaptation of [Zero-Mem: Zero-Token Memory Operations for LLM Agents](https://arxiv.org/abs/2607.29377).

It keeps Pi's original session messages as the source of record, retrieves relevant old turns through graph and temporal views, and sends only the retrieved evidence plus the current turn to the reader model. spaCy extracts typed entities and BGE-M3 supplies dense context, entity-alignment, and episode-continuity scores without LLM calls.

## Install

```bash
pi install .
```

Set up the Python NLP worker once (BGE-M3 downloads from Hugging Face on first use):

```bash
uv sync --python 3.12
uv run python -m spacy download en_core_web_sm
export ZERO_MEM_PYTHON="$PWD/.venv/bin/python"
```

For development:

```bash
pi -e ./extensions/zero-mem.ts
```

Use `/zero-mem` to show the retrieval route, engine, selected count, and blocked historical instructions. If the Python environment is unavailable, retrieval falls back to BM25 and reports the reason.

## Check

```bash
bun run check
```

## Paper mapping

- **Raw trace substrate:** original messages on the active branch strictly before the current user turn, including messages hidden by compaction. Sibling branches and the ongoing tool loop are outside the historical boundary.
- **Entity-context graph:** frequency-weighted spaCy/deterministic entities, dense query-entity alignment, relevance-weighted co-occurrence propagation, tree-adjacency edges, and convergent personalized PageRank.
- **Temporal hierarchy:** semantic/time-bounded episodes, sliding windows, turns, and local spans searched coarse-to-fine.
- **Routing and closure:** a deterministic subject/keyword/type/temporal/boundary profile selects the primary view; normalized scores use the paper's `rho = 0.6`; five primary traces receive bounded graph bridges and local neighbors.
- **Calibration:** provenance and boundary filtering, deduplication, temporal conflict resolution, type-aware ranking, XML escaping, and deterministic rejection of instruction-like historical traces.
- **Answer checks:** final short answers are conservatively calibrated when evidence provides a unique number/date correction or supports pruning an extractive list. Tool calls, long prose, code, and ambiguous answers are left unchanged; current-turn tool results participate only in these checks.
- **Compaction:** Pi compaction uses a fixed non-generative checkpoint; original branch traces remain retrievable.

This is a Pi-oriented implementation of the architecture, not a reproduction of the paper's reported benchmark. Thresholds that the paper does not publish use deterministic local defaults.
