# pi-zero-mem

A Pi adaptation of [Zero-Mem: Zero-Token Memory Operations for LLM Agents](https://arxiv.org/abs/2607.29377).

It keeps Pi's original session messages as the source of record, retrieves relevant old turns through graph and temporal views, and sends only the retrieved evidence plus the current turn to the reader model. spaCy extracts entities and BGE-M3 supplies dense relevance scores without LLM calls.

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

- **Raw trace substrate:** all message entries in the current Pi session tree, including sibling branches and messages hidden by compaction.
- **Entity-context graph:** spaCy NER plus deterministic file/name extraction, co-occurrence edges, tree-adjacency edges, and personalized PageRank.
- **Temporal hierarchy:** turn, sliding-window, and timestamp-gap episode scores.
- **Routing and closure:** query cues select the primary view; normalized scores use the paper's `rho = 0.6`; five primary traces receive bounded graph and local neighbors.
- **Calibration:** provenance filtering, deduplication, hybrid BM25/BGE-M3 ranking, XML escaping, and deterministic rejection of instruction-like historical traces.
- **Compaction:** Pi compaction uses a fixed non-generative checkpoint; original branch traces remain retrievable.

This is a Pi-oriented implementation of the architecture, not a reproduction of the paper's reported benchmark. It does not rewrite final agent answers.
