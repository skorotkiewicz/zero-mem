# pi-zero-mem

A dependency-free Pi adaptation of [Zero-Mem: Zero-Token Memory Operations for LLM Agents](https://arxiv.org/abs/2607.29377).

It keeps Pi's original session messages as the source of record, retrieves relevant old turns through graph and temporal views, and sends only the retrieved evidence plus the current turn to the reader model. Memory operations make no LLM calls.

## Install

```bash
pi install .
```

For development:

```bash
pi -e ./extensions/zero-mem.ts
```

Use `/zero-mem` to show the most recent retrieval statistics.

## Check

```bash
bun run check
```

## Paper mapping

- **Raw trace substrate:** current Pi session branch, including messages hidden by compaction.
- **Entity-context graph:** deterministic entity extraction, co-occurrence edges, trace-adjacency edges, and personalized PageRank.
- **Temporal hierarchy:** turn, sliding-window, and timestamp-gap episode scores.
- **Routing and closure:** query cues select the primary view; normalized scores use the paper's `rho = 0.6`; five primary traces receive bounded graph and local neighbors.
- **Calibration:** provenance filtering, deduplication, relevance ranking, and an explicit instruction that historical traces are evidence rather than current commands.
- **Compaction:** Pi compaction uses a fixed non-generative checkpoint; original branch traces remain retrievable.

This is a Pi-oriented implementation of the architecture, not a reproduction of the paper's reported benchmark. It replaces spaCy and BGE-M3 with deterministic lexical heuristics to avoid extra runtimes and encoder downloads. It indexes the current session branch only and does not rewrite final agent answers.
