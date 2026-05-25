# TMCRA Long-Memory Runtime

TMCRA is a graph-based long-memory runtime for agent systems. It is designed to help an LLM retrieve, connect, and reason over long dialogue histories without exposing the full conversation context on every turn.

This release contains a frozen TMCRA baseline package with model weights, runtime code snapshots, and LongMemEval S500 benchmark results.

## Why TMCRA

Long-running agents need more than simple vector recall. They need to preserve user facts, preferences, timeline changes, cross-session events, and multi-step evidence chains.

TMCRA organizes memory into graph nodes and learned retrieval paths, then surfaces compact evidence to the answer model. The goal is to let external agents use long-term memory through a runtime/API layer while keeping the memory algorithm and model weights independently deployable.

## Benchmark Result

This package includes a full LongMemEval S500 run.

- Benchmark: LongMemEval S set, 500 samples
- Evaluation: official-compatible LongMemEval judge prompt
- Judge model: `gpt-4o`, resolved as `gpt-4o-2024-08-06`
- Answer layer used in this run: GPT5.4-compatible API
- Overall accuracy: `310 / 500 = 62.00%`

## Results by Task Type

| task type | accuracy | count |
| --- | ---: | ---: |
| single-session-user | 81.43% | 70 |
| single-session-assistant | 78.57% | 56 |
| knowledge-update | 70.51% | 78 |
| temporal-reasoning | 63.16% | 133 |
| single-session-preference | 56.67% | 30 |
| multi-session | 39.85% | 133 |

## Included Artifacts

- `code/`: runtime and evaluation code snapshot for this baseline.
- `models/action_frame_tunnel_graph548_tunnel_fusion_train_20260524_042557/`: full trained graph-model output directory.
- `results/`: predictions, judge output, summary metrics, and compressed run artifacts.
- `docs/`: baseline record and result notes.

## Model Package

The included model package preserves the full training output for the graph scorer stack:

- `node_scorer.pt` and `path_scorer.pt`: runtime graph scoring weights.
- `node_scorer_best.pt` and `path_scorer_best.pt`: best checkpoint aliases.
- `node_scorer_last.pt` and `path_scorer_last.pt`: final training aliases.
- `checkpoints/`: epoch and step checkpoints.
- `export_manifest.json`, `train_summary.json`, and `train.log`: model metadata and training trace.

## Current Strengths

- Strong direct user-fact recall in single-session settings.
- Strong assistant-detail recall.
- Competitive knowledge-update behavior for changing facts.
- Working temporal and preference retrieval layers with clear room for further specialization.

## Active Improvement Areas

- Multi-session aggregation and unit coverage.
- Deeper time-graph reasoning.
- Preference-profile abstraction and cross-session tunneling.
- Query-graph to memory-graph matching for complex questions.

## Intended Use

This repository is a public-facing evidence package for TMCRA's long-memory runtime work. It is suitable for:

- Benchmark review.
- Model and result inspection.
- Reproducing the frozen baseline.
- Demonstrating how TMCRA can be packaged as an external memory runtime for agents.
