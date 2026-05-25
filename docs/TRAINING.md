# TMCRA Training Notes

This document summarizes the training direction behind the graph scorer package included in this release.

## Training Goal

TMCRA trains graph-scoring components for long-memory retrieval. The goal is to help the runtime decide which memory nodes and graph paths should be surfaced to an answer model for a given user query.

The trained model is not intended to replace the answer LLM. It is responsible for memory selection:

- identify relevant memory nodes
- score graph paths between related memory events
- preserve useful cross-turn and cross-session links
- reduce noisy or stale evidence before answer generation

## Model Components

The released model directory contains two main runtime scorers:

- `node_scorer.pt`: scores candidate memory nodes.
- `path_scorer.pt`: scores graph paths and tunnel links between memory nodes.

The training output also includes:

- best checkpoints
- last checkpoints
- epoch and step checkpoints
- training summary and logs
- export manifest

## Training Data Direction

Training data is built around dialogue-memory behavior rather than isolated QA pairs. Samples are designed to teach the model how memory should connect across turns and sessions.

The major training directions include:

- direct user facts
- assistant-provided details
- preference/profile signals
- temporal state changes
- old-value vs current-value selection
- cross-session event links
- multi-evidence aggregation
- evidence-positive vs noise/negative memory separation
- unit-to-unit coverage for count/sum/compare tasks

## Graph Memory Supervision

Each training example is converted into graph-oriented supervision. Instead of only asking whether a text chunk is relevant, TMCRA trains over:

- memory node relevance
- event-unit relevance
- path usefulness
- tunnel/link usefulness
- evidence role
- currentness and temporal state
- whether a candidate should be injected into answer context

This allows the runtime to learn memory structure, not only lexical similarity.

## Writer and Scorer Separation

TMCRA separates memory writing from graph scoring.

The writer extracts candidate memory records from dialogue. The graph model then learns how those records should be selected and connected during retrieval.

This separation is important because a long-memory system needs two different abilities:

- write useful memory units from conversation
- retrieve and connect the right units later under noise

## Training Output Included

The packaged model output is located at:

```text
models/action_frame_tunnel_graph548_tunnel_fusion_train_20260524_042557/
```

Runtime files:

```text
node_scorer.pt
path_scorer.pt
export_manifest.json
```

Full training trace:

```text
checkpoints/
node_scorer_best.pt
path_scorer_best.pt
node_scorer_last.pt
path_scorer_last.pt
train_summary.json
train.log
training_issues.jsonl
```

## Current Training Lessons

The current baseline shows that TMCRA has strong single-session fact recall and assistant-detail recall. It also has working temporal and preference layers.

The main remaining training targets are:

- stronger multi-session aggregation
- better unit coverage for count/sum/compare questions
- deeper temporal graph planning
- query-graph to memory-graph matching
- more stable preference abstraction under indirect user requests

These directions are the next step for improving TMCRA from a working long-memory runtime into a stronger general agent-memory layer.
