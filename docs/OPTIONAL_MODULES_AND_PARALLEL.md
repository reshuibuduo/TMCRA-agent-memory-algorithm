# Optional Modules and Parallel Evaluation Plan

This document describes two optional extension points already preserved in the current TMCRA package:

- Embedder interface
- LLM planner interface

It also summarizes the parallel evaluation pattern used for the S500 baseline so later deployment, evaluation, and ablation runs can reuse the same structure.

## 1. Current Main Path

The frozen S500 baseline uses this core path:

```text
dialogue -> writer layer -> graph memory -> learned node/path scorer -> evidence selection -> answer layer
```

The responsibilities are:

- The writer layer converts dialogue into memory nodes, event units, profile signals, and temporal signals.
- The graph memory layer stores nodes, paths, and tunnel links.
- `node_scorer.pt` and `path_scorer.pt` perform learned node/path scoring.
- Evidence selection converts candidate memories into compact evidence.
- The answer-layer LLM produces the final response from the selected evidence.

Embedder and LLM planner modules are optional enhancement modules. They should not replace the main graph model. They are better treated as auxiliary channels, ablation switches, or higher-cost deployment paths.

## 2. Embedder Interface

The current code exposes three embedder integration points.

### 2.1 Write-time Indexing

At write time, TMCRA can build an embedding index for newly written memory nodes. That index can later serve as an auxiliary candidate source during retrieval.

Relevant configuration:

```bash
export TMCRA_EMBEDDER_MODEL_PATH="BAAI/bge-m3"
export TMCRA_EMBEDDER_DEVICE="cuda"
export TMCRA_EMBEDDER_MODEL_MAX_LENGTH="512"
export TMCRA_WRITE_EMBEDDER_INDEX_MODE="bge_m3"
export TMCRA_WRITE_EMBEDDER_INDEX_MAX_TERMS="96"
```

Purpose:

- Build semantic indexes after the writer stores memory nodes.
- Keep the original graph structure unchanged.
- Add a semantic candidate channel without replacing the learned node/path scorers.

### 2.2 Pre-recall Candidate Expansion

Before graph retrieval, the embedder can find candidate event ids, which are then passed into graph retrieval and scorer ranking.

Relevant configuration:

```bash
export TMCRA_EMBEDDER_PRE_RECALL_MODE="bge_m3"
export TMCRA_EMBEDDER_PRE_RECALL_K="16"
export TMCRA_EMBEDDER_INDEX_RECALL_MODE="bge_m3"
export TMCRA_EMBEDDER_INDEX_RECALL_K="24"
```

Purpose:

- Expand the candidate range before retrieval.
- Help when query wording and memory wording differ.
- Provide an auxiliary path for semantically close memories with weak graph paths.

### 2.3 Post-recall Fusion

After retrieval, embedder-matched events can be fused with graph-model results so semantically relevant nodes receive a limited boost.

Relevant configuration:

```bash
export TMCRA_EMBEDDER_FUSION_MODE="on"
export TMCRA_EMBEDDER_FUSION_WEIGHT="0.35"
export TMCRA_EMBEDDER_FUSION_SCORE_FLOOR="0.62"
export TMCRA_EMBEDDER_FUSION_TOP_K="16"
export TMCRA_EMBEDDER_FUSION_SELECT_K="4"
export TMCRA_EMBEDDER_FUSION_MAX_BOOST="0.42"
```

Purpose:

- Give semantically similar candidates a bounded score boost.
- Prevent the embedder from directly replacing main evidence ranking.
- Use embedding as an auxiliary recall layer for the learned graph scorer.

## 3. LLM Planner Interface

The current code exposes three main LLM planner paths. They run either after retrieval or before query-side retrieval expansion.

### 3.1 Evidence-unit Planner

The evidence-unit planner runs after retrieval and uses an LLM to normalize retrieved windows into evidence units.

Relevant configuration:

```bash
export TMCRA_EVIDENCE_UNIT_PLANNER_MODE="on"
export TMCRA_EVIDENCE_UNIT_PLANNER_BASE_URL="<openai-compatible-base-url>"
export TMCRA_EVIDENCE_UNIT_PLANNER_MODEL="<planner-model>"
export TMCRA_EVIDENCE_UNIT_PLANNER_API_KEY="<planner-api-key>"
export TMCRA_EVIDENCE_UNIT_PLANNER_MAX_CANDIDATES="10"
export TMCRA_EVIDENCE_UNIT_PLANNER_CHARS="1100"
export TMCRA_EVIDENCE_UNIT_PLANNER_MAX_TOKENS="760"
export TMCRA_EVIDENCE_UNIT_PLANNER_REORDER="0"
```

If planner-specific base/model/key values are not configured, this planner inherits answer-layer configuration:

```bash
export TMCRA_ANSWER_BASE_URL="<openai-compatible-base-url>"
export TMCRA_ANSWER_MODEL="<answer-model>"
export TMCRA_ANSWER_API_KEY="<answer-api-key>"
```

Purpose:

- Mark answer units, positive evidence, temporal anchors, current values, old values, constraints, and negative evidence.
- Help the final answer layer understand how to use the retrieved evidence.
- Organize evidence without replacing graph retrieval.

### 3.2 LLM Channel Planner

The LLM channel planner runs before final evidence is sent to the answer layer. It separates main evidence, coverage evidence, support evidence, and suppressed evidence.

Relevant configuration:

```bash
export TMCRA_LLM_CHANNEL_PLANNER_MODE="on"
export TMCRA_LLM_CHANNEL_PLANNER_MAX_WINDOWS="16"
export TMCRA_LLM_CHANNEL_PLANNER_WINDOW_CHARS="520"
export TMCRA_LLM_CHANNEL_PLANNER_MAX_TOKENS="700"
```

Purpose:

- Make coverage evidence supplement main facts instead of replacing them.
- Improve count/sum/ratio/duration/multi-unit tasks.
- Provide a higher-cost quality mode for experiments and selected deployments.

In the frozen S500 baseline record, this module was:

```text
llm_channel_planner=off
```

### 3.3 Query Graph Builder

The query graph builder runs before retrieval. It converts the user question into a compact query graph and can expand that graph into sidecar retrieval queries.

Relevant configuration:

```bash
export TMCRA_QUERY_GRAPH_BUILDER_MODE="on"
export TMCRA_QUERY_GRAPH_BASE_URL="<openai-compatible-base-url>"
export TMCRA_QUERY_GRAPH_MODEL="<query-graph-model>"
export TMCRA_QUERY_GRAPH_API_KEY="<query-graph-api-key>"
export TMCRA_QUERY_GRAPH_MAX_TOKENS="700"
export TMCRA_QUERY_GRAPH_SIDECAR_RETRIEVAL_MODE="on"
export TMCRA_QUERY_GRAPH_SIDECAR_MAX_QUERIES="6"
export TMCRA_QUERY_GRAPH_SIDECAR_TOP_K="4"
```

Purpose:

- Convert the question into task intent, required units, operation, and tunnel needs.
- Give complex multi-session, temporal, and profile questions a clearer retrieval direction.
- Test whether building a query graph before retrieval improves candidate recall.

## 4. Local Model Planner vs LLM Planner

The code also contains local model planner interfaces, for example:

```bash
export TMCRA_ANSWER_WINDOW_PLANNER_MODE="on"
export TMCRA_ANSWER_WINDOW_PLANNER_MODEL_PATH="<planner-checkpoint>"
export TMCRA_UNIFIED_OPERATION_PLANNER_MODE="on"
export TMCRA_UNIFIED_OPERATION_PLANNER_MODEL_PATH="<planner-checkpoint>"
export TMCRA_INJECTION_PLANNER_MODE="guided"
export TMCRA_INJECTION_PLANNER_MODEL_PATH="<planner-checkpoint>"
```

These are local model interfaces, not LLM planner interfaces.

The distinction is:

- LLM planner: calls an external or local LLM; higher cost; useful for validating capability ceilings.
- Local model planner: lower cost and better for productization, but requires targeted training and stability validation.

Recommended workflow:

```text
validate behavior with an LLM planner -> distill or train the useful behavior into the graph model or a local planner head
```

## 5. Parallel Evaluation Plan

The S500 baseline used shard-level parallelism:

```text
500 samples -> 10 shards -> 50 samples per shard
```

Each shard runs independently:

```text
input_shard_N.json -> shard_N/ -> predictions/debug/summary
```

Core parallelization principles:

- One independent process per shard.
- One independent output directory per shard.
- Writer key pool is rotated by shard index.
- Main model weights are read-only and shared.
- Predictions, samples_debug, and judge results are merged after all shards complete.

Key S500 baseline runtime configuration:

```text
samples=500
shards=10
per_shard=50
writer=DeepSeek v4 Flash
answer_layer=GPT5.4
llm_channel_planner=off
history_mode=controlled_answer_plus_distractors
```

### 5.1 Reusable Parallel Template

Recommended baseline template:

```bash
export TMCRA_RETRIEVAL_MODE="hybrid_node_scored"
export TMCRA_REQUIRE_LEARNED_SCORER="1"
export TMCRA_NODE_MODEL_DEVICE="cuda"
export TMCRA_NODE_MODEL_PATH="models/action_frame_tunnel_graph548_tunnel_fusion_train_20260524_042557/node_scorer.pt"
export TMCRA_PATH_MODEL_PATH="models/action_frame_tunnel_graph548_tunnel_fusion_train_20260524_042557/path_scorer.pt"

export TMCRA_WRITER_MODEL="deepseek-chat"
export TMCRA_WRITER_MAX_TOKENS="512"
export TMCRA_WRITER_TIMEOUT_SECONDS="180"
export TMCRA_WRITER_TEMPERATURE="0"
export TMCRA_WRITER_INPUT_MODE="delta"
export TMCRA_WRITER_MAX_PROPOSALS="2"

export TMCRA_ANSWER_MAX_TOKENS="512"
```

Single-shard execution shape:

```bash
python code/run_lme_s10_native_tmcra.py \
  --data "<run-root>/input_shard_N.json" \
  --repo "<tmcra-repo-root>" \
  --service-root "<tmcra-service-root>" \
  --out "<run-root>/shard_N" \
  --limit 50 \
  --top-k 10 \
  --max-distractor-sessions 5 \
  --max-distractor-chunks 1 \
  --max-answer-chunks 4 \
  --chunk-chars 7000
```

### 5.2 Suggested Rollout Order

Do not enable all optional modules at once. Use staged A/B testing:

1. **Baseline scorer-only**
   - embedder off
   - LLM channel planner off
   - query graph builder off
   - confirms frozen baseline stability

2. **Embedder pre-recall A/B**
   - enable write-time indexing and pre-recall candidate expansion only
   - measure candidate hit rate, retrieval latency, and error-type shifts

3. **Embedder fusion A/B**
   - enable fusion only after pre-recall is stable
   - keep boost bounded so embedder does not override the main graph scorer

4. **Evidence-unit planner A/B**
   - enable LLM evidence-unit planner
   - measure whether the answer layer uses retrieved evidence better

5. **LLM channel planner A/B**
   - test mainly on multi/aggregation/temporal error clusters
   - verify coverage evidence supplements main facts instead of replacing them

6. **Query graph builder A/B**
   - validate the ceiling of query-graph-first retrieval
   - if effective, distill the behavior into query-understanding or graph scorer training

### 5.3 Parallel Scale Guidance

Parallelism should not be determined only by the number of API keys. Also watch:

- GPU memory
- CPU memory
- writer latency
- answer-layer latency
- graph ingest / SQLite write overhead
- average writer calls per shard

Scale gradually:

```text
5 shards smoke -> 10 shards stable -> 20 shards stress -> 30 shards only if no memory/API/IO issue
```

If error rate rises, memory drops sharply, API 402/429 appears, chunk errors occur, or shards stall, reduce parallelism first and then resume missing samples.

## 6. Recommended Experiment Matrix

Minimal interpretable matrix:

| experiment | Embedder | LLM planner | purpose |
| --- | --- | --- | --- |
| baseline | off | off | fixed main graph-model baseline |
| embedder-pre | pre-recall on | off | test candidate expansion |
| embedder-fusion | pre-recall + fusion on | off | test semantic fusion |
| evidence-unit | off | evidence-unit on | test pre-answer evidence organization |
| channel-planner | off | channel planner on | test main/coverage separation |
| query-graph | off | query graph on | test query-graph retrieval |
| combined-light | pre-recall on | evidence-unit on | test lower-cost combined path |
| combined-heavy | pre-recall + fusion on | evidence-unit + channel planner on | test capability ceiling |

Each run should preserve:

- predictions
- samples_debug
- judge output
- by-task accuracy
- writer calls
- retrieval latency
- answer latency
- per-sample error type

This makes it possible to separate recall errors, evidence-selection errors, planner errors, answer-layer errors, and parallel-runtime instability.
