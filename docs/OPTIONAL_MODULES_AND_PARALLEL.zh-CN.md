# 可开启模块接入与并行推进说明

本文说明当前 TMCRA 包中已经保留的两个可开启接入口：

- Embedder 接入口
- LLM planner 接入口

同时说明 S500 基线测试中使用过的并行推进方式，方便后续在部署、评估或消融实验中复用。

## 1. 当前主链路

冻结 S500 基线的核心链路是：

```text
dialogue -> writer layer -> graph memory -> learned node/path scorer -> evidence selection -> answer layer
```

其中：

- 写入层负责把对话转成记忆节点、事件单元、profile 信号、时间信号。
- 图记忆层保存节点、路径和隧穿关系。
- `node_scorer.pt` 和 `path_scorer.pt` 负责学习式节点/路径打分。
- evidence selection 把候选记忆整理成紧凑证据。
- 回答层 LLM 根据证据生成最终回复。

Embedder 和 LLM planner 都是可开启增强模块，不应该替代主图模型。它们更适合作为辅助通道、对比实验或高成本部署路径。

## 2. Embedder 接入口

Embedder 当前有三类接入位置。

### 2.1 写入阶段索引

写入阶段可以为新写入的记忆建立 embedding 索引，后续召回时作为辅助候选来源。

相关配置：

```bash
export TMCRA_EMBEDDER_MODEL_PATH="BAAI/bge-m3"
export TMCRA_EMBEDDER_DEVICE="cuda"
export TMCRA_EMBEDDER_MODEL_MAX_LENGTH="512"
export TMCRA_WRITE_EMBEDDER_INDEX_MODE="bge_m3"
export TMCRA_WRITE_EMBEDDER_INDEX_MAX_TERMS="96"
```

作用：

- 在 writer 写入记忆节点后，为节点文本建立语义索引。
- 不改变原图结构。
- 不替代 node/path scorer，只是给召回增加一条语义候选通道。

### 2.2 召回前候选补充

召回前可以先用 embedder 找到一批候选 event id，再交给图召回和 scorer 做后续排序。

相关配置：

```bash
export TMCRA_EMBEDDER_PRE_RECALL_MODE="bge_m3"
export TMCRA_EMBEDDER_PRE_RECALL_K="16"
export TMCRA_EMBEDDER_INDEX_RECALL_MODE="bge_m3"
export TMCRA_EMBEDDER_INDEX_RECALL_K="24"
```

作用：

- 帮助召回阶段扩大候选范围。
- 对语义相近但图路径弱的记忆提供补充入口。
- 适合测试 query 与 memory 表达不完全一致的场景。

### 2.3 召回后融合加权

召回后可以把 embedder 命中的 event 与主图模型结果融合，让高语义相关的节点获得有限 boost。

相关配置：

```bash
export TMCRA_EMBEDDER_FUSION_MODE="on"
export TMCRA_EMBEDDER_FUSION_WEIGHT="0.35"
export TMCRA_EMBEDDER_FUSION_SCORE_FLOOR="0.62"
export TMCRA_EMBEDDER_FUSION_TOP_K="16"
export TMCRA_EMBEDDER_FUSION_SELECT_K="4"
export TMCRA_EMBEDDER_FUSION_MAX_BOOST="0.42"
```

作用：

- 给语义相似候选增加有限分数。
- 避免 embedder 直接重排主证据。
- 适合作为主图 scorer 的辅助召回层。

## 3. LLM Planner 接入口

LLM planner 当前主要有三类接入方式。它们都位于召回之后或 query 进入召回之前，用于增强证据组织能力。

### 3.1 Evidence-unit planner

Evidence-unit planner 在召回后运行，用 LLM 把候选窗口整理成 evidence unit。

相关配置：

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

如果不单独设置 planner 的 base/model/key，它会继承回答层配置：

```bash
export TMCRA_ANSWER_BASE_URL="<openai-compatible-base-url>"
export TMCRA_ANSWER_MODEL="<answer-model>"
export TMCRA_ANSWER_API_KEY="<answer-api-key>"
```

作用：

- 标注候选窗口里的 answer unit、positive evidence、temporal anchor、current value、old value、constraint、negative evidence。
- 帮助最终回答层理解“这批证据应该怎么用”。
- 默认更适合做证据整理，不建议让它直接替代图召回。

### 3.2 LLM channel planner

LLM channel planner 在最终证据进入回答层前运行，用 LLM 区分 main evidence、coverage evidence、support evidence 和 suppress evidence。

相关配置：

```bash
export TMCRA_LLM_CHANNEL_PLANNER_MODE="on"
export TMCRA_LLM_CHANNEL_PLANNER_MAX_WINDOWS="16"
export TMCRA_LLM_CHANNEL_PLANNER_WINDOW_CHARS="520"
export TMCRA_LLM_CHANNEL_PLANNER_MAX_TOKENS="700"
```

作用：

- 让 coverage 证据补充主事实，而不是替代主事实。
- 对 count/sum/ratio/duration/multi-unit 问题特别有用。
- 成本高于纯模型 scorer，适合高质量模式或实验开关。

冻结 S500 基线记录中，该项为：

```text
llm_channel_planner=off
```

### 3.3 Query graph builder

Query graph builder 在召回前运行，把用户问题转成 query graph，再扩展为 sidecar retrieval queries。

相关配置：

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

作用：

- 把问题拆成 task intent、required units、operation、tunnel needs。
- 对复杂 multi-session、temporal、profile 问题提供更明确的召回方向。
- 适合做对比实验，观察“问题建图后再召回”是否提升候选命中。

## 4. 本地模型 planner 与 LLM planner 的区别

当前代码里也有本地模型 planner 接口，例如：

```bash
export TMCRA_ANSWER_WINDOW_PLANNER_MODE="on"
export TMCRA_ANSWER_WINDOW_PLANNER_MODEL_PATH="<planner-checkpoint>"
export TMCRA_UNIFIED_OPERATION_PLANNER_MODE="on"
export TMCRA_UNIFIED_OPERATION_PLANNER_MODEL_PATH="<planner-checkpoint>"
export TMCRA_INJECTION_PLANNER_MODE="guided"
export TMCRA_INJECTION_PLANNER_MODEL_PATH="<planner-checkpoint>"
```

这些是本地模型接入口，不是 LLM planner。区别是：

- LLM planner：调用外部或本地 LLM，成本更高，适合验证能力上限。
- 本地模型 planner：成本更低，更适合产品化，但需要专项训练和稳定性验证。

建议流程是：

```text
先用 LLM planner 验证能力是否有效 -> 再把有效行为蒸馏/训练进本地图模型或 planner head
```

## 5. 并行推进方案

S500 基线采用过分片并行方式：

```text
500 samples -> 10 shards -> 50 samples per shard
```

每个 shard 独立运行：

```text
input_shard_N.json -> shard_N/ -> predictions/debug/summary
```

核心并行原则：

- 每个 shard 独立进程。
- 每个 shard 独立输出目录。
- writer key pool 按 shard index 轮转。
- 主模型权重只读共享。
- 最终合并 predictions、samples_debug、judge 结果。

冻结 S500 记录中的关键运行配置：

```text
samples=500
shards=10
per_shard=50
writer=DeepSeek v4 Flash
answer_layer=GPT5.4
llm_channel_planner=off
history_mode=controlled_answer_plus_distractors
```

### 5.1 复用的并行模板

推荐的并行模板：

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

单 shard 执行形态：

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

### 5.2 推进顺序

建议按下面顺序推进，不要一次把所有模块全打开：

1. **Baseline scorer-only**
   - embedder off
   - LLM channel planner off
   - query graph builder off
   - 用于确认 frozen baseline 是否稳定。

2. **Embedder pre-recall A/B**
   - 只打开写入索引和召回前候选补充。
   - 观察候选命中率、召回延迟、错误类型是否变化。

3. **Embedder fusion A/B**
   - 在 pre-recall 稳定后打开 fusion。
   - 控制 boost，不允许 embedder 直接压过主图 scorer。

4. **Evidence-unit planner A/B**
   - 打开 LLM evidence-unit planner。
   - 观察 answer 层是否更会使用召回证据。

5. **LLM channel planner A/B**
   - 只在 multi/aggregation/temporal 错误集中验证。
   - 重点观察 coverage 证据是否补充主事实，而不是替换主事实。

6. **Query graph builder A/B**
   - 用于验证“问题建图后再召回”的上限。
   - 如果有效，再考虑训练进 query-understanding 或 graph scorer。

### 5.3 并行规模建议

并行数不要只看 API 数量，还要看：

- GPU 显存
- CPU 内存
- writer 延迟
- answer 层延迟
- graph ingest/SQLite 写入开销
- 每 shard 平均 writer calls

建议从小到大：

```text
5 shards smoke -> 10 shards stable -> 20 shards stress -> 30 shards only if no memory/API/IO issue
```

如果出现错误率升高、内存下降明显、API 402/429、chunk error 或 shard 卡住，应先降并行，再补跑缺失样本。

## 6. 推荐实验矩阵

最小可解释矩阵：

| 实验 | Embedder | LLM planner | 目的 |
| --- | --- | --- | --- |
| baseline | off | off | 固定主图模型基线 |
| embedder-pre | pre-recall on | off | 测候选扩展是否提升 |
| embedder-fusion | pre-recall + fusion on | off | 测语义融合是否提升 |
| evidence-unit | off | evidence-unit on | 测回答前证据整理 |
| channel-planner | off | channel planner on | 测 main/coverage 分离 |
| query-graph | off | query graph on | 测问题建图召回 |
| combined-light | pre-recall on | evidence-unit on | 测较低成本组合 |
| combined-heavy | pre-recall + fusion on | evidence-unit + channel planner on | 测能力上限 |

每一组都应保留：

- predictions
- samples_debug
- judge output
- by-task accuracy
- writer calls
- retrieval latency
- answer latency
- per-sample error type

这样后续可以判断问题来自召回、证据选择、planner、回答层，还是并行运行不稳定。
