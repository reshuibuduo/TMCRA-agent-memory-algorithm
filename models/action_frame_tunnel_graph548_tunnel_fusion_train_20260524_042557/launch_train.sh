#!/usr/bin/env bash
set -euo pipefail
cd <tmcra-repo-root>
DATA=outputs/action_frame_tunnel_graph_dataset_548_event_units_20260524/dataset
OUT="$1"
<tmcra-service-root>/.venv/bin/python scripts/train_locomo_node_memory.py \
  --data-dir "$DATA" \
  --output-dir "$OUT" \
  --resume-checkpoint <tmcra-service-root>/models/tmcra_node_memory_v11_priorfusion_zeroload_20260517/node_scorer.pt \
  --resume-weights-only \
  --trainable-stage tunnel_fusion_only \
  --epochs 8 \
  --batch-size 8 \
  --lr 8e-05 \
  --fail-on-bad-query-rows \
  --epoch-train-eval-max-rows 80 \
  --epoch-val-eval-max-rows 32 \
  --graph-memory-cache-size 64 \
  --lazy-prefetch-workers 6 \
  --lazy-prefetch-window 24 \
  --batch-prepare-workers 16 \
  --batch-prepare-lookahead-batches 64 \
  --graph-prefetch-lookahead-batches 3 \
  --reserve-cpu-cores 2 \
  --torch-cpu-threads 4 \
  --torch-interop-threads 4 \
  --log-every-steps 20 \
  --checkpoint-every-steps 50 \
  --keep-step-checkpoints 2 \
  --train-sampling-mode source_aware_balanced \
  --loss-group-balancing-mode supervision_bucket \
  --loss-source-alpha 0.4 \
  --loss-blend-uniform-ratio 0.25 \
  --loss-weight-power 0.5 \
  --loss-time-boost 1.45 \
  --loss-multi-evidence-boost 2.0 \
  --loss-temporal-positive-boost 1.2 \
  --loss-min-example-weight 0.65 \
  --loss-max-example-weight 1.85 \
  --l2sp-loss-weight 0.02 \
  --event-selection-positive-coverage-count 3 \
  --path-selection-positive-coverage-count 3 \
  --multi-positive-coverage-fraction 0.7 \
  --multi-positive-recall-coverage-count 6 \
  --multi-positive-event-coverage-count 5 \
  --multi-positive-path-coverage-count 3 \
  --multi-positive-final-event-set-coverage-count 5 \
  > "$OUT/train.log" 2>&1
