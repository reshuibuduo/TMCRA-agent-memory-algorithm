# TMCRA S500 Frozen Baseline 38 Release

This repository snapshot packages the TMCRA frozen baseline used for the LongMemEval S500 full-history run completed on 2026-05-25.

## Result

- Benchmark: LongMemEval S set, 500 samples
- Evaluation prompt: official-compatible LongMemEval judge
- Judge model: `gpt-4o`, resolved as `gpt-4o-2024-08-06`
- Answer layer: GPT5.4-compatible API during the run
- Overall accuracy: `310 / 500 = 62.00%`

## By Question Type

| question type | correct rate | count |
| --- | ---: | ---: |
| single-session-user | 81.43% | 70 |
| single-session-assistant | 78.57% | 56 |
| knowledge-update | 70.51% | 78 |
| temporal-reasoning | 63.16% | 133 |
| single-session-preference | 56.67% | 30 |
| multi-session | 39.85% | 133 |

## Packaged Contents

- `code/`: key runtime/evaluation code snapshot used around this baseline.
- `models/action_frame_tunnel_graph548_tunnel_fusion_train_20260524_042557/`: full graph model output directory, including final, best, last, and checkpoint weights.
- `results/`: S500 predictions, official-compatible judge outputs, summary, and the compressed full result archive.
- `docs/`: notes and baseline documentation.

## Model Files

The full model directory is intentionally included. It contains:

- `node_scorer.pt`, `path_scorer.pt`: runtime model weights.
- `node_scorer_best.pt`, `path_scorer_best.pt`: best checkpoint aliases.
- `node_scorer_last.pt`, `path_scorer_last.pt`: last checkpoint aliases.
- `checkpoints/`: epoch and step checkpoints for full training trace preservation.
- `export_manifest.json`, `train_summary.json`, `train.log`: model metadata and training logs.

Large binary artifacts are tracked through Git LFS.

## Notes

This is a baseline/reproducibility package, not a polished public SDK. It is intended to preserve the exact run artifacts and model weights for later comparison, regression checks, and paper/project evidence.
