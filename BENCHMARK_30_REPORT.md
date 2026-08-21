# 30-Question Retrieval Benchmark

## Scope

The benchmark evaluates the existing 191-chunk WHO hypertension corpus on 30 questions:

- 10 primary questions with reviewed source pages.
- 10 held-out paraphrase questions with the same reviewed labels.
- 10 out-of-scope negative controls.

Command:

```powershell
python benchmark_30.py
```

Results are stored in `benchmark_30_results.json`.

## Results at K=3

| Architecture | Primary P@3 | Primary PageRecall@3 | Paraphrase P@3 | Paraphrase PageRecall@3 | Negative abstain | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense bge-small | 56.7% | 100% | 43.3% | 80% | 100% | 44.1 / 33.1 ms |
| BM25 | 43.3% | 90% | 36.7% | 80% | Not calibrated | 0.9 / 0.7 ms |
| RRF Dense + BM25 | **60.0%** | **100%** | **46.7%** | **100%** | **100%** | 32.7 / 43.7 ms |

### Weighted RRF experiment

The benchmark also tested Dense/Sparse weights `1.5/1.0`, `2.0/1.0`, and `1.0/1.5` after caching the candidate lists. None improved both positive sets over equal-weight RRF:

- `1.5/1.0`: primary P@3 `60.0%`, paraphrase P@3 `43.3%`.
- `2.0/1.0`: primary P@3 `60.0%`, paraphrase P@3 `43.3%`.
- `1.0/1.5`: primary P@3 `53.3%`, paraphrase P@3 `46.7%`.

Equal-weight RRF remains the selected fusion policy. A medical-tokenizer experiment was also rejected because it reduced equal-weight RRF to primary P@3 `56.7%` and paraphrase P@3 `40.0%` on this corpus.

### Query expansion experiment

The opt-in clinical synonym expansion variant produced primary P@3 `56.7%`, paraphrase P@3 `46.7%`, and negative abstention `90%`. Because it reduced primary precision and introduced a false-positive negative control, it is rejected for production. The expansion helper remains available only for future per-query routing experiments.

### Page diversification experiment

Selecting at most one result per document/page from the top-15 RRF candidates produced primary P@3 `53.3%`, paraphrase P@3 `46.7%`, and negative abstention `100%`. It is rejected because it lowers primary precision; repeated chunks from one page can carry distinct supporting details and should not be removed blindly.

### Chunk ablation status

`chunk_ablation.py` was started against temporary collections and stopped during the first Chroma collection initialization after repeated telemetry errors, before any configuration produced a result. The production vectorstore was not modified, so the current `500/75` configuration remains unchanged until this experiment can run in a clean compatible Chroma environment.

The two latency values are primary/paraphrase mean latency respectively.

## Decision

RRF is the best measured quality path for this corpus:

- It improves primary Precision@3 over Dense by 3.3 percentage points.
- It improves paraphrase Precision@3 by 3.4 percentage points.
- It raises paraphrase PageRecall@3 from 80% to 100%.
- It preserves 100% negative-control abstention when the dense score is retained for the guard.
- Its latency remains in the same practical range as Dense for this 191-chunk corpus.

RRF scores are rank-fusion scores, not probabilities. Confidence and abstention must continue to use the preserved `dense_score`, while `rrf_score` is used for ranking only.

## Important Caveats

- The benchmark has 20 positive questions and 10 negatives; it is stronger than the original 10-question development set but is still not an external clinical validation set.
- The current runtime result differs from older report values (`63.3%` Dense P@3); the fresh benchmark output is authoritative for the current 191-chunk index and code.
- FlashRank reranker benchmarking was attempted with `python benchmark_30.py --with-rerank`, but the process stalled after model initialization in the current environment. It was stopped and is not used in the production recommendation.
- BM25 alone does not provide a calibrated dense relevance score for the abstention guard.

## Recommended Rollout

1. Keep Dense as a fallback and baseline.
2. Use RRF for an offline benchmarked retrieval mode after adding a feature flag.
3. Keep the dense score as the safety/abstention signal.
4. Do not enable FlashRank by default until its 30-question benchmark completes reliably and its latency/quality tradeoff is documented.
5. Expand the evaluation set with independently reviewed questions before calling the system clinically validated.
