# Quire eval report

## Retrieval ablation (60 hand-checked questions, PEP-level relevance)

| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | p50 latency |
|---|---|---|---|---|---|---|
| BM25 | 66.7% | 83.3% | 91.7% | 95.0% | 0.771 | 0.4 ms |
| Dense (bge-small) | 78.3% | 95.0% | 98.3% | 100.0% | 0.875 | 65.8 ms |
| Hybrid (RRF) | 75.0% | 88.3% | 93.3% | 98.3% | 0.830 | 56.1 ms |
| Hybrid + cross-encoder rerank | 85.0% | 93.3% | 96.7% | 100.0% | 0.899 | 754.4 ms |
