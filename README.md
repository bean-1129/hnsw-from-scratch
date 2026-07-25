# hnsw-from-scratch
From-scratch C++17 implementation of HNSW (Malkov &amp; Yashunin, TPAMI 2018) for approximate nearest neighbor search. Algorithms 1-5 straight from the paper, including the diversity heuristic — no hnswlib, no FAISS. Brute-force oracle, Recall@10 evaluator, serialization, unit tests and an efSearch benchmark sweep, measured on glove-100-angular.
