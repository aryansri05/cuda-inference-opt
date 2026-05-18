# Sarvam-2B Inference Benchmark — T4 vs M2

Benchmarking Sarvam-2B inference across Hindi, Tamil, and code-mixed Hindi-English on two architectures:
- NVIDIA T4 GPU (Google Colab) — discrete GPU, 320 GB/s memory bandwidth
- Apple M2 8GB (MacBook Air) — unified memory architecture, 100 GB/s memory bandwidth

---

## Key Findings

### 1. Tamil P95 Tail Latency Issue (T4)
1 in 20 Tamil queries takes over 5 seconds on T4. For production voice workloads, that's an SLA problem.
Hypothesis: attention layer memory access patterns for longer Indic tokenization sequences.

### 2. M2 8GB is Memory Constrained
Sarvam-2B peaks at 5.053 GB — hitting the ceiling of 8GB unified memory.
Result: 4.6 tok/s vs T4's 24.8 tok/s. Not production viable on base tier Apple Silicon.

### 3. Minimum Viable Hardware Profile
| Hardware | Median Latency | Throughput | Viable? |
|---|---|---|---|
| M2 8GB (unified memory) | ~30,000ms | 4.6 tok/s | ❌ Memory constrained |
| T4 GPU (cloud) | ~2,471ms | 24.8 tok/s | ✅ But Tamil P95 issue |
| M2 16GB (estimated) | ~8,000ms | ~12 tok/s | ✅ Likely viable |
| H100 (cloud) | TBD | TBD | ✅ Optimal |

---

## T4 Results (Google Colab)
30 prompts per language — Hindi, Tamil, code-mixed Hindi-English

| Language | Median Latency | Throughput | P95 Latency |
|---|---|---|---|
| Hindi | 2,471ms | 24.8 tok/s | — |
| Tamil | 2,536ms | 22.5 tok/s | 5,477ms ⚠️ |
| Code-mixed | 2,469ms | 25.0 tok/s | — |

**Tamil P95: 5,477ms** — 1 in 20 Tamil queries takes 5+ seconds.

---

## M2 8GB Results (Apple Silicon — MLX)
Single prompt test — memory constrained environment

| Metric | Value |
|---|---|
| Throughput | 4.645 tok/s |
| Time for 50 tokens | 30,433ms |
| Peak memory usage | 5.053 GB |
| Available unified memory | 8 GB |

**Finding:** Sarvam-2B requires minimum 16GB unified memory for viable on-device inference on Apple Silicon.
The 8GB M2 hits memory ceiling, causing significant slowdown for longer generations (Hindi: ~88s, Tamil: ~16s).

---

## Architecture Comparison

| | NVIDIA T4 | Apple M2 8GB |
|---|---|---|
| Memory bandwidth | 320 GB/s | ~100 GB/s |
| Memory available | 16 GB VRAM | 8 GB unified |
| Sarvam-2B headroom | 11 GB free | 3 GB free |
| Throughput | 24.8 tok/s | 4.6 tok/s |
| Tamil P95 | 5,477ms ⚠️ | N/A (memory bound) |

---

## Hypothesis — Tamil P95 Root Cause

Tamil has significantly longer subword sequences than Hindi for equivalent semantic content.
At P95, tokenized sequence length is 2–3x longer than median.
This pushes attention computation into a memory bandwidth bottleneck regime — KV cache reads scale quadratically with sequence length.

**Proposed fix:** Flash Attention 2 for sequences above a threshold length.

---

## Setup

**T4 benchmark:** Google Colab — stock HuggingFace transformers, 30 prompts per language

**M2 benchmark:** Apple MacBook Air M2 8GB — MLX framework
```bash
pip install mlx-lm
python3 benchmark_m2.py
```

---

## Next Steps

- [ ] Run benchmark on M2 16GB to validate estimated throughput
- [ ] Implement Flash Attention 2 kernel for long Indic sequences
- [ ] Profile exact attention layer memory access pattern for Tamil P95 queries
- [ ] H100 benchmark for full hardware comparison

---
