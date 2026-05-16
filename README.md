# Sarvam-2B Inference Benchmark
Benchmarking Sarvam-2B on Hindi, Tamil, and code-mixed inference using HuggingFace transformers on T4 GPU.

## Key finding
Tamil P95 tail latency: 5477ms — 1 in 20 Tamil queries takes 5+ seconds.
Likely cause: attention layer memory access pattern for longer Indic tokenization sequences.

## Results
| Language | Median latency | Throughput |
|---|---|---|
| Hindi | 2471ms | 24.8 tok/s |
| Tamil | 2536ms | 22.5 tok/s |
| Code-mixed | 2469ms | 25.0 tok/s |
| Tamil P95 | 5477ms | — |

## Setup
GPU: T4 (Google Colab) | Prompts: 30 per language | Baseline: stock HuggingFace transformers
