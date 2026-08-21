# LLM-Benchmarks
# LLM Inference Batching Benchmark
 The goal of this project is to show how batching makes LLm
**Headline result:** batching lifts throughput **~73×** (25 → 1,850 tokens/sec) as
batch size grows from 1 to 256, while single-request latency stays flat until the
GPU saturates around batch 32–64.