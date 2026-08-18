
import time
import csv
import torch

from model import ModelRunner
from engine import static_batch


def make_prompts(n):
    """n copies of a prompt. This is done so that its all standardized bc dif prompts have dif lengths."""
    return ["Explain italian brainrot."] * n


@torch.inference_mode()
def benchmark_batch_size(runner, batch_size, max_new_tokens=128):

    static_batch(make_prompts(batch_size), runner.model, runner.tok, max_new_tokens=8)

    torch.cuda.synchronize()              
    t0 = time.perf_counter()

    static_batch(make_prompts(batch_size), runner.model, runner.tok,
                 max_new_tokens=max_new_tokens)

    torch.cuda.synchronize()           
    wall = time.perf_counter() - t0

    total_tokens = batch_size * max_new_tokens

    throughput = total_tokens / wall 
    latency = wall

    return {
        "batch_size": batch_size,
        "wall_s": round(wall, 3),
        "throughput_tok_s": round(throughput, 1),
        "latency_s": round(latency, 3),
    }


def sweep(runner, batch_sizes=(1, 2, 4, 8, 16, 32), max_new_tokens=128):
    rows = []
    for b in batch_sizes:
        row = benchmark_batch_size(runner, b, max_new_tokens)
        print(row)
        rows.append(row)
    return rows


if __name__ == "__main__":
    runner = ModelRunner()
    rows = sweep(runner)
    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nsaved results.csv")