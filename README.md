# Self-Evolving Inference System

**$100 of Claude doubles the throughput of Qwen3-0.6B on an RTX PRO 6000 Blackwell — no human intervention required.**

A minimal scaffold that lets [Claude Code](https://claude.com/claude-code) iteratively optimize an LLM inference engine across sequential trials. Over 10 trials, Claude evolved [mini-sglang](https://github.com/smth-fun/mini-sglang) from **512 → 1,016 tok/s (+98%)** — writing fused Triton/CUDA kernels, adding INT8 quantization, restructuring CUDA graphs, and tuning the scheduler — all autonomously.

## How it works

The scaffold has four components:

1. **Fresh sessions** — each trial is a new Claude Code session with no conversation history carried over. The agent starts clean every time.
2. **Frozen benchmark** — `mini-sglang/benchmark/offline/bench_simple.py` is the single source of truth. The agent cannot modify it. This prevents the agent from gaming the metric.
3. **Scalar signal** — the only optimization target is throughput (tokens/second). Correctness is enforced by exact output match under greedy decoding.
4. **Shared `learning.md`** — the agent's memory between sessions. Each trial reads prior learnings, runs experiments, and appends what it discovered. Over time this file accumulates a growing "do NOT try" list alongside proven techniques.

## Results

| Phase | Trials | Throughput | Gain | Key techniques |
|---|---|---|---|---|
| Baseline | — | 512 tok/s | — | Vanilla mini-sglang |
| Kernel fusion | 1–3 | 676 tok/s | +32% | Fused RMSNorm+GEMV, fused QKNorm+RoPE+KV, split-K attention |
| Quantization | 4–6 | 860 tok/s | +27% | INT8 weight quantization, fused dequant |
| System-level | 7–10 | 1,016 tok/s | +18% | Multi-step CUDA graphs, in-graph argmax, decode metadata pre-allocation |

See the evolved code on the [`trial-10`](https://github.com/smth-fun/mini-sglang/tree/trial-10) branch.

## Reproducing

```bash
# Clone (with submodule)
git clone --recurse-submodules https://github.com/smth-fun/self-evolving-inference-system.git
cd self-evolving-inference-system

# Set up the environment
bash setup.sh

# (Optional) Edit run_trial.sh to configure GPU, number of trials, budget per trial

# Run the evolution
bash run_trial.sh
```

**Requirements:**
- NVIDIA GPU with CUDA support
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (for venv/package management)
- [Claude Code](https://claude.com/claude-code) CLI installed and authenticated

## Blog posts

- [Self-Evolving Inference: When AI Agents Write GPU Kernels](posts/post-1-kernel.md) — technical analysis of the 3,700 lines of Triton/CUDA code Claude wrote, including 8 clever optimizations and 3 significant bugs
- [Optimizing Mini-Sglang Inference as an Evolution Problem](posts/post-2-evolve.md) — analysis of the optimization process itself: emergent behaviors, failure memory, and why scaffold design matters more than orchestration

## Acknowledgments

This project was inspired by Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch), which demonstrated the potential of letting AI agents autonomously drive research and optimization loops.

## License

MIT
