# Generation Scripts

Interactive streaming chat scripts that read prompts from stdin and stream generated tokens to stdout.

## Scripts

| Script | Backend | Description |
|--------|---------|-------------|
| `stream_chat_mini_sgl.py` | mini-sglang | Uses the local mini-sglang submodule |
| `stream_chat_trt.py` | TensorRT-LLM | Uses NVIDIA TensorRT-LLM |

## Setup

### mini-sglang

From the repo root, run:

```bash
./setup.sh
```

This creates a `.venv` and installs mini-sglang into it. Then run:

```bash
.venv/bin/python generation/stream_chat_mini_sgl.py --model Qwen/Qwen3-0.6B --chat
```

### TensorRT-LLM (Optional)

TensorRT-LLM requires NVIDIA GPUs and CUDA. First run `./setup.sh` to create the venv (if you haven't already), then install TRT-LLM:

```bash
./setup.sh
uv pip install --python .venv/bin/python tensorrt-llm -U --pre --extra-index-url https://pypi.nvidia.com
```

Then run:

```bash
.venv/bin/python generation/stream_chat_trt.py --model Qwen/Qwen3-0.6B --chat
```

## Usage

Both scripts share the same core flags:

```
--model MODEL       Model name or path (default: Qwen/Qwen3-0.6B)
--max-tokens N      Max tokens to generate per response (default: 256)
--chat              Apply the model's chat template
```

`stream_chat_mini_sgl.py` has additional flags for disabling specific optimizations (run with `--help` to see them).

Once running, type a prompt at the `> ` marker and press Enter. Tokens stream to stdout as they are generated. Press Ctrl+D to quit.
