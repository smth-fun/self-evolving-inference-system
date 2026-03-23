"""Interactive streaming chat using mini-sglang.

Reads prompts from stdin, streams generated tokens to stdout.
Usage: .venv/bin/python stream_chat.py [--model MODEL] [--max-tokens N] [--chat] [--port PORT]
"""

import argparse
import random
import sys
import time
from typing import List

import torch

from minisgl.core import SamplingParams
from minisgl.llm import LLM
from minisgl.message import DetokenizeMsg


class StreamingLLM(LLM):
    """LLM subclass that prints tokens to stdout as they are generated."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._decode_start = None
        self._decode_tokens = 0

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        for msg in reply:
            status = self.status_map[msg.uid]
            if not (msg.finished and msg.next_token == self.eos_token_id):
                if self._decode_start is None:
                    self._decode_start = time.perf_counter()
                self._decode_tokens += 1
                status.output_ids.append(msg.next_token)
                token_text = self.tokenizer.decode(
                    status.output_ids, skip_special_tokens=True
                )
                # Print only the new characters since last decode
                prev_text = getattr(status, "_prev_text", "")
                new_text = token_text[len(prev_text):]
                if new_text:
                    sys.stdout.write(new_text)
                    sys.stdout.flush()
                status._prev_text = token_text


def main():
    parser = argparse.ArgumentParser(description="Interactive streaming chat")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--chat", action="store_true", help="Apply chat template")
    parser.add_argument("--port", type=int, default=2333, help="Port for torch.distributed rendezvous")
    parser.add_argument("--no-quant", action="store_true", help="Disable INT8/INT4 quantization on all layers")
    parser.add_argument("--full-precision-lm-head", action="store_true", help="Disable INT8/INT4 quantization on lm_head only (decoder layers stay quantized)")
    parser.add_argument("--no-fused-decode", action="store_true", help="Disable fused RMSNorm+GEMV decode path (use original separate-kernel path)")
    parser.add_argument("--no-splitk", action="store_true", help="Disable Split-K attention (use FlashInfer instead)")
    parser.add_argument("--no-triton-gemv", action="store_true", help="Disable Triton GEMV (use cuBLAS F.linear instead)")
    parser.add_argument("--no-fused-qk-rope", action="store_true", help="Disable fused QKNorm+RoPE (use separate kernels)")
    parser.add_argument("--no-fused-lm-head", action="store_true", help="Disable fused final-norm+lm_head (use original model.forward path)")
    parser.add_argument("--no-fast-metadata", action="store_true", help="Disable fast FlashInfer metadata init (use original synchronous path)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set all RNG seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Override the hardcoded distributed port
    if args.port != 2333:
        from minisgl.engine.config import EngineConfig
        EngineConfig.distributed_addr = property(
            lambda self: f"tcp://127.0.0.1:{args.port}"
        )

    # Disable quantization on all layers
    if args.no_quant:
        import minisgl.kernel.quantize as _qmod
        _qmod.quantize_model_weights = lambda *a, **kw: None
    elif args.full_precision_lm_head:
        import minisgl.kernel.quantize as _qmod
        _orig_quantize = _qmod.quantize_model_weights
        def _quantize_skip_lm_head(*a, **kw):
            _orig_quantize(*a, skip_decoder=False, **kw)
            # Remove quantized weights from lm_head so it falls back to bf16 GEMM
            model = a[0]
            for attr in ('weight_int8', 'weight_scale', 'weight_int4', 'weight_scale_int4'):
                if hasattr(model.lm_head, attr):
                    delattr(model.lm_head, attr)
        _qmod.quantize_model_weights = _quantize_skip_lm_head

    # Disable fused RMSNorm+GEMV decode path
    if args.no_fused_decode:
        from minisgl.models.qwen3 import Qwen3DecoderLayer
        _orig_layer_init = Qwen3DecoderLayer.__init__
        def _patched_layer_init(self, *a, **kw):
            _orig_layer_init(self, *a, **kw)
            self._use_fused_decode = False
        Qwen3DecoderLayer.__init__ = _patched_layer_init

    # Disable Split-K attention (fall back to FlashInfer)
    if args.no_splitk:
        from minisgl.attention.fi import FlashInferBackend
        _orig_prepare = FlashInferBackend.prepare_metadata
        def _patched_prepare(self, batch):
            _orig_prepare(self, batch)
            self._use_splitk = False
        FlashInferBackend.prepare_metadata = _patched_prepare

    # Disable Triton GEMV (fall back to cuBLAS F.linear)
    if args.no_triton_gemv:
        import minisgl.layers.linear as _lmod
        import torch.nn.functional as F
        _lmod._should_use_triton = lambda *a, **kw: False

    # Disable fused QKNorm+RoPE in AttentionLayer (use separate norm, rope, store)
    if args.no_fused_qk_rope:
        from minisgl.layers.attention import AttentionLayer
        _orig_attn_forward = AttentionLayer.forward
        def _patched_attn_forward(self, qkv):
            from minisgl.core import get_global_ctx
            ctx = get_global_ctx()
            q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
            if self.q_norm is not None:
                self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
            if self.k_norm is not None:
                self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
            q, k = self.rotary.forward(ctx.batch.positions, q, k)
            q = q.view(-1, self.num_qo_heads, self.head_dim)
            o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
            return o.view(-1, self.qo_attn_dim)
        AttentionLayer.forward = _patched_attn_forward

    # Disable fused final-norm+lm_head (revert to original model.forward path)
    if args.no_fused_lm_head:
        from minisgl.models.qwen3 import Qwen3ForCausalLM
        def _patched_causal_forward(self):
            from minisgl.core import get_global_ctx
            output = self.model.forward(get_global_ctx().batch.input_ids)
            return self.lm_head.forward(output)
        Qwen3ForCausalLM.forward = _patched_causal_forward

    # Disable fast FlashInfer metadata init (use original synchronous path)
    if args.no_fast_metadata:
        from minisgl.attention.fi import FlashInferBackend
        FlashInferBackend._initialize_metadata_fast = FlashInferBackend._initialize_metadata_once

    llm = StreamingLLM(
        args.model,
        max_seq_len_override=4096,
        max_extend_tokens=16384,
        cuda_graph_max_bs=1,
        attention_backend="fi",
    )

    # Warm up
    llm.generate(["warmup"], SamplingParams(temperature=0.0, max_tokens=1))

    print(f"Model loaded: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    if args.chat:
        print("Chat template: enabled")
    print("Enter a prompt (Ctrl+D to quit):\n")

    history = []

    while True:
        try:
            sys.stdout.write("> ")
            sys.stdout.flush()
            prompt = input()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt.strip():
            continue

        if args.chat:
            history.append({"role": "user", "content": prompt})
            formatted = llm.tokenizer.apply_chat_template(
                history, add_generation_prompt=True, tokenize=False
            )
        else:
            formatted = prompt

        sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

        # Reset streaming state
        for status in llm.status_map.values():
            if hasattr(status, "_prev_text"):
                del status._prev_text

        llm._decode_start = None
        llm._decode_tokens = 0
        result = llm.generate([formatted], sampling_params)
        if llm._decode_start is not None and llm._decode_tokens > 0:
            elapsed = time.perf_counter() - llm._decode_start
            tps = llm._decode_tokens / elapsed
            sys.stdout.write(f"\n[{llm._decode_tokens} tokens, {elapsed:.2f}s, {tps:.1f} tok/s]\n\n")
        else:
            sys.stdout.write("\n\n")
        sys.stdout.flush()

        if args.chat:
            history.append({"role": "assistant", "content": result[0]["text"]})


if __name__ == "__main__":
    main()
