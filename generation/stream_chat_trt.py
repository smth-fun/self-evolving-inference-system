"""Interactive streaming chat using TensorRT-LLM.

Reads prompts from stdin, streams generated tokens to stdout.
Usage: python stream_chat_trt.py [--model MODEL] [--max-tokens N] [--chat]
"""

import argparse
import sys
import time

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import RequestOutput


def main():
    parser = argparse.ArgumentParser(description="Interactive streaming chat (TRT-LLM)")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--chat", action="store_true", help="Apply chat template")
    args = parser.parse_args()

    llm = LLM(model=args.model)
    tokenizer = llm.tokenizer

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
            formatted = tokenizer.apply_chat_template(
                history, add_generation_prompt=True, tokenize=False
            )
        else:
            formatted = prompt

        sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

        decode_start = None
        decode_tokens = 0
        prev_text = ""

        for output in llm.generate([formatted], sampling_params=sampling_params, streaming=True):
            if isinstance(output, RequestOutput):
                text = output.outputs[0].text
            else:
                text = output[0].outputs[0].text

            new_text = text[len(prev_text):]
            if new_text:
                if decode_start is None:
                    decode_start = time.perf_counter()
                decode_tokens += len(new_text.split()) if decode_tokens == 0 else 1
                sys.stdout.write(new_text)
                sys.stdout.flush()
                prev_text = text

        # Stats
        if decode_start is not None:
            # Use token count from output if available
            if isinstance(output, RequestOutput):
                n_tokens = len(output.outputs[0].token_ids)
            else:
                n_tokens = len(output[0].outputs[0].token_ids)
            elapsed = time.perf_counter() - decode_start
            tps = n_tokens / elapsed if elapsed > 0 else 0
            sys.stdout.write(f"\n[{n_tokens} tokens, {elapsed:.2f}s, {tps:.1f} tok/s]\n\n")
        else:
            sys.stdout.write("\n\n")
        sys.stdout.flush()

        if args.chat:
            history.append({"role": "assistant", "content": prev_text})


if __name__ == "__main__":
    main()
