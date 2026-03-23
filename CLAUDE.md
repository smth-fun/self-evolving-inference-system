You are an autonomous deep researcher. You are a team and orchestrate as many agents as you want.

I want to automatically optimize sglang.

# How to run it

- "mini-sglang" folder contains mini-sglang as a git submodule (from https://github.com/smth-fun/mini-sglang). Always use this.
- Always use `.venv/bin/python` to run things. Do NOT activate the venv, do NOT use system python, do NOT use docker.
  - Example: `.venv/bin/python mini-sglang/benchmark/offline/bench_simple.py`

# How to benchmark it

- Run `.venv/bin/python mini-sglang/benchmark/offline/bench_simple.py`, which gives you:
```
[{'text': 'OMER\n\nProcess object for transporting the data. The process handles the data flow, processing, and transformation. It can be used to create new data or modify existing data as needed. The process can be initialized with parameters that define the input and output data. The process can also be customized to handle different data types and processing logic.\n\nThe process can be implemented in various programming languages, such as Python, Java, C#, or other similar languages. The process can be used in different contexts, such as data processing pipelines, data transformation pipelines, or data flow processing.\n\nThe process is a fundamental component in data processing and transformation. It allows', 'token_ids': [1898, 640, 271, 7423, 1633, 369, 66657, 279, 821, 13, 576, 1882, 13469, 279, 821, 6396, 11, 8692, 11, 323, 17991, 13, 1084, 646, 387, 1483, 311, 1855, 501, 821, 476, 5602, 6350, 821, 438, 4362, 13, 576, 1882, 646, 387, 17271, 448, 5029, 429, 6979, 279, 1946, 323, 2550, 821, 13, 576, 1882, 646, 1083, 387, 31689, 311, 3705, 2155, 821, 4494, 323, 8692, 12218, 382, 785, 1882, 646, 387, 11537, 304, 5257, 15473, 15459, 11, 1741, 438, 13027, 11, 7943, 11, 356, 60778, 476, 1008, 4428, 15459, 13, 576, 1882, 646, 387, 1483, 304, 2155, 37597, 11, 1741, 438, 821, 8692, 57673, 11, 821, 17991, 57673, 11, 476, 821, 6396, 8692, 382, 785, 1882, 374, 264, 15811, 3692, 304, 821, 8692, 323, 17991, 13, 1084, 6147]}]
Total: 128tok, Time: 0.27s, Throughput: 475.65tok/s
```
- `[{'text':` is the output text. Your output needs to be EXACTLY the same as this.
- Your goal is to optimize Throughput (i.e., the 475.65tok/s part)
- YOU CANNOT CHANGE ANYTHING IN `mini-sglang/benchmark/offline/bench_simple.py`

- After each optimization pass, you should test both correctness -- i.e., it gets the same result, and throughput
- For correctness -- the output need to be EXACTLY the same.
- Report both correctness and the throughput you are getting after each trial.

# Optimization Hints

## Technical Guidelines

- In this optimization journey, let's focus on megakernel/persistent kernel -- I.e., let's fuse the whole foward pass into as few as kernels as possible to eliminate kernel launch overhead and data movement between kernel runs. 

## Principles

- Be strict about correctness. Under greedy decoding, your output needs to be exactly the same.
- Every 1% matter. But let's be systematic -- identify bottlenecks via benchmarking and profiling, and sysystematically remove bottlenecks. 
- Try to do deep things instead of only tuning. Do the right thing no matter how hard it might look like.
- Make sure you test BOTH performance and quality.

- Be systematic!

- Wrap up and summarize your learning once you used 90% of your context -- NEVER COMPACT CONTEXT -- and write the learning down for future trials.

- Each trial creates a git branch **inside the `mini-sglang/` submodule** (e.g. `cd mini-sglang && git checkout -b trial-<ID>`). The submodule is its own git repo. Do NOT run `git init` anywhere. Do NOT create branches in the parent scaffolding repo.

## Processes

1. Before you do things, read all trials before you, criticize and improve

2. Each trial creates a git branch **inside `mini-sglang/`** (e.g. `cd mini-sglang && git checkout -b trial-<ID>`) from the current state, and commits changes before finishing

3. Document all things in .md files; make sure future trials can learn about your progress and improve

4. Always produce `trial_<ID>/plan.md` first such that you and I know what you plan to do

5. For each trial, summarize the end to end test result, of performance you are getting, make sure we can track the progress

6. After you finish a trial, update `learning.md` (root directory) about this trial and what you learned and what you think the next trial should do.

7. The previous session's history is in trial_<ID>/history.jsonl  All trials are started with `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 claude --dangerously-skip-permissions --max-turns 300 -p "You are trial $TRIAL_ID, learn from previous trials and do the task in CLAUDE.md" --verbose --output-format=stream-json > trial_$TRIAL_ID/history.jsonl`

8. **IMPORTANT: Only work on YOUR OWN trial.** You are trial <ID>. Only create branch `trial-<ID>`, only write to `trial_<ID>/` directory (except for `learning.md` in root). Do NOT create branches or directories for other trials. Do NOT start working on trial <ID+1>. When you are done, STOP.

9. **Commit early, commit often.** Every time you run a benchmark and get **correct output** (output matches reference exactly), **document it in `trial_<ID>/results.md`** — whether throughput improved, regressed, or stayed the same. This creates a trace of what worked and what didn't. If correctness passes AND throughput improved, **STOP and do these steps before doing anything else**:
   - Git commit your code changes
   - Update `trial_<ID>/results.md` with the latest throughput numbers and what you did (create the file if it doesn't exist; append each result)
   - Update `learning.md` with key learnings
   - **Only after all three are done**, continue to the next optimization attempt
   Do NOT wait until the end to commit and document. Your session may be terminated at any time.
   **Common mistake**: Seeing improvement (e.g. +10 tok/s) and immediately trying the next idea without committing. If the next attempt fails or regresses, you lose the proven gain. NEVER skip the commit+document step.

10. **Checkpoint before each new attempt.** Before starting a new optimization idea, ask: "Did I just get a throughput improvement with correct output?" If yes and you haven't committed it yet, do Rule 9 first. No exceptions.