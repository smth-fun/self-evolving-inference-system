GPU=0
for LOCAL_ID in $(seq 1 10); do
  TRIAL_ID="g${GPU}_${LOCAL_ID}"
  mkdir -p trial_$TRIAL_ID
  CUDA_VISIBLE_DEVICES=$GPU CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 claude --dangerously-skip-permissions --max-turns 300 --max-budget-usd 20 -p "You are trial $TRIAL_ID, learn from previous trials and do the task in CLAUDE.md" --verbose --output-format=stream-json > trial_$TRIAL_ID/history.jsonl
done
