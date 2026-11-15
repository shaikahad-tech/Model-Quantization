#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -c 'import torch; assert torch.cuda.is_available(), "Linux NVIDIA CUDA required"'
run_dir="${1:-runs/$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -e "$run_dir" ]]; then
  echo "Refusing to overwrite existing run directory: $run_dir" >&2
  exit 1
fi
mkdir -p "$run_dir/logs"
python -m pip freeze > "$run_dir/logs/environment.txt"
cp docs/memory.md "$run_dir/logs/budget_before_run.md"
date -u +%FT%TZ > "$run_dir/logs/budget_recorded_at.txt"
sampler_pid=''
cleanup() {
  if [[ -n "$sampler_pid" ]]; then
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
    sampler_pid=''
  fi
}
trap cleanup EXIT
for stage in train merge quantize verify; do
  (
    while true; do
      date -u +%FT%TZ
      nvidia-smi
      sleep 1
    done
  ) > "$run_dir/logs/nvidia_smi_${stage}.txt" 2>&1 &
  sampler_pid=$!
  python -u code/rerun.py "$stage" --run-dir "$run_dir" 2>&1 |
    python -u -c 'import datetime,sys
for line in sys.stdin:
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(), line, end="", flush=True)' |
    tee "$run_dir/logs/${stage}.log"
  cleanup
done
printf '%s\n' "Workflow finished: $run_dir. Safety and shuffled-label checks remain outstanding; DON'T SHIP."
