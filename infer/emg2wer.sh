#!/bin/bash
# EMG -> mel -> wav -> PER / CER / WER.
#
#   ./emg2wer.sh [split] [extra hydra overrides...]
set -euo pipefail
split=${1:-test}
shift $(( $# < 1 ? $# : 1 ))
cd "$(dirname "$0")"

python emg2feat.py split="$split" "$@"
python mel2wav.py split="$split" "$@"

run_dir=$(python - "$split" "$@" <<'PY'
import sys
from hydra import compose, initialize
with initialize(version_base=None, config_path='../configs'):
    cfg = compose(config_name='mel2wav', overrides=[f'split={sys.argv[1]}', *sys.argv[2:]])
print(f'{cfg.exp_path}/{cfg.exp_name}/synth/{cfg.split}')
PY
)

python evaluate.py asr --run-dir "$run_dir" --split "$split"
python evaluate.py align --run-dir "$run_dir" --mfa-bin "${MFA_BIN:-mfa}"
python evaluate.py report --run-dir "$run_dir" --split "$split"
