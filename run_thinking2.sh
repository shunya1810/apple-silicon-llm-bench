#!/bin/zsh
# Re-run of the prompt-file tasks from run_thinking.sh (the first pass passed the
# prompt-file option as one word and every run exited on an argparse error).
cd "$(dirname "$0")"
PY=$(python3 -c 'import json;print(json.load(open("config/local.json"))["MTPLX_PYTHON"])')
caffeinate -dimsu -w $$ &
battery() { pmset -g batt | grep -o '[0-9]*%' | tr -d %; }
run() {  # engine tag n prompt-file
  until [ "$(battery)" -ge 80 ]; do sleep 60; done
  echo "$(date +%T) $2 $1 x$3 battery $(battery)%"
  "$PY" scripts/pelican.py --engine engines/erp/$1.json --out results/pelican --n $3 --max-tokens 131072 --tag $2 --prompt-file $4 2>&1 | grep -E '^\{|Error|error:'
  sleep 120
}
run splash-pelican heptagon 2 prompts/heptagon.txt
run fork-fix-pelican heptagon 2 prompts/heptagon.txt
run splash-pelican codereview 2 prompts/codereview-cold-tier.txt
run fork-fix-pelican codereview 2 prompts/codereview-cold-tier.txt
touch results/pelican/THINKING2_DONE
