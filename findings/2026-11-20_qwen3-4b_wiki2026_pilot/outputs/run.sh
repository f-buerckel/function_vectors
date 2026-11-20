cd ~/research-project/function_vectors/src
export HF_HUB_OFFLINE=1
PY=../../venvs/fv/bin/python
R=../results/wiki2026_pilot
run() { name=$1; shift; mkdir -p $R/$name; s=$(date +%s); $PY learn_document_intervention.py --fast --early_stop "$@" --save_path $R/$name > $R/$name/run.log 2>&1; echo "$name $(( $(date +%s) - s )) s" >> $R/timings.txt; }
for w in 1 0; do
  run kv_top10_4slots_w$w        --mode kv --lr 0.05 --n_slots 4 --contrast_weight $w
  run heads_last_w$w             --mode heads --lr 0.5 --contrast_weight $w
  run layers_head_layers_w$w     --mode layers --layers 24 26 29 31 32 33 34 35 --contrast_weight $w
  run vector_layer12_w$w         --mode layers --layers 12 --contrast_weight $w
done
touch $R/ALL_DONE
