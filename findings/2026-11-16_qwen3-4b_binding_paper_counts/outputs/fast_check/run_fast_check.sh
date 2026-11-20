cd ~/research-project/function_vectors/src
export HF_HUB_OFFLINE=1
PY=../../venvs/fv/bin/python
R=../results/knowledge_fast_check
while [ ! -f ../results/PAPER_COUNTS_DONE ]; do sleep 30; done
run() { name=$1; shift; mkdir -p $R/$name; s=$(date +%s); $PY learn_knowledge_contrast.py --steps 300 "$@" --save_path $R/$name > $R/$name/run.log 2>&1; echo "$name $(( $(date +%s) - s )) s" >> $R/timings.txt; }
run heads_last_w1_fast         --mode heads --position last --lr 0.5 --contrast_weight 1 --fast
run heads_last_w1_fast_stop    --mode heads --position last --lr 0.5 --contrast_weight 1 --fast --early_stop
run kv_top10_4slots_w1_fast_stop --mode kv --lr 0.05 --n_slots 4 --contrast_weight 1 --fast --early_stop
run layers_head_layers_last_w1_fast_stop --mode layers --layers 24 26 29 31 32 33 34 35 --position last --contrast_weight 1 --fast --early_stop
run vector_layer12_name_w1_fast_stop --mode layers --layers 12 --position name --contrast_weight 1 --fast --early_stop
touch $R/ALL_DONE
