cd ~/research-project/function_vectors/src
export HF_HUB_OFFLINE=1
PY=../../venvs/fv/bin/python
R=../results/knowledge_capacity_kv_memory
run() { name=$1; shift; mkdir -p $R/$name; $PY learn_kv_memory.py --n_facts 8 "$@" --save_path $R/$name > $R/$name/run.log 2>&1; }
run top10_heads_4_slots  --n_slots 4
run top10_heads_16_slots --n_slots 16
run layers_6_12_18_24_all_heads_2_slots --layers_all_heads 6 12 18 24 --n_slots 2
touch $R/ALL_DONE
