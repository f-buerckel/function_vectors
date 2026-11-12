cd ~/research-project/function_vectors/src
export HF_HUB_OFFLINE=1
PY=../../venvs/fv/bin/python
R=../results/knowledge_capacity_intervention
run() { name=$1; shift; mkdir -p $R/$name; $PY learn_knowledge_intervention.py --n_facts 8 "$@" --save_path $R/$name > $R/$name/run.log 2>&1; }
run heads_last_lr0.1          --mode heads --position last
run heads_last_lr0.5_300steps --mode heads --position last --lr 0.5 --steps 300
run heads_name_lr0.5_300steps --mode heads --position name --lr 0.5 --steps 300
run layers_head_layers_last   --mode layers --layers 24 26 29 31 32 33 34 35 --position last
run layers_6_12_18_24_last    --mode layers --layers 6 12 18 24 --position last
run layers_6_12_18_24_name    --mode layers --layers 6 12 18 24 --position name
run layers_all_last           --mode layers --layers $(seq 0 35) --position last
touch $R/ALL_DONE
