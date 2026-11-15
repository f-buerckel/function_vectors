cd ~/research-project/function_vectors/src
export HF_HUB_OFFLINE=1
PY=../../venvs/fv/bin/python
R=../results/knowledge_contrast
run() { name=$1; shift; mkdir -p $R/$name; $PY learn_knowledge_contrast.py --steps 300 "$@" --save_path $R/$name > $R/$name/run.log 2>&1; }
for w in 0 1; do
  run vector_layer12_last_w$w   --mode layers --layers 12 --position last --contrast_weight $w
  run vector_layer12_name_w$w   --mode layers --layers 12 --position name --contrast_weight $w
  run heads_last_w$w            --mode heads --position last --lr 0.5 --contrast_weight $w
  run kv_top10_4slots_w$w       --mode kv --lr 0.05 --n_slots 4 --contrast_weight $w
  run layers_head_layers_last_w$w --mode layers --layers 24 26 29 31 32 33 34 35 --position last --contrast_weight $w
done
touch $R/ALL_DONE
