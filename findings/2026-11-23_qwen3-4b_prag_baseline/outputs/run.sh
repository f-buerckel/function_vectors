cd ~/research-project/function_vectors/src
export HF_HUB_OFFLINE=1
PY=../../venvs/fv/bin/python
R=../results/prag_baseline
L8="--mode layers --layers 24 26 29 31 32 33 34 35"
s=$(date +%s); $PY augment_documents_prag.py > $R/augment.log 2>&1; echo "augment $(( $(date +%s) - s )) s" >> $R/timings.txt
vec() { name=$1; shift; mkdir -p $R/$name; s=$(date +%s); $PY learn_document_intervention.py --questions filtered_v2.json --fast --early_stop --contrast_weight 1 "$@" --save_path $R/$name > $R/$name/run.log 2>&1; echo "$name $(( $(date +%s) - s )) s" >> $R/timings.txt; }
lora() { name=$1; shift; mkdir -p $R/$name; s=$(date +%s); $PY learn_document_lora.py "$@" --save_path $R/$name > $R/$name/run.log 2>&1; echo "$name $(( $(date +%s) - s )) s" >> $R/timings.txt; }
vec  vec_prag_layers8_all  $L8 --position all --train_source prag
lora lora_prag             --data prag
lora lora_qdoc_c1          --data questions --doc_lm_weight 1 --epochs 5 --contrast_weight 1
lora lora_prag_c1          --data prag --contrast_weight 1
touch $R/ALL_DONE
