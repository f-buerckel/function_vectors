cd ~/research-project/function_vectors/src
export HF_HUB_OFFLINE=1
PY=../../venvs/fv/bin/python
R=../results/wiki2026_coverage
L8='--mode layers --layers 24 26 29 31 32 33 34 35'
run() { name=$1; shift; mkdir -p $R/$name; s=$(date +%s); $PY learn_document_intervention.py --questions filtered_v2.json --fast --early_stop --contrast_weight 1 "$@" --save_path $R/$name > $R/$name/run.log 2>&1; echo "$name $(( $(date +%s) - s )) s" >> $R/timings.txt; }
run qa_kv                --mode kv --lr 0.05
run qa_doc_layers8_all   $L8 --position all --doc_lm_weight 1
run qa_doc_kv            --mode kv --lr 0.05 --doc_lm_weight 1
run doc_layers8_all      $L8 --position all --qa_weight 0 --doc_lm_weight 1
run doc_kv               --mode kv --lr 0.05 --qa_weight 0 --doc_lm_weight 1
touch $R/ALL_DONE
