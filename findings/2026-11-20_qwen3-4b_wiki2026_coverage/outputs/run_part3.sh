cd ~/research-project/function_vectors/src
export HF_HUB_OFFLINE=1
PY=../../venvs/fv/bin/python
R=../results/wiki2026_coverage
while [ ! -f $R/ALL_DONE ]; do sleep 30; done
mkdir -p $R/qa_doc_kv; s=$(date +%s)
$PY learn_document_intervention.py --questions filtered_v2.json --fast --early_stop --contrast_weight 1 --mode kv --lr 0.05 --doc_lm_weight 1 --save_path $R/qa_doc_kv > $R/qa_doc_kv/run.log 2>&1
echo "qa_doc_kv $(( $(date +%s) - s )) s" >> $R/timings.txt
touch $R/ALL_DONE_PART3
