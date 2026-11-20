cd ~/research-project/function_vectors/src
export HF_HUB_OFFLINE=1
PY=../../venvs/fv/bin/python
for effect in binding logprob; do
  R=../results/knowledge_smoke_${effect}_paper_counts
  $PY extract_knowledge_fv.py --effect $effect --corruption other_person --n_mean_trials 100 --n_ie_trials 25 --head_batch_size 128 --save_path $R > $R.log 2>&1 || exit 1
  $PY inject_knowledge_fv.py --fv_path $R/fv_extraction.pt --save_path ${R}_injection > ${R}_injection.log 2>&1 || exit 1
done
touch ../results/PAPER_COUNTS_DONE
