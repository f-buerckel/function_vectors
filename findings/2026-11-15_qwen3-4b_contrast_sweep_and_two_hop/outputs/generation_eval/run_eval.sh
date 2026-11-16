cd ~/research-project/function_vectors/src
export HF_HUB_OFFLINE=1
../../venvs/fv/bin/python eval_contrast_interventions.py ../results/knowledge_contrast/*/ ../results/knowledge_contrast_sweep/*/
touch ../results/EVAL_DONE
