cd ~/research-project/function_vectors/src
export HF_HUB_OFFLINE=1
PY=../../venvs/fv/bin/python
$PY extract_knowledge_fv.py --effect binding --corruption other_person --n_ie_trials 4 --save_path ../results/knowledge_smoke_binding > ../results/knowledge_smoke_binding.log 2>&1 || exit 1
$PY inject_knowledge_fv.py --fv_path ../results/knowledge_smoke_binding/fv_extraction.pt --save_path ../results/knowledge_smoke_binding_injection > ../results/knowledge_smoke_binding_injection.log 2>&1
touch ../results/BINDING_DONE
