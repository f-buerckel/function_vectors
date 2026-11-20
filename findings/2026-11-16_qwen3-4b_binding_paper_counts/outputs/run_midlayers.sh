cd ~/research-project/function_vectors/src
export HF_HUB_OFFLINE=1
PY=../../venvs/fv/bin/python
while [ ! -f ../results/knowledge_fast_check/ALL_DONE ]; do sleep 30; done
mkdir -p ../results/knowledge_smoke_binding_midlayers
for effect in binding logprob; do
  $PY restrict_fv_heads.py --fv_path ../results/knowledge_smoke_${effect}_paper_counts/fv_extraction.pt --max_layer 24 --save_path ../results/knowledge_smoke_binding_midlayers/fv_${effect}_below24.pt > ../results/knowledge_smoke_binding_midlayers/restrict_${effect}.log 2>&1 || exit 1
  $PY inject_knowledge_fv.py --fv_path ../results/knowledge_smoke_binding_midlayers/fv_${effect}_below24.pt --save_path ../results/knowledge_smoke_binding_midlayers/injection_${effect}_below24 > ../results/knowledge_smoke_binding_midlayers/injection_${effect}.log 2>&1 || exit 1
done
touch ../results/knowledge_smoke_binding_midlayers/ALL_DONE
