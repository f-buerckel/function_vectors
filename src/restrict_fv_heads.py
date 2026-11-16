"""
Rebuilds person FVs from an extraction run's saved mean activations and indirect effects, choosing the top heads only
among layers in [min_layer, max_layer). Used to test whether mid-layer heads carry binding without the late
answer-writing heads dominating the sum. Writes an fv_extraction.pt-style file for inject_knowledge_fv.py.

Usage (from src/): python restrict_fv_heads.py --fv_path <run>/fv_extraction.pt --max_layer 24 --save_path <file>
"""
import argparse, json

import torch

from utils.model_utils import load_gpt_model_and_tokenizer
from utils.extract_utils import compute_function_vector

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B')
    parser.add_argument('--fv_path', type=str, required=True)
    parser.add_argument('--min_layer', type=int, default=0)
    parser.add_argument('--max_layer', type=int, required=True, help='Heads in layers >= max_layer are excluded')
    parser.add_argument('--n_top_heads', type=int, default=10)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    saved = torch.load(args.fv_path)
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    indirect_effect = saved['indirect_effect'].clone()
    excluded = torch.ones(indirect_effect.shape[1], dtype=torch.bool)
    excluded[args.min_layer:args.max_layer] = False
    indirect_effect[:, excluded] = -1e9  # never selected

    function_vectors = {}
    for name, last_token in saved['mean_activations_last_token'].items():
        mean_activations = last_token.unsqueeze(2).to(model.device)  # (layers, heads, 1 token, head_dim); FV uses the last token
        function_vectors[name], top_heads = compute_function_vector(mean_activations, indirect_effect, model, model_config,
                                                                    n_top_heads=args.n_top_heads)
    top_heads = [(int(l), int(h), float(s)) for l, h, s in top_heads]
    print("Top heads:", top_heads)
    torch.save({'function_vectors': {n: v.cpu() for n, v in function_vectors.items()}, 'top_heads': top_heads,
                'args': vars(args)}, args.save_path)
    with open(args.save_path.replace('.pt', '.json'), 'w') as f:
        json.dump({'args': vars(args), 'top_heads': top_heads,
                   'fv_norms': {n: v.norm().item() for n, v in function_vectors.items()}}, f, indent=2)
