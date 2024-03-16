from typing import Optional
import numpy as np
import torch
import math
import torch.nn.functional as F

#copied from https://github.com/huggingface/transformers/blob/b71f20a7c9f3716d30f6738501559acf863e2c5c/src/transformers/models/llama/modeling_llama.py#L200
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

#copied from https://github.com/huggingface/transformers/blob/a0857740c0e6127485c11476650314df3accc2b6/src/transformers/models/llama/modeling_llama.py#L180
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SkeletonLlamaForCausalLM():
    def __init__(self, model):
        self.model = model.model
        self.lm_head = model.lm_head
        self.hidden_size = model.config.hidden_size
        self.num_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
    
    def swap_reps(self, interv_info, orignl, layer_id, rep_type):
        new_state = orignl.clone()
        d = self.head_dim
        for h, p, s in interv_info.get(f'{rep_type}_{layer_id}', []):
            # Ensure the intervention is within the bounds of the tensor
            if s[0] < new_state.size(0) and s[1] < new_state.size(0) and p[0] < new_state.size(1) and p[1] < new_state.size(1):
                new_state[s[0], :, d*h:d*(h+1)][p[0], :] = orignl[s[1], :, d*h:d*(h+1)][p[1], :]
                new_state[s[1], :, d*h:d*(h+1)][p[1], :] = orignl[s[0], :, d*h:d*(h+1)][p[0], :]
        return new_state

    @torch.no_grad()
    def __call__(self, input_ids, attention_mask, interventions):
        output_hidden = []
        hidden = self.model.embed_tokens(input_ids)
        output_hidden.append(hidden)
        (batch_size, seq_len) = hidden.shape[:2]
        attention_mask = self.model._update_causal_mask(attention_mask, hidden)
        
        for layer_id, layer in enumerate(self.model.layers):
            res = hidden
            hidden = layer.input_layernorm(hidden)

            qry = layer.self_attn.q_proj(hidden)
            key = layer.self_attn.k_proj(hidden)
            val = layer.self_attn.v_proj(hidden)

            # Apply interventions for 'layer', 'query', 'key', 'value'
            res = self.swap_reps(interventions, res, layer_id, 'layer')
            qry = self.swap_reps(interventions, qry, layer_id, 'query')
            key = self.swap_reps(interventions, key, layer_id, 'key')
            val = self.swap_reps(interventions, val, layer_id, 'value')

            split_qry, split_key = apply_rotary_pos_emb(qry, key, *layer.self_attn.rotary_emb(hidden, seq_len))

            # Calculate attention and apply any 'attention' interventions if needed
            # This part would need to be adapted based on how you want to handle attention interventions

            trfm = torch.nn.functional.scaled_dot_product_attention(split_qry, split_key, val, attn_mask=attention_mask, dropout_p=0.0)
            trfm = trfm.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

            # Apply 'trfm' interventions
            trfm = self.swap_reps(interventions, trfm, layer_id, 'trfm')

            trfm = layer.self_attn.o_proj(trfm)
            hidden = res + trfm
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden) + hidden

            output_hidden.append(hidden)

        hidden = self.model.norm(hidden)
        logits = self.lm_head(hidden)
        return {'logits': logits, 'hidden_states': output_hidden}
