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
    
    def swap_reps(self, layer_interventions, orignl):
        new_state = orignl.clone()
        d = self.head_dim
        # Directly iterate over the interventions for the specific representation type
        for h, p, s in layer_interventions:  # layer_interventions is already a list of tuples for a specific rep_type
            # Assuming p and s are correctly structured for indexing
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
            
            # Retrieve interventions for the current layer
            layer_interventions = interventions[layer_id]
            
            # Apply interventions for 'layer', 'query', 'key', 'value'
            # Note: The swap_reps method should directly accept the list of tuples for interventions
            res = self.swap_reps(layer_interventions['lay'], res)
            qry = self.swap_reps(layer_interventions['qry'], qry)
            key = self.swap_reps(layer_interventions['key'], key)
            val = self.swap_reps(layer_interventions['val'], val)

            split_qry, split_key = apply_rotary_pos_emb(qry, key, *layer.self_attn.rotary_emb(hidden, seq_len))

            # Calculate attention and apply any 'attention' interventions if needed
            # This part would need to be adapted based on how you want to handle attention interventions

            trfm = torch.nn.functional.scaled_dot_product_attention(split_qry, split_key, val, attn_mask=attention_mask, dropout_p=0.0)
            trfm = trfm.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

            # Apply 'trfm' interventions
            trfm = self.swap_reps(layer_interventions['trfm'], trfm)

            trfm = layer.self_attn.o_proj(trfm)
            hidden = res + trfm
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden) + hidden

            output_hidden.append(hidden)

        hidden = self.model.norm(hidden)
        logits = self.lm_head(hidden)
        return {'logits': logits, 'hidden_states': output_hidden}
