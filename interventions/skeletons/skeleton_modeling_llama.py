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
        self.num_key_value_heads = model.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
    
    def swap_reps(self, interv_info, orignl):
        new_state = orignl.clone()
        d = self.head_dim
        for h, p, s in interv_info:
            new_state[s[0],:,d*h:d*(h+1)][p[0],:] = orignl[s[1],:,d*h:d*(h+1)][p[1],:]
            new_state[s[1],:,d*h:d*(h+1)][p[1],:] = orignl[s[0],:,d*h:d*(h+1)][p[0],:]
        return new_state.clone()

    def __call__(
            self,
            input_ids,
            attention_mask,
            interventions,
            ):
        output_hidden = []
        with torch.no_grad():
            hidden = self.model.embed_tokens(input_ids)
            output_hidden.append(hidden)
            (batch_size, seq_len) = hidden.shape[:2]
            #attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=hidden.device)
            #attention_mask = self.model._prepare_decoder_attention_mask(attention_mask, (batch_size, seq_len), hidden, 0)
            attention_mask = self.model._update_causal_mask(attention_mask, hidden)
            for layer_id, layer in enumerate(self.model.layers):
                res = hidden
                hidden = layer.input_layernorm(hidden)

                qry = layer.self_attn.q_proj(hidden)
                key = layer.self_attn.k_proj(hidden)
                val = layer.self_attn.v_proj(hidden)

                res = self.swap_reps(interventions[layer_id]['lay'], res)
                qry = self.swap_reps(interventions[layer_id]['qry'], qry)
                key = self.swap_reps(interventions[layer_id]['key'], key)
                val = self.swap_reps(interventions[layer_id]['val'], val)

                split_qry = qry.view(*(qry.size()[:-1]+(self.num_heads,self.head_dim))).permute(0,2,1,3)
                split_key = key.view(*(key.size()[:-1]+(self.num_heads,self.head_dim))).permute(0,2,1,3)
                split_val = val.view(*(val.size()[:-1]+(self.num_heads,self.head_dim))).permute(0,2,1,3)

                position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden.device)
                position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
                cos, sin = layer.self_attn.rotary_emb(split_val, position_ids)
                split_qry, split_key = apply_rotary_pos_emb(split_qry, split_key, cos, sin)

                #split_key = repeat_kv(split_key, self.num_key_value_groups)
                #split_val = repeat_kv(split_val, self.num_key_value_groups)

                cache_position = torch.arange(seq_len, dtype=torch.long, device=hidden.device)
                attention_mask = attention_mask[:, :, cache_position, :split_key.shape[-2]]


                #raw_attn = split_qry@split_key.permute(0,1,3,2)/math.sqrt(self.head_dim)
                #raw_attn = raw_attn + attention_mask.to(hidden.device)
                #attn = F.softmax(raw_attn, dim=-1, dtype=torch.float32).to(split_qry.dtype)
                #trfm_indiv = attn@split_val
                #trfm = trfm_indiv.permute(0,2,1,3).reshape(*hidden.size())
                trfm = torch.nn.functional.scaled_dot_product_attention(
                    split_qry,
                    split_key,
                    split_val,
                    attn_mask=attention_mask,
                    dropout_p=0.0
                    )

                trfm = trfm.transpose(1, 2).contiguous()
                trfm = trfm.view(batch_size, seq_len, self.hidden_size)

                trfm = self.swap_reps(interventions[layer_id]['trfm'], trfm)

                trfm = layer.self_attn.o_proj(trfm)

                hidden = res.to(hidden.device) + trfm
                res = hidden
                hidden = layer.post_attention_layernorm(hidden)
                hidden = layer.mlp(hidden)
                hidden = res.to(hidden.device) + hidden

                if layer_id!=self.num_layers-1:
                    output_hidden.append(hidden)
            hidden = self.model.norm(hidden)
            assert layer_id==self.num_layers-1
            output_hidden.append(hidden)
            logits = self.lm_head(hidden)
        return {'logits':logits,'hidden_states':output_hidden}