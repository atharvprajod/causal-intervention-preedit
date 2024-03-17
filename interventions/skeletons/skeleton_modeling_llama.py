import torch
import torch.nn.functional as F

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
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
    
    def swap_reps(self, interventions, states):
        """Swap representations according to specified interventions."""
        for intervention in interventions:
            h, p, s = intervention['head'], intervention['positions'], intervention['sequences']
            d = self.head_dim
            # Swap operation
            temp = states[s[0], :, d*h:d*(h+1)][p[0], :].clone()
            states[s[0], :, d*h:d*(h+1)][p[0], :] = states[s[1], :, d*h:d*(h+1)][p[1], :]
            states[s[1], :, d*h:d*(h+1)][p[1], :] = temp
        return states

    @torch.no_grad()
    def __call__(self, input_ids, attention_mask, interventions):
        output_hidden_states = []
        hidden_states = self.model.embed_tokens(input_ids)
        output_hidden_states.append(hidden_states)

        batch_size, seq_len = hidden_states.shape[:2]
        attention_mask = self.model._update_causal_mask(attention_mask, hidden_states)
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand_as(input_ids)
        print(f"Type of position_ids before forward call: {type(position_ids)}")
        if isinstance(position_ids, torch.Tensor):
          print(f"Shape of position_ids: {position_ids.shape}")
        else:
            print(f"position_ids is not a tensor: {position_ids}")

        for layer_idx, layer_module in enumerate(self.model.layers):
            hidden_states = layer_module.input_layernorm(hidden_states)

            query = layer_module.self_attn.q_proj(hidden_states)
            key = layer_module.self_attn.k_proj(hidden_states)
            value = layer_module.self_attn.v_proj(hidden_states)

            if interventions.get(layer_idx):
                layer_interventions = interventions[layer_idx]
                if 'query' in layer_interventions:
                    query = self.swap_reps(layer_interventions['query'], query)
                if 'key' in layer_interventions:
                    key = self.swap_reps(layer_interventions['key'], key)
                if 'value' in layer_interventions:
                    value = self.swap_reps(layer_interventions['value'], value)

            cos, sin = layer_module.self_attn.rotary_emb(hidden_states, position_ids)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

            attention_output = torch.matmul(query, key.transpose(-1, -2))
            attention_output = F.softmax(attention_output, dim=-1)
            attention_output = torch.matmul(attention_output, value)

            attention_output = attention_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
            attention_output = layer_module.self_attn.o_proj(attention_output)

            hidden_states = hidden_states + attention_output
            hidden_states = layer_module.post_attention_layernorm(hidden_states)

            mlp_output = layer_module.mlp(hidden_states)
            hidden_states = hidden_states + mlp_output

            output_hidden_states.append(hidden_states)

        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return {'logits': logits, 'hidden_states': output_hidden_states}
