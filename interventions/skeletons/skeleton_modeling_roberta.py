import numpy as np
import torch
import torch.nn.functional as F
import math

@torch.no_grad()
def SkeletonRobertaLayer(layer_id,layer,hidden,interventions):
    attention_layer = layer.attention.self
    num_heads = attention_layer.num_attention_heads
    head_dim = attention_layer.attention_head_size
    assert num_heads*head_dim == hidden.shape[2]

    qry = attention_layer.query(hidden)
    key = attention_layer.key(hidden)
    val = attention_layer.value(hidden)

    assert qry.shape == hidden.shape
    assert key.shape == hidden.shape
    assert val.shape == hidden.shape

    # swap representations
    reps = {
            'lay': hidden,
            'qry': qry,
            'key': key,
            'val': val,
            }
    for rep_type in ['lay','qry','key','val']:
        interv_rep = interventions[layer_id][rep_type]
        new_state = reps[rep_type].clone()
        for head_id, pos, swap_ids in interv_rep:
            new_state[swap_ids[0],:,head_dim*head_id:head_dim*(head_id+1)][pos,:] = reps[rep_type][swap_ids[1],:,head_dim*head_id:head_dim*(head_id+1)][pos,:]
            new_state[swap_ids[1],:,head_dim*head_id:head_dim*(head_id+1)][pos,:] = reps[rep_type][swap_ids[0],:,head_dim*head_id:head_dim*(head_id+1)][pos,:]
        reps[rep_type] = new_state.clone()

    hidden = reps['lay'].clone()
    qry = reps['qry'].clone()
    key = reps['key'].clone()
    val = reps['val'].clone()


    #split into multiple heads
    split_qry = qry.view(*(qry.size()[:-1]+(num_heads,head_dim))).permute(0,2,1,3)
    split_key = key.view(*(key.size()[:-1]+(num_heads,head_dim))).permute(0,2,1,3)
    split_val = val.view(*(val.size()[:-1]+(num_heads,head_dim))).permute(0,2,1,3)

    #calculate the attention matrix
    attn_mat = F.softmax(split_qry@split_key.permute(0,1,3,2)/math.sqrt(head_dim),dim=-1)

    z_rep_indiv = attn_mat@split_val
    z_rep = z_rep_indiv.permute(0,2,1,3).reshape(*hidden.size())

    hidden_post_attn_res = layer.attention.output.dense(z_rep)+hidden # residual connection
    hidden_post_attn = layer.attention.output.LayerNorm(hidden_post_attn_res) # layer_norm

    hidden_post_interm = layer.intermediate(hidden_post_attn) # massive feed forward
    hidden_post_interm_res = layer.output.dense(hidden_post_interm)+hidden_post_attn # residual connection
    new_hidden =  layer.output.LayerNorm(hidden_post_interm_res) # layer_norm
    return new_hidden

def SkeletonRobertaForMaskedLM(model,input_ids,interventions):
    core_model = model.roberta
    lm_head = model.lm_head
    output_hidden = []
    with torch.no_grad():
        hidden = core_model.embeddings(input_ids)
        output_hidden.append(hidden)
        for layer_id in range(model.config.num_hidden_layers):
            layer = core_model.encoder.layer[layer_id]
            hidden = SkeletonRobertaLayer(layer_id,layer,hidden,interventions)
            output_hidden.append(hidden)
        logits = lm_head(hidden)
    return {'logits':logits,'hidden_states':output_hidden}
