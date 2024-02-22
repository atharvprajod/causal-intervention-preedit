import numpy as np
import torch
import torch.nn.functional as F
import math

from transformers.modeling_utils import apply_chunking_to_forward

@torch.no_grad()
def SkeletonAlbertLayer(layer_id,layer,hidden,interventions):
    attention_layer = layer.attention
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

    hidden_post_attn_res = layer.attention.dense(z_rep)+hidden
    hidden_post_attn = layer.attention.LayerNorm(hidden_post_attn_res)

    ffn_output = apply_chunking_to_forward(layer.ff_chunk,layer.chunk_size_feed_forward,
                                            layer.seq_len_dim,hidden_post_attn)
    new_hidden = layer.full_layer_layer_norm(ffn_output+hidden_post_attn)
    return new_hidden

def SkeletonAlbertForMaskedLM(model,input_ids,interventions):
    core_model = model.albert
    lm_head = model.predictions
    output_hidden = []
    with torch.no_grad():
        hidden = core_model.embeddings(input_ids)
        hidden = core_model.encoder.embedding_hidden_mapping_in(hidden)
        output_hidden.append(hidden)
        for layer_id in range(model.config.num_hidden_layers):
            layer = core_model.encoder.albert_layer_groups[0].albert_layers[0]
            hidden = SkeletonAlbertLayer(layer_id,layer,hidden,interventions)
            output_hidden.append(hidden)
        logits = lm_head(hidden)
    return {'logits':logits,'hidden_states':output_hidden}
