from typing import Optional, Tuple, Union, List
import numpy as np

import torch
import torch.nn.functional as F

from .utils import load_model, mask_out, extract_from_config, InterventionBase

class InterchangeInterventionArgs():
    '''
        Config class for the 'InterchangeIntervention'
    '''
    def __init__(
        self,
        sents: Tuple[str, str],
        options: Tuple[str, str],
        pron_locs: Tuple[List[int],List[int]],
        ):
        '''
            sents: the list of the two sentences
            options: the list of the two option noun phrases; they don't have to be in the sentences
            pron_locs: location of the pronouns for each sentence; word-level, not token-level, indices
        '''
        self.sents = sents
        self.options = options
        self.pron_locs = pron_locs

class InterchangeIntervention(InterventionBase):
    '''
        Performs interchange interventions
    '''
    def __init__(self,model_name,interv_args):
        super().__init__('interchange')

        tokenizer,model,skeleton,mask_id = load_model(model_name)
        self.tokenizer = tokenizer
        self.model = model
        self.skeleton = skeleton
        self.mask_id = model_args.mask_id
        self.num_layers, self.num_heads = extract_from_config(self.model.config)

        self.sents = interv_args.sents
        self.options = interv_args.options
        self.pron_locs = interv_args.pron_locs

        self.option_tokens = [self.tokenize_without_sp_tokens(option) for option in self.options]
        self.pron_locs_tokens = [self.convert_to_tokens(loc,sent) for loc,sent in zip(self.pron_locs,self.sents)]

    def create_interventions(
        self,
        target_tokens: List[int],
        rep_types: List[str],
        multihead: bool=True,
        heads: List[int]=[],
        ):
        interventions = {}
        for token_id in target_tokens:
            for rep in ['lay','qry','key','val']:
                if rep in rep_types:
                    if multihead:
                        interventions[rep] = [(head_id,token_id,[0,1]) for head_id in range(self.num_heads)]
                    else:
                        interventions[rep] = [(head_id,token_id,[i,i+len(heads)]) for i,head_id in enumerate(heads)]
                else:
                    interventions[rep] = []
        return interventions

    def run_interventions(
        self,
        interventions: dict,
        batch_size: int,
        ):
        probs = []
        for option in self.option_tokens:
            masked_ids_1 = mask_out(self.input_ids_1,self.pron_locs_tokens[0],option,self.mask_id)
            masked_ids_2 = mask_out(self.input_ids_2,self.pron_locs_tokens[1],option,self.mask_id)

            input_ids_all = torch.tensor([
                                    *[masked_ids_1 for _ in range(batch_size)],
                                    *[masked_ids_2 for _ in range(batch_size)]
                                    ])
            outputs = self.skeleton(self.model,input_ids_all,interventions)
            logprobs = F.log_softmax(outputs['logits'], dim = -1)
            logprobs_1, logprobs_2 = logprobs[:batch_size], logprobs[batch_size:]
            evals_1 = [logprobs_1[:,self.pron_locs_tokens[0][0]+i,token].numpy() for i,token in enumerate(option)]
            evals_2 = [logprobs_2[:,self.pron_locs_tokens[1][0]+i,token].numpy() for i,token in enumerate(option)]
            probs.append([np.exp(np.mean(evals_1,axis=0)),np.exp(np.mean(evals_2,axis=0))])
        probs = np.array(probs)
        assert probs.shape[0]==2 and probs.shape[1]==2 and probs.shape[2]==batch_size
        return probs

    def run(
        self,
        target: [List[int],List[int]],
        rep_types: List[str],
        multihead: bool=True,
        heads: List[int]=[],
        ):

        target_tokens = [self.context_to_tokens(loc,sent) for loc,sent in zip(target,self.sents)]
        assert np.all(np.array(target_tokens[0])==np.array(target_tokens[1]))

        self.input_ids_1 = self.tokenizer(self.sents[0]).input_ids
        self.input_ids_2 = self.tokenizer(self.sents[1]).input_ids

        interventions_empty = {'lay':[],'qry':[],'key':[],'val':[]}
        interventions_layer = self.create_interventions(target_tokens[0],rep_types,multihead,heads)
        batch_size = 1 if multihead else len(heads)

        # run without intervention
        interventions = [interventions_empty  for i in range(self.num_layers)]
        probs_origin = self.run_interventions(interventions,batch_size=1)

        # run intervention for each layer
        effect_list = []
        for layer_id in range(self.num_layers):
            interventions = [interventions_layer if i==layer_id else interventions_empty for i in range(num_layers)]
            probs_interv = self.run_interventions(interventions,batch_size=batch_size)

            effect = ((probs_origin-probs_interv)[0,0]
                        + (probs_origin-probs_interv)[1,1]
                        + (probs_interv-probs_origin)[0,1]
                        + (probs_interv-probs_origin)[1,0])/4

            effect_list.append(effect)
        return np.array(effect_list)
