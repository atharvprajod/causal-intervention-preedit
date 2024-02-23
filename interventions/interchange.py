from typing import Optional, Tuple, Union, List
import numpy as np

import torch
import torch.nn.functional as F

from .utils import extract_from_config, InterventionBase

class InterchangeInterventionConfig:
    """
    Config class for the 'InterchangeIntervention'
    """
    def __init__(
        self,
        sents: Tuple[str, str],
        conts: List[str],
    ):
        """
        sents: minimal pair of sentences
        conts: list of possible continuations
        """
        self.sents = sents
        self.conts = conts
        self.contexts = self.find_context()

    def find_context(self):
        sent_1 = self.sents[0]
        sent_2 = self.sents[1]
        if sent_1==sent_2:
            return 0, "", ""
        else:
            split_sent_1 = sent_1.split(' ')
            split_sent_2 = sent_2.split(' ')
            min_sent_len = min(len(split_sent_1),len(split_sent_2))
            # remove each word from the start until they are different
            word_id = 0
            while split_sent_1[word_id]==split_sent_2[word_id]:
                word_id += 1
                if word_id==min_sent_len:
                    word_id -= 1
                    break
            start_id = word_id
            # remove each word from the end until they are different
            word_id = -1
            while split_sent_1[word_id]==split_sent_2[word_id]:
                word_id -= 1
                if word_id==-min_sent_len-1:
                    word_id += 1
                    break
            end_id = word_id+1 if word_id!=-1 else None

            context_words_1 = split_sent_1[start_id:end_id]
            context_words_2 = split_sent_2[start_id:end_id]

            context_1 = ' '.join(context_words_1)
            context_2 = ' '.join(context_words_2)
            return [context_1, context_2]

class InterchangeIntervention(InterventionBase):
    """
    Performs interchange interventions
    """

    def __init__(self, model_name, interv_config, **model_args):
        super().__init__("interchange")

        self.sents = interv_config.sents
        self.conts = interv_config.conts

        self.load_model(model_name, **model_args)
        self.num_layers, self.num_heads = extract_from_config(self.model.config)

    def create_interventions(
        self,
        targets: Tuple[List[str],List[str]], # tuple of target phrases (list of words)
        rep_types: List[str],
        multihead: bool = True,
        heads: List[int] = [],
    ):
        # convert targets to tokens
        target_ids = []
        for phrase, sent in zip(targets, self.sents):
            token_ids = []
            for word in phrase:
                token_ids.extend(self.convert_to_tokens(word, sent))
            target_ids.append(token_ids)
        assert len(target_ids[0])==len(target_ids[1]), "targets do not have the same number of tokens"

        interventions = {}
        for rep in ["lay", "qry", "key", "val", "trfm"]:
            if rep in rep_types:
                if multihead:
                    interventions[rep] = [
                        (head_id, [token_1,token_2], [0, 1])
                        for token_1 in target_ids[0]
                        for token_2 in target_ids[1]
                        for head_id in range(self.num_heads)
                    ]
                else:
                    interventions[rep] = [
                        (head_id, [token_1,token_2], [i, i + len(heads)])
                        for token_1 in target_ids[0]
                        for token_2 in target_ids[1]
                        for i, head_id in enumerate(heads)
                    ]
            else:
                interventions[rep] = []
        return interventions

    def run_interventions(
        self,
        interventions: dict,
        batch_size: int,
    ):
        probs = []
        for cont in self.conts:
            cont_start_1 = len(self.tokenizer(self.sents[0]).input_ids)-1 # assumes gpt2 type tokenizer/model
            cont_start_2 = len(self.tokenizer(self.sents[1]).input_ids)-1 # assumes gpt2 type tokenizer/model
            cont_tokens = self.tokenize_without_sp_tokens(cont)

            sent_1 = self.sents[0] + " " + cont
            sent_2 = self.sents[1] + " " + cont

            input_ids_all = self.tokenizer(
                [
                    *[sent_1 for _ in range(batch_size)],
                    *[sent_2 for _ in range(batch_size)],
                ],
                return_tensors="pt", 
                padding="longest")["input_ids"]

            outputs = self.skeleton_model(self.model, input_ids_all, interventions)
            logprobs = F.log_softmax(outputs["logits"], dim=-1)
            logprobs_1, logprobs_2 = logprobs[:batch_size], logprobs[batch_size:]

            evals_1 = [
                logprobs_1[:, cont_start_1+i, token].numpy()
                for i, token in enumerate(cont_tokens)
            ]
            evals_2 = [
                logprobs_2[:, cont_start_2+i, token].numpy()
                for i, token in enumerate(cont_tokens)
            ]
            probs.append(
                [np.exp(np.mean(evals_1, axis=0)), np.exp(np.mean(evals_2, axis=0))]
            )
        probs = np.array(probs)
        assert (
            probs.shape[0] == len(self.conts) and probs.shape[1] == 2 and probs.shape[2] == batch_size
        )
        return probs

    def run(
        self,
        targets: Tuple[str, str],
        rep_types: List[str],
        multihead: bool = True,
        heads: List[int] = [],
    ):
        '''
        below needs to be updated
        '''
        self.input_ids_1 = self.tokenizer(self.sents[0]).input_ids
        self.input_ids_2 = self.tokenizer(self.sents[1]).input_ids

        target_tokens_1 = self.convert_to_tokens(targets[0], self.sents[0])
        target_tokens_2 = self.convert_to_tokens(targets[1], self.sents[1])
        assert len(target_tokens_1) == len(target_tokens_2), "targets do not have the same number of tokens"
        interventions_empty = {"lay": [], "qry": [], "key": [], "val": []}
        batch_size = 1 if multihead else len(heads)

        # run without intervention
        interventions = [interventions_empty for i in range(self.num_layers)]
        probs_origin = self.run_interventions(interventions, batch_size=1)

        # run intervention for each layer
        effect_list = []
        for layer_id in range(self.num_layers):
            interventions = [
                interventions_layer if i == layer_id else interventions_empty
                for i in range(num_layers)
            ]
            probs_interv = self.run_interventions(interventions, batch_size=batch_size)

            effect = (
                (probs_origin - probs_interv)[0, 0]
                + (probs_origin - probs_interv)[1, 1]
                + (probs_interv - probs_origin)[0, 1]
                + (probs_interv - probs_origin)[1, 0]
            ) / 4

            effect_list.append(effect)
        return np.array(effect_list)
