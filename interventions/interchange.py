from typing import Optional, Tuple, Union, List
import numpy as np

import torch
import torch.nn.functional as F

from .utils import custom_split, extract_from_config, InterventionBase

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

    # essentially use it to identify the differences between 2 sentences
    # so in case of com2sense, determine which is major variations that lead t
    # false
    def find_context(self):
        sent_1 = self.sents[0]
        sent_2 = self.sents[1]
        if sent_1==sent_2:
            return 0, "", ""
        else:
            # tokenize
            split_sent_1 = custom_split(sent_1)
            split_sent_2 = custom_split(sent_2)
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

    def __init__(
            self,
            model_name:str=None,
            interv_config:InterchangeInterventionConfig=None,
            cache_dir:str=None,
            device:torch.device=torch.device("cpu"),
            ):
        super().__init__("interchange")

        self.sents = interv_config.sents
        self.conts = interv_config.conts

        self.device = device

        self.load_model(model_name, cache_dir, device)
        self.num_layers, self.num_heads = extract_from_config(self.model.config)

    def create_interventions(
    self,
    targets: Union[Tuple[List[str],List[str]], Tuple[List[int],List[int]]], # tuple of target phrases (list of words or token ids)
    rep_types: List[str],
    multihead: bool = True,
    heads: List[int] = [],
    ):
    # Check if the first element of the first tuple is a string to determine if conversion is needed
        if isinstance(targets[0][0], str):
            target_ids = []
            for target_group in targets:
                token_ids_group = []
                for word in target_group:
                    tokenized_word = self.tokenizer(word, add_special_tokens=False)['input_ids']
                    token_ids_group.extend(tokenized_word)
                # Append the group as a single list if pairing is intended
                target_ids.append(token_ids_group)
            print(f'Target IDs (paired): {target_ids}')
        else:
            target_ids = targets
        assert len(target_ids[0])==len(target_ids[1]), "targets do not have the same number of tokens"

        interventions = {}
        for rep in ["lay", "qry", "key", "val", "trfm"]:
            if rep in rep_types:
                if multihead:
                    interventions[rep] = [
                        (head_id, [token_1,token_2], [0, 1])
                        for token_1, token_2 in zip(target_ids[0], target_ids[1])
                        for head_id in range(self.num_heads)
                    ]
                else:
                    interventions[rep] = [
                        (head_id, [token_1,token_2], [i, i + len(heads)])
                        for token_1, token_2 in zip(target_ids[0], target_ids[1])
                        for i, head_id in enumerate(heads)
                    ]
            else:
                interventions[rep] = []
        return interventions
    # where big chunk of actual causal intervention is actually hapening
    
    def run_interventions(
        self,
        interventions: dict,
        batch_size: int,
    ):
        probs = []
        # go over all possible extensions to the sentence and assign
        # probs to each continuation
        for cont in self.conts:
            cont_start_1 = len(self.tokenizer(self.sents[0]).input_ids)-1 # assumes gpt2 type tokenizer/model
            cont_start_2 = len(self.tokenizer(self.sents[1]).input_ids)-1 # assumes gpt2 type tokenizer/model
            cont_tokens = self.tokenize_without_sp_tokens(cont)

            sent_1 = self.sents[0] + " " + cont
            sent_2 = self.sents[1] + " " + cont
            # process input data as tensors for processing by Llama
            inputs_all = self.tokenizer(
                [
                    *[sent_1 for _ in range(batch_size)],
                    *[sent_2 for _ in range(batch_size)],
                ],
                return_tensors="pt", 
                padding="longest").to(self.device)
            # print(inputs_all)
            print(f'interventions: {interventions}')
            outputs = self.skeleton_model(inputs_all["input_ids"], inputs_all["attention_mask"], interventions)
            logprobs = F.log_softmax(outputs["logits"], dim=-1, dtype=torch.float32)
            logprobs_1, logprobs_2 = logprobs[:batch_size], logprobs[batch_size:]

            evals_1 = [
                logprobs_1[:, cont_start_1+i, token].cpu().numpy()
                for i, token in enumerate(cont_tokens)
            ]
            evals_2 = [
                logprobs_2[:, cont_start_2+i, token].cpu().numpy()
                for i, token in enumerate(cont_tokens)
            ]
            probs.append(
                [np.exp(np.mean(evals_1, axis=0)), np.exp(np.mean(evals_2, axis=0))]
            )
        probs = np.array(probs)
        assert (
            probs.shape[0] == len(self.conts) and probs.shape[1] == 2 and probs.shape[2] == batch_size
        )
        return np.array(probs)

    def run(
        self,
        targets: Union[Tuple[List[str],List[str]], Tuple[List[int],List[int]]],
        rep_types: List[str],
        multihead: bool = True,
        heads: List[int] = [],
    ):
        probs = {}
        batch_size = 1 if multihead else len(heads)
        interventions_empty = {"lay": [], "qry": [], "key": [], "val": [], "trfm": []}
        interventions = [interventions_empty for i in range(self.num_layers)]
        probs["original"] = self.run_interventions(interventions, batch_size=1)

        interventions_layer = self.create_interventions(targets,rep_types,multihead,heads)
        
        # run intervention for each layer
        for layer_id in range(self.num_layers):
            interventions = [
                interventions_layer if i == layer_id else interventions_empty
                for i in range(self.num_layers)
            ]
            probs_interv = self.run_interventions(interventions, batch_size)
            probs[f"interv_layer_{layer_id}"] = probs_interv
        return probs
