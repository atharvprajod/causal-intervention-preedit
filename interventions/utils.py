from typing import Optional, Tuple, Union, List
import numpy as np
import itertools
from transformers.configuration_utils import PretrainedConfig


def mask_out(input_ids, pron_loc, option, mask_id):
    # note annotations are shifted by 1 because special tokens were omitted
    return (
        input_ids[: pron_loc[0]]
        + [mask_id for _ in range(len(option))]
        + input_ids[pron_loc[-1] :]
    )


def extract_from_config(config: PretrainedConfig):
    if config.model_type in ["albert", "bert", "roberta"]:
        return config.num_hidden_layers, config.num_attention_heads
    elif config.model_type in ["gpt2"]:
        return config.n_layer, config.n_head
    else:
        raise NotImplementedError


class InterventionBase:
    """
    Base class for interventions
    """

    def __init__(self, interv_type):
        self.interv_type = interv_type

    def tokenize_without_sp_tokens(self, text: str):
        tokenized = self.tokenizer(text).input_ids
        if "bert" in self.model.config.model_type:
            return tokenized[1:-1]
        elif "gpt2" in self.model.config.model_type:
            return tokenized

    def convert_to_tokens(
        self,
        word: str,
        sent: str,
    ):
        word_ids, messages = self.find_word(word, sent)
        if messages=='no match':
            raise ValueError(f"No match found for {word} in {sent}")
        elif messages=='multiple matches':
            print(f"Multiple matches found for {word} in {sent}")
        sent_before = sent.split(" ")[:word_ids[0]]
        sent_after = sent.split(" ")[:(word_ids[-1]+1)]
        tokens_before = self.tokenizer(" ".join(sent_before)).input_ids
        tokens_after = self.tokenizer(" ".join(sent_after)).input_ids
        start_id = len(tokens_before)
        end_id = len(tokens_after)
        self.ckeck_align(word, sent, start_id, end_id)
        return [i for i in range(start_id, end_id)]
    
    def ckeck_align(
        self,
        word: str,
        sent: str,
        start_id: int,
        end_id: int,
    ):
        tokenized = self.tokenizer(sent).input_ids
        target = " ".join([self.tokenizer.decode(token) for token in tokenized[start_id,:end_id]]).strip(' ,.;:!?').lower()
        if word.lower() not in [target,target,target.replace("'s","").replace(";s","").replace("’s","")]:
            print(f"There seems to be misalignment for {word} in {sent}")


    def load_model(self, model_name: str):
        if model_name.startswith("albert"):
            from transformers import AlbertTokenizer, AlbertForMaskedLM
            from skeletons.skeleton_modeling_albert import SkeletonAlbertForMaskedLM

            self.tokenizer = AlbertTokenizer.from_pretrained(model_name)
            self.model = AlbertForMaskedLM.from_pretrained(model_name)
            self.skeleton_model = SkeletonAlbertForMaskedLM
            self.mask_id = self.tokenizer(self.tokenizer.mask_token).input_ids[1]

        elif model_name.startswith("bert"):
            from transformers import BertTokenizer, BertForMaskedLM
            from skeletons.skeleton_modeling_bert import SkeletonBertForMaskedLM

            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForMaskedLM.from_pretrained(model_name)
            self.skeleton_model = SkeletonBertForMaskedLM
            self.mask_id = self.tokenizer(self.tokenizer.mask_token).input_ids[1]

        elif model_name.startswith("roberta"):
            from transformers import RobertaTokenizer, RobertaForMaskedLM
            from skeletons.skeleton_modeling_roberta import SkeletonRobertaForMaskedLM

            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaForMaskedLM.from_pretrained(model_name)
            self.skeleton_model = SkeletonRobertaForMaskedLM
            self.mask_id = self.tokenizer(self.tokenizer.mask_token).input_ids[1]

        else:
            raise NotImplementedError

    def find_word(
        self, 
        word: str, 
        sent: str,
        ):
        split_word = word.split(" ")
        split_sent = sent.split(" ")
        # find each word in the phrase
        find_phrase = []
        for w in split_word:
            find_w = []
            for word_id, sent_word in enumerate(split_sent):
                stripped = sent_word.strip(' ,.;:!?').lower()
                stripped_list = [sent_word,stripped,stripped.replace("'s","").replace(";s","").replace("’s","")]
                if w.lower() in stripped_list:
                    find_w.append(word_id)
            find_phrase.append(find_w)
        # consider all possible combinations
        candidates = np.array(list(itertools.product(*find_phrase)))
        # find ones that are in a sequence
        candidate_test = np.array([np.all(np.diff(candidate)==1) for candidate in candidates])
        if candidate_test.sum()==0:
            return (None, 'no match')
        elif candidate_test.sum()>1:
            return (candidates[candidate_test][0], 'multiple matches')
        else:
            # return the id of the first word
            return (candidates[candidate_test][0], 'unique match!')
        
    def calc_logprob_skeleton(
        sent: str,
        cont: str,
        intervention
    ):