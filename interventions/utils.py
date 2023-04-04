from transformers.configuration_utils import PretrainedConfig

def load_model(model_name):
    if model_name.startswith('albert'):
        from transformers import AlbertTokenizer, AlbertForMaskedLM
        from skeletons.skeleton_modeling_albert import SkeletonAlbertForMaskedLM
        tokenizer = AlbertTokenizer.from_pretrained(model_name)
        model = AlbertForMaskedLM.from_pretrained(model_name)
        skeleton_model = SkeletonAlbertForMaskedLM
    elif model_name.startswith('bert'):
        from transformers import BertTokenizer, BertForMaskedLM
        from skeletons.skeleton_modeling_bert import SkeletonBertForMaskedLM
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForMaskedLM.from_pretrained(model_name)
        skeleton_model = SkeletonBertForMaskedLM
    elif model_name.startswith('roberta'):
        from transformers import RobertaTokenizer, RobertaForMaskedLM
        from skeletons.skeleton_modeling_roberta import SkeletonRobertaForMaskedLM
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForMaskedLM.from_pretrained(model_name)
        skeleton_model = SkeletonRobertaForMaskedLM
    else:
        raise NotImplementedError
    mask_id = tokenizer(tokenizer.mask_token).input_ids[1]
    return tokenizer,model,skeleton_model,mask_id

def mask_out(input_ids,pron_loc,option,mask_id):
    # note annotations are shifted by 1 because special tokens were omitted
    return input_ids[:pron_loc[0]] + [mask_id for _ in range(len(option))] + input_ids[pron_loc[-1]:]

def extract_from_config(
    config: PretrainedConfig
    ):
    if config.model_type in ['albert','bert','roberta']:
        return config.num_hidden_layers, config.num_attention_heads
    elif config.model_type in ['gpt2']:
        return config.n_layer, config.n_head
    else:
        raise NotImplementedError

class InterventionBase():
    '''
        Base class for interventions
    '''
    def __init__(self,interv_type):
        self.interv_type = interv_type

    def tokenize_without_sp_tokens(
        self,
        text: str
        ):
        tokenized = self.tokenizer(text).input_ids
        if 'bert' in self.model.config.model_type:
            return tokenized[1:-1]
        elif 'gpt2' in self.model.config.model_type:
            return tokenized

    def convert_to_tokens(
        self,
        word_ids: List[int],
        sent: str,
        splitter: str=' ',
        ):
        sent_before = sent.split(splitter)[:word_ids[0]]
        sent_after = sent.split(splitter)[word_ids[-1]+1:]
        tokens_before = self.tokenizer(sent_before).input_ids
        tokens_after = self.tokenizer(sent_after).input_ids
        if 'bert' in self.model.config.model_type:
            start_id = len(tokens_before) - 1
            end_id = len(tokens_after) - 1
        elif 'gpt2' in self.model.config.model_type:
            start_id = len(tokens_before)
            end_id = len(tokens_after)
        return [i for i in range(start_id,end_id)]
