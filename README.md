# causal-intervention
Work in progress...

### features:
- Based on the causal-intervention-demo (https://huggingface.co/spaces/taka-yamakoshi/causal-intervention-demo) implementation
    - Package interventions as a tuple (head_id, position, swap_ids) where swap_ids is a pair of batch_ids to swap
    - Batchify interventions across layers and heads into one pass (batch_size=num_layers*num_heads_per_layer*2)
- Adjusted for autoregressive models (GPT2, LLaMA, etc.)
- Accommodate cases with different number of tokens

### intended usage:
```{python}
from interventions import interchange

sent_1 = "In one word Yes/No, is the property is_small.1 true for the concept salamander?"
sent_2 = "In one word Yes/No, is the property is_small.1 true for the concept dinosaur?"
cont_1 = "Yes"
cont_2 = "No"

config = interchange.InterchangeInterventionConfig(sents=[sent_1,sent_2],conts=[cont_1,cont_2])
intervention_model = InterchangeIntervention("gpt2", config)
effects = intervention_model.run(
    targets=["slamander", "dinosaur"],
    rep_type=["lay","qry","key","val"],
    multihead=True)
```

### TODOs
- 