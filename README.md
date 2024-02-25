# causal-intervention

### Features:
- Accommodates any task with a pair of prompts and different continuations for each prompt.
- Increases speed by batchfying interventions for different heads into one pass (batch_size=num_target_heads\*2).
- Decreases RAM usage by representing interventions as a tuple (head_id, target_ids, swap_ids) where target_ids is a pair of token positions to swap and swap_ids is a pair of batch_ids to swap.

### Intended usage:
Details are provided in the README inside `interventions` directory
```{python}
from interventions import interchange

# Define a pair of prompts and the continuations
# Example 1: situation model (Winograd)
sent_1 = "Paul tried to call George but he wasn't successful . Who wasn't successful ?"
sent_2 = "Paul tried to call George but he wasn't available . Who wasn't available ?"
conts = ["Paul", "George"]

# Example 2: feature norms
sent_1 = "In one word 'Yes' or 'No', is the property is_small true for the concept salamander ?"
sent_2 = "In one word 'Yes' or 'No', is the property is_small true for the concept dinosaur ?"
conts = ["Yes","No"] 

# Example 3: emotional reaction
sent_1 = "Paul scored a goal on George, which made Paul"
sent_2 = "Paul scored a goal on George, which made George"
conts = ["happy", "sad", "angry", "scared"] # continuations can be more than two

# set the config
config = interchange.InterchangeInterventionConfig(sents=[sent_1,sent_2],conts=conts)

# create the intervention model
intervention_model = InterchangeIntervention("meta-llama/Llama-2-7b-chat-hf", config)

# run the intervention
output = intervention_model.run(
    targets=["salamander", "dinosaur"], # target words to swap
    rep_type=["lay","qry","key","val"], # representations to swap
    multihead=True)
```

### TODOs
- add more model options (potentially masked language models)
- add more intervention options
- accommodate cases where targets have different number of tokens