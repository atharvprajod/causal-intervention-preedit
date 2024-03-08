---
title: causal intervention for prompts
emoji: 🤗
colorFrom: yellow
colorTo: red
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# causal-intervention

## Features:
- Accommodates any task with a pair of prompts and different continuations for each prompt.
- Increases speed by batchfying interventions for different heads into one pass (batch_size=num_target_heads\*2).
- Decreases RAM usage by representing interventions as a tuple (head_id, target_ids, swap_ids) where target_ids is a pair of token positions to swap and swap_ids is a pair of batch_ids to swap.

## Usage:
### (Only for gated models) Get access to the model
In order to use a gated model like Llama, make sure that you have access to it on hugginface (e.g. at https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

You also need to create an access token (https://huggingface.co/docs/hub/en/security-tokens).

Then run the following command in your terminal to log in to your huggingface account and add your token to the machine.
```{shell}
huggingface-cli login
```

### Sample python script
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

# Set the config
config = interchange.InterchangeInterventionConfig(sents=[sent_1,sent_2],conts=conts)

# Create the intervention model
intervention_model = InterchangeIntervention("meta-llama/Llama-2-7b-chat-hf", config)

# Run the intervention
# "output" is a dictionary with the following keys:
# list(output.keys()) = ["original", "interv_layer_0", "interv_layer_1",...]
# The values are numpy arrays of the shape (num_conts, num_sents, num_heads) 
# containing probability of each continuation for each prompt given the intervention
output = intervention_model.run(
    targets=["salamander", "dinosaur"], # target words to swap
    rep_types=["lay","qry","key","val"], # representations to swap
    multihead=True) # set multihead to True if swap all heads; otherwise set head=[list_of_head_ids]
```

## TODOs
- add more model options (potentially masked language models)
- add more intervention options
- accommodate cases where targets have different number of tokens