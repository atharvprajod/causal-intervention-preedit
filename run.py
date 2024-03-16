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
intervention_model = interchange.InterchangeIntervention("meta-llama/Llama-2-7b-chat-hf", config)

# Run the intervention
# "output" is a dictionary with the following keys:
# list(output.keys()) = ["original", "interv_layer_0", "interv_layer_1",...]
# The values are numpy arrays of the shape (num_conts, num_sents, num_heads) 
# containing probability of each continuation for each prompt given the intervention
output = intervention_model.run(
    targets=["salamander", "dinosaur"], # target words to swap
    rep_types=["lay","qry","key","val"], # representations to swap
    multihead=True) # set multihead to True if swap all heads; otherwise set head=[list_of_head_ids]

import json
file_name = 'output.json'

# Open a new file in write mode and use json.dump to write the data
with open(file_name, 'w') as json_file:
    json.dump(output, json_file, indent=4)

print(f"Data has been written to {file_name}")