import torch
import torch.nn.functional as F
import numpy as np
import pdb

def process_long_input(model, input_ids, attention_mask):
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        return_dict=True
    )
    sequence_output = output['last_hidden_state']
    attention = (output["attentions"][-1] + output["attentions"][-2] + output["attentions"][-3]) / 3.0
    # attention = torch.cat(output["attentions"][-4:], dim=1)
    hidden_states = output["hidden_states"][7:]
    return sequence_output, attention, hidden_states

