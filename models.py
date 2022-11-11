import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup

# -------------------------------------------------------
#   BERT model
# -------------------------------------------------------
class BertMaxPool(nn.Module):
    def __init__(self, pretrained_model):
        super(BertMaxPool, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
    
    def forward(self, input_ids, attention_mask):
        # get bert outputs (last_hidden_state, pooler_output)
        last_hidden_state = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]

        # path 1. get cls hidden state
        # cls_hidden_state = last_hidden_state[:,0,:]

        # path 2. mean pooling, ref: sbert
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.shape).float() # count by attention mask size (input size)
        sum_mask = input_mask_expanded.sum(1) # sum and reduce one dimension -> (1, 768)
        sum_mask = torch.clamp(sum_mask, min=1e-9) # make sure no zero
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        outputs = sum_embeddings / sum_mask
        
        return outputs