import torch
from torch.utils.data import Dataset

# -------------------------------------------------------
#   BERT basic
# -------------------------------------------------------
class BertDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        super(BertDataset, self).__init__()
        self.data=data
        self.tokenizer=tokenizer
        self.max_length=max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(
            text=self.data[index],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'input_ids': torch.as_tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.as_tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.as_tensor(token_type_ids, dtype=torch.long),
        }

    # stack data to batch manually
    # dynamic pad to max length of batch
    def collate_fn(batch):
        max_length = max([d['input_ids'].shape[1] for d in batch])
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []

        for d in batch:
            diff = max_length - d['input_ids'].shape[1]
            batch_input_ids.append(d['input_ids'][0].tolist() + [0] * diff)
            batch_attention_mask.append(d['attention_mask'][0].tolist() + [0] * diff)
            batch_token_type_ids.append(d['token_type_ids'][0].tolist() + [0] * diff)
        
        return {
            'input_ids': torch.tensor(batch_input_ids),
            'attention_mask': torch.tensor(batch_attention_mask),
            'token_type_ids': torch.tensor(batch_token_type_ids)
        }

# -------------------------------------------------------
#   Triplet BERT
# -------------------------------------------------------
class TripletBertDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        super(TripletBertDataset, self).__init__()
        self.data=data # with format ([[anchor, pos, neg], ...])
        self.tokenizer=tokenizer
        self.max_length=max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        anchor_inputs = self.tokenizer.encode_plus(
            text=self.data[index][0],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        pos_inputs = self.tokenizer.encode_plus(
            text=self.data[index][1],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        neg_inputs = self.tokenizer.encode_plus(
            text=self.data[index][2],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'anchor_input_ids': anchor_inputs['input_ids'],
            'anchor_attention_mask': anchor_inputs['attention_mask'],
            'pos_input_ids': pos_inputs['input_ids'],
            'pos_attention_mask': pos_inputs['attention_mask'],
            'neg_input_ids': neg_inputs['input_ids'],
            'neg_attention_mask': neg_inputs['attention_mask'],
        }

    # stack data to batch manually
    # dynamic pad to max length of batch
    def collate_fn(batch):
        # find max length from all anchers, poses & negs in batch
        max_length = max([d[ids].shape[1] for d in batch for ids in ['anchor_input_ids', 'pos_input_ids', 'neg_input_ids']])
        
        # zero padding
        batch_anchor_input_ids = []
        batch_anchor_attention_mask = []
        batch_pos_input_ids = []
        batch_pos_attention_mask = []
        batch_neg_input_ids = []
        batch_neg_attention_mask = []
        for d in batch:
            # anchor
            diff = max_length - d['anchor_input_ids'].shape[1]
            batch_anchor_input_ids.append(d['anchor_input_ids'][0].tolist() + [0] * diff)
            batch_anchor_attention_mask.append(d['anchor_attention_mask'][0].tolist() + [0] * diff)

            # pos
            diff = max_length - d['pos_input_ids'].shape[1]
            batch_pos_input_ids.append(d['pos_input_ids'][0].tolist() + [0] * diff)
            batch_pos_attention_mask.append(d['pos_attention_mask'][0].tolist() + [0] * diff)

            # neg
            diff = max_length - d['neg_input_ids'].shape[1]
            batch_neg_input_ids.append(d['neg_input_ids'][0].tolist() + [0] * diff)
            batch_neg_attention_mask.append(d['neg_attention_mask'][0].tolist() + [0] * diff)
        
        return {
            'anchor_input_ids': torch.tensor(batch_anchor_input_ids),
            'anchor_attention_mask': torch.tensor(batch_anchor_attention_mask),
            'pos_input_ids': torch.tensor(batch_pos_input_ids),
            'pos_attention_mask': torch.tensor(batch_pos_attention_mask),
            'neg_input_ids': torch.tensor(batch_neg_input_ids),
            'neg_attention_mask': torch.tensor(batch_neg_attention_mask),
        }