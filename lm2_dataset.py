import json
from time import time

# torch
import torch
from torch.utils.data import Dataset, DataLoader


class LM2_Dataset(Dataset):
    def __init__(self, json_path, tokenizer):

        with open(json_path, 'r', encoding="utf-8") as file:
            self.ytn_data = json.load(file)

        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.ytn_data)
    
    def __getitem__(self, index):
        title = self.ytn_data[index]["metadata"]["title"]
        article = self.ytn_data[index]["document"]

        prompt_format = f"### TITLE: {title}\n### ARTICLE:"
        prompt_complete = f"{prompt_format} {article}"

        encoding = self.tokenizer(text=prompt_complete, 
                                  truncation=True,
                                  max_length=1024,
                                  padding="max_length",
                                  add_special_tokens=True,
                                  return_tensors="pt")
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        prompt_enc = self.tokenizer(text=prompt_format, add_special_tokens=False)
        prompt_len = len(prompt_enc["input_ids"])
        labels[:prompt_len] = -100 # prompt 학습 제외
        labels[labels == self.tokenizer.pad_token_id] = -100  # 패딩된 부분 학습 제외

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels}
    
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    return {"input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels}