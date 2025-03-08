from typing import Optional

import torch
import torch.nn as nn

from function.lm2.memory_module import LM2MemoryModule
from function.lm2.memory_llama_model import LlamaModelWithMemory

class LM2ForCausalLM(nn.Module):
    def __init__(self,
                 config,
                 memory_module: LM2MemoryModule):
        super().__init__()
        # base model
        self.model = LlamaModelWithMemory(config,
                                          memory_module=memory_module)
        # lm_head
        self.lm_head = nn.Linear(config.hidden_size, 
                                 config.vocab_size, 
                                 bias=False)
        
        # tie_word_embeddings => skip for simplicity
        # typically self.model.embed_tokens.weight => self.lm_head.weight

    def forward(self,
                input_ids: Optional[torch.LongTensor]=None,
                attention_mask: Optional[torch.Tensor]=None,
                memory_states: Optional[torch.Tensor]=None,
                labels: Optional[torch.LongTensor]=None,
                output_attentions: bool=False,
                use_cache: bool=False,
                **kwargs):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             memory_states=memory_states,
                             output_attentions=output_attentions,
                             use_cache=use_cache,
                             **kwargs)
        hidden_states = outputs[0]
        new_memory = outputs[-1]

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        # return (logits, loss, new_memory, [attentions?])
        if output_attentions:
            all_attn = outputs[1]
            return (logits, loss, new_memory, all_attn)
        return (logits, loss, new_memory)