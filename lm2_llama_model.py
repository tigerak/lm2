from typing import Optional, Any

import torch
import torch.nn as nn

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaModel, LlamaConfig

from function.lm2.lm2_memory_module import LM2MemoryModule
from function.lm2.lm2_decoder_layer import LlamaDecoderLayerWithMemory
from rope import build_sin_cos_position_embeddings

class LlamaModelWithMemory(LlamaModel):
    def __init__(self, 
                 config, 
                 memory_module: LM2MemoryModule):
        super().__init__(config)

        # 기존 llama decoder block 전부 교체
        new_layers = nn.ModuleList()
        for idx, layer in enumerate(self.layers):
            block = LlamaDecoderLayerWithMemory(config=config,
                                                layer_idx=idx,
                                                memory_module=memory_module)
            block.load_state_dict(layer.state_dict(), strict=False)
            new_layers.append(block)
        self.layers = new_layers

        self.memory_module = memory_module

    def forward(self, 
                input_ids: Optional[torch.LongTensor]=None, 
                inputs_embeds: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.Tensor]=None, 
                position_ids: Optional[torch.LongTensor]=None,
                past_key_values: Optional[Any]=None,
                cache_position: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool]=None,
                output_attentions: Optional[bool]=None,
                output_hidden_status: Optional[bool]=None,
                memory_states: torch.Tensor=None,
                **kwargs):
        if input_ids is None and inputs_embeds is None:
            raise ValueError("input_ids나 inputs_embeds가 없다.")
        
        if input_ids is not None:
            bsz, seq_len = input_ids.shape
            input_embeds = self.embed_tokens(input_ids)
            ### 일단 차원 맞춰주자
            num_heads = 4
            attention_mask = attention_mask.unsqueeze(-1).expand(-1, -1, seq_len)
            attention_mask = attention_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
            attention_mask = attention_mask.to(dtype=input_embeds.dtype)
        else:
            bsz, seq_len = input_embeds.shape[:2]
            input_embeds = input_embeds
            ### 일단 차원 맞춰주자

        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(input_embeds, position_ids)
        position_embeddings = (cos, sin)

        # layer stack
        all_attentions = () if output_attentions else None
        new_past_key_values = () if use_cache else None

        hidden_states = input_embeds
        current_mem = memory_states

        for i, layer in enumerate(self.layers):
            # hidden_states, layer_weights, layer_mem = layer(
            layer_out = layer(hidden_states=hidden_states,
                              attention_mask=attention_mask,
                              position_ids=position_ids,
                              position_embeddings=position_embeddings,
                              past_key_value=past_key_values,
                              memory_states=current_mem,
                              output_attentions=output_attentions,
                              cache_position=cache_position,
                              use_cache=use_cache,
                              **kwargs)
            hidden_states = layer_out[0] 
            current_mem = layer_out[-1]
            if output_attentions:
                all_attentions = all_attentions + (layer_out[1],)

        # return
        output = (hidden_states, )
        if output_attentions:
            output += (all_attentions, )
        output += (current_mem, )

        return output
        # return hidden_states, mem