from typing import Optional, Tuple, Any

import torch

from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from function.lm2.lm2_memory_module import LM2MemoryModule
# from rope_attention import LlamaAttentionWithPoPR


class LlamaDecoderLayerWithMemory(LlamaDecoderLayer):
    def __init__(self, 
                 config, 
                 layer_idx: int,
                 memory_module: LM2MemoryModule):
        super().__init__(config=config, 
                         layer_idx=layer_idx)
        self.memory_module = memory_module

    def forward(self, 
                hidden_states: torch.Tensor,  
                attention_mask: Optional[torch.Tensor]=None, 
                position_ids: Optional[torch.LongTensor] = None,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, # (cos, sin) for RoPE
                past_key_value: Optional[Any]=None,
                cache_position: Optional[torch.LongTensor]=None,
                output_attentions: Optional[bool]=False,
                use_cache: Optional[bool]=False,
                memory_states: Optional[torch.Tensor]=None, 
                **kwargs):

        # Llama self-attention
        residual = hidden_states # (B, S, d)
        hidden_states  = self.input_layernorm(hidden_states)
        attn_output, attn_weights = self.self_attn(
                                        hidden_states=hidden_states,
                                        position_embeddings=position_embeddings,
                                        attention_mask=attention_mask,
                                        past_key_value=past_key_value,
                                        cache_position=cache_position,
                                        output_attentions=output_attentions, #
                                        use_cache=use_cache, #
                                        **kwargs
                                        ) # output: attn_output, attn_weights
        hidden_states = residual + attn_output
        
        # LM2 memory module 추가
        residual2 = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        E_out, updated_mem = self.memory_module(hidden_states, 
                                                memory_states) # memory_module => (E_out, updated_mem)
        hidden_states = residual2 + E_out
        
        # Llama MLP
        residual3 = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual3 + hidden_states 
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,) 
        outputs += (updated_mem,)
    
        return outputs