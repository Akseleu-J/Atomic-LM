import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from modeule import ModelConfig, HybridBlock, RMSNorm
class Atomic(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        self.embed = nn.Embedding(config.vocab_size, d_model)
        self.blocks = nn.ModuleList()
        for i, layer_type in enumerate(config.layer_types):
            use_moe = (i + 1) % config.moe_every == 0
            block = HybridBlock(
                config=config,
                layer_type=layer_type,
                use_moe=use_moe)
            self.blocks.append(block)
        self.final_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, config.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self._init_weights()
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    def forward(self, tokens, targets=None):
        x = self.embed(tokens)
        for i, block in enumerate(self.blocks):
            if block.layer_type == "mla":
                use_rope_here = (i == 5)
                x = block(x, use_rope=use_rope_here)
            else:
                x = block(x)
        x = self.final_norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            pad_idx = getattr(self.config, 'pad_token_id', 0)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=pad_idx
            ) 
            
        return logits, loss
