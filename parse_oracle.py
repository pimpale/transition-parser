import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

# gpt 2 style positional encoding
def compute_positional_encoding(ctx_len: int, d_embed: int):
    # compute the positional encoding
    # pe in (ctx_len, d_embed)
    pe = torch.zeros(ctx_len, d_embed)
    for pos in range(ctx_len):
        for i in range(0, d_embed, 2):
            pe[pos, i] = np.sin(pos / (10000 ** ((2 * i)/d_embed)))
            pe[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1))/d_embed)))
    return pe

@dataclass
class ParseOracleConfig:
    """
    Configuration for the ParseOracle model
    """
    stack_size: int
    buffer_size: int
    d_embed: int


class ParseOracle(nn.Module):
    """
    Neural network using embeddings for parsing
    Has an attention layer and a feedfoward layer
    """
    def __init__(self, config: ParseOracleConfig):
        super().__init__()
        self.config = config
        ctx_len = config.stack_size + config.buffer_size
        self.register_buffer("positional_encoding", compute_positional_encoding(ctx_len, config.d_embed))
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=config.d_embed, nhead=8), num_layers=1)
        self.lin1 = nn.Linear(config.d_embed, 64)
        self.lin2 = nn.Linear(64 * ctx_len, 3)

    def forward(self, buf, stack):
        """
        Forward pass of the model
        @param buf: input tensor (batch_size, buffer_size, emb_size)
        @param stack: input tensor (batch_size, stack_size, emb_size)
        """
        # concat the buffer and stack
        x = torch.cat([buf, stack], dim=1)
        # add the positional encoding
        x = x + self.positional_encoding
        # transformer encoder
        x = self.transformer_encoder(x)
        # linear layer
        x = F.relu(self.lin1(x))
        # flatten
        x = x.flatten(1)
        # linear layer
        x = self.lin2(x)