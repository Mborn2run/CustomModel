import torch
import torch.nn as nn

class Transformer_Encoder(nn.Module):
    def __init__(self, embed_dim, heads, dropout, forward_expansion):
        super(Transformer_Encoder, self).__init__()
        self.multi_attention = nn.MultiheadAttention(embed_dim, heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.transformer_block = nn.TransformerEncoderLayer(embed_dim, heads, dropout, forward_expansion)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)