import torch
import torch.nn as nn
import math

class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=384, nhead=6, num_layers=6, dim_feedforward=1024, dropout=0.1, max_len=2048):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, x, mem=None):
        # x: (B, T)
        B, T = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.tok_emb(x) + self.pos_emb(pos)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        h = self.dec(tgt=h, memory=None, tgt_mask=tgt_mask)  # decoder-only
        h = self.ln(h)
        logits = self.out(h)
        return logits
