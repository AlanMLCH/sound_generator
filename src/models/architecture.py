import torch
import torch.nn as nn

def causal_mask(T: int, device: torch.device):
    # Upper-triangular mask with -inf above the diagonal (block future tokens)
    mask = torch.full((T, T), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)

class MusicTransformer(nn.Module):
    """
    Decoder-only (GPT-like) using TransformerEncoder + causal mask.
    No cross-attention / memory needed.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 768,
        dropout: float = 0.1,
        max_len: int = 1536,
    ):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T) token ids
        returns logits: (B, T, V)
        """
        B, T = x.shape
        device = x.device
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} > max_len {self.max_len}")

        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        h = self.tok_emb(x) + self.pos_emb(positions)

        # causal mask for autoregressive LM
        mask = causal_mask(T, device)
        h = self.enc(h, mask=mask)
        h = self.ln(h)
        return self.out(h)
