import torch
import torch.nn as nn
import math
from xctopus.modules.lora import LoRA

# -----------------------------
# LoRA Linear Adapter
# -----------------------------
class LoRALinear(nn.Module):
    """
    Linear layer with LoRA.
    Small parameters A and B are trained, base W is frozen.
    """
    def __init__(self, in_features, out_features, r=4, alpha=1.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha

        # Base weights (frozen)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01, requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # LoRA adapters
        if r > 0:
            self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
            self.B = nn.Parameter(torch.randn(out_features, r) * 0.01)
            self.scaling = alpha / r
        else:
            self.A = None
            self.B = None
            self.scaling = 0

    def forward(self, x):
        base_out = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.r > 0:
            lora_out = torch.nn.functional.linear(x, self.A.T)  # [batch, r]
            lora_out = torch.nn.functional.linear(lora_out, self.B) * self.scaling
            return base_out + lora_out
        else:
            return base_out

    def lora_parameters(self):
        if self.r > 0:
            return [self.A, self.B]
        else:
            return []

# -----------------------------
# Modifications to MultiHeadAttention for LoRA
# -----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, use_lora=False, lora_r=4, lora_alpha=1.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_lora = use_lora

        # Create base linear layers
        base_W_q = nn.Linear(d_model, d_model)
        base_W_k = nn.Linear(d_model, d_model)
        base_W_v = nn.Linear(d_model, d_model)
        base_W_o = nn.Linear(d_model, d_model)

        # Apply LoRA if enabled
        if use_lora:
            self.W_q = LoRA(base_W_q, rank=lora_r, alpha=lora_alpha)
            self.W_k = LoRA(base_W_k, rank=lora_r, alpha=lora_alpha)
            self.W_v = LoRA(base_W_v, rank=lora_r, alpha=lora_alpha)
            self.W_o = LoRA(base_W_o, rank=lora_r, alpha=lora_alpha)
        else:
            self.W_q = base_W_q
            self.W_k = base_W_k
            self.W_v = base_W_v
            self.W_o = base_W_o

    def split_heads(self, x):
        batch, seq_len, d_model = x.size()
        return x.view(batch, seq_len, self.num_heads, self.d_k).transpose(1,2)

    def combine_heads(self, x):
        batch, _, seq_len, d_k = x.size()
        return x.transpose(1,2).contiguous().view(batch, seq_len, self.d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)

    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations (with or without LoRA)
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_out = self.scaled_dot_product_attention(Q,K,V,mask)
        return self.W_o(self.combine_heads(attn_out))

    def lora_parameters(self):
        """Returns only LoRA parameters if enabled"""
        params = []
        if self.use_lora:
            for linear in [self.W_q, self.W_k, self.W_v, self.W_o]:
                if isinstance(linear, LoRA):
                    params.append(linear.lora_A)
                    params.append(linear.lora_B)
        return params

# -----------------------------
# Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# -----------------------------
# Position-wise Feed Forward
# -----------------------------
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# -----------------------------
# Encoder Layer
# -----------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, use_lora=False, lora_r=4, lora_alpha=1.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# -----------------------------
# Decoder Layer
# -----------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, use_lora=False, lora_r=4, lora_alpha=1.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

# -----------------------------
# Simplified forward in Transformer with external embeddings and LoRA
# -----------------------------
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff,
                 max_seq_length, dropout, embedding_dim=None, use_lora=False, lora_r=4, lora_alpha=1.0):
        super().__init__()
        self.d_model = d_model
        self.use_lora = use_lora

        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)

        self.input_projection = nn.Linear(embedding_dim, d_model) if embedding_dim else None

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha) 
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha) 
            for _ in range(num_layers)
        ])

    def encode(self, external_embeddings=None):
        if external_embeddings is None:
            raise ValueError("Must pass external embeddings for incremental forward")
        x = self.input_projection(external_embeddings) if self.input_projection else external_embeddings
        x = self.dropout(self.positional_encoding(x))
        src_mask = (x.abs().sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x.mean(dim=1)

    def lora_parameters(self):
        """Returns only trainable LoRA parameters from the entire Transformer"""
        params = []
        for layer in self.encoder_layers + self.decoder_layers:
            for module in layer.modules():
                if isinstance(module, MultiHeadAttention) and module.use_lora:
                    params.extend(module.lora_parameters())
                elif hasattr(module, 'lora_parameters'):
                    params.extend(module.lora_parameters())
        return params
    
    def freeze_base_parameters(self):
        """Freezes all base parameters, leaving only LoRA trainable"""
        for param in self.parameters():
            param.requires_grad = False
        # Ensure LoRA remains trainable
        for param in self.lora_parameters():
            param.requires_grad = True
