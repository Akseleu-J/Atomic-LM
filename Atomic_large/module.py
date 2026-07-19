import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
@dataclass
class ModelConfig:
    vocab_size: int=8000
    d_model: int = 256
    max_seq_len: int = 512
    dropout: float = 0.1
    layer_types: list[str] = field(default_factory=lambda: [ 
        "mamba", "mamba", "mla", "mamba", "mamba", "mla", "mamba", "mamba", "mamba", "attn" 
   ]) 
    d_state: int = 16
    n_heads: int = 8
    mla_kv_dim: int = 64
    moe_every: int=2
    num_experts: int=8
    top_k: int = 2
    d_ff: int=1024
    @property
    def n_layers(self) -> int:
        return len(self.layer_types)
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight *x/rms
class MambaMixer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        d_model, d_state = config.d_model, config.d_state
        d_inner = d_model * 2
        self.in_proj = nn.Linear(d_model, d_inner*2, bias=False)
        self.conv = nn.Conv1d(
            d_inner, d_inner, kernel_size=4,
            groups=d_inner, padding=3)
        self.A_log = nn.Parameter(torch.randn(d_inner, d_state))
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1)
        self.out_proj = nn.Linear(d_inner, d_model)
        self.d_inner, self.d_state = d_inner, d_state
    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = self.conv(x_in.transpose(1, 2)).transpose(1, 2)
        x_conv = x_conv[:, :L, :] 
        x_conv = F.silu(x_conv)
        params = self.x_proj(x_conv)
        dt, B_ssm, C_ssm = params.split([1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(dt)
        A = -torch.exp(self.A_log)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        outputs = []
        for t in range(L): 
            dt_t = dt[:, t, :].unsqueeze(-1)
            B_t = B_ssm[:, t, :].unsqueeze(1)      
            C_t = C_ssm[:, t, :].unsqueeze(1)      
            
            dA = torch.exp(dt_t * A.unsqueeze(0))  
            dB = dt_t * B_t                        
            
            h = dA * h + dB * x_conv[:, t, :].unsqueeze(-1) 
            
            outputs.append((C_t * h).sum(-1))
        y = torch.stack(outputs, dim=1) * F.silu(z)
        return self.out_proj(y)
class CasualAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        d_model, n_heads = config.d_model, config.n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout, self.n_heads = nn.Dropout(config.dropout), n_heads
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        out = self.dropout(F.softmax(scores, dim=-1)) @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)
class MoEFF(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        d_model = config.d_model
        self.router = nn.Linear(d_model, self.num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, config.d_ff), nn.SiLU(),
            nn.Linear(config.d_ff, d_model))
            for _ in range(self.num_experts)
        ])
    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        top_probs, top_idx = torch.topk(router_probs, self.top_k, dim=-1)
        top_probs = top_probs/ top_probs.sum(-1, keepdim=True)
        outputs = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            mask = (top_idx == e).any(dim=-1)
            if not mask.any(): continue
            idx = mask.nonzero(as_tuple=True)
            weight_mask = (top_idx[idx] == e)
            weights = (top_probs[idx] * weight_mask).sum(-1, keepdim=True)
            outputs[idx] += weights * self.experts[e](x_flat[idx])
        return outputs.view(B, T, D)
class DenseFF(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        d_model, d_ff = config.d_model, config.d_ff
        dropout = config.dropout
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x): return self.net(x)
class MLA(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = self.d_model // self.n_heads
        self.rope_dim = 16
        self.total_head_dim = self.head_dim + self.rope_dim
        self.kv_dim = config.mla_kv_dim
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.total_head_dim, bias=False)
        self.kv_proj_down = nn.Linear(self.d_model, self.kv_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_dim)
        self.kv_up_proj = nn.Linear( 
            self.kv_dim, 
            self.n_heads * self.total_head_dim + self.n_heads * self.head_dim, 
            bias=False 
        ) 
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False) 
    def _apply_rope(self, x_rope, seq_len, device):
        pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        dim = torch.arange(self.rope_dim // 2, dtype=torch.float32, device=device).unsqueeze(0)
        inv_freq = 1.0 / (10000 **(2 * dim / self.rope_dim))
        sinusoid = pos @ inv_freq
        sin = torch.sin(sinusoid).repeat(1, 2).view(1, 1, seq_len, self.rope_dim)
        cos = torch.cos(sinusoid).repeat(1, 2).view(1, 1, seq_len, self.rope_dim)
        x_rotated = (x_rope * cos) + (self._rotate_half(x_rope) * sin)
        return x_rotated
    def _rotate_half(self, x):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    def forward(self, x, use_rope=True):
        B, S, _ = x.shape
        q_all = self.q_proj(x).view(B, S, self.n_heads, self.total_head_dim) 
        q_main, q_rope = q_all.split([self.head_dim, self.rope_dim], dim=-1) 
        compressed_kv = self.kv_norm(self.kv_proj_down(x))
        kv_unpacked = self.kv_up_proj(compressed_kv)
        k_all, v = kv_unpacked.split([self.n_heads * self.total_head_dim, self.n_heads * self.head_dim], dim=-1)
        k_all = k_all.view(B, S, self.n_heads, self.total_head_dim)
        k_main, k_rope = k_all.split([self.head_dim, self.rope_dim], dim=-1)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        q_main = q_main.transpose(1, 2)
        k_main = k_main.transpose(1, 2)
        scores = torch.matmul(q_main, k_main.transpose(-2, -1))
        if use_rope:
            q_rope = q_rope.transpose(1, 2)
            k_rope = k_rope.transpose(1, 2)
            
            q_rope = self._apply_rope(q_rope, S, x.device)
            k_rope = self._apply_rope(k_rope, S, x.device)
            
            scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))
            scores = (scores + scores_rope) / math.sqrt(self.total_head_dim)
        else:
            scores = scores / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(S, S, device=x.device), 1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, S, self.n_heads * self.head_dim) 
        return self.out_proj(context)
class HybridBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_type: str, use_moe: bool):
        super().__init__()
        d_model = config.d_model
        self.layer_type = layer_type
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        if layer_type == "mamba":
            self.mixer = MambaMixer(config)
        elif layer_type == "mla":
            self.mixer = MLA(config)
        elif layer_type == "attn":
            self.mixer = CasualAttention(config)
        else: 
            raise ValueError (f"Unknown type of mixer: {layer_type}")
        self.ffn = MoEFF(config) if use_moe else DenseFF(config)
    def forward(self, x, use_rope: bool = True):
        if self.layer_type == "mla":
            mixer_out = self.mixer(self.norm1(x), use_rope=use_rope)
        else:
            mixer_out = self.mixer(self.norm1(x))
        x = x + mixer_out
        x = x + self.ffn(self.norm2(x))
        return x
