from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from dataset import save_emb
import random

class RotaryPositionEmbedding(torch.nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Pre-compute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Pre-compute cos and sin cache
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len):
        """Pre-compute cache for cos and sin values"""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)
    
    def forward(self, q, k, seq_len=None, position_ids=None):
        """Apply rotary position encoding"""
        if seq_len is None:
            seq_len = q.shape[2]
        seq_len = min(seq_len, self.max_position_embeddings)
        if seq_len > getattr(self, "max_seq_len_cached", 0):
            self._set_cos_sin_cache(seq_len)

        if position_ids is None:
            pos = torch.arange(seq_len, device=q.device).unsqueeze(0).expand(q.size(0), -1)
        else:
            pos = position_ids.clamp_(0, self.max_seq_len_cached - 1).to(q.device)

        # Fix: use reshape instead of view
        cos = self.cos_cached.index_select(0, pos.reshape(-1)).reshape(q.size(0), seq_len, -1).to(q.dtype)
        sin = self.sin_cached.index_select(0, pos.reshape(-1)).reshape(q.size(0), seq_len, -1).to(q.dtype)

        cos = cos.unsqueeze(1).expand(-1, q.size(1), -1, -1)
        sin = sin.unsqueeze(1).expand(-1, q.size(1), -1, -1)

        q_rot = self.apply_rotary_pos_emb(q, cos, sin)
        k_rot = self.apply_rotary_pos_emb(k, cos, sin)
        return q_rot, k_rot
    
    @staticmethod
    def apply_rotary_pos_emb(x, cos, sin):
        """Apply rotary position embedding"""
        device, dtype = x.device, x.dtype
        B, H, L, D = x.shape
        assert D % 2 == 0, "Head dimension must be even for RoPE"

        if cos.dim() == 2:
            cos = cos[:L].to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
            sin = sin[:L].to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        else:
            assert cos.shape[-2] == L and sin.shape[-2] == L, "RoPE cos/sin seq_len mismatch"
            cos = cos.to(device=device, dtype=dtype)
            sin = sin.to(device=device, dtype=dtype)

        x_even = x[..., ::2]
        x_odd  = x[..., 1::2]

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos

        x_out = torch.empty_like(x)
        x_out[..., ::2] = x_rot_even
        x_out[..., 1::2] = x_rot_odd
        return x_out

def print_memory_usage(stage_name, device):
    """Print memory usage"""
    if torch.cuda.is_available() and device == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Memory {stage_name} - VRAM: {allocated:.2f}GB (allocated) / {reserved:.2f}GB (reserved) / {max_allocated:.2f}GB (peak)")


class HSTUBlock(torch.nn.Module):
    """
    Simplified HSTU Block, based on the implementation in dlrm_hstu.py
    Implements the formula:
    - U(X), V(X), Q(X), K(X) = Split(φ₁(f₁(X)))
    - A(X)V(X) = φ₂(Q(X)K(X)ᵀ)V(X)  
    - Y(X) = f₂(Norm(A(X)V(X))) ⊙ U(X)
    where both φ₁ and φ₂ use SiLU activation function
    """
    
    def __init__(self, hidden_units, num_heads, dropout_rate, hidden_dim=None, attn_dim=None, use_rope=False, rope_base=10000, max_position_embeddings=2048):
        super(HSTUBlock, self).__init__()
        
        self.hidden_units = hidden_units      # embedding_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_rope = use_rope
        
        # If not specified, use default values: hidden_dim * num_heads = hidden_units
        if hidden_dim is None:
            self.hidden_dim = hidden_units // num_heads
        else:
            self.hidden_dim = hidden_dim
            
        if attn_dim is None:
            self.attn_dim = hidden_units // num_heads
        else:
            self.attn_dim = attn_dim
        
        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"
        assert self.hidden_dim * num_heads == hidden_units, "hidden_dim * num_heads must equal hidden_units"

        if self.use_rope:
            self.rope = RotaryPositionEmbedding(
                dim=self.attn_dim,
                max_position_embeddings=max_position_embeddings,
                base=rope_base)
                
        # Input normalization layer (increase eps)
        self.input_norm = torch.nn.LayerNorm(hidden_units, eps=1e-5)
        
        # Linear transformation to compute U, V, Q, K
        # According to original HSTU code: (hidden_dim * 2 + attention_dim * 2) * num_heads
        # U: hidden_units -> hidden_dim * num_heads = hidden_units
        # V: hidden_units -> hidden_dim * num_heads = hidden_units  
        # Q: hidden_units -> attn_dim * num_heads
        # K: hidden_units -> attn_dim * num_heads
        total_dim = (self.hidden_dim * 2 + self.attn_dim * 2) * self.num_heads
        self.uvqk_linear = torch.nn.Linear(hidden_units, total_dim)
        
        # Output projection layer - project from concatenated dimension back to hidden_units
        # Concatenated dimension: u + x + normed_attn = hidden_units + hidden_units + hidden_units = 3 * hidden_units
        self.output_proj = torch.nn.Linear(3 * hidden_units, hidden_units)
        
        # Output normalization layer (increase eps)
        self.output_norm = torch.nn.LayerNorm(self.hidden_dim * self.num_heads, eps=1e-5)
        
        # Initialize weights - use more stable initialization
        torch.nn.init.xavier_uniform_(self.uvqk_linear.weight, gain=0.1)
        torch.nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
        torch.nn.init.zeros_(self.uvqk_linear.bias)
        torch.nn.init.zeros_(self.output_proj.bias)

        
        
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: [batch_size, seq_len, hidden_units]
            attention_mask: [batch_size, seq_len] for padding mask
        Returns:
            output: [batch_size, seq_len, hidden_units]
        """
        batch_size, seq_len, _ = x.shape
        
        # Numerical stability check
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: Input contains NaN or Inf values")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 1. Input normalization
        normed_x = self.input_norm(x)
        
        # 2. Compute U, V, Q, K
        uvqk = self.uvqk_linear(normed_x)  # [batch_size, seq_len, total_dim]
        
        # 3. Apply SiLU activation function (φ₁) - apply SiLU to entire uvqk
        uvqk = F.silu(uvqk)  # [batch_size, seq_len, total_dim]
        
        # Split U, V, Q, K
        u_dim = self.hidden_dim * self.num_heads      # hidden_dim * num_heads
        v_dim = self.hidden_dim * self.num_heads      # hidden_dim * num_heads
        q_dim = self.attn_dim * self.num_heads        # attn_dim * num_heads
        k_dim = self.attn_dim * self.num_heads        # attn_dim * num_heads
        
        u, v, q, k = torch.split(uvqk, [u_dim, v_dim, q_dim, k_dim], dim=-1)
        
        # 4. Reshape Q, K, V to multi-head format
        q = q.view(batch_size, seq_len, self.num_heads, self.attn_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.attn_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.hidden_dim).transpose(1, 2)

        if self.use_rope:
            try:
                q, k = self.rope(q, k, seq_len)
            except Exception as e:
                print(f"HSTU RoPE error at step: {e}")
                print(f"q shape: {q.shape}, k shape: {k.shape}, seq_len: {seq_len}")
                # If RoPE fails, fallback to non-RoPE mode
                print("Falling back to non-RoPE mode for this batch")
                pass
        
        # 5. Compute attention (A(X)V(X) = φ₂(Q(X)K(X)ᵀ)V(X)), perform main computation in FP32 for numerical stability
        scale = float(self.attn_dim) ** 0.5
        q32 = q.to(torch.float32); k32 = k.to(torch.float32); v32 = v.to(torch.float32)

        scores = torch.matmul(q32, k32.transpose(-2, -1)) / scale  # [B, H, L, L]
        
        # Apply mask first, then activate, to avoid non-linear operations on invalid positions
        if attention_mask is not None:
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(-1)  # [B,1,L,1]
            padding_mask = padding_mask & attention_mask.unsqueeze(1).unsqueeze(-2)  # [B,1,L,L]
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=0)
            valid_mask = padding_mask & causal_mask.unsqueeze(0).unsqueeze(0)        # [B,1,L,L]
            valid_mask_expanded = valid_mask.expand(-1, self.num_heads, -1, -1)      # [B,H,L,L]
            scores = scores.masked_fill(~valid_mask_expanded, -30.0)
        else:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=0)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)                       # [1,1,L,L]
            causal_mask_expanded = causal_mask.expand(batch_size, self.num_heads, -1, -1)
            scores = scores.masked_fill(~causal_mask_expanded, -30.0)

        # Clamp to avoid FP16/BF16 overflow
        scores = scores.clamp_(-30.0, 30.0)

        # φ₂: SiLU activation (still in fp32)
        attn_weights = F.silu(scores)

        # A(X)V(X)
        attn_output = torch.matmul(attn_weights, v32)  # [B,H,L,hidden_dim] in fp32
        
        # 6. Reshape attention output
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_heads, hidden_dim]
        attn_output = attn_output.view(batch_size, seq_len, -1)  # [batch_size, seq_len, num_heads * hidden_dim]
        attn_output = attn_output.to(x.dtype)
        
        # 7. Compute final output Y(X) = f₂(Norm(A(X)V(X)) ⊙ U(X))
        # Normalize attention output
        normed_attn = self.output_norm(attn_output)
        
        # Implement Norm(A(X)V(X)) ⊙ U(X)
        # normed_attn: [batch_size, seq_len, hidden_units]
        # u: [batch_size, seq_len, hidden_units]
        # Element-wise multiplication (⊙)
        normed_attn_mul_u = normed_attn * u
        
        # According to original HSTU code, concatenate u, attn, normed_attn_mul_u
        # u: [batch_size, seq_len, hidden_units]
        # attn_output: [batch_size, seq_len, hidden_units] (attention output)
        # normed_attn_mul_u: [batch_size, seq_len, hidden_units] (u * LayerNorm(attn))
        concatenated = torch.cat([u, attn_output, normed_attn_mul_u], dim=-1)  # [batch_size, seq_len, 3 * hidden_units]
        
        # Apply dropout after concatenation, following original HSTU code exactly
        concatenated = F.dropout(concatenated, p=self.dropout_rate, training=self.training)
        
        # Step-by-step implementation for clarity
        # 1. Linear transformation: directly use Linear layer
        linear_output = self.output_proj(concatenated)
        
        # 2. Residual connection: x + linear_output
        output = x + linear_output
        
        # Final numerical check
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"Warning: Output contains NaN or Inf values, replacing with input")
            output = x
        
        return output


class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(FlashMultiHeadAttention, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()
    
        # Compute Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
    
        # Reshape to multi-head format
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ use built-in Flash Attention
            # Handle attention mask dimensions
            if attn_mask is not None:
                # attn_mask shape: [batch_size, seq_len] 
                # Need to expand to [batch_size, num_heads, seq_len, seq_len]
                # First create causal mask
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool))
                # Combine with padding mask
                # attn_mask: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
                padding_mask = attn_mask.unsqueeze(1).unsqueeze(1)
                # Broadcast to [batch_size, 1, seq_len, seq_len]
                padding_mask = padding_mask & attn_mask.unsqueeze(1).unsqueeze(-2)
                # Combine with causal mask
                combined_mask = padding_mask & causal_mask.unsqueeze(0).unsqueeze(0)
                # Expand to all heads [batch_size, num_heads, seq_len, seq_len]
                mask_for_attention = combined_mask.expand(-1, self.num_heads, -1, -1)
            else:
                # Only use causal mask
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool))
                mask_for_attention = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
            
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, attn_mask=mask_for_attention
            )
        else:
            # Fallback to standard attention mechanism
            scale = (self.head_dim) ** -0.5
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
            # Apply causal mask
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool))
            scores.masked_fill_(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
            if attn_mask is not None:
                # Apply padding mask
                # attn_mask: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len] 
                padding_mask = attn_mask.unsqueeze(1).unsqueeze(1)
                # Broadcast to [batch_size, 1, seq_len, seq_len]
                padding_mask = padding_mask & attn_mask.unsqueeze(1).unsqueeze(-2)
                # Expand to all heads
                padding_mask = padding_mask.expand(-1, self.num_heads, -1, -1)
                scores.masked_fill_(~padding_mask, float('-inf'))
    
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
            attn_output = torch.matmul(attn_weights, V)
    
        # Reshape back to original format
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)
    
        # Final linear transformation
        output = self.out_linear(attn_output)
    
        return output, None


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate, h):
        super(PointWiseFeedForward, self).__init__()
        self.h = h
        self.conv1 = torch.nn.Conv1d(hidden_units, 2*h, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.conv2 = torch.nn.Conv1d(h, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = inputs.transpose(-1, -2)  # Conv1D expects (N, C, Length)
        outputs = self.dropout1(self.conv1(outputs))
        a, b = torch.split(outputs, self.h, dim=1)
        outputs = a * F.silu(b)
        outputs = self.dropout2(self.conv2(outputs))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        return outputs


class BaselineModel(torch.nn.Module):
    """
    Args:
        user_num: Number of users
        item_num: Number of items
        feat_statistics: Feature statistics, key is feature ID, value is feature count
        feat_types: Feature types for each feature, key is feature type name, value is list of included feature IDs
        args: Global parameters
    """

    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):
        super(BaselineModel, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen
        self.use_hstu = getattr(args, 'use_hstu', False)  # Add HSTU flag
        self.use_rope = getattr(args, 'use_rope', False)
        
        # For parameter count statistics
        self.embedding_params = 0
        self.hstu_params = 0

        # Use the same compression strategy as the first model: uniformly use 1/4 dimension
        embedding_dim = args.hidden_units // 4  # Use 1/4 of the dimension
        
        self.item_emb = torch.nn.Embedding(self.item_num + 1, embedding_dim, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, embedding_dim, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, embedding_dim, padding_idx=0)
        
        # Add projection layers to expand embeddings to hidden_units
        self.item_proj = torch.nn.Linear(embedding_dim, args.hidden_units)
        self.user_proj = torch.nn.Linear(embedding_dim, args.hidden_units)
        self.pos_proj = torch.nn.Linear(embedding_dim, args.hidden_units)
        
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()
        self.continual_transform = torch.nn.ModuleDict()

        # Choose architecture based on use_hstu parameter
        if self.use_hstu:
            # Use HSTU Block
            self.hstu_blocks = torch.nn.ModuleList()
            for _ in range(args.num_blocks):
                hstu_block = HSTUBlock(
                    hidden_units=args.hidden_units,
                    num_heads=args.num_heads,
                    dropout_rate=args.dropout_rate,
                    hidden_dim=getattr(args, 'hstu_hidden_dim', None),  # Use new parameter or default value
                    attn_dim=getattr(args, 'hstu_attn_dim', None),       # Use new parameter or default value
                    use_rope=self.use_rope,                             # New addition
                    max_position_embeddings=args.maxlen * 2,            # New addition, consistent with rope version
                    rope_base=getattr(args, 'rope_base', 10000)
                )
                self.hstu_blocks.append(hstu_block)
            print(f"[INFO] num_blocks: {len(self.hstu_blocks)}")
        else:
            # Use original attention layers and FFN layers
            self.attention_layernorms = torch.nn.ModuleList()
            self.attention_layers = torch.nn.ModuleList()
            self.forward_layernorms = torch.nn.ModuleList()
            self.forward_layers = torch.nn.ModuleList()
            self.middle_layernorms = torch.nn.ModuleList()
            
            for _ in range(args.num_blocks):
                new_attn_layernorm = torch.nn.RMSNorm(args.hidden_units, eps=1e-8)
                self.attention_layernorms.append(new_attn_layernorm)

                new_attn_layer = FlashMultiHeadAttention(
                    args.hidden_units, args.num_heads, args.dropout_rate
                )
                self.attention_layers.append(new_attn_layer)
                middle_layernorm = torch.nn.RMSNorm(args.hidden_units, eps=1e-8)
                self.middle_layernorms.append(middle_layernorm)

                new_fwd_layernorm = torch.nn.RMSNorm(args.hidden_units, eps=1e-8)
                self.forward_layernorms.append(new_fwd_layernorm)

                new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate, args.hidden_units * 3)
                self.forward_layers.append(new_fwd_layer)

        self._init_feat_info(feat_statistics, feat_types)

        # Calculate dimensions for various feature types - use concat approach
        for k in feat_types['user_continual']:
            self.continual_transform[k] = torch.nn.Linear(1, args.hidden_units)
        for k in feat_types['item_continual']:
            self.continual_transform[k] = torch.nn.Linear(1, args.hidden_units)
        
        # Use concat feature fusion instead of SENet fusion
        userdim = args.hidden_units * (len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT)) + args.hidden_units * len(self.USER_CONTINUAL_FEAT)
        itemdim = (
            args.hidden_units * (len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT))
            + args.hidden_units * len(self.ITEM_CONTINUAL_FEAT)
            + args.hidden_units * len(self.ITEM_EMB_FEAT)
        )

        self.userdnn = torch.nn.Linear(userdim, args.hidden_units)
        self.itemdnn = torch.nn.Linear(itemdim, args.hidden_units)

        self.last_layernorm = torch.nn.RMSNorm(args.hidden_units, eps=1e-8)

        # Sparse feature embeddings - maintain original dimensions (no compression)
        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
            
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.hidden_units)
        
        # ===== User-only DropoutNet (cold-start) =====
        self.use_dropoutnet = bool(getattr(args, 'use_dropoutnet', False))
        self.user_dropout_rate = float(getattr(args, 'user_dropout_rate', 0.0))
        self.entity_level_dropout = bool(getattr(args, 'entity_level_dropout', True))  # Entity-level priority
        self.dropout_curriculum = bool(getattr(args, 'dropout_curriculum', False))

        # Consistent with original dimension of user_emb (embedding_dim = hidden_units // 4)
        embedding_dim = args.hidden_units // 4
        self.default_user_emb = torch.nn.Parameter(torch.randn(1, 1, embedding_dim) * 0.01)

        # (Optional) Curriculum: linearly ramp up in first half of training, then constant
        self._drop_epoch, self._drop_total = 0, 1
        def set_dropout_progress(ep:int, tot:int):
            self._drop_epoch = max(0, int(ep)); self._drop_total = max(1, int(tot))
        self.set_dropout_progress = set_dropout_progress

        def _eff_rate(base: float) -> float:
            if not self.dropout_curriculum or base <= 0.0: return base
            ramp = min(1.0, (self._drop_epoch / self._drop_total) / 0.5)
            return float(base) * ramp
        self._eff_rate = _eff_rate
        
        # Count and print parameters
        self._count_and_print_parameters()

    def _init_feat_info(self, feat_statistics, feat_types):
        """Initialize feature information"""
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        # EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        # self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}
        self.ITEM_EMB_FEAT = {k: int(feat_statistics[k]) for k in feat_types['item_emb']}

    def feat2emb(self, seq, feature_array, mask=None, include_user=False):
        """
        Enhanced feat2emb with concat-based feature fusion (instead of SENet)
        """
        seq = seq.to(self.dev)
        
        # Check if it's already processed tensor dictionary
        if isinstance(feature_array, dict):
            processed_features = feature_array
        else:
            processed_features = self._process_features_to_tensors_in_model(feature_array)
        
        # Collect user features
        user_feat_list = []
        if include_user:
            user_mask = (mask == 2).to(self.dev)
            raw_user_embedding = self.user_emb(user_mask * seq)

            # ===== User-only DropoutNet: BEFORE projection =====
            if self.use_dropoutnet and self.training and self.user_dropout_rate > 0.0:
                rate = self._eff_rate(self.user_dropout_rate)
                if rate > 0.0:
                    B, L, E = raw_user_embedding.shape
                    # Entity-level (whole user) or position-level (choose one, default entity-level)
                    if self.entity_level_dropout:
                        # Each batch sample one Bernoulli trial: replace entire sequence with default user vector
                        drop_flag = (torch.rand(B, 1, 1, device=raw_user_embedding.device) < rate)
                        default_u = self.default_user_emb.expand(B, L, E)
                        raw_user_embedding = torch.where(drop_flag, default_u, raw_user_embedding)
                    else:
                        # Position-level: position-wise Bernoulli trial
                        pos_mask = (torch.rand(B, L, 1, device=raw_user_embedding.device) < rate)
                        default_u = self.default_user_emb.expand(B, L, E)
                        raw_user_embedding = torch.where(pos_mask, default_u, raw_user_embedding)

            user_embedding = self.user_proj(raw_user_embedding)  # ← Projection after dropout
            user_feat_list.append(user_embedding)
            
            for k in self.USER_SPARSE_FEAT:
                if k in processed_features:
                    tensor_feature = processed_features[k].to(self.dev)
                    user_feat_list.append(self.sparse_emb[k](tensor_feature))
                    
            for k in self.USER_ARRAY_FEAT:
                if k in processed_features:
                    tensor_feature = processed_features[k].to(self.dev)
                    user_feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                    
            # Fix: continuous features need to be transformed to hidden_units dimension
            for k in self.USER_CONTINUAL_FEAT:
                if k in processed_features:
                    tensor_feature = processed_features[k].to(self.dev)
                    # Expand dimension and transform through linear layer
                    tensor_feature = tensor_feature.unsqueeze(-1)  # [batch, seq_len, 1]
                    transformed_feature = self.continual_transform[k](tensor_feature)  # [batch, seq_len, hidden_units]
                    user_feat_list.append(transformed_feature)
        
        # Collect item features
        item_feat_list = []
        if include_user:
            item_mask = (mask == 1).to(self.dev)
            item_embedding = self.item_emb(item_mask * seq)
        else:
            item_embedding = self.item_emb(seq)
        item_embedding = self.item_proj(item_embedding)  # Project to hidden_units
        item_feat_list.append(item_embedding)
        
        for k in self.ITEM_SPARSE_FEAT:
            if k in processed_features:
                tensor_feature = processed_features[k].to(self.dev)
                item_feat_list.append(self.sparse_emb[k](tensor_feature))
                
        for k in self.ITEM_ARRAY_FEAT:
            if k in processed_features:
                tensor_feature = processed_features[k].to(self.dev)
                item_feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                
        # Fix: continuous features need to be transformed to hidden_units dimension
        for k in self.ITEM_CONTINUAL_FEAT:
            if k in processed_features:
                tensor_feature = processed_features[k].to(self.dev)
                # Expand dimension and transform through linear layer
                tensor_feature = tensor_feature.unsqueeze(-1)  # [batch, seq_len, 1]
                transformed_feature = self.continual_transform[k](tensor_feature)  # [batch, seq_len, hidden_units]
                item_feat_list.append(transformed_feature)
                
        for k in self.ITEM_EMB_FEAT:
            if k in processed_features:
                tensor_feature = processed_features[k].to(self.dev)
                item_feat_list.append(self.emb_transform[k](tensor_feature))
        
        # Use concat feature fusion (replace SENet)
        # Merge features
        all_item_emb = torch.cat(item_feat_list, dim=2)
        all_item_emb = F.silu(self.itemdnn(all_item_emb))
        
        if include_user and len(user_feat_list) > 0:
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = F.silu(self.userdnn(all_user_emb))
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb
            
        return seqs_emb

    def _process_features_to_tensors_in_model(self, feature_array):
        """Process features to tensors in model (backward compatibility)"""
        processed_features = {}
        
        # Helper function: convert features to tensor
        def feat2tensor(seq_feature, k, default_value, is_array=False, is_continual=False):
            batch_size = len(seq_feature)
            
            if is_array:
                # Array feature processing
                max_array_len = 0
                max_seq_len = 0
                
                # First calculate maximum dimensions
                for i in range(batch_size):
                    seq_data = [item[k] if k in item else default_value 
                               for item in seq_feature[i]]
                    max_seq_len = max(max_seq_len, len(seq_data))
                    for item_data in seq_data:
                        if isinstance(item_data, list):
                            max_array_len = max(max_array_len, len(item_data))
                
                # Initialize batch_data - this is the key fix!
                batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
                
                # Fill data
                for i in range(batch_size):
                    seq_data = [item[k] if k in item else default_value 
                               for item in seq_feature[i]]
                    for j, item_data in enumerate(seq_data):
                        if isinstance(item_data, list):
                            actual_len = min(len(item_data), max_array_len)
                            batch_data[i, j, :actual_len] = item_data[:actual_len]
                
                return torch.from_numpy(batch_data)
            
            else:
                # Non-array feature processing
                max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
                
                # Choose data type based on feature type
                if is_continual:
                    dtype = np.float32
                else:
                    dtype = np.int64
                    
                batch_data = np.zeros((batch_size, max_seq_len), dtype=dtype)
                
                for i in range(batch_size):
                    seq_data = [item[k] if k in item else default_value 
                               for item in seq_feature[i]]
                    batch_data[i, :len(seq_data)] = seq_data
                
                return torch.from_numpy(batch_data)
        
        # Process various features
        for k in self.USER_SPARSE_FEAT:
            processed_features[k] = feat2tensor(feature_array, k, 0)
        for k in self.ITEM_SPARSE_FEAT:
            processed_features[k] = feat2tensor(feature_array, k, 0)
        for k in self.USER_ARRAY_FEAT:
            processed_features[k] = feat2tensor(feature_array, k, [0], is_array=True)
        for k in self.ITEM_ARRAY_FEAT:
            processed_features[k] = feat2tensor(feature_array, k, [0], is_array=True)
        
        # Continuous features need float type
        for k in self.USER_CONTINUAL_FEAT:
            processed_features[k] = feat2tensor(feature_array, k, 0.0, is_continual=True)
        for k in self.ITEM_CONTINUAL_FEAT:
            processed_features[k] = feat2tensor(feature_array, k, 0.0, is_continual=True)
            
        # Process emb features
        # EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        for k in self.ITEM_EMB_FEAT:
            batch_size = len(feature_array)
            # emb_dim = EMB_SHAPE_DICT[k]
            emb_dim = int(self.ITEM_EMB_FEAT[k])
            seq_len = len(feature_array[0])
            
            batch_emb_data = np.zeros((batch_size, seq_len, emb_dim), dtype=np.float32)
            
            for i, seq in enumerate(feature_array):
                for j, item in enumerate(seq):
                    if k in item:
                        batch_emb_data[i, j] = item[k]
                        
            processed_features[k] = torch.from_numpy(batch_emb_data)
            
        return processed_features

    def _count_and_print_parameters(self):
        """Count and print parameter count for embedding layers and HSTU model"""
        # Count embedding layer parameters
        embedding_params = 0
        embedding_components = []
        
        # Basic embedding layers (compressed)
        embedding_params += self.item_emb.weight.numel()
        embedding_components.append(f"item_emb: {self.item_emb.weight.numel():,}")
        
        embedding_params += self.user_emb.weight.numel()
        embedding_components.append(f"user_emb: {self.user_emb.weight.numel():,}")
        
        embedding_params += self.pos_emb.weight.numel()
        embedding_components.append(f"pos_emb: {self.pos_emb.weight.numel():,}")
        
        # Projection layers
        embedding_params += sum(p.numel() for p in self.item_proj.parameters())
        embedding_components.append(f"item_proj: {sum(p.numel() for p in self.item_proj.parameters()):,}")
        
        embedding_params += sum(p.numel() for p in self.user_proj.parameters())
        embedding_components.append(f"user_proj: {sum(p.numel() for p in self.user_proj.parameters()):,}")
        
        embedding_params += sum(p.numel() for p in self.pos_proj.parameters())
        embedding_components.append(f"pos_proj: {sum(p.numel() for p in self.pos_proj.parameters()):,}")
        
        # Sparse feature embeddings
        for k, emb in self.sparse_emb.items():
            param_count = emb.weight.numel()
            embedding_params += param_count
            embedding_components.append(f"sparse_emb[{k}]: {param_count:,}")
        
        # Continuous feature transformation layers
        for k, linear in self.continual_transform.items():
            param_count = sum(p.numel() for p in linear.parameters())
            embedding_params += param_count
            embedding_components.append(f"continual_transform[{k}]: {param_count:,}")
        
        # Feature fusion layers
        userdnn_params = sum(p.numel() for p in self.userdnn.parameters())
        itemdnn_params = sum(p.numel() for p in self.itemdnn.parameters())
        embedding_params += userdnn_params + itemdnn_params
        embedding_components.append(f"userdnn: {userdnn_params:,}")
        embedding_components.append(f"itemdnn: {itemdnn_params:,}")
        
        # Pre-trained embedding transformation layers
        for k, linear in self.emb_transform.items():
            param_count = sum(p.numel() for p in linear.parameters())
            embedding_params += param_count
            embedding_components.append(f"emb_transform[{k}]: {param_count:,}")
        
        # Count HSTU model parameters
        hstu_params = 0
        hstu_components = []
        
        if self.use_hstu:
            for i, hstu_block in enumerate(self.hstu_blocks):
                param_count = sum(p.numel() for p in hstu_block.parameters())
                hstu_params += param_count
                hstu_components.append(f"HSTU_block_{i}: {param_count:,}")
        else:
            # Original attention mechanism parameters
            for i, (attn_layer, fwd_layer) in enumerate(zip(self.attention_layers, self.forward_layers)):
                attn_params = sum(p.numel() for p in attn_layer.parameters())
                fwd_params = sum(p.numel() for p in fwd_layer.parameters())
                hstu_params += attn_params + fwd_params
                hstu_components.append(f"attention_layer_{i}: {attn_params:,}")
                hstu_components.append(f"forward_layer_{i}: {fwd_params:,}")
            
            # Normalization layer parameters
            for i, (attn_norm, fwd_norm, middle_norm) in enumerate(zip(self.attention_layernorms, self.forward_layernorms, self.middle_layernorms)):
                attn_norm_params = sum(p.numel() for p in attn_norm.parameters())
                fwd_norm_params = sum(p.numel() for p in fwd_norm.parameters())
                middle_norm_params = sum(p.numel() for p in middle_norm.parameters())
                hstu_params += attn_norm_params + fwd_norm_params + middle_norm_params
                hstu_components.append(f"attention_norm_{i}: {attn_norm_params:,}")
                hstu_components.append(f"forward_norm_{i}: {fwd_norm_params:,}")
                hstu_components.append(f"middle_norm_{i}: {middle_norm_params:,}")
        
        # Final normalization layer
        last_norm_params = sum(p.numel() for p in self.last_layernorm.parameters())
        hstu_params += last_norm_params
        hstu_components.append(f"last_layernorm: {last_norm_params:,}")
        
        # Save parameter counts
        self.embedding_params = embedding_params
        self.hstu_params = hstu_params
        
        # Calculate memory usage (assuming float32, 4 bytes per parameter)
        embedding_memory_mb = embedding_params * 4 / (1024 * 1024)
        hstu_memory_mb = hstu_params * 4 / (1024 * 1024)
        total_memory_mb = (embedding_params + hstu_params) * 4 / (1024 * 1024)
        
        # Print results
        print("=" * 80)
        print("Model Parameter Count Statistics")
        print("=" * 80)
        print(f"Embedding Layer Parameters: {embedding_params:,} ({embedding_memory_mb:.2f} MB)")
        print("Embedding Layer Component Details:")
        for component in embedding_components:
            print(f"  - {component}")
        
        print(f"\nHSTU Model Parameters: {hstu_params:,} ({hstu_memory_mb:.2f} MB)")
        print("HSTU Model Component Details:")
        for component in hstu_components:
            print(f"  - {component}")
        
        print(f"\nTotal Parameters: {embedding_params + hstu_params:,} ({total_memory_mb:.2f} MB)")
        print(f"Parameter Ratio - Embedding: {embedding_params/(embedding_params + hstu_params)*100:.1f}%, HSTU: {hstu_params/(embedding_params + hstu_params)*100:.1f}%")
        print("=" * 80)

    def log2feats(self, log_seqs, mask, seq_feature):
        """Calculate sequence representations"""
        batch_size = log_seqs.shape[0]
        maxlen = log_seqs.shape[1]
        seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True)
        # Remove scaling factor to avoid numerical instability
        # seqs *= self.item_emb.embedding_dim**0.5

        if not self.use_rope:
            poss = torch.arange(1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).clone()
            poss *= log_seqs != 0
            pos_emb = self.pos_emb(poss)
            pos_emb = self.pos_proj(pos_emb)  # Project to hidden_units
            seqs += pos_emb
        
        # Apply dropout after adding position encoding, before entering Transformer
        seqs = self.emb_dropout(seqs)
        
        maxlen = seqs.shape[1]
        # Create attention mask for padding - only need padding mask, causal mask is handled inside HSTU
        attention_mask = (mask != 0).to(self.dev)  # [batch_size, seq_len]
        # print(f"[INFO] seqs: {seqs[0]}")
        # print(f"[INFO] attention_mask: {attention_mask[0]}")

        # Use HSTU Block to replace original attention layers and FFN layers
        if self.use_hstu:
            # More aggressive gradient checkpointing strategy: all layers use gradient checkpointing
            for hstu_block in self.hstu_blocks:
                if self.norm_first:
                    # Pre-norm: normalize first, then compute
                    # All layers use gradient checkpointing for maximum memory savings
                    seqs = checkpoint(hstu_block, seqs, attention_mask, use_reentrant=False)
                else:
                    # Post-norm: compute first, then normalize
                    # All layers use gradient checkpointing for maximum memory savings
                    seqs = checkpoint(hstu_block, seqs, attention_mask, use_reentrant=False)
        else:
            # Use original attention layers and FFN layers
            for i in range(len(self.attention_layers)):
                if self.norm_first:
                    x = self.attention_layernorms[i](seqs)
                    mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                    seqs = seqs + mha_outputs
                    seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
                else:
                    mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                    seqs = self.attention_layernorms[i](seqs + mha_outputs)
                    seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type, 
                seq_feature, pos_feature, neg_feature):
        """Called during training, compute logits for positive and negative samples"""
        log_feats = self.log2feats(user_item, mask, seq_feature)
        loss_mask = (next_mask == 1).to(self.dev)

        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        pos_logits = pos_logits * loss_mask
        neg_logits = neg_logits * loss_mask

        return log_feats, pos_embs, neg_embs, loss_mask, next_action_type

    def predict(self, log_seqs, seq_feature, mask):
        """Calculate user sequence representations"""
        log_feats = self.log2feats(log_seqs, mask, seq_feature)
        final_feat = log_feats[:, -1, :]
        final_feat = F.normalize(final_feat, p=2, dim=-1, eps=1e-6)
        return final_feat

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        """Generate candidate pool item embeddings for retrieval"""
        all_embs = []
        
        print(f"Starting save_item_emb - Total items: {len(item_ids)}, batch_size: {batch_size}")
        print(f"Estimated batches needed: {len(item_ids) // batch_size + 1}")

        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))
            
            print(f"Processing batch {start_idx//batch_size + 1}: items {start_idx}-{end_idx-1}")
            print_memory_usage(f"Batch {start_idx//batch_size + 1} start", 'cuda')

            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)
            batch_feat = []
            for i in range(start_idx, end_idx):
                batch_feat.append(feat_dict[i])

            batch_feat = np.array(batch_feat, dtype=object)
            print_memory_usage(f"Batch {start_idx//batch_size + 1} data prepared", 'cuda')

            batch_emb = self.feat2emb(item_seq, [batch_feat], include_user=False).squeeze(0)
            print_memory_usage(f"Batch {start_idx//batch_size + 1} feat2emb done", 'cuda')
            
            batch_emb = self.last_layernorm(batch_emb)
            batch_emb = F.normalize(batch_emb, p=2, dim=-1, eps=1e-6)
            print_memory_usage(f"Batch {start_idx//batch_size + 1} normalize done", 'cuda')

            batch_emb_cpu = batch_emb.detach().cpu().numpy().astype(np.float32)
            all_embs.append(batch_emb_cpu)
            print_memory_usage(f"Batch {start_idx//batch_size + 1} to CPU done", 'cuda')
            
            # Clean up memory
            del batch_emb, batch_emb_cpu, item_seq, batch_feat
            torch.cuda.empty_cache()
            print_memory_usage(f"Batch {start_idx//batch_size + 1} cleanup done", 'cuda')

        print("Starting to merge all batch results...")
        print_memory_usage("Before merging", 'cuda')
        
        # Merge all batch results and save
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        print_memory_usage("final_ids created", 'cuda')
        
        final_embs = np.concatenate(all_embs, axis=0)
        print_memory_usage("final_embs merged", 'cuda')
        
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))
        print_memory_usage("Files saved", 'cuda')