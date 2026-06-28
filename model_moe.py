import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.nn.attention import sdpa_kernel, SDPBackend

class Expert(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    #public_forward = True
    def forward(self, x):
        #if self.training:
        #    return checkpoint_sequential(self.net, 1, x, use_reentrant=False)
        return self.net(x)

class Router(nn.Module):
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        logits = self.gate(x) # (batch_size, seq_len, num_experts)

        # Top-k gating
        probs = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(probs, self.top_k, dim=-1)

        # Re-normalize top-k weights
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Auxiliary Loss (Load Balancing)
        # Use bincount instead of histc for better performance on GPU
        #tokens_per_expert = torch.bincount(topk_indices.flatten(), minlength=self.num_experts).float() / (batch_size * seq_len * self.top_k)
        indices_one_hot = F.one_hot(topk_indices, num_classes=self.num_experts).float()
        tokens_per_expert = indices_one_hot.sum(dim=(0,1,2)) / (batch_size * seq_len * self.top_k)
        avg_prob_per_expert = probs.mean(dim=(0, 1))
        aux_loss = self.num_experts * torch.sum(tokens_per_expert * avg_prob_per_expert)

        return topk_indices, topk_weights, aux_loss
    

class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = Router(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(d_model, d_ff, dropout=dropout) for _ in range(num_experts)])

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        topk_indices, topk_weights, aux_loss = self.router(x)

        # Flatten batch and seq dimensions for processing
        flat_x = x.view(-1, d_model)
        flat_indices = topk_indices.view(-1, self.top_k)
        flat_weights = topk_weights.view(-1, self.top_k)

        combined_output = torch.zeros_like(flat_x)

        # Optimized gather/scatter processing
        for i, expert in enumerate(self.experts):
            # Find indices where this expert is selected
            token_idx, k_idx = torch.where(flat_indices == i)
            if token_idx.numel() > 0:
                # Gather inputs for the expert
                expert_input = flat_x[token_idx]
                # Compute expert output
                expert_output = expert(expert_input)
                # Scale by routing weights
                expert_output = expert_output * flat_weights[token_idx, k_idx].unsqueeze(-1)
                # Scatter add the results back to the combined output (index_add_ is faster and safer than +=)
                combined_output.index_add_(0, token_idx, expert_output)

        return combined_output.view(batch_size, seq_len, d_model), aux_loss


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Fused QKV projection to speed up compute
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, is_causal=True):
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V in a single matrix multiplication, then split
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Use PyTorch SDPA instead of manual attention implementation for better performance and memory efficiency
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH
        ], set_priority=True):
        
        #with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            context = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal
            )
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_linear(context)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_experts, top_k=2, dropout=0.1, use_moe=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.use_moe = use_moe
        if use_moe:
            self.moe = MoELayer(d_model, d_ff, num_experts, top_k, dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(d_ff, d_model)
            )

    def forward(self, x, is_causal=True):
        attn_out = self.attn(self.ln1(x), is_causal=is_causal)
        x = x + attn_out
        if self.use_moe:
            moe_out, aux_loss = self.moe(self.ln2(x))
            x = x + moe_out
            return x, aux_loss
        else:
            x = x + self.ffn(self.ln2(x))
            return x, x.new_zeros(1).squeeze() #torch.tensor(0.0, device=x.device)

class MoETransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, num_experts, max_seq_len=512, top_k=2, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, num_experts, top_k, dropout,
                                    use_moe=(i != 0 and i != num_layers - 1))  # dense first + last
                                    for i in range(num_layers)])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight Tying: share weights between embedding and output layer
        self.head.weight = self.token_emb.weight
        self.max_seq_len = max_seq_len

    def forward(self, idx):
        batch_size, seq_len = idx.shape
        positions = torch.arange(0, seq_len, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(positions)

        total_aux_loss = 0
        for block in self.blocks:
            x, aux_loss = block(x, is_causal=True)
            total_aux_loss += aux_loss

        x = self.ln_f(x)
        logits = self.head(x)
        #if targets is not None:
        #    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        #    loss = loss + 0.01 * total_aux_loss

        return logits, total_aux_loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def get_batch(split, accel_device):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    device_type = accel_device
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    x, y = x.pin_memory().to(accel_device, non_blocking=True), y.pin_memory().to(accel_device, non_blocking=True)
    
    return x, y

if __name__ == "__main__":
    config = dict()
    vocab_size = config.get("vocab_size", 50304)
    d_model = config.get("d_model", 384)
    num_heads = config.get("num_heads", 16)
    d_ff = config.get("d_ff", 1024)
    num_layers = config.get("num_layers", 16)
    num_experts = config.get("num_experts", 16)
    batch_size = config.get("batch_size", 4)
    block_size = config.get("block_size", 2048)
    data_dir = config.get("data_dir", "./")
    model = MoETransformer(vocab_size, d_model, num_heads, d_ff, num_layers, num_experts, max_seq_len=block_size, top_k=2, dropout=0.2)
    print(f"# trainable parameters: {sum([p.numel() for p in model.parameters()])}")
    #import numpy as np
    #import os
    #X, y = get_batch("val", "cpu")
    #print(f"Sample data shapes:\t{X.shape}, {y.shape} ")