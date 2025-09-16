import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
from model import BaselineModel
from model_ae import UnifiedModel


def evaluate_ranking_metrics(model, valid_loader, dataset, num_users=10000, num_negatives=999, k=10):
    """
    Calculate HR@K and NDCG@K
    
    Args:
        model: trained model
        valid_loader: validation data loader
        dataset: dataset instance
        num_users: number of users to evaluate
        num_negatives: number of negative samples per user
        k: top-k for metrics
    
    Returns:
        hr: Hit Rate@K
        ndcg: NDCG@K
    """
    model.eval()
    
    hr_list = []
    ndcg_list = []
    
    # Randomly select users to evaluate
    all_users = list(range(len(valid_loader.dataset)))
    random.shuffle(all_users)
    eval_users = all_users[:min(num_users, len(all_users))]
    
    with torch.no_grad():
        for user_idx in tqdm(eval_users, desc="Computing ranking metrics"):
            # Get user data
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = valid_loader.dataset[user_idx]
            
            # Find the last valid positive sample as ground truth
            valid_pos_indices = np.where(pos > 0)[0]
            if len(valid_pos_indices) == 0:
                continue
            
            last_pos_idx = valid_pos_indices[-1]
            ground_truth = pos[last_pos_idx]
            
            if ground_truth == 0:
                continue
            
            # Prepare user sequence (remove last item for prediction)
            eval_seq = seq.copy()
            eval_seq[last_pos_idx] = 0  # mask the position to predict
            
            # Convert to batch format
            eval_seq_batch = torch.from_numpy(eval_seq).unsqueeze(0).to(model.dev)
            token_type_batch = torch.from_numpy(token_type).unsqueeze(0).to(model.dev)
            
            # Process features - use correct format
            seq_feat_batch = [seq_feat]  # wrap as list
            seq_feat_tensors = dataset.process_features_to_tensors(seq_feat_batch)
            
            # Get user representation
            user_emb = model.predict(eval_seq_batch, seq_feat_tensors, token_type_batch)
            
            # Randomly sample negative items
            neg_items = []
            while len(neg_items) < num_negatives:
                neg_item = np.random.randint(1, dataset.itemnum + 1)
                if neg_item != ground_truth and str(neg_item) in dataset.item_feat_dict:
                    neg_items.append(neg_item)
            
            # All candidate items (1 positive + 999 negatives)
            candidate_items = [ground_truth] + neg_items
            candidate_ids = torch.tensor(candidate_items, device=model.dev).unsqueeze(0)
            
            # Get features for all candidate items
            candidate_feats = []
            for item_id in candidate_items:
                if str(item_id) in dataset.item_feat_dict:
                    feat = dataset.fill_missing_feat(dataset.item_feat_dict[str(item_id)], item_id)
                else:
                    feat = dataset.feature_default_value.copy()
                candidate_feats.append(feat)
            
            # Process candidate item features
            candidate_feat_tensors = dataset.process_features_to_tensors([candidate_feats])
            
            # Calculate candidate item embeddings
            item_embs = model.feat2emb(candidate_ids, candidate_feat_tensors, include_user=False)
            item_embs = model.last_layernorm(item_embs.squeeze(0))
            item_embs = F.normalize(item_embs, p=2, dim=-1)
            
            # Calculate scores
            scores = torch.matmul(user_emb, item_embs.T).squeeze(0)
            
            # Get top-k
            _, top_indices = torch.topk(scores, k=min(k, len(candidate_items)))
            top_indices = top_indices.cpu().numpy()
            
            # Calculate HR@K
            if 0 in top_indices:  # ground truth at position 0
                hr_list.append(1.0)
                
                # Calculate NDCG@K
                rank = np.where(top_indices == 0)[0][0] + 1  # rank (1-based)
                ndcg = 1.0 / np.log2(rank + 1)
                ndcg_list.append(ndcg)
            else:
                hr_list.append(0.0)
                ndcg_list.append(0.0)
    
    hr = np.mean(hr_list) if hr_list else 0.0
    ndcg = np.mean(ndcg_list) if ndcg_list else 0.0
    
    return hr, ndcg

class RowWiseAdamWOffload(Optimizer):
    """
    Row-wise AdamW with optimizer states on CPU (DRAM).
    - Only supports 2D parameters (Embedding: [num_rows, dim])
    - Maintains only two CPU scalar states per row: m_row, v_row
    - Parameters on GPU; only transfers per-row statistics to CPU for updates, then broadcasts back to GPU
    """
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.0,
                 pin_memory=True, cpu_dtype=torch.float32, max_row_norm=5.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        pin_memory=pin_memory, cpu_dtype=cpu_dtype, max_row_norm=max_row_norm)
        super().__init__(params, defaults)

        # Lazy initialization of CPU states for each param: state[p]['m'], state[p]['v'] shape=[num_rows]
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad and p.ndim == 2:
                    st = self.state[p]
                    st['step'] = 0
                    # Delay allocation until first step (need to know number of rows)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']; beta1, beta2 = group['betas']
            eps = group['eps']; wd = group['weight_decay']
            pin = group['pin_memory']; cpu_dtype = group['cpu_dtype']; max_row_norm = group['max_row_norm']

            for p in group['params']:
                if p.grad is None or p.ndim != 2:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    grad = grad.to_dense()

                # Per-row gradient norm clipping
                row_norm = grad.norm(p=2, dim=1, keepdim=True)         # [R,1]
                scale = (max_row_norm / (row_norm + 1e-6)).clamp(max=1.0)
                grad = grad * scale

                # Statistics (on GPU)
                g_mean = grad.mean(dim=1).nan_to_num_(0.0)             # [R]
                g2_mean = (grad*grad).mean(dim=1).nan_to_num_(0.0)     # [R]

                st = self.state[p]
                if 'm' not in st:
                    R = p.shape[0]
                    st['m'] = torch.zeros(R, dtype=cpu_dtype, device='cpu', pin_memory=pin)
                    st['v'] = torch.zeros(R, dtype=cpu_dtype, device='cpu', pin_memory=pin)
                    st['step'] = 0

                m_cpu = st['m']; v_cpu = st['v']
                st['step'] += 1; t = st['step']

                # decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Update states on CPU
                m_cpu.mul_(beta1).add_(g_mean.to('cpu', dtype=cpu_dtype), alpha=(1.0 - beta1))
                v_cpu.mul_(beta2).add_(g2_mean.to('cpu', dtype=cpu_dtype), alpha=(1.0 - beta2))

                # Bias correction
                m_hat = m_cpu / (1.0 - beta1**t)
                v_hat = v_cpu / (1.0 - beta2**t)

                # Per-row step size and broadcast
                denom = v_hat.sqrt().add_(eps)                         # [R]
                step_row = (lr * m_hat / denom).to(device=p.device, dtype=p.dtype).view(-1,1)
                p.add_(-step_row)

        return loss


def set_random_seed(seed=3407):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_embed', default=None, type=float,
                       help='Learning rate for embedding parameters (default: same as --lr)')
    parser.add_argument('--maxlen', default=101, type=int)
    
    # DataLoader params
    parser.add_argument('--num_workers', default=4, type=int, 
                       help='Number of data loading workers (default: 4, set to 0 for debugging)')

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_blocks', default=4, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--dropout_rate', default=0.00, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')

    # Enhanced negative sampling params
    parser.add_argument('--use_enhanced_negatives', action='store_true', 
                       help='Enable enhanced negative sampling with more global negatives')
    parser.add_argument('--num_extra_negatives', default=10000, type=int, 
                       help='Number of extra negative samples to add')

    parser.add_argument('--senet_reduction_ratio', default=16, type=int,help='SENet reduction ratio for feature fusion (default: 16)')

    # Action type weighting params
    parser.add_argument('--use_action_weights', action='store_true',
                       help='Enable action type weighting for InfoNCE loss')
    parser.add_argument('--click_weight', default=4.0, type=float,
                       help='Weight for click actions (default: 2.0)')
    parser.add_argument('--default_weight', default=1.0, type=float,
                       help='Weight for other actions (default: 1.0)')

    # Random seed
    parser.add_argument('--seed', default=3407, type=int,
                       help='Random seed for reproducibility (default: 3407)')

    # HSTU Block specific parameters
    parser.add_argument('--use_hstu', action='store_true', help='Enable HSTU Block in model')
    parser.add_argument('--hstu_hidden_dim', default=None, type=int, 
                       help='HSTU hidden dimension per head (default: hidden_units // num_heads)')
    parser.add_argument('--hstu_attn_dim', default=None, type=int,
                       help='HSTU attention dimension per head (default: hidden_units // num_heads)')
    parser.add_argument('--hstu_use_silu', action='store_true', default=True,
                       help='Use SiLU activation in HSTU (default: True)')
    parser.add_argument('--hstu_use_causal_mask', action='store_true', default=True,
                       help='Use causal mask in HSTU attention (default: True)')
    parser.add_argument('--hstu_use_padding_mask', action='store_true', default=True,
                       help='Use padding mask in HSTU attention (default: True)')

    # RoPE parameters
    parser.add_argument('--use_rope', action='store_true', default = True,help='Enable RoPE on HSTU Q/K')
    parser.add_argument('--rope_base', default=10000, type=int, help='RoPE base')

    parser.add_argument('--mlp_ratio', default=2, type=int,
                       help='MLP hidden dimension ratio (default: 2, means hidden_dim = hidden_units * 2)')
    parser.add_argument('--mlp_layers', default=2, type=int,
                       help='Number of MLP layers (default: 2)')
    parser.add_argument('--mlp_activation', default='gelu', type=str, choices=['gelu', 'relu', 'silu'],
                       help='MLP activation function (default: gelu)')
    parser.add_argument('--mlp_dropout', default=0.3, type=float,
                       help='MLP dropout rate (default: None, will use dropout_rate if not specified)')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    # Autoencoder params 
    parser.add_argument('--use_autoencoder', default=True, action='store_true', help='Enable Autoencoder for feature compression')
    parser.add_argument('--autoencoder_dim', default=32, type=int, help='Dimension of the Autoencoder bottleneck')
    parser.add_argument('--autoencoder_epochs', default=3, type=int, help='Number of epochs to train autoencoder')
    parser.add_argument('--autoencoder_batch_size', default=128, type=int, help='Batch size for autoencoder training')

    args = parser.parse_args()

    return args


def save_unified_model(model, dataset, save_path):
    """
    Save unified model state, including baseline model and autoencoder
    """
    unified_state_dict = {}
    
    # Save baseline model parameters
    for key, value in model.baseline_model.state_dict().items():
        unified_state_dict[f'baseline_model.{key}'] = value
    
    # Save autoencoder parameters
    if hasattr(dataset, 'autoencoder') and dataset.autoencoder is not None:
        for key, value in dataset.autoencoder.state_dict().items():
            unified_state_dict[f'autoencoder.{key}'] = value
        print("Autoencoder parameters added to save file")
    
    torch.save(unified_state_dict, save_path)
    print(f"Unified model saved to: {save_path}")


def load_unified_model(model_path, usernum, itemnum, feat_statistics, feat_types, args, dataset):
    """
    Load unified model, including baseline model and autoencoder
    """
    print(f"Loading unified model from {model_path}...")
    
    # Load state dictionary
    state_dict = torch.load(model_path, map_location=args.device)
    
    # Create baseline model
    baseline_model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args)
    
    # Create unified model
    unified_model = UnifiedModel(baseline_model, dataset.autoencoder if hasattr(dataset, 'autoencoder') else None)
    
    # Load baseline model parameters
    baseline_state_dict = {}
    autoencoder_state_dict = {}
    
    for key, value in state_dict.items():
        if key.startswith('baseline_model.'):
            # Remove 'baseline_model.' prefix
            baseline_key = key[len('baseline_model.'):]
            baseline_state_dict[baseline_key] = value
        elif key.startswith('autoencoder.'):
            # Collect autoencoder parameters, but don't load here (already handled in dataset)
            autoencoder_key = key[len('autoencoder.'):]
            autoencoder_state_dict[autoencoder_key] = value
        else:
            # Backward compatibility: if no prefix, assume baseline_model parameters
            baseline_state_dict[key] = value
    
    # Load baseline model parameters
    unified_model.baseline_model.load_state_dict(baseline_state_dict)
    
    # If autoencoder parameters exist and dataset has autoencoder, load autoencoder parameters
    if autoencoder_state_dict and hasattr(dataset, 'autoencoder') and dataset.autoencoder is not None:
        dataset.autoencoder.load_state_dict(autoencoder_state_dict)
        print("Autoencoder parameters loaded from model.pt")
    
    print(f"Unified model loading completed")
    return unified_model


def create_dataset_with_model_state(data_path, args, model_state_dict=None):
    """
    Create dataset with model state for autoencoder initialization
    """
    return MyDataset(data_path, args, model_state_dict=model_state_dict)


def info_loss_with_more_negatives(seq, pos, neg, loss_mask, model, dataset, num_extra_negatives=1000, action_weights=None):
    """
    InfoNCE loss function with additional global negative sampling and action type weighting
    
    Args:
        seq: sequence embedding
        pos: positive sample embedding  
        neg: negative sample embedding
        loss_mask: loss mask
        model: model instance for computing complete embeddings of additional negative samples
        dataset: dataset instance for getting item features
        num_extra_negatives: number of additional negative samples
        action_weights: action type weights, shape [B, L], for weighting positive samples
    """
    tau = 0.03
    
    # L2 normalize
    seq = F.normalize(seq, p=2, dim=-1, eps=1e-6)
    pos = F.normalize(pos, p=2, dim=-1, eps=1e-6)
    neg = F.normalize(neg, p=2, dim=-1, eps=1e-6)
    
    # Reduce invalid computations
    valid = loss_mask.bool().unsqueeze(-1).expand_as(seq)
    seq = seq[valid].view(-1, seq.size(-1))
    pos = pos[valid].view(-1, pos.size(-1))
    neg = neg[valid].view(-1, neg.size(-1))
    
    # Extract action weights for valid positions
    valid_weights = None
    if action_weights is not None:
        action_weights = action_weights.to(seq.device)
        valid_mask = loss_mask.bool()
        valid_weights = action_weights[valid_mask]  # [N_valid]
    
    # Positive sample similarity
    pos_logit = F.cosine_similarity(seq, pos, dim=-1).unsqueeze(-1)
    
    # Original batch negative samples
    neg_pool = neg.reshape(-1, neg.size(-1))
    
    # Additional random global negative sampling (full feature version)
    if num_extra_negatives > 0:
        # Randomly sample item IDs (excluding padding id=0)
        random_item_ids = torch.randint(1, dataset.itemnum + 1, (num_extra_negatives,), device=seq.device)
        
        # Construct complete item features
        batch_feat = []
        valid_ids = []
        for item_id in random_item_ids:
            item_id_int = item_id.item()
            if str(item_id_int) in dataset.item_feat_dict:
                feat = dataset.fill_missing_feat(dataset.item_feat_dict[str(item_id_int)], item_id_int)
                batch_feat.append(feat)
                valid_ids.append(item_id_int)
        
        if len(valid_ids) > 0:
            # Convert to tensor format
            valid_ids_tensor = torch.tensor(valid_ids, device=seq.device).unsqueeze(0)  # [1, N]
            
            # Process features to tensor dictionary format
            processed_features = dataset.process_features_to_tensors([batch_feat])
            
            # Calculate complete item embeddings
            with torch.no_grad():
                # Use baseline_model of unified model
                if hasattr(model, 'baseline_model'):
                    extra_neg_embs = model.baseline_model.feat2emb(valid_ids_tensor, processed_features, include_user=False)
                    extra_neg_embs = extra_neg_embs.squeeze(0)  # [N, hidden_dim]
                    extra_neg_embs = model.baseline_model.last_layernorm(extra_neg_embs)
                else:
                    extra_neg_embs = model.feat2emb(valid_ids_tensor, processed_features, include_user=False)
                    extra_neg_embs = extra_neg_embs.squeeze(0)  # [N, hidden_dim]
                    extra_neg_embs = model.last_layernorm(extra_neg_embs)

                extra_neg_embs = F.normalize(extra_neg_embs, p=2, dim=-1, eps=1e-6)
            
            # Merge batch negatives and additional negatives
            all_neg_pool = torch.cat([neg_pool, extra_neg_embs], dim=0)
        else:
            all_neg_pool = neg_pool
    else:
        all_neg_pool = neg_pool
    
    # Chunked similarity computation to avoid OOM
    def mm(A, B, chunk=8192):
        out = []
        for i in range(0, B.size(0), chunk):
            B_chunk = B[i:i+chunk]
            out.append(A @ B_chunk.T)
        return torch.cat(out, dim=-1)
    
    neg_logit = mm(seq, all_neg_pool)
    
    # Concatenate positive and negative samples
    logits = torch.cat([pos_logit, neg_logit], dim=-1) / tau
    labels = torch.zeros_like(logits[:, 0], dtype=torch.long)
    
    # Calculate basic loss
    if valid_weights is not None:
        # Use action weight weighted loss
        raw_loss = F.cross_entropy(logits, labels, reduction='none')  # [N_valid]
        weighted_loss = raw_loss * valid_weights  # Click actions have higher weight
        loss = weighted_loss.mean()
    else:
        loss = F.cross_entropy(logits, labels)
    
    return loss


def info_loss(seq, pos, neg, loss_mask, action_weights=None):
    """
    Original InfoNCE loss function with action type weighting support
    
    Args:
        seq: sequence embedding
        pos: positive sample embedding
        neg: negative sample embedding
        loss_mask: loss mask
        action_weights: action type weights, shape [B, L]
    """
    tau = 0.03
    # L2 normalize
    seq = F.normalize(seq, p=2, dim=-1, eps=1e-6)
    pos = F.normalize(pos, p=2, dim=-1, eps=1e-6)
    neg = F.normalize(neg, p=2, dim=-1, eps=1e-6)
    
    # Reduce invalid computations
    valid = loss_mask.bool().unsqueeze(-1).expand_as(seq)  # [B, L, D]
    seq = seq[valid].view(-1, seq.size(-1))   # [N_valid, D]
    pos = pos[valid].view(-1, pos.size(-1))
    neg = neg[valid].view(-1, neg.size(-1))
    
    # Extract action weights for valid positions
    valid_weights = None
    if action_weights is not None:
        action_weights = action_weights.to(seq.device)
        valid_mask = loss_mask.bool()
        valid_weights = action_weights[valid_mask]  # [N_valid]
    
    # Positive sample cosine similarity
    pos_logit = F.cosine_similarity(seq, pos, dim=-1).unsqueeze(-1)
    
    # Negative samples in batch     
    neg_pool = neg.reshape(-1, neg.size(-1))
    
    def mm(A, B, chunk=8192):
        out = []
        for i in range(0, B.size(0), chunk):
            B_chunk = B[i:i+chunk]     
            out.append(A @ B_chunk.T)   
        return torch.cat(out, dim=-1)
    
    neg_logit = mm(seq, neg_pool)
    
    # Concatenate positive + negative
    logits = torch.cat([pos_logit, neg_logit], dim=-1) / tau
    labels = torch.zeros_like(logits[:, 0], dtype=torch.long)
    
    # Calculate weighted loss
    if valid_weights is not None:
        # Use action weight weighted loss
        raw_loss = F.cross_entropy(logits, labels, reduction='none')  # [N_valid]
        weighted_loss = raw_loss * valid_weights  # Click actions have higher weight
        loss = weighted_loss.mean()
    else:
        loss = F.cross_entropy(logits, labels)
    
    return loss


def print_memory_usage(stage_name, device):
    """Print GPU memory usage"""
    if torch.cuda.is_available() and device == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Memory {stage_name} - GPU: {allocated:.2f}GB (allocated) / {reserved:.2f}GB (reserved) / {max_allocated:.2f}GB (peak)")

class TrueCPUOptimizer:
    """True CPU optimizer that completely avoids optimizer computation on GPU"""
    
    def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.model = model
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        
        # Store optimizer states on CPU
        self.exp_avg = {}
        self.exp_avg_sq = {}
        
        # For GradScaler compatibility, need param_groups attribute
        self.param_groups = [{'params': list(model.parameters())}]
        
    def zero_grad(self):
        """Zero gradients"""
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self):
        """Execute Adam optimizer step on CPU"""
        self.step_count += 1
        
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
                
            # Move parameters and gradients to CPU
            param_cpu = param.data.cpu().clone()
            grad_cpu = param.grad.cpu().clone()
            
            # Weight decay
            if self.weight_decay != 0:
                grad_cpu = grad_cpu + self.weight_decay * param_cpu
            
            # Initialize optimizer states
            if name not in self.exp_avg:
                self.exp_avg[name] = torch.zeros_like(param_cpu)
                self.exp_avg_sq[name] = torch.zeros_like(param_cpu)
            
            # Adam update on CPU
            self.exp_avg[name] = self.beta1 * self.exp_avg[name] + (1 - self.beta1) * grad_cpu
            self.exp_avg_sq[name] = self.beta2 * self.exp_avg_sq[name] + (1 - self.beta2) * grad_cpu ** 2
            
            # Bias correction
            bias_correction1 = 1 - self.beta1 ** self.step_count
            bias_correction2 = 1 - self.beta2 ** self.step_count
            
            step_size = self.lr / bias_correction1
            denom = (self.exp_avg_sq[name].sqrt() / (bias_correction2 ** 0.5)).add_(self.eps)
            
            # Update parameters
            param_cpu = param_cpu - step_size * self.exp_avg[name] / denom
            
            # Move back to GPU
            param.data = param_cpu.cuda()
            
        # Clean CPU memory
        torch.cuda.empty_cache()

def print_detailed_memory_usage(stage_name, device, optimizer_dense=None, optimizer_row=None):
    """Print detailed GPU memory usage including gradient information"""
    if torch.cuda.is_available() and device == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        # Calculate gradient memory usage
        gradient_memory = 0
        param_memory = 0
        optimizer_memory = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradient_memory += param.grad.numel() * param.grad.element_size() / 1024**3
            param_memory += param.numel() * param.element_size() / 1024**3
        
        # Calculate optimizer state memory (only dense optimizer, row-wise states on CPU)
        if optimizer_dense and hasattr(optimizer_dense, 'state'):
            for param_id, state in optimizer_dense.state.items():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        optimizer_memory += value.numel() * value.element_size() / 1024**3
        
        print(f"Memory {stage_name} - Details:")
        print(f"  Total Memory: {allocated:.2f}GB (allocated) / {reserved:.2f}GB (reserved) / {max_allocated:.2f}GB (peak)")
        print(f"  Parameter Memory: {param_memory:.2f}GB")
        print(f"  Gradient Memory: {gradient_memory:.2f}GB")
        print(f"  Optimizer State Memory: {optimizer_memory:.2f}GB (dense only, row-wise on CPU)")
        print(f"  Other Memory: {allocated - param_memory - gradient_memory - optimizer_memory:.2f}GB")
        
        # Predict total memory needed for optimizer
        if optimizer_memory == 0:  # Optimizer states not created yet
            predicted_optimizer_memory = 2 * param_memory  # Adam needs 2x parameter memory
            total_predicted = param_memory + gradient_memory + predicted_optimizer_memory
            print(f"  Predicted Total Optimizer Memory: {total_predicted:.2f}GB")
            print(f"  Memory Remaining: {38.23 - total_predicted:.2f}GB (assuming 40GB memory)")

def get_action_weights(next_action_type, click_weight=2.0, default_weight=1.0):
    """
    Generate weights based on action types
    
    Args:
        next_action_type: action type tensor, shape [B, L]
        click_weight: weight for click actions
        default_weight: weight for other actions
        
    Returns:
        weights: weight tensor, shape [B, L]
    """
    # Assume action_type encoding: 1=click, 0=other/unknown
    # You can adjust this mapping based on actual encoding
    weights = torch.where(next_action_type == 1, click_weight, default_weight)
    return weights.float()


# Fixed training loop code (replace original optimizer section and training loop)

if __name__ == '__main__':
    # Don't set multiprocessing method here, let load_mm_emb handle it itself
    # torch.multiprocessing.set_start_method('spawn', force=True)

    args = get_args()
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    
    # global dataset
    data_path = os.environ.get('TRAIN_DATA_PATH')

    # Create dataset and model
    if args.state_dict_path is not None:
        print(f"Loading model from: {args.state_dict_path}")
        
        # First create a temporary dataset to get basic information
        temp_dataset = MyDataset(data_path, args)
        usernum, itemnum = temp_dataset.usernum, temp_dataset.itemnum
        feat_statistics, feat_types = temp_dataset.feat_statistics, temp_dataset.feature_types
        
        # Load state dictionary
        state_dict = torch.load(args.state_dict_path, map_location=args.device)
        
        # Recreate dataset with state_dict (so autoencoder will load from state_dict)
        dataset = MyDataset(data_path, args, model_state_dict=state_dict)
        
        # Create and load unified model
        model = load_unified_model(
            args.state_dict_path, 
            usernum, itemnum, feat_statistics, feat_types, 
            args, dataset
        )
        
        # Extract epoch information for continued training
        tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
        epoch_start_idx = int(tail[:tail.find('.')]) + 1 if 'epoch=' in args.state_dict_path else 1
        
    else:
        print("Creating new model and dataset...")
        
        # Create new dataset
        dataset = MyDataset(data_path, args)
        usernum, itemnum = dataset.usernum, dataset.itemnum
        feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types
        
        # Create new model
        baseline_model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args)
        model = UnifiedModel(baseline_model, dataset.autoencoder if hasattr(dataset, 'autoencoder') else None)
        
        epoch_start_idx = 1

    # After dataset creation, ensure multiprocessing method is set to fork (for DataLoader)
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('fork', force=True)
        print("Dataset loading completed, setting multiprocessing start method to fork (for DataLoader)")
    except RuntimeError as e:
        print(f"Cannot set fork method: {e}")
        print("Will use default multiprocessing method")

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.95, 0.05])
    
    # Use command line argument to set num_workers
    if args.num_workers == -1:
        import multiprocessing
        args.num_workers = min(8, multiprocessing.cpu_count() - 1)
        print(f"Auto-setting num_workers to {args.num_workers}")
    
    print(f"Using {args.num_workers} workers for data loading")
    
    # Set other optimization parameters for DataLoader
    use_pin_memory = True if args.device == 'cuda' else False
    persistent_workers = True if args.num_workers > 0 else False
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        collate_fn=dataset.collate_fn,
        pin_memory=use_pin_memory,
        prefetch_factor=2 if args.num_workers > 0 else 2,  
        persistent_workers=persistent_workers
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        collate_fn=dataset.collate_fn,
        pin_memory=use_pin_memory,
        prefetch_factor=2 if args.num_workers > 0 else 2,
        persistent_workers=persistent_workers
    )
    model = model.to(args.device)

    # Selective initialization: xavier for linear layers, small variance normal for Embedding, keep Norm default
    def _init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    model.baseline_model.apply(_init_weights)

    model.baseline_model.pos_emb.weight.data[0, :] = 0
    model.baseline_model.item_emb.weight.data[0, :] = 0
    model.baseline_model.user_emb.weight.data[0, :] = 0

    for k in model.baseline_model.sparse_emb:
        model.baseline_model.sparse_emb[k].weight.data[0, :] = 0
    
    # Print GPU memory usage after model initialization
    print_memory_usage("After Model Initialization", args.device)

    # Create optimizer and scheduler (only optimize baseline_model parameters)
    # === Use single standard AdamW to optimize all parameters ===
    optimizer = torch.optim.AdamW(
        model.baseline_model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,)
    
    print("=" * 80)
    print("Optimizer Configuration: All parameters use standard AdamW")
    print(f"  Total Parameters: {sum(p.numel() for p in model.baseline_model.parameters() if p.requires_grad)}")
    print(f"  Learning Rate: {args.lr}")
    print("=" * 80)
    
    # Mixed precision training configuration - Fix: use configuration matching autocast
    use_bf16 = True  # Use BF16 instead of FP16
    if use_bf16:
        # BF16 doesn't need GradScaler
        print("Enabling BF16 mixed precision training (no GradScaler needed)")
        scaler = None
    else:
        # FP16 needs GradScaler
        print("Enabling FP16 mixed precision training (using GradScaler)")
        scaler = GradScaler()
    
    # Two schedulers (keep consistent with respective optimizers)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.num_epochs, eta_min=1e-6)
  

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    
    print("Start training")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break

        dataset.use_popularity_sampling = True
            
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            
            # Monitor embedding stage memory
            if step == 0:  # Only print for first batch
                print(f"\nTraining Stage Memory Monitoring (Epoch {epoch}, Step {step})")
                print_memory_usage("After Data Loading", args.device)
            
            # Use mixed precision training
            if use_bf16:
                # BF16 autocast
                with autocast(dtype=torch.bfloat16):
                    seq_emb, pos_emb, neg_emb, loss_mask, next_action_type = model(
                        seq, pos, neg, token_type, next_token_type, 
                        next_action_type, seq_feat, pos_feat, neg_feat
                    )
                    
                    # Monitor memory after model forward pass
                    if step == 0:
                        print_memory_usage("After Model Forward Pass (BF16)", args.device)
                    
                    # Generate action weights (if enabled)
                    action_weights = None
                    if args.use_action_weights:
                        action_weights = get_action_weights(
                            next_action_type, 
                            click_weight=args.click_weight, 
                            default_weight=args.default_weight
                        )
                    
                    # Use enhanced InfoNCE loss
                    if args.use_enhanced_negatives:
                        loss = info_loss_with_more_negatives(
                            seq_emb, pos_emb, neg_emb, loss_mask,
                            model=model, 
                            dataset=dataset,
                            num_extra_negatives=args.num_extra_negatives,
                            action_weights=action_weights
                        )
                    else:
                        loss = info_loss(seq_emb, pos_emb, neg_emb, loss_mask, action_weights=action_weights)
                    
                    # Monitor memory after loss computation
                    if step == 0:
                        print_memory_usage("After Loss Computation (BF16)", args.device)
                    
                    # L2 regularization
                    for param in model.baseline_model.item_emb.parameters():
                        loss += args.l2_emb * torch.norm(param)
            else:
                # FP16 autocast with GradScaler
                with autocast(dtype=torch.float16):
                    seq_emb, pos_emb, neg_emb, loss_mask, next_action_type = model(
                        seq, pos, neg, token_type, next_token_type, 
                        next_action_type, seq_feat, pos_feat, neg_feat
                    )
                    
                    if step == 0:
                        print_memory_usage("After Model Forward Pass (FP16)", args.device)
                    
                    # Generate action weights (if enabled)
                    action_weights = None
                    if args.use_action_weights:
                        action_weights = get_action_weights(
                            next_action_type, 
                            click_weight=args.click_weight, 
                            default_weight=args.default_weight
                        )
                    
                    # Use enhanced InfoNCE loss
                    if args.use_enhanced_negatives:
                        loss = info_loss_with_more_negatives(
                            seq_emb, pos_emb, neg_emb, loss_mask,
                            model=model, 
                            dataset=dataset,
                            num_extra_negatives=args.num_extra_negatives,
                            action_weights=action_weights
                        )
                    else:
                        loss = info_loss(seq_emb, pos_emb, neg_emb, loss_mask, action_weights=action_weights)
                    
                    if step == 0:
                        print_memory_usage("After Loss Computation (FP16)", args.device)
                    
                    # L2 regularization
                    for param in model.baseline_model.item_emb.parameters():
                        loss += args.l2_emb * torch.norm(param)

            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1

            # Zero gradients
            optimizer.zero_grad(set_to_none=True)
            
            # Monitor memory before backward pass
            if step == 0:
                print_memory_usage("Before Backward Pass", args.device)
            
            # Backward
            if use_bf16:
                loss.backward()
            else:
                scaler.scale(loss).backward()

            # (Optional) gradient clipping: for FP16, do it after unscale_
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Monitor memory after backward pass
            if step == 0:
                print_detailed_memory_usage("After Backward Pass (Gradients Computed)", args.device, optimizer, None)
            
            # step
            if use_bf16:
                optimizer.step()
            else:
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            
            # Monitor memory after optimizer update
            if step == 0:
                print_memory_usage("After Optimizer Update", args.device)
            
            # Reduce memory cleanup frequency, only when necessary
            if step % 50 == 0:
                torch.cuda.empty_cache()
        
        # Update learning rate after each epoch
        scheduler.step()
        print(f"Epoch {epoch} completed - lr={scheduler.get_last_lr()[0]:.6e}")

        # Validation phase
        model.eval()
        valid_loss_sum = 0
        with torch.no_grad():
            for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader), desc="Validation"):
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
                seq = seq.to(args.device)
                pos = pos.to(args.device)
                neg = neg.to(args.device)
                
                seq_emb, pos_emb, neg_emb, loss_mask, next_action_type = model(
                    seq, pos, neg, token_type, next_token_type, 
                    next_action_type, seq_feat, pos_feat, neg_feat
                )
                
                # Generate action weights (if enabled)
                action_weights = None
                if args.use_action_weights:
                    action_weights = get_action_weights(
                        next_action_type, 
                        click_weight=args.click_weight, 
                        default_weight=args.default_weight
                    )
                
                if args.use_enhanced_negatives:
                    loss = info_loss_with_more_negatives(
                        seq_emb, pos_emb, neg_emb, loss_mask,
                        model=model, 
                        dataset=dataset,
                        num_extra_negatives=args.num_extra_negatives,
                        action_weights=action_weights
                    )
                else:
                    loss = info_loss(seq_emb, pos_emb, neg_emb, loss_mask, action_weights=action_weights)
                    
                valid_loss_sum += loss.item()
                
        valid_loss_sum /= len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)
        
        print(f"Computing ranking metrics for epoch {epoch}...")
        hr_at_10, ndcg_at_10 = evaluate_ranking_metrics(
            model=model,
            valid_loader=valid_loader, 
            dataset=dataset,
            num_users=1000,  # Evaluate 1000 users
            num_negatives=999,  # 999 negative samples
            k=10
        )
        
        # Log to tensorboard
        writer.add_scalar('Metrics/HR@10', hr_at_10, global_step)
        writer.add_scalar('Metrics/NDCG@10', ndcg_at_10, global_step)

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / "model.pt"
        save_unified_model(model, dataset, save_path)
        
        print(f"Model saved to: {save_path}")

    print("Done")
    writer.close()
    log_file.close()