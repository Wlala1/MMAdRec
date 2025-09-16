import subprocess
import sys

try:
    import orjson
except ImportError:
    print("Installing orjson...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "orjson"])
    import orjson
import argparse
import json
import math
import os
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyTestDataset, save_emb
from model import BaselineModel


def _row_l2_normalize(x: np.ndarray) -> np.ndarray:
    # x: (N, D) float32/float64
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / denom).astype(np.float32)


def _load_fbin(path: Path) -> np.ndarray:
    """
    Load .fbin vector files written by save_emb().
    Compatible with two common formats:
      A) [uint32 num][uint32 dim][float32 data...]
      B) [float32 data...] (no header)
    """
    file_size = os.path.getsize(path)
    with open(path, 'rb') as f:
        if file_size >= 8:
            import struct
            num = struct.unpack('I', f.read(4))[0]
            dim = struct.unpack('I', f.read(4))[0]
            expected = 8 + num * dim * 4
            if expected == file_size and num > 0 and dim > 0 and dim < 32768:
                arr = np.fromfile(f, dtype=np.float32, count=num*dim)
                return arr.reshape(num, dim)
    raise RuntimeError(f"Unknown .fbin format for {path}. Please ensure it has header [num, dim].")


def _load_u64bin(path: Path) -> np.ndarray:
    """
    Load id.u64bin (usually consecutive uint64, no header).
    """
    return np.fromfile(str(path), dtype=np.uint64)


def compute_similarity_topk(all_emb, candidate_emb, k=10, batch_size=128, device='cuda'):
    """
    all_emb: (num_users, D)
    candidate_emb: (num_candidates, D)
    return: topk_indices (num_users, k), topk_scores (num_users, k)
    """
    if not isinstance(all_emb, torch.Tensor):
        all_emb = torch.tensor(all_emb, dtype=torch.float32)
    if not isinstance(candidate_emb, torch.Tensor):
        candidate_emb = torch.tensor(candidate_emb, dtype=torch.float32)

    all_emb = all_emb.to(device)
    candidate_emb = candidate_emb.to(device)

    num_users = all_emb.shape[0]
    topk_scores_list, topk_indices_list = [], []

    with torch.no_grad():
        for i in tqdm(range(0, num_users, batch_size), desc="Computing similarities"):
            batch_emb = all_emb[i:i+batch_size]                  # (b, D)
            similarity = torch.matmul(batch_emb, candidate_emb.t())  # (b, C)
            scores, indices = torch.topk(similarity, k=k, dim=1)     # (b, k)
            topk_scores_list.append(scores.cpu())
            topk_indices_list.append(indices.cpu())

    topk_scores = torch.cat(topk_scores_list, dim=0)
    topk_indices = torch.cat(topk_indices_list, dim=0)
    return topk_indices, topk_scores


# Enable expandable segment allocator for inference as well
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def print_memory_usage(stage_name, device):
    """Print GPU memory usage"""
    if torch.cuda.is_available() and device == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[Memory] {stage_name} - GPU Memory: {allocated:.2f}GB (allocated) / {reserved:.2f}GB (reserved) / {max_allocated:.2f}GB (peak)")


def print_detailed_memory_usage(stage_name, device, model=None):
    """Print detailed GPU memory usage"""
    if torch.cuda.is_available() and device == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        param_memory = 0
        gradient_memory = 0
        
        if model:
            for name, param in model.named_parameters():
                param_memory += param.numel() * param.element_size() / 1024**3
                if param.grad is not None:
                    gradient_memory += param.grad.numel() * param.grad.element_size() / 1024**3
        
        print(f"[Memory Detail] {stage_name}:")
        print(f"  Total GPU Memory: {allocated:.2f}GB (allocated) / {reserved:.2f}GB (reserved) / {max_allocated:.2f}GB (peak)")
        print(f"  Parameter Memory: {param_memory:.2f}GB")
        print(f"  Gradient Memory: {gradient_memory:.2f}GB")
        print(f"  Other Memory: {allocated - param_memory - gradient_memory:.2f}GB")


def get_ckpt_path():
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--lr_embed', default=0.00, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    
    # DataLoader params
    parser.add_argument('--num_workers', default=4, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=256, type=int)
    parser.add_argument('--num_blocks', default=8, type=int)
    parser.add_argument('--num_epochs', default=8, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--dropout_rate', default=0.00, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true', default=True)

    # Enhanced negative sampling params
    parser.add_argument('--use_enhanced_negatives', action='store_true')
    parser.add_argument('--num_extra_negatives', default=10000, type=int)
    parser.add_argument('--senet_reduction_ratio', default=16, type=int)

    # Action type weighting params
    parser.add_argument('--use_action_weights', action='store_true')
    parser.add_argument('--click_weight', default=4.0, type=float)
    parser.add_argument('--default_weight', default=1.0, type=float)

    # Random seed
    parser.add_argument('--seed', default=3407, type=int)

    # HSTU Block specific parameters
    parser.add_argument('--use_hstu', action='store_true', default=True)
    parser.add_argument('--hstu_hidden_dim', default=32, type=int)
    parser.add_argument('--hstu_attn_dim', default=32, type=int)
    parser.add_argument('--hstu_use_silu', action='store_true', default=True)
    parser.add_argument('--hstu_use_causal_mask', action='store_true', default=True)
    parser.add_argument('--hstu_use_padding_mask', action='store_true', default=True)

    # RoPE parameters
    parser.add_argument('--use_rope', action='store_true', default=True)
    parser.add_argument('--rope_base', default=10000, type=int)

    parser.add_argument('--mlp_ratio', default=2, type=int)
    parser.add_argument('--mlp_layers', default=2, type=int)
    parser.add_argument('--mlp_activation', default='gelu', type=str)
    parser.add_argument('--mlp_dropout', default=0.3, type=float)

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81', '82'], type=str)
    
    # Autoencoder params
    parser.add_argument('--use_autoencoder', action='store_true', default=True)
    parser.add_argument('--autoencoder_dim', default=32, type=int)
    parser.add_argument('--autoencoder_epochs', default=4, type=int)
    parser.add_argument('--autoencoder_batch_size', default=8192, type=int)

    args = parser.parse_args()
    return args


def process_cold_start_feat(feat):
    """
    Process cold start features
    """
    processed_feat = {}
    for feat_id, feat_value in feat.items():
        if type(feat_value) == list:
            value_list = []
            for v in feat_value:
                if type(v) == str:
                    value_list.append(0)
                else:
                    value_list.append(v)
            processed_feat[feat_id] = value_list
        elif type(feat_value) == str:
            processed_feat[feat_id] = 0
        else:
            processed_feat[feat_id] = feat_value
    return processed_feat


def get_candidate_emb(indexer, feat_types, feat_default_value, compressed_emb_dict, model, autoencoder_dim=32):
    """
    Generate candidate item IDs and embeddings
    Using compressed embeddings
    """
    candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
    item_ids, creative_ids, retrieval_ids, features = [], [], [], []
    retrieve_id2creative_id = {}

    print("Starting to load candidate data...")
    print_memory_usage("Before loading candidate data", 'cuda')

    with open(candidate_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            feature = line['features']
            creative_id = line['creative_id']
            retrieval_id = line['retrieval_id']
            item_id = indexer[creative_id] if creative_id in indexer else 0
            
            # Process basic features
            missing_fields = set(
                feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual']
            ) - set(feature.keys())
            feature = process_cold_start_feat(feature)
            for feat_id in missing_fields:
                feature[feat_id] = feat_default_value[feat_id]
            
            # Use compressed embeddings
            for feat_id in feat_types['item_emb']:
                if compressed_emb_dict and feat_id in compressed_emb_dict:
                    if creative_id in compressed_emb_dict[feat_id]:
                        feature[feat_id] = compressed_emb_dict[feat_id][creative_id]
                    else:
                        # Use compressed dimensions
                        feature[feat_id] = np.zeros(autoencoder_dim, dtype=np.float32)
                else:
                    feature[feat_id] = np.zeros(autoencoder_dim, dtype=np.float32)

            item_ids.append(item_id)
            creative_ids.append(creative_id)
            retrieval_ids.append(retrieval_id)
            features.append(feature)
            retrieve_id2creative_id[retrieval_id] = creative_id

    print(f"Candidate data loading completed, total {len(item_ids)} items")
    print_memory_usage("After loading candidate data", 'cuda')
    print_detailed_memory_usage("After loading candidate data details", 'cuda', model)

    # Save candidate embeddings and IDs
    print("Starting to generate candidate embeddings...")
    model.save_item_emb(item_ids, retrieval_ids, features, os.environ.get('EVAL_RESULT_PATH'))
    print_memory_usage("After generating candidate embeddings", 'cuda')
    
    with open(Path(os.environ.get('EVAL_RESULT_PATH'), "retrive_id2creative_id.json"), "w") as f:
        json.dump(retrieve_id2creative_id, f)
    return retrieve_id2creative_id


def infer():
    args = get_args()
    
    # Set parameters consistent with training
    args.use_hstu = True
    args.norm_first = True
    args.hstu_hidden_dim = 32
    args.hstu_attn_dim = 32
    args.hstu_use_silu = True
    args.hstu_use_causal_mask = True
    args.hstu_use_padding_mask = True
    args.use_rope = True
    args.mm_emb_id = ['81', '82']  # Keep consistent with training
    args.use_autoencoder = True
    args.autoencoder_dim = 32
    args.seed = 3407
    
    print("Inference started - Initialization phase")
    print_memory_usage("Inference start", 'cuda')
    
    # 1. First load model weights file to get autoencoder parameters
    ckpt_path = get_ckpt_path()
    print(f"Loading checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    
    # 2. Create test dataset, pass state_dict to load autoencoder and compress embeddings one by one
    data_path = os.environ.get('EVAL_DATA_PATH')
    print("Starting to create test dataset (including loading and compressing embeddings one by one)...")
    test_dataset = MyTestDataset(data_path, args, model_state_dict=state_dict)
    print("Test dataset creation completed, all embeddings have been compressed")
    print_memory_usage("After dataset loading (with autoencoder compression)", 'cuda')
    
    # 3. Create DataLoader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=min(args.num_workers, 2),
        collate_fn=test_dataset.collate_fn
    )
    print_memory_usage("After DataLoader creation", 'cuda')
    
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types
    
    # 4. Create BaselineModel and load weights
    # First create model on CPU
    orig_device = args.device
    args.device = 'cpu'
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args)
    model.eval()
    print_memory_usage("After model creation on CPU", 'cuda')
    
    # 5. Extract baseline_model parameters from state_dict
    baseline_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('baseline_model.'):
            baseline_key = key[len('baseline_model.'):]
            baseline_state_dict[baseline_key] = value
        elif not key.startswith('autoencoder.'):
            # Backward compatibility
            baseline_state_dict[key] = value
    
    model.load_state_dict(baseline_state_dict, strict=True)
    del state_dict, baseline_state_dict
    
    # 6. Move model to GPU
    model.to('cuda')
    model.dev = 'cuda'
    args.device = orig_device
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print_memory_usage("Model loading completed", 'cuda')
    print_detailed_memory_usage("Model loading completed details", 'cuda', model)
    
    # 7. Generate user embeddings
    all_embs = []
    user_list = []
    
    print("Starting to generate user embeddings...")
    with torch.no_grad():
        for step, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Generating user embeddings"):
            if step == 0:
                print_memory_usage("Before first user batch", 'cuda')
            
            seq, token_type, seq_feat, user_id = batch
            seq = seq.to(args.device)
            token_type = token_type.to(args.device)
            
            if step == 0:
                print_memory_usage("After moving data to GPU", 'cuda')
            
            logits = model.predict(seq, seq_feat, token_type)
            
            if step == 0:
                print_memory_usage("After generating user embedding", 'cuda')
            
            for i in range(logits.shape[0]):
                emb = logits[i].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
                all_embs.append(emb)
            user_list += user_id

    print(f"Generated embeddings for {len(user_list)} users")
    print_memory_usage("After generating all user embeddings", 'cuda')

    # 8. Generate candidate embeddings
    print("Generating candidate item embeddings...")
    
    torch.cuda.empty_cache()
    print_memory_usage("After clearing GPU memory", 'cuda')
    
    # Generate candidates using compressed embeddings
    retrieve_id2creative_id = get_candidate_emb(
        test_dataset.indexer['i'],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.compressed_emb_dict if hasattr(test_dataset, 'compressed_emb_dict') else None,
        model,
        autoencoder_dim=args.autoencoder_dim
    )
    
    all_embs = np.concatenate(all_embs, axis=0)
    print(f"User embeddings shape: {all_embs.shape}")
    print_memory_usage("After merging user embeddings", 'cuda')
    
    # Save query file
    save_emb(all_embs, Path(os.environ.get('EVAL_RESULT_PATH'), 'query.fbin'))
    print_memory_usage("After saving query file", 'cuda')
    
    # 9. PyTorch top-k retrieval
    print("Loading candidate embeddings/id from disk...")
    result_dir = Path(os.environ.get("EVAL_RESULT_PATH"))
    cand_emb_path = result_dir / "embedding.fbin"
    cand_id_path = result_dir / "id.u64bin"

    # Load candidate vectors and IDs
    cand_emb = _load_fbin(cand_emb_path)
    cand_ids = _load_u64bin(cand_id_path)

    # Safety check
    assert cand_emb.ndim == 2 and all_embs.ndim == 2, "Embeddings must be 2D"
    assert cand_emb.shape[1] == all_embs.shape[1], f"Dim mismatch: cand {cand_emb.shape}, user {all_embs.shape}"

    # L2 normalization
    all_embs_norm = _row_l2_normalize(all_embs)
    cand_emb_norm = _row_l2_normalize(cand_emb)

    print("Computing similarity topk...")
    topk_indices, topk_scores = compute_similarity_topk(
        all_emb=all_embs_norm,
        candidate_emb=cand_emb_norm,
        k=10,
        batch_size=128,
        device=args.device
    )

    # indices -> retrieval_id -> creative_id
    print("Mapping indices -> retrieval_id -> creative_id...")
    top10s_untrimmed = []
    for row in topk_indices:
        for rel_idx in row.tolist():
            rid = int(cand_ids[rel_idx])
            top10s_untrimmed.append(retrieve_id2creative_id.get(rid, 0))

    top10s = [top10s_untrimmed[i:i+10] for i in range(0, len(top10s_untrimmed), 10)]
    print(f"Generated recommendations for {len(top10s)} users (PyTorch top-k)")
    
    return top10s, user_list


if __name__ == "__main__":
    top10s, user_list = infer()
    print("Inference completed successfully!")
    print(f"Generated recommendations for {len(user_list)} users")
    print(f"Sample recommendations for first user: {top10s[0] if top10s else 'No recommendations'}")