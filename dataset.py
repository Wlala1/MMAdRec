import json
import orjson
import pickle
import struct
from pathlib import Path
import gc
import time

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from model_ae import AutoencoderTrainer, MultiModalAutoencoder

class AliasMethod:
    """O(1) time complexity sampler"""
    def __init__(self, probs):
        self.n = len(probs)
        self.prob = np.zeros(self.n)
        self.alias = np.zeros(self.n, dtype=int)
        
        # Build Alias table
        scaled_probs = np.array(probs) * self.n
        small = []
        large = []
        
        for i, p in enumerate(scaled_probs):
            if p < 1.0:
                small.append(i)
            else:
                large.append(i)
        
        while small and large:
            s = small.pop()
            l = large.pop()
            self.prob[s] = scaled_probs[s]
            self.alias[s] = l
            scaled_probs[l] -= (1.0 - scaled_probs[s])
            if scaled_probs[l] < 1.0:
                small.append(l)
            else:
                large.append(l)
        
        while large:
            self.prob[large.pop()] = 1.0
        while small:
            self.prob[small.pop()] = 1.0
    
    def sample(self):
        """O(1) sampling"""
        i = np.random.randint(0, self.n)
        return i if np.random.random() < self.prob[i] else self.alias[i]


class MyDataset(torch.utils.data.Dataset):
    """
    User sequence dataset - FGSM version, supports temporal features and Autoencoder compression
    """

    def __init__(self, data_dir, args, model_state_dict=None):
        """
        Initialize dataset
        
        Args:
            model_state_dict: Complete model state dictionary containing autoencoder parameters
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self._load_data_and_offsets()
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id
        self.device = args.device
        self.use_popularity_sampling = False
        self.use_autoencoder = args.use_autoencoder
        self.autoencoder_dim = args.autoencoder_dim
        self.autoencoder_epochs = args.autoencoder_epochs
        self.autoencoder_batch_size = args.autoencoder_batch_size
        
        # Important: Extract autoencoder from externally passed model_state_dict
        self.model_state_dict = model_state_dict

        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        
        # Load multimodal features
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        
        # Initialize or load Autoencoder from model_state_dict
        self.autoencoder = None
        if self.use_autoencoder and self.mm_emb_dict:
            self._init_autoencoder_from_state_dict()
        
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer

        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()
        
        # Initialize feature processing related information
        self._init_feat_processing_info()

    def _init_autoencoder_from_state_dict(self):
        """Initialize autoencoder from model_state_dict"""
        # Calculate input dimensions
        input_dims = {}
        for feat_id, emb_dict in self.mm_emb_dict.items():
            if emb_dict:
                sample_emb = next(iter(emb_dict.values()))
                input_dims[feat_id] = sample_emb.shape[0]
        
        # Create autoencoder
        self.autoencoder = MultiModalAutoencoder(input_dims, self.autoencoder_dim).to(self.device)
        
        if self.model_state_dict is not None:
            # Extract autoencoder parameters from complete model_state_dict
            autoencoder_state_dict = {}
            for key, value in self.model_state_dict.items():
                if key.startswith('autoencoder.'):
                    # Remove 'autoencoder.' prefix
                    autoencoder_key = key[len('autoencoder.'):]
                    autoencoder_state_dict[autoencoder_key] = value
            
            if autoencoder_state_dict:
                print("Loading pretrained autoencoder from model.pt...")
                self.autoencoder.load_state_dict(autoencoder_state_dict)
                self.autoencoder.eval()
                print("Autoencoder loaded successfully!")
                # Pre-compress all embeddings and cleanup original data
                self._precompress_and_cleanup_embeddings()
                return
        
        # If no pretrained autoencoder, train a new one
        print("No pretrained autoencoder found, starting to train new one...")
        trainer = AutoencoderTrainer(
            self.mm_emb_dict, 
            target_dim=self.autoencoder_dim, 
            device=self.device
        )
        trainer.train_all_features(epochs=self.autoencoder_epochs, batch_size=self.autoencoder_batch_size)
        
        self.autoencoder = trainer.autoencoder
        self.autoencoder.eval()
        
        # Cleanup trainer
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        
        # Pre-compress all embeddings and cleanup original data
        self._precompress_and_cleanup_embeddings()

        self.autoencoder = self.autoencoder.cpu()

    def _precompress_and_cleanup_embeddings(self):
        """Pre-compress all embeddings and cleanup original data to save memory"""
        if self.autoencoder is None:
            print("Warning: Autoencoder does not exist, skipping pre-compression")
            return
        
        print("Starting to pre-compress all embeddings...")
        
        # Calculate original memory usage
        original_memory = 0
        compressed_memory = 0
        
        # Create compressed dictionary
        self.compressed_emb_dict = {}
        
        for feat_id in self.mm_emb_ids:
            if feat_id not in self.mm_emb_dict or not self.mm_emb_dict[feat_id]:
                continue
                
            print(f"  Compressing feature {feat_id}...")
            self.compressed_emb_dict[feat_id] = {}
            
            # Batch compress embeddings
            emb_dict = self.mm_emb_dict[feat_id]
            total_items = len(emb_dict)
            batch_size = 16384  # Batch processing to save GPU memory
            
            processed = 0
            for i in range(0, total_items, batch_size):
                batch_items = list(emb_dict.items())[i:i+batch_size]
                batch_embeddings = []
                batch_item_ids = []
                
                for item_id, embedding in batch_items:
                    if isinstance(embedding, np.ndarray):
                        original_memory += embedding.nbytes
                        batch_embeddings.append(embedding)
                        batch_item_ids.append(item_id)
                
                if not batch_embeddings:
                    continue
                
                # Batch compression
                try:
                    with torch.no_grad():
                        batch_tensor = torch.from_numpy(np.array(batch_embeddings)).float().to(self.device)
                        compressed_batch = self.autoencoder.encode(batch_tensor, feat_id)
                        compressed_batch_cpu = compressed_batch.cpu().numpy()
                        
                        # Store compression results
                        for j, item_id in enumerate(batch_item_ids):
                            compressed_emb = compressed_batch_cpu[j]
                            self.compressed_emb_dict[feat_id][item_id] = compressed_emb
                            compressed_memory += compressed_emb.nbytes
                        
                        processed += len(batch_item_ids)
                        if processed % 5000 == 0:
                            print(f"    Processed {processed}/{total_items} items")
                        
                        # Clear GPU memory
                        del batch_tensor, compressed_batch, compressed_batch_cpu
                        
                except Exception as e:
                    print(f"    Warning: Batch compression failed: {e}")
                    # Process failed batch one by one
                    for item_id, embedding in batch_items:
                        try:
                            compressed_emb = self._compress_embedding_single(embedding, feat_id)
                            self.compressed_emb_dict[feat_id][item_id] = compressed_emb
                            compressed_memory += compressed_emb.nbytes
                        except Exception as e2:
                            print(f"    Warning: Single embedding compression failed {item_id}: {e2}")
                            # Use zero vector as fallback
                            self.compressed_emb_dict[feat_id][item_id] = np.zeros(self.autoencoder_dim, dtype=np.float32)
                            compressed_memory += self.autoencoder_dim * 4
            
            print(f"  Feature {feat_id} compression completed: {len(self.compressed_emb_dict[feat_id])} items")
        
        # Cleanup original mm_emb_dict to free memory
        print("Cleaning up original embedding data...")
        original_size_mb = original_memory / (1024 * 1024)
        compressed_size_mb = compressed_memory / (1024 * 1024)
        
        self.mm_emb_dict.clear()  # Clear original dictionary
        del self.mm_emb_dict      # Delete reference
        self.mm_emb_dict = None   # Set to None
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"Memory optimization completed:")
        print(f"  Original memory: {original_size_mb:.1f} MB")
        print(f"  Compressed memory: {compressed_size_mb:.1f} MB") 
        print(f"  Memory saved: {original_size_mb - compressed_size_mb:.1f} MB ({(1 - compressed_size_mb/original_size_mb)*100:.1f}%)")
        
        # GPU memory usage monitoring
        self._print_memory_usage("After Autoencoder memory cleanup")

    def _print_memory_usage(self, stage_name):
        """Print GPU and system memory usage"""
        print(f"Memory usage at {stage_name}:")
        
        # GPU memory usage
        if torch.cuda.is_available() and self.device == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  GPU memory: {allocated:.2f}GB (allocated) / {reserved:.2f}GB (reserved) / {max_allocated:.2f}GB (peak)")
        
        # System memory usage (if available)
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_gb = memory_info.rss / 1024**3
            print(f"  System memory: {memory_gb:.2f}GB (current process)")
        except ImportError:
            print("  System memory: psutil not installed, cannot monitor")
        except Exception as e:
            print(f"  System memory: monitoring failed ({e})")

    def _compress_embedding_single(self, embedding, feat_id):
        """Helper method to compress single embedding"""
        if isinstance(embedding, np.ndarray):
            embedding_tensor = torch.from_numpy(embedding).float().to(self.device)
        else:
            embedding_tensor = embedding.float().to(self.device)
        
        if embedding_tensor.dim() == 1:
            embedding_tensor = embedding_tensor.unsqueeze(0)
        
        compressed = self.autoencoder.encode(embedding_tensor, feat_id)
        return compressed.cpu().numpy().squeeze()
    
        
    def _compute_item_popularity(self, alpha=0.25, smoothing=1.0):
        """Optimized version of popularity calculation"""
        print("Computing item popularity for negative sampling...")
        
        # Count frequencies (same as before)
        item_counts = {}
        for uid in range(len(self.seq_offsets)):
            user_data = self._load_user_data(uid)
            for record in user_data:
                item_id = record[1]
                if item_id and item_id <= self.itemnum:
                    item_counts[item_id] = item_counts.get(item_id, 0) + 1
        
        # Build valid items list and probabilities
        self.valid_items = []
        valid_probs = []
        
        # Pre-build string set for faster lookup
        item_feat_dict_keys = set(self.item_feat_dict.keys())
        
        for item_id in range(1, self.itemnum + 1):
            if str(item_id) in item_feat_dict_keys:  # Use set lookup, O(1)
                count = item_counts.get(item_id, 0)
                prob = (count + smoothing) ** alpha
                self.valid_items.append(item_id)
                valid_probs.append(prob)
        
        self.valid_items = np.array(self.valid_items)
        
        # Normalize probabilities
        valid_probs = np.array(valid_probs)
        valid_probs = valid_probs / valid_probs.sum()
        
        # Build Alias sampler
        self.alias_sampler = AliasMethod(valid_probs)
        
        print(f"Popularity computed: {len(self.valid_items)} valid items")
        print(f"Using Alias Method for O(1) sampling")

    def _init_feat_processing_info(self):
        """Initialize information needed for feature processing"""
        self.USER_SPARSE_FEAT = {k: self.feat_statistics[k] for k in self.feature_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = self.feature_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: self.feat_statistics[k] for k in self.feature_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = self.feature_types['item_continual']
        self.USER_ARRAY_FEAT = {k: self.feat_statistics[k] for k in self.feature_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: self.feat_statistics[k] for k in self.feature_types['item_array']}
        
        # Update EMB feature dimensions - use compressed dimensions if using autoencoder
        if self.use_autoencoder:
            self.ITEM_EMB_FEAT = {k: self.autoencoder_dim for k in self.feature_types['item_emb']}
        else:
            EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
            self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in self.feature_types['item_emb']}

    def _load_data_and_offsets(self):
        """
        Load user sequence data and file offsets for each row (preprocessed), used for fast random data access and I/O
        """
        self.data_file_path = self.data_dir / "seq.jsonl"
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _load_user_data(self, uid):
        """
        Load single user's data from data file
        """
        with open(self.data_file_path, 'rb') as f:
            f.seek(self.seq_offsets[uid])
            line = f.readline()
            data = json.loads(line)
        return data

    def _add_time_features(self, user_sequence, tau=86400):
        """
        Add temporal features to user sequence
        
        Args:
            user_sequence: Original user sequence
            tau: Time decay parameter, default 86400 seconds (1 day)
        
        Returns:
            ts_array: Timestamp array
            new_sequence: New sequence with added temporal features
        """
        # Extract timestamp array
        ts_array = np.array([r[5] for r in user_sequence], dtype=np.int64)
        
        # Calculate time_gap and log_gap
        prev_ts_array = np.roll(ts_array, 1)
        prev_ts_array[0] = ts_array[0]
        time_gap = ts_array - prev_ts_array
        time_gap[0] = 0
        log_gap = np.log1p(time_gap)  # log1p avoids log(0)
        
        # Calculate hour, weekday, month
        ts_utc8 = ts_array + 8 * 3600
        hours = (ts_utc8 % 86400) // 3600
        weekdays = ((ts_utc8 // 86400 + 4) % 7).astype(np.int32)
        months = pd.to_datetime(ts_utc8, unit='s').to_numpy().astype('datetime64[M]').astype(int) % 12 + 1
        
        # Calculate time_decay
        last_ts = ts_array[-1]
        delta_t = last_ts - ts_array
        delta_scaled = np.log1p(delta_t / tau)
        
        # Build new sequence, adding temporal features to user_feat
        new_sequence = []
        for idx, record in enumerate(user_sequence):
            u, i, user_feat, item_feat, action_type, ts = record
            if user_feat is None:
                user_feat = {}
            
            # Add temporal features to user_feat
            user_feat["200"] = int(hours[idx])      # hour
            user_feat["201"] = int(weekdays[idx])    # weekday
            # user_feat["202"] = float(time_gap[idx])  # time_gap (if needed)
            user_feat["203"] = float(log_gap[idx])   # log_gap
            user_feat["204"] = int(months[idx])      # month
            user_feat["205"] = float(delta_scaled[idx])  # time_decay
            
            new_sequence.append((u, i, user_feat, item_feat, action_type, ts))
        
        return ts_array, new_sequence

    def _transfer_context_features(self, user_feat, item_feat, cols_to_trans):
        """
        Transfer context features from user_feat to item_feat
        This is a key step: ensure temporal features enter sequence but not pos/neg
        
        Args:
            user_feat: User feature dictionary
            item_feat: Item feature dictionary
            cols_to_trans: List of features to transfer
        
        Returns:
            item_feat: Updated item feature dictionary
        """
        if item_feat is None:
            item_feat = {}
        
        for col in cols_to_trans:
            if col in user_feat:
                item_feat[col] = user_feat[col]
        
        return item_feat

    def _random_neq_uniform(self, l, r, s):
        """
        Uniform random sampling: generate a random integer not in sequence s, used for negative sampling during training
        """
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def _random_neq_popularity(self, l, r, s):
        """
        Popularity sampling: high-speed negative sampling using Alias Method
        """
        # Convert to set for faster lookup
        if not isinstance(s, set):
            s = set(s)
        
        max_attempts = 100  # Can reduce attempts since sampling is more efficient
        
        for _ in range(max_attempts):
            if hasattr(self, 'alias_sampler'):
                # O(1) sampling
                idx = self.alias_sampler.sample()
                t = self.valid_items[idx]
            else:
                t = np.random.randint(l, r)
            
            if t not in s:  # No need to check item_feat_dict due to pre-filtering
                return t
        
        # fallback: randomly choose one from valid_items not in s
        if hasattr(self, 'valid_items'):
            valid_set = set(self.valid_items) - s
            if valid_set:
                return np.random.choice(list(valid_set))
        
        # Final fallback
        t = np.random.randint(l, r)
        while t in s:
            t = np.random.randint(l, r)
        return t

    def _random_neq(self, l, r, s):
        """
        Choose sampling strategy based on current settings
        """
        if self.use_popularity_sampling:
            return self._random_neq_popularity(l, r, s)
        else:
            return self._random_neq_uniform(l, r, s)

    def __getitem__(self, uid):
        """
        Get single user's data and perform padding to generate model-required data format
        """
        user_sequence = self._load_user_data(uid)
        
        # Add temporal features
        ts_array, user_sequence = self._add_time_features(user_sequence)
        
        # Temporal feature columns
        time_feature_cols = ["200", "201", "203", "204", "205"]

        user_token_data = None
        item_sequence = []
        
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            
            # Collect user token (only take the first one)
            if u and user_feat and user_token_data is None:
                user_token_data = (u, user_feat, {}, 2, action_type)
            
            # Collect item tokens
            if i and item_feat:
                # Transfer temporal features to current item
                i_feat = self._transfer_context_features(
                    user_feat, item_feat, time_feature_cols
                )
                item_sequence.append((i, {}, i_feat, 1, action_type))
    
        # Initialize arrays
        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)
    
        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1], dtype=object)
    
        # First place user token at position 0
        if user_token_data:
            u_id, u_feat, _, type_, act_type = user_token_data
            seq[0] = u_id
            token_type[0] = type_
            seq_feat[0] = self.fill_missing_feat(u_feat, u_id)
        
        # Collect item IDs for negative sampling
        ts = set()
        for i, _, i_feat, type_, act_type in item_sequence:
            if i:
                ts.add(i)
    
        # Fill item tokens starting from position 1 (left-padding strategy)
        idx = self.maxlen  # Start from rightmost position
        nxt = None
        
        # Reverse traverse item sequence for left-padding
        for i in range(len(item_sequence) - 1, -1, -1):
            if idx <= 0:  # Position 0 reserved for user token
                break
                
            current = item_sequence[i]
            i_id, _, i_feat, type_, act_type = current
            
            seq[idx] = i_id
            token_type[idx] = type_
            seq_feat[idx] = self.fill_missing_feat(i_feat, i_id)
            
            # Process next token (for training)
            if nxt is not None:
                next_i, _, next_feat, next_type, next_act_type = nxt
                next_token_type[idx] = next_type
                if next_act_type is not None:
                    next_action_type[idx] = next_act_type
                
                if next_type == 1 and next_i != 0:
                    pos[idx] = next_i
                    pos_feat[idx] = self.fill_missing_feat(next_feat, next_i)
                    neg_id = self._random_neq(1, self.itemnum + 1, ts)
                    neg[idx] = neg_id
                    neg_feat[idx] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
            
            nxt = current
            idx -= 1
    
        # Fill default values
        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)
    
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

    def __len__(self):
        """
        Return dataset length, i.e., number of users

        Returns:
            usernum: Number of users
        """
        return len(self.seq_offsets)
        
    def _init_feat_info(self):
        """
        Initialize feature information, including feature default values and feature types
        Note: Temporal features are defined as user_sparse type
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        
        # Original features + temporal features
        feat_types['user_sparse'] = ['103', '104', '105', '109', '200', '201', '204']  # hour, weekday, month
        feat_types['user_continual'] = ['203', '205']  # log_gap, time_decay etc
        
        feat_types['item_sparse'] = [
            '100', '117', '111', '118', '101', '102', '119', '120',
            '114', '112', '121', '115', '122', '116',
        ]
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['item_continual'] = []

        # Initialize feature default values and statistics
        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            if feat_id in self.indexer['f']:
                feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
            else:
                # Statistics for temporal features
                if feat_id == '200':  # hour: 0-23
                    feat_statistics[feat_id] = 24
                elif feat_id == '201':  # weekday: 0-6
                    feat_statistics[feat_id] = 7
                elif feat_id == '204':  # month: 1-12
                    feat_statistics[feat_id] = 13
                    
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0.0
            
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
            
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
            
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
            
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
            
        for feat_id in feat_types['item_emb']:
            if self.use_autoencoder:
                feat_default_value[feat_id] = np.zeros(self.autoencoder_dim, dtype=np.float32)
                feat_statistics[feat_id] = self.autoencoder_dim
            else:
                in_dim = list(self.mm_emb_dict[feat_id].values())[0].shape[0]
                feat_default_value[feat_id] = np.zeros(in_dim, dtype=np.float32)
                feat_statistics[feat_id] = in_dim

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        """
        Fill default values for missing features in original data
        """
        if feat == None:
            feat = {}
        filled_feat = {}
        for k in feat.keys():
            filled_feat[k] = feat[k]

        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)
        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]
        
        # Process multimodal embedding features
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0:
                item_key = self.indexer_i_rev[item_id]
                
                if self.use_autoencoder:
                    # Get data from pre-compressed dictionary
                    if hasattr(self, 'compressed_emb_dict') and self.compressed_emb_dict is not None:
                        if feat_id in self.compressed_emb_dict and item_key in self.compressed_emb_dict[feat_id]:
                            filled_feat[feat_id] = self.compressed_emb_dict[feat_id][item_key]
                        # If not in compressed dictionary, use default value (zero vector)
                        # else: will use zero vector from feature_default_value
                    else:
                        # Compatibility mode: if not pre-compressed, compress from original data
                        if (self.mm_emb_dict is not None and 
                            feat_id in self.mm_emb_dict and 
                            item_key in self.mm_emb_dict[feat_id]):
                            original_emb = self.mm_emb_dict[feat_id][item_key]
                            if isinstance(original_emb, np.ndarray):
                                compressed_emb = self._compress_embedding(original_emb, feat_id)
                                filled_feat[feat_id] = compressed_emb
                else:
                    # Not using autoencoder, get from original data
                    if (self.mm_emb_dict is not None and 
                        feat_id in self.mm_emb_dict and 
                        item_key in self.mm_emb_dict[feat_id]):
                        original_emb = self.mm_emb_dict[feat_id][item_key]
                        if isinstance(original_emb, np.ndarray):
                            filled_feat[feat_id] = original_emb

        return filled_feat

    # Other methods remain unchanged...
    def feat2tensor(self, seq_feature, k):
        """Convert features to tensor"""
        batch_size = len(seq_feature)

        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
            max_array_len = 0
            max_seq_len = 0

            for i in range(batch_size):
                seq_data = [item[k] if k in item else self.feature_default_value[k] 
                           for item in seq_feature[i]]
                max_seq_len = max(max_seq_len, len(seq_data))
                for item_data in seq_data:
                    if isinstance(item_data, list):
                        max_array_len = max(max_array_len, len(item_data))

            batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] if k in item else self.feature_default_value[k] 
                           for item in seq_feature[i]]
                for j, item_data in enumerate(seq_data):
                    if isinstance(item_data, list):
                        actual_len = min(len(item_data), max_array_len)
                        batch_data[i, j, :actual_len] = item_data[:actual_len]

            return torch.from_numpy(batch_data)
        else:
            max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
            
            # Choose data type based on feature type
            if k in self.USER_CONTINUAL_FEAT or k in self.ITEM_CONTINUAL_FEAT:
                dtype = np.float32
            else:
                dtype = np.int64
                
            batch_data = np.zeros((batch_size, max_seq_len), dtype=dtype)

            for i in range(batch_size):
                seq_data = [item[k] if k in item else self.feature_default_value[k] 
                           for item in seq_feature[i]]
                batch_data[i, :len(seq_data)] = seq_data

            return torch.from_numpy(batch_data)

    def process_features_to_tensors(self, feature_array):
        """Batch process features and convert to tensor dictionary"""
        processed_features = {}
        
        for k in self.USER_SPARSE_FEAT:
            processed_features[k] = self.feat2tensor(feature_array, k)
        for k in self.ITEM_SPARSE_FEAT:
            processed_features[k] = self.feat2tensor(feature_array, k)
        for k in self.USER_ARRAY_FEAT:
            processed_features[k] = self.feat2tensor(feature_array, k)
        for k in self.ITEM_ARRAY_FEAT:
            processed_features[k] = self.feat2tensor(feature_array, k)
        for k in self.USER_CONTINUAL_FEAT:
            processed_features[k] = self.feat2tensor(feature_array, k)
        for k in self.ITEM_CONTINUAL_FEAT:
            processed_features[k] = self.feat2tensor(feature_array, k)
        for k in self.ITEM_EMB_FEAT:
            batch_size = len(feature_array)
            emb_dim = self.ITEM_EMB_FEAT[k]
            seq_len = len(feature_array[0])
            
            batch_emb_data = np.zeros((batch_size, seq_len, emb_dim), dtype=np.float32)
            
            for i, seq in enumerate(feature_array):
                for j, item in enumerate(seq):
                    if k in item:
                        batch_emb_data[i, j] = item[k]
                        
            processed_features[k] = torch.from_numpy(batch_emb_data)
            
        return processed_features

    def collate_fn(self, batch):
        """
        Process batch data into tensors and transfer to GPU
        Actually perform feature preprocessing here, utilizing multiprocessing parallelism
        
        Args:
            batch: Multiple data returned by __getitem__

        Returns:
            seq: User sequence IDs, in torch.Tensor form
            pos: Positive sample IDs, in torch.Tensor form
            neg: Negative sample IDs, in torch.Tensor form
            token_type: User sequence types, in torch.Tensor form
            next_token_type: Next token types, in torch.Tensor form
            seq_feat: User sequence features, processed into tensor dictionary
            pos_feat: Positive sample features, processed into tensor dictionary
            neg_feat: Negative sample features, processed into tensor dictionary
        """
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        
        # Actually perform feature preprocessing here! Utilizing multiprocessing parallelism
        seq_feat_tensors = self.process_features_to_tensors(seq_feat)
        pos_feat_tensors = self.process_features_to_tensors(pos_feat)
        neg_feat_tensors = self.process_features_to_tensors(neg_feat)
        
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat_tensors, pos_feat_tensors, neg_feat_tensors


class MyTestDataset(MyDataset):
    """
    Test dataset - supports temporal features and Autoencoder compression
    """

    def __init__(self, data_dir, args, model_state_dict=None):
        super().__init__(data_dir, args, model_state_dict)

    def _load_data_and_offsets(self):
        """Override parent method to use prediction data file"""
        self.data_file_path = self.data_dir / "predict_seq.jsonl"
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _add_time_features_for_inference(self, user_sequence, tau=86400):
        ts_array = np.array([r[5] for r in user_sequence], dtype=np.int64)
        
        # Basic temporal feature calculation (keep unchanged)
        prev_ts_array = np.roll(ts_array, 1)
        prev_ts_array[0] = ts_array[0]
        time_gap = ts_array - prev_ts_array
        time_gap[0] = 0
        log_gap = np.log1p(time_gap)
        
        # Calculate hour, weekday, month
        ts_utc8 = ts_array + 8 * 3600
        hours = (ts_utc8 % 86400) // 3600
        weekdays = ((ts_utc8 // 86400 + 4) % 7).astype(np.int32)
        months = pd.to_datetime(ts_utc8, unit='s').to_numpy().astype('datetime64[M]').astype(int) % 12 + 1
        
        # Calculate time_decay
        last_ts = ts_array[-1]
        delta_t = last_ts - ts_array
        delta_scaled = np.log1p(delta_t / tau)
        
        # Build new sequence, adding temporal features to user_feat
        new_sequence = []
        for idx, record in enumerate(user_sequence):
            u, i, user_feat, item_feat, action_type, ts = record
            if user_feat is None:
                user_feat = {}
            
            # Add temporal features to user_feat
            user_feat["200"] = int(hours[idx])      # hour
            user_feat["201"] = int(weekdays[idx])    # weekday
            # user_feat["202"] = float(time_gap[idx])  # time_gap (if needed)
            user_feat["203"] = float(log_gap[idx])   # log_gap
            user_feat["204"] = int(months[idx])      # month
            user_feat["205"] = float(delta_scaled[idx])  # time_decay
            
            new_sequence.append((u, i, user_feat, item_feat, action_type, ts))
        
        return ts_array, new_sequence

    def _process_cold_start_feat(self, feat):
        """
        Process cold start features, including all temporal features
        """
        if feat is None:
            return {}
            
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            # Process all continuous temporal features
            if feat_id in ["203", "205"]:  # All continuous temporal features
                if isinstance(feat_value, (int, float)):
                    processed_feat[feat_id] = float(feat_value)
                else:
                    processed_feat[feat_id] = 0.0
            # Process discrete temporal features
            elif feat_id in ["200", "201", "204"]:  # hour, weekday, month
                if isinstance(feat_value, (int, float)):
                    processed_feat[feat_id] = int(feat_value)
                else:
                    processed_feat[feat_id] = 0
            # ... process other features ...
            elif type(feat_value) == list:
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

    def __getitem__(self, uid):
        """
        Get single user's data and perform padding to generate model-required data format
        
        Args:
            uid: User's row number stored in self.data_file
            
        Returns:
            seq: User sequence IDs
            token_type: User sequence types, 1 for item, 2 for user
            seq_feat: User sequence features, each element is a dictionary
            user_id: user_id, for subsequent answer verification
        """
        user_sequence = self._load_user_data(uid)
        
        # Add temporal features - use inference mode
        ts_array, user_sequence = self._add_time_features_for_inference(user_sequence)
        
        # Temporal feature list - keep consistent with training
        time_feature_cols = ["200", "201", "203", "204", "205"]
        
        # Build extended sequence
        user_token_data = None
        item_sequence = []
        user_id = None
        
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            
            # Process user information
            if u:
                if type(u) == str:  # String indicates user_id
                    user_id = u
                    u = 0  # Assign 0 for cold start users
                else:  # int indicates re_id
                    user_id = self.indexer_u_rev[u]
                
                # Collect user token (only take the first one)
                if user_feat and user_token_data is None:
                    user_feat = self._process_cold_start_feat(user_feat)
                    user_token_data = (u, user_feat, {}, 2)
            
            # Process item tokens - transfer temporal features from user_feat to item_feat (consistent with training)
            if i and item_feat:
                # For items not seen during training, assign value 0
                if i > self.itemnum:
                    i = 0
                item_feat = self._process_cold_start_feat(item_feat)
                
                # Transfer temporal features to item (keep consistent with training)
                item_feat = self._transfer_context_features(
                    user_feat, item_feat, time_feature_cols
                )
                
                item_sequence.append((i, {}, item_feat, 1))

        # Initialize arrays
        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)

        # Step 1: Place user token at position 0
        if user_token_data:
            u_id, u_feat, _, type_ = user_token_data
            seq[0] = u_id
            token_type[0] = type_
            seq_feat[0] = self.fill_missing_feat(u_feat, u_id)

        # Step 2: Fill items starting from position 1 (using left-padding)
        idx = self.maxlen  # Start from rightmost position
        
        # Reverse traverse item sequence for left-padding
        for i in range(len(item_sequence) - 1, -1, -1):
            if idx <= 0:  # Position 0 reserved for user token
                break
            
            item_id, _, item_feat, type_ = item_sequence[i]
            seq[idx] = item_id
            token_type[idx] = type_
            seq_feat[idx] = self.fill_missing_feat(item_feat, item_id)
            idx -= 1

        # Fill default values
        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)

        return seq, token_type, seq_feat, user_id

    def __len__(self):
        """
        Return dataset length
        
        Returns:
            Number of users
        """
        return len(self.seq_offsets)

    def collate_fn(self, batch):
        """
        Concatenate multiple __getitem__ returned data into one batch
        Keep consistent with training, need to process features into tensors
        
        Args:
            batch: Multiple data returned by __getitem__
            
        Returns:
            seq: User sequence IDs, in torch.Tensor form
            token_type: User sequence types, in torch.Tensor form
            seq_feat: User sequence features, processed into tensor dictionary
            user_ids: List of user_ids
        """
        seq, token_type, seq_feat, user_id = zip(*batch)
        
        # Convert to tensor
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        
        # Use parent class's process_features_to_tensors method to process features
        # This ensures consistency with training-time feature processing
        seq_feat_tensors = self.process_features_to_tensors(seq_feat)
        
        return seq, token_type, seq_feat_tensors, user_id


def save_emb(emb, save_path):
    """
    Save Embedding as binary file

    Args:
        emb: Embedding to save, shape [num_points, num_dimensions]
        save_path: Save path
    """
    num_points = emb.shape[0]  # Number of data points
    num_dimensions = emb.shape[1]  # Vector dimensions
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
    """
    Load multimodal feature embeddings

    Args:
        mm_path: Multimodal feature embedding path
        feat_ids: List of multimodal feature IDs to load

    Returns:
        mm_emb_dict: Multimodal feature embedding dictionary, key is feature ID, value is feature embedding dictionary (key is item ID, value is embedding)
    """
    # Set spawn method inside load_mm_emb to avoid CUDA multiprocessing issues
    original_start_method = mp.get_start_method(allow_none=True)
    try:
        mp.set_start_method('spawn', force=True)
        print("load_mm_emb: Set multiprocessing start method to spawn")
    except RuntimeError as e:
        print(f"Warning: Cannot set spawn method: {e}")
    
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        start_time = time.time()
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        if feat_id != '81':
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                json_files = list(base_path.glob('part-*'))

                # Parse JSON files in parallel
                with Pool(processes=min(cpu_count(), 8)) as pool:  # 8 processes in parallel, don't open too many
                    results = pool.map(_parse_json_file, json_files)

                # Merge results
                for r in results:
                    emb_dict.update(r)
            except Exception as e:
                print(f"transfer error: {e}")
        if feat_id == '81':
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
        end_time = time.time()
        mm_count = len(emb_dict)
        print(f"Loading completed: {end_time - start_time:.2f} seconds, {mm_count} embeddings")
    
    # Restore original multiprocessing start method
    try:
        if original_start_method is not None:
            mp.set_start_method(original_start_method, force=True)
            print(f"load_mm_emb: Restored multiprocessing start method to {original_start_method}")
        else:
            # If no original method, set to fork (more efficient on Linux)
            mp.set_start_method('fork', force=True)
            print("load_mm_emb: Set multiprocessing start method to fork (for DataLoader)")
    except RuntimeError as e:
        print(f"Warning: Cannot restore multiprocessing method: {e}")
    
    return mm_emb_dict

def _parse_json_file(json_file):
    """Child process reads single JSON file"""
    emb_dict = {}
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            obj = orjson.loads(line)
            emb = np.array(obj['emb'], dtype=np.float32)
            emb_dict[obj['anonymous_cid']] = emb
    return emb_dict