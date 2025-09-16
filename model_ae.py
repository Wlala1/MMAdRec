import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class MultiModalAutoencoder(nn.Module):
    """
    Multi-modal Autoencoder that compresses multi-modal features of different dimensions into a unified low-dimensional space
    """
    
    def __init__(self, input_dims, target_dim=64, hidden_ratio=0.5):
        """
        Args:
            input_dims: dict, {feat_id: input_dimension}
            target_dim: int, target compression dimension
            hidden_ratio: float, hidden layer dimension ratio
        """
        super().__init__()
        self.input_dims = input_dims
        self.target_dim = target_dim
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        
        for feat_id, input_dim in input_dims.items():
            # Calculate hidden layer dimension
            hidden_dim = max(int(input_dim * hidden_ratio), target_dim * 2)
            
            # Encoder: input_dim -> hidden_dim -> target_dim
            self.encoders[feat_id] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, target_dim),
                nn.LayerNorm(target_dim)
            )
            
            # Decoder: target_dim -> hidden_dim -> input_dim
            self.decoders[feat_id] = nn.Sequential(
                nn.Linear(target_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, input_dim)
            )
    
    def encode(self, x, feat_id):
        """Encode single feature"""
        return self.encoders[feat_id](x)
    
    def decode(self, z, feat_id):
        """Decode single feature"""
        return self.decoders[feat_id](z)
    
    def forward(self, x, feat_id):
        """Forward propagation"""
        z = self.encode(x, feat_id)
        x_recon = self.decode(z, feat_id)
        return z, x_recon


class AutoencoderTrainer:
    """
    Autoencoder trainer that supports memory-efficient training
    """
    
    def __init__(self, mm_emb_dict, target_dim=64, device='cuda'):
        self.mm_emb_dict = mm_emb_dict
        self.target_dim = target_dim
        self.device = device
        
        # Calculate input dimensions
        input_dims = {}
        for feat_id, emb_dict in mm_emb_dict.items():
            if emb_dict:
                sample_emb = next(iter(emb_dict.values()))
                input_dims[feat_id] = sample_emb.shape[0]
        
        self.autoencoder = MultiModalAutoencoder(input_dims, target_dim).to(device)

        # self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)
        # Use AdamW optimizer for better generalization performance
        self.optimizer = torch.optim.AdamW(
            self.autoencoder.parameters(), 
            lr=3e-3,  # Increase learning rate
            weight_decay=1e-4,
            betas=(0.9, 0.95)  # Adjust momentum parameters
        )

        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode='min', factor=0.5, patience=5
        # )
        # Use more aggressive learning rate scheduling
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=3e-3,
            epochs=15,  # Significantly reduce epoch count
            steps_per_epoch=100,  # Estimated value, will be adjusted during training
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
    def create_training_data(self, feat_id, batch_size=1024):
        """Create training data generator for specified feature"""
        emb_dict = self.mm_emb_dict[feat_id]
        embeddings = list(emb_dict.values())
        
        # Convert to numpy array and randomly shuffle
        embeddings = np.array(embeddings, dtype=np.float32)
        np.random.shuffle(embeddings)

        # Pre-move to GPU to reduce data transfer overhead
        total_batches = (len(embeddings) + batch_size - 1) // batch_size
        
        # Batch generator
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size]
            # Non-blocking transfer
            yield torch.from_numpy(batch).to(self.device, non_blocking=True), total_batches
    
    def train_feature(self, feat_id, epochs=50, batch_size=1024):
        """Train autoencoder for specified feature"""
        print(f"Training autoencoder for feature {feat_id}...")
        
        self.autoencoder.train()
        best_loss = float('inf')
        patience_counter = 0
        
        total_batches = 0

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # Use tqdm to show progress
            data_gen = self.create_training_data(feat_id, batch_size)

            if epoch == 0:
                # Calculate total_batches in first epoch to adjust scheduler
                first_batch, estimated_total = next(data_gen)
                total_batches = estimated_total
                
                # Re-initialize scheduler with correct steps_per_epoch
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=3e-3,
                    epochs=epochs,
                    steps_per_epoch=total_batches,
                    pct_start=0.3,
                    anneal_strategy='cos'
                )
                
                # Process first batch
                batch_data = first_batch
            else:
                data_gen = self.create_training_data(feat_id, batch_size)
            
            pbar = tqdm(data_gen, total=total_batches, desc=f"Epoch {epoch+1}/{epochs}")

            for batch_data, _ in pbar:
                self.optimizer.zero_grad()
                
                # Forward propagation
                z, x_recon = self.autoencoder(batch_data, feat_id)
                
                # Calculate reconstruction loss
                # print("Start calculating reconstruction loss...")
                recon_loss = F.mse_loss(x_recon, batch_data)
                # print("Reconstruction loss calculated.")
                
                # Add regularization term
                l2_reg = 0
                for param in self.autoencoder.encoders[feat_id].parameters():
                    l2_reg += torch.norm(param, 2)
                for param in self.autoencoder.decoders[feat_id].parameters():
                    l2_reg += torch.norm(param, 2)
                
                loss = recon_loss + 1e-4 * l2_reg
                
                # Backward propagation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.6f}', 
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                })

                # Clean GPU memory
                del z, x_recon, batch_data
                torch.cuda.empty_cache()
            
            avg_loss = total_loss / num_batches

            # Manually print learning rate changes
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(avg_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"[Scheduler] ReduceLROnPlateau: reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")
            
            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.6f}, LR: {self.scheduler.get_last_lr()[0]:.2e}")
            
            # Early stopping mechanism
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    def train_all_features(self, epochs=50, batch_size=1024):
        """Train autoencoder for all features"""
        for idx, feat_id in enumerate(self.mm_emb_dict.keys()):
            if self.mm_emb_dict[feat_id]:  # Ensure feature dictionary is not empty

                # Re-initialize optimizer for each feature to avoid parameter pollution
                self.optimizer = torch.optim.AdamW(
                    [p for name, p in self.autoencoder.named_parameters() 
                     if name.startswith(f'encoders.{feat_id}') or name.startswith(f'decoders.{feat_id}')],
                    lr=3e-3,
                    weight_decay=1e-4,
                    betas=(0.9, 0.95)
                )

                self.train_feature(feat_id, epochs, batch_size)
                # Force garbage collection
                gc.collect()
                torch.cuda.empty_cache()

                print(f"Feature {feat_id} training completed!")
        print("\nAll features training completed!")
    

class UnifiedModel(nn.Module):
    """
    Unified model class that contains BaselineModel and Autoencoder
    """
    
    def __init__(self, baseline_model, autoencoder=None):
        super().__init__()
        self.baseline_model = baseline_model
        self.autoencoder = autoencoder
        
        # If autoencoder exists, register its parameters but don't participate in main training
        if self.autoencoder is not None:
            for param in self.autoencoder.parameters():
                param.requires_grad = False
    
    def forward(self, *args, **kwargs):
        """Directly call baseline_model's forward method"""
        return self.baseline_model(*args, **kwargs)
    
    def __getattr__(self, name):
        """Proxy attribute access to baseline_model"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.baseline_model, name)