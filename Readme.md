# MMAdRec: Multi-Modal Advertisement Recommendation with HSTU Architecture


A generative advertisement recommendation system that leverages multi-modal user behavior data for next-item prediction. The model evolves from SASRec baseline with significant architectural improvements including HSTU blocks, autoencoder-based feature compression, and temporal modeling.

## 🎯 Key Features

- **Multi-Modal Fusion**: Integrates collaborative signals with visual (image) and textual embeddings
- **HSTU Architecture**: Novel attention mechanism with SiLU activation for better gradient flow
- **Autoencoder Compression**: Efficient dimensionality reduction for high-dimensional features (e.g., 4096→32)
- **Temporal Dynamics**: Captures time-decay, hourly/weekly patterns, and temporal gaps
- **Memory Efficient**: Optimized for large-scale datasets with gradient checkpointing and mixed precision training
- **Cold-Start Handling**: DropoutNet-inspired mechanism for robust user representation

## 📊 Performance

Evaluation metrics on proprietary e-commerce dataset:
- **HR@10**: Hit Rate at 10
- **NDCG@10**: Normalized Discounted Cumulative Gain at 10

## 🏗️ Architecture

### Core Components

1. **Feature Processing Pipeline**
   - Sparse features: User/Item categorical attributes
   - Dense features: Pre-trained multi-modal embeddings (vision, text)
   - Temporal features: Time-aware contextual signals
   - Array features: Multi-valued categorical attributes

2. **HSTU Block** (Hierarchical Self-attention with Temporal Understanding)
   ```
   U(X), V(X), Q(X), K(X) = Split(φ₁(f₁(X)))
   A(X)V(X) = φ₂(Q(X)K(X)ᵀ)V(X)  
   Y(X) = f₂(Norm(A(X)V(X))) ⊙ U(X)
   ```
   Where φ₁, φ₂ use SiLU activation

3. **Multi-Modal Autoencoder**
   - Compresses high-dimensional embeddings (1024/3584/4096 → 32)
   - Feature-specific encoder-decoder pairs
   - Preserves semantic information while reducing memory footprint

4. **Training Strategy**
   - InfoNCE loss with enhanced negative sampling
   - Action-type weighted loss (click actions weighted higher)
   - Mixed precision training (BF16/FP16)
   - Curriculum learning for user dropout

## 🚀 Quick Start

### Requirements

```bash
pip install torch>=2.0.0
pip install numpy pandas tqdm orjson
pip install tensorboard
```

### Training

```python
python train.py \
    --batch_size 128 \
    --lr 0.001 \
    --num_epochs 3 \
    --hidden_units 256 \
    --num_blocks 8 \
    --use_hstu \
    --use_rope \
    --use_autoencoder \
    --autoencoder_dim 32 \
    --mm_emb_id 81 82
```

### Inference

```python
python predict.py \
    --batch_size 256 \
    --maxlen 101
```

## 📁 Project Structure

```
MMAdRec/
├── dataset.py          # Data loading and preprocessing
├── model.py            # Main model architecture (HSTU, attention blocks)
├── model_ae.py         # Autoencoder for feature compression
├── train.py            # Training script with memory optimization
├── predict.py          # Inference and candidate generation
└── README.md
```

## 💡 Technical Highlights

### Memory Optimization
- **Gradient Checkpointing**: Reduces memory usage in deep networks
- **Row-wise AdamW**: CPU-offloaded optimizer states for large embeddings
- **Dynamic Batching**: Adaptive batch processing for inference

### Temporal Modeling
- Hour-of-day, day-of-week, month encoding
- Logarithmic time gaps between interactions
- Time-decay weighting with τ=86400 (1 day)

### Negative Sampling
- Popularity-based sampling with Alias method (O(1) complexity)
- Enhanced global negatives (10K+ samples)
- In-batch negative sharing

## 📈 Scalability

- Handles millions of users and items
- Supports sequences up to 101 interactions
- Efficient on datasets with 95/5 train/val split
- GPU memory optimized (runs on 40GB A100)

## 🔬 Research Contributions

1. **HSTU Architecture**: Novel attention mechanism combining gating and self-attention
2. **Multi-Modal Compression**: Feature-specific autoencoders with minimal information loss
3. **Temporal Feature Engineering**: Comprehensive time-aware context modeling
4. **Memory-Efficient Training**: Techniques for scaling to industrial datasets


## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.


---

**Note**: This implementation uses anonymized e-commerce data. The feature IDs and embeddings have been desensitized for privacy protection.