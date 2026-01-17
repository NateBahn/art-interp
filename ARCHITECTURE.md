# Architecture: Vision Model Interpretability with SAEs

This document describes the technical design and methodology of the interpretability toolkit.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Input Images                                │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CLIP ViT-B-32 Encoder                            │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Layer 0 → Layer 7 → Layer 8 → ... → Layer 11 → Output       │   │
│  │              │          │                │                    │   │
│  │           [hook]     [hook]           [hook]                  │   │
│  │              │          │                │                    │   │
│  │              ▼          ▼                ▼                    │   │
│  │         768-dim     768-dim          768-dim                  │   │
│  │       activation   activation       activation               │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Sparse Autoencoder (SAE)                        │
│                                                                     │
│     activation (768) ──→ ReLU(W_enc @ x + b_enc) ──→ features (49K) │
│                                                                     │
│     W_enc: 768 × 49152        b_enc: 49152                         │
│     Most outputs are 0 (sparse!)                                    │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Analysis Pipeline                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐        │
│  │ Monosemanticity│  │  Correlation   │  │    Feature     │        │
│  │    Scoring     │  │   Analysis     │  │   Selection    │        │
│  └────────────────┘  └────────────────┘  └────────────────┘        │
│           │                   │                   │                 │
│           └───────────────────┴───────────────────┘                 │
│                               │                                     │
│                               ▼                                     │
│                    ┌────────────────┐                               │
│                    │  VLM Labeling  │                               │
│                    │   (Gemini)     │                               │
│                    └────────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. SAE Feature Extractor

**Purpose**: Transform CLIP activations into sparse, interpretable features.

**Architecture**:
```python
class SAEFeatureExtractor:
    # CLIP model (frozen)
    clip_model: ViT-B-32  # 12 transformer blocks, 768 hidden dim

    # SAE weights (from Prisma-Multimodal)
    W_enc: Tensor[768, 49152]   # Encoder weights
    b_enc: Tensor[49152]        # Encoder bias
    W_dec: Tensor[49152, 768]   # Decoder weights (for reconstruction)
    b_dec: Tensor[768]          # Decoder bias
```

**Forward Pass**:
```python
def extract_from_image(self, image):
    # 1. Preprocess image (resize, normalize)
    x = self.preprocess(image)  # [1, 3, 224, 224]

    # 2. Forward through CLIP with activation hook
    self.clip_model.encode_image(x)
    activation = self._activations[f"layer_{self.layer}"]  # [1, 50, 768]

    # 3. Extract CLS token (global representation)
    cls = activation[0, 0, :]  # [768]

    # 4. Pass through SAE encoder
    features = torch.relu(cls @ self.W_enc + self.b_enc)  # [49152]

    # 5. Return sparse result
    return SAEFeatureResult(
        features=features,
        num_active=(features > 0).sum(),
        top_indices=topk_indices,
        top_values=topk_values,
    )
```

**Spatial SAE Variant**:
For localized features, we also support spatial SAEs that operate on the 49 spatial tokens (7×7 grid):

```python
def extract_spatial(self, image):
    # Extract patch activations (tokens 1-49, excluding CLS)
    patch_activations = activation[0, 1:, :]  # [49, 768]

    # Apply SAE to each patch
    patch_features = torch.relu(patch_activations @ self.W_enc + self.b_enc)  # [49, 49152]

    # Can visualize as 7×7 heatmap per feature
    feature_heatmap = patch_features[:, feature_idx].reshape(7, 7)
```

### 2. Monosemanticity Scoring

**Problem**: Not all SAE features are interpretable. Some fire on unrelated concepts (polysemantic).

**Method** (from SAE-for-VLM, NeurIPS 2025):

1. For feature k, get top-20 images by activation value
2. Get embeddings for those images from an INDEPENDENT model (DINOv2)
3. Compute activation-weighted pairwise similarity

**Formula**:
```
MS(k) = (1/Z) Σ_n Σ_m (ã_n × ã_m) × sim(emb_n, emb_m)

where:
  ã_n = normalized activation (min-max to [0,1])
  sim = cosine similarity of DINOv2 embeddings
  Z = Σ_n Σ_m (ã_n × ã_m)  (normalization)
```

**Why DINOv2?**

Using CLIP embeddings would be circular - the SAE was trained on CLIP, so features that select coherent CLIP regions aren't necessarily coherent concepts. DINOv2 provides an independent measure of visual similarity.

**Calibrated Thresholds**:

We calibrate against random image pair similarity (computed on 50k pairs):
- Random baseline: mean=0.195, std=0.154
- **Polysemantic** (MS < 0.35): Within +1σ of random
- **Moderate** (0.35 ≤ MS < 0.50): +1σ to +2σ above random
- **Monosemantic** (MS ≥ 0.50): +2σ above random (clearly coherent)

### 3. Correlation Analysis

**Purpose**: Find which features predict human ratings.

**Method**: Pearson correlation between feature activations and rating values across dataset.

**Multiple Testing Correction**: Benjamini-Hochberg FDR at α=0.05.

```python
def compute_correlations(features, ratings):
    for feature_idx in range(49152):
        for rating_dim in ["mirror_self", "drawn_to", ...]:
            # Get activation values and ratings for all items
            activations = [features[id][feature_idx] for id in items]
            rating_values = [ratings[id][rating_dim] for id in items]

            # Compute Pearson correlation
            r, p = pearsonr(activations, rating_values)

            # Store for FDR correction
            correlations.append((feature_idx, rating_dim, r, p))

    # Apply BH correction
    corrected = benjamini_hochberg(correlations, alpha=0.05)
```

### 4. Feature Selection

**Tiered Selection Strategy**:

| Tier | Criteria | Purpose |
|------|----------|---------|
| 1 (Elite) | MS ≥ 0.50 AND \|r\| ≥ 0.25 | Interpretable AND predictive |
| 2 (Good) | MS ≥ 0.35 OR \|r\| ≥ 0.15 | Either interpretable or predictive |
| 3 (Diverse) | Top by MS | Exploratory coverage |

This ensures we label:
1. The most clearly interpretable, predictive features
2. Features useful for understanding ratings
3. A diverse sample across the feature space

### 5. VLM Feature Labeling

**Approach**: Show a VLM (Gemini 2.0 Flash) the top-activating images for a feature and ask it to identify the common visual property.

**Prompt Structure**:
```
These images all strongly activate SAE feature #{idx}.
The feature has monosemanticity score {ms:.2f} and correlates
{corr:.2f} with "{rating_dim}" ratings.

Analyze what visual property these images share:
1. Short label (2-5 words)
2. Description (1-2 sentences)
3. Visual elements (list)
4. Why it might predict the rating
```

## Protocol-Based Abstraction

The toolkit uses Python protocols for extensibility:

```python
class ImageProvider(Protocol):
    """Provides images from any source."""
    def get_image(self, item_id: str) -> ImageItem | None: ...
    def iter_images(self, filter_labeled: bool, limit: int) -> Iterator[ImageItem]: ...

class FeatureStore(Protocol):
    """Stores/loads features in any format."""
    def save_features(self, item_id: str, features: np.ndarray, layer: int): ...
    def load_features(self, item_id: str, layer: int) -> np.ndarray | None: ...

class EmbeddingProvider(Protocol):
    """Provides embeddings for monosemanticity scoring."""
    def get_embedding(self, item_id: str) -> np.ndarray | None: ...
```

This allows the same analysis code to work with:
- Databases (SQLAlchemy, Supabase)
- Files (JSON, NPZ)
- APIs (remote services)
- In-memory data (testing)

## Design Decisions

### Why CLIP ViT-B-32?

- Well-studied architecture with available SAE weights
- Good balance of quality and efficiency
- DataComp-trained variant has strong visual representations

### Why 49,152 features?

Prisma-Multimodal SAEs use 64× expansion (768 × 64 = 49,152). This provides enough capacity to decompose the representation while maintaining interpretable sparsity.

### Why layers 7, 8, 11?

- **Layer 7**: Early enough for textures/patterns, late enough for semantic
- **Layer 8**: Middle ground, balanced abstraction
- **Layer 11**: Near output, captures high-level semantics

### Why sparse JSON storage?

SAE features are ~99% zero. Sparse storage reduces disk usage by ~50× while maintaining readability and debugging capability.

## Performance Characteristics

| Operation | Time (GPU) | Time (CPU) |
|-----------|-----------|------------|
| Load extractor | ~5s | ~5s |
| Extract single image | ~50ms | ~500ms |
| Extract 1000 images | ~50s | ~8min |
| Monosemanticity (1 feature) | ~10ms | ~10ms |
| Correlations (49K features × 8 ratings) | ~30s | ~30s |

## Future Directions

1. **Cross-modal SAEs**: Apply to text encoder, compare features
2. **Causal interventions**: Modify features, measure output change
3. **Fine-grained spatial**: Higher resolution than 7×7
4. **Temporal SAEs**: Video model interpretability
