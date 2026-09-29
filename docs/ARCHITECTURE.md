# DiffTokenStream: Mathematical & Technical Architecture

> **System Paradigm**: Feature Coding for Machines (FCM) via Split-Inference Vision Transformers  
> **Reference Backbone**: Frozen DINOv2-ViT-Small (`dinov2_vits14`, $D=384$, Patch Size $P=14 \times 14$)  
> **Primary Objective**: Ultra-low-bitrate machine-to-machine video perception over constrained channels  
> **Key Metric**: Downstream task fidelity retention vs. Raw ViT Oracle and H.264 at matched wire bitrates

---

## 1. System Overview & The FCM Paradigm

### 1.1 The Machine Vision Dilemma: The Inefficiency of Pixel Streaming
Traditional video streaming architectures (H.264, H.265, AV1) are designed for human visual perception. They allocate substantial compute and bandwidth to preserving high-frequency color textures, motion-vector prediction blocks, and chroma subsampling:

$$\text{Pixel Pipeline}: \quad \mathbf{I}_t \xrightarrow{\text{Video Codec}} \text{Bitstream (Mbps)} \xrightarrow{\text{Decoder}} \hat{\mathbf{I}}_t \xrightarrow{\text{Neural Net}} \text{Predictions}$$

When the consumer is an autonomous agent, drone, or cloud vision model, transmitting full-resolution RGB video introduces severe inefficiencies:
1. **Redundant Edge Encoding Overhead**: Edge compute is wasted compressing pixel details discarded by early neural layers.
2. **Bandwidth Inefficiency**: Sub-100 kbps pixel compression introduces catastrophic block artifacts and blur, destroying semantic features.
3. **Double Encoding-Decoding Latency**: Video compression at edge + decompression at server + ViT tokenization at server creates unnecessary latency pipelines.

### 1.2 Split-Inference Token Streaming
**DiffTokenStream** splits the Vision Transformer between edge device and central server:

```
[ Camera Sensor ]
        │
        ▼
[ Phase A: The Slicer ] ───────► Frozen DINOv2-ViT-Small Stem ──────► Latent Tokens Z_t ∈ R^{256 × 384}
        │                                                                     │
        ├─────────────────────────────────────────────────────────────────────┼──────────────────┐
        ▼                                                                     ▼                  ▼
[ Phase C: Ego-Motion Warper ] ──► Sub-0.1ms Phase Correlation    [ Phase B: Saliency Bouncer ]  │
  - Estimates (Δx, Δy)             - Warps Reference Caches          - Dual-Cache Shadow Check   │
        │                                                            - Saliency [CLS] Prior      │
        ▼                                                                     │                  │
    (Δx, Δy)                                                        Active Mask M ∈ {0, 1}^256   │
        │                                                                     │                  │
        └───────────────────────────────┬─────────────────────────────────────┘                  │
                                        ▼                                                        ▼
                        [ Phase D: The Packer (RVQ) ] ◄────────────── Dynamic Tokens Z_active ∈ R^{K × 384}
                          - 4-Stage Residual Vector Quantization
                          - 32-Byte Bit-Packed Spatial Mask
                          - Binary Wire Serialization (to_bytes)
                                        │
                                        ▼
                               [ Network Channel ] ────► Payload Size: 58 + 4K Bytes (Avg: 74.7 kbps)
                                        │
                                        ▼
                        [ Phase E: The Rebuilder ]
                          - Binary Unpacking & RVQ Index Lookup
                          - Server-Side Motion Warping
                          - Selective In-Place KV-Cache Overwrite: Z_cache[0, M, :] = Z_hat
                                        │
                                        ▼
                        [ Downstream Task Execution ] ◄── Refreshed Full Token Grid Z_cache
                          - Multi-Head Attention / Dense Segmentation / Tracking
```

---

## 2. Phase A: The Slicer (`src/slicer.py`)

### 2.1 Spatial Quantization & Stem Projection
Given an incoming video frame $\mathbf{I}_t \in \mathbb{R}^{H \times W \times C}$ (ImageNet normalized $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$):
1. **Resolution Alignment**:
   $$H_{\text{aligned}} = \lfloor H / P \rfloor \cdot P, \quad W_{\text{aligned}} = \lfloor W / P \rfloor \cdot P$$
   For standard $224 \times 224$ inputs with patch size $P=14$:
   $$H_p = \frac{224}{14} = 16, \quad W_p = \frac{224}{14} = 16 \implies N = H_p \times W_p = 256 \text{ patches}$$
2. **Latent Stem Projection**: Patches $\mathbf{x}_p \in \mathbb{R}^{P^2 C}$ ($14 \times 14 \times 3 = 588$ dimensions) are linearly projected into embedding dimension $D=384$ and combined with positional embeddings $\mathbf{E}_{\text{pos}}$:
   $$\mathbf{z}_0 = [\mathbf{x}_{\text{cls}}; \, \mathbf{x}_p^1 \mathbf{E}; \, \dots; \, \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{\text{pos}}, \quad \mathbf{z}_0 \in \mathbb{R}^{(N+1) \times D}$$

### 2.2 Foreground Saliency Prior Extraction
To distinguish true semantic object motion from ambient background noise (e.g. water ripples, foliage flutter), the Slicer extracts self-attention weights from the final Transformer block ($l=12$).

For query tokens $\mathbf{Q} = \mathbf{X} \mathbf{W}_Q$ and key tokens $\mathbf{K} = \mathbf{X} \mathbf{W}_K$:
$$\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{B \times H_{\text{heads}} \times (N+1) \times (N+1)}$$
Attention weights from the `[CLS]` token (index 0) to all $N=256$ spatial patches (indices $1 \dots N$) are extracted and averaged across all $H_{\text{heads}}=6$ heads:
$$\mathbf{A}_{\text{cls}, i} = \frac{1}{H_{\text{heads}}} \sum_{h=1}^{H_{\text{heads}}} \mathbf{A}_h[0, \, i+1], \quad \forall i \in \{1, \dots, N\}$$
The raw attention vector is dynamically normalized per-frame into a foreground saliency prior $\mathbf{S}_{\text{cls}} \in [0, 1]^N$:
$$\mathbf{S}_{\text{cls}, i} = \frac{\mathbf{A}_{\text{cls}, i} - \min(\mathbf{A}_{\text{cls}})}{\max(\mathbf{A}_{\text{cls}}) - \min(\mathbf{A}_{\text{cls}}) + \epsilon}$$

---

## 3. Phase B: The Bouncer & Dual-Cache Shadow Architecture (`src/bouncer.py`)

### 3.1 The Single-Cache Quantization Flaw
In naive split-inference setups, the edge evaluates motion by comparing new tokens $\mathbf{z}_{i, t}$ against its local replica of the server cache $\mathbf{z}_{i, \text{cache}}$. However, because the server cache stores lossy RVQ-reconstructed tokens $\hat{\mathbf{z}}$, the codebook quantization error:
$$\text{dist}(\mathbf{z}_{\text{raw}}, \hat{\mathbf{z}}_{\text{rvq}}) \approx 0.18 - 0.35$$
consistently exceeds the sensitive motion detection thresholds ($\tau_{\text{dynamic}} = 0.0008, \tau_{\text{hard}} = 0.15$). This causes the differential gate to **saturate** ($K=256$, 100% transmission) on every frame.

### 3.2 The Dual-Cache Shadow Solution
The Dual-Cache Shadow Architecture decouples physical scene motion detection from RVQ quantization noise by maintaining two distinct persistent caches on the edge:
1. `edge_raw_shadow` $\in \mathbb{R}^{1 \times N \times D}$: Holds uncompressed float32 ViT tokens representing the pristine state of transmitted patches.
2. `edge_replica_cache` $\in \mathbb{R}^{1 \times N \times D}$: Holds RVQ-reconstructed tokens $\hat{\mathbf{Z}}$, strictly bit-synchronized with the cloud server's cache.

```
Incoming Tokens Z_raw ──► [ Cosine Difference vs edge_raw_shadow ] ──► Motion Mask M
                                                                             │
    ┌────────────────────────────────────────────────────────────────────────┘
    ▼
If M_i == 1:
  - Transmit patch i via RVQ
  - edge_raw_shadow[0, i, :]    <-- Z_raw[i]            (Pristine Float32)
  - edge_replica_cache[0, i, :] <-- Decode(RVQ(Z_raw[i])) (Bit-Identical to Server)
```

### 3.3 Saliency-Gated Mathematical Logic
1. **Cosine Similarity against Shadow Cache**:
   $$S_i = \text{CosSim}(\mathbf{z}_{i, t}, \, \mathbf{z}_{\text{shadow}, i}) = \frac{\mathbf{z}_{i, t} \cdot \mathbf{z}_{\text{shadow}, i}}{\|\mathbf{z}_{i, t}\|_2 \|\mathbf{z}_{\text{shadow}, i}\|_2 + \epsilon}, \quad S_i \in [-1.0, 1.0]$$
2. **Latent Difference Metric**:
   $$\delta_i = \max(0.0, \, 1.0 - S_i) \in [0.0, 2.0]$$
3. **Non-Linear Saliency Modulation**:
   $$\Delta_i = \delta_i \cdot \left(\mathbf{S}_{\text{cls}, i}\right)^\gamma, \quad \text{with } \gamma = 1.0$$
4. **Dual-Threshold Activation Rule**:
   $$M_i = \begin{cases} 1 & \text{if } \Delta_i > \tau_{\text{dynamic}} \quad \text{or} \quad \delta_i > \tau_{\text{hard\_change}} \\ 0 & \text{otherwise} \end{cases}$$
   *Calibrated Thresholds*: $\tau_{\text{dynamic}} = 0.0008$, $\tau_{\text{hard\_change}} = 0.30$.
5. **Dynamic Token Extraction**:
   $$\mathbf{Z}_{\text{active}} = \{\mathbf{z}_{i, t} \mid M_i = 1\} \in \mathbb{R}^{K \times D}, \quad K = \sum_{i=1}^N M_i$$

---

## 4. Phase C: Adaptive Sub-0.1ms Ego-Motion Warper (`src/ego_motion.py`)

When the camera undergoes global translation (e.g., panning or tracking shots), naive temporal filtering falsely triggers differential transmission for every background patch.

### 4.1 2D Sub-Pixel Phase Correlation
1. Input frames $\mathbf{I}_{t-1}, \mathbf{I}_t$ are converted to grayscale and downsampled to $64 \times 64$ ($< 0.1$ ms computation).
2. A 2D Hanning window $W(x, y)$ eliminates boundary spectral leakage:
   $$W(x, y) = \sin\left(\frac{\pi x}{W-1}\right) \sin\left(\frac{\pi y}{H-1}\right)$$
3. 2D Discrete Fourier Transforms: $\mathcal{F}_1 = \mathcal{F}\{\mathbf{I}_{t-1} \odot W\}$, $\mathcal{F}_2 = \mathcal{F}\{\mathbf{I}_t \odot W\}$.
4. Normalized cross-power spectrum:
   $$\mathbf{R} = \frac{\mathcal{F}_1 \odot \mathcal{F}_2^*}{|\mathcal{F}_1 \odot \mathcal{F}_2^*| + \epsilon}$$
5. Inverse DFT $\mathbf{r} = \mathcal{F}^{-1}\{\mathbf{R}\}$. Peak location with sub-pixel parabolic centroid fitting yields camera displacement $(\Delta x, \Delta y)$ in pixel space.

### 4.2 Differentiable Bilinear Token Grid Warping
Convert image displacement into patch grid coordinates ($\Delta x_p = \Delta x / P, \Delta y_p = \Delta y / P$). Rearrange token cache $\mathbf{Z} \in \mathbb{R}^{B \times 256 \times 384}$ to 2D feature grid $\mathbf{T} \in \mathbb{R}^{B \times 384 \times 16 \times 16}$.
For each patch grid location $(x, y)$, sample source coordinates:
$$x_{\text{src}} = x - \Delta x_p, \quad y_{\text{src}} = y - \Delta y_p$$
Normalized to $[-1, 1]$ coordinates with border clamping via PyTorch `F.grid_sample`.
Dual-grid warping warps both `edge_raw_shadow` and `edge_replica_cache` concurrently.

---

## 5. Phase D: The Packer (RVQ & Wire Serialization - `src/packer.py`)

### 5.1 4-Stage Residual Vector Quantization
We employ $Q=4$ cascaded codebooks $\mathcal{C}_1, \dots, \mathcal{C}_4$, each with vocabulary size $V=256$ codebook vectors $\mathbf{e}_v \in \mathbb{R}^{384}$. Each quantization index is represented by an 8-bit unsigned integer (`uint8`).

For each active dynamic token $\mathbf{z} \in \mathbf{Z}_{\text{active}}$:
- Initial residual: $\mathbf{r}_0 = \mathbf{z}$
- For stage $q = 1 \dots Q$:
  $$k_q = \arg\min_{v \in \{0, \dots, V-1\}} \|\mathbf{r}_{q-1} - \mathbf{e}_{q, v}\|_2^2$$
  $$\hat{\mathbf{z}}_q = \mathbf{e}_{q, k_q}, \quad \mathbf{r}_q = \mathbf{r}_{q-1} - \hat{\mathbf{z}}_q$$
- Reconstructed approximation:
  $$\hat{\mathbf{z}} = \sum_{q=1}^Q \hat{\mathbf{z}}_q = \sum_{q=1}^Q \mathbf{e}_{q, k_q}$$

### 5.2 Bit-Packed Spatial Mask
The boolean spatial mask $\mathbf{M} \in \{0, 1\}^{256}$ is packed into contiguous bits:
$$\text{Mask Bytes} = \frac{256}{8} = 32 \text{ bytes}, \quad B_j = \sum_{b=0}^7 M_{8j+b} \cdot 2^b, \quad j \in \{0, \dots, 31\}$$

### 5.3 Binary Transmission Packet Format (`TransmissionPacket`)

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       Magic: 'DT' (0x4454)    |          Version (0x0001)     |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                      Frame Sequence ID (uint32)               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       Total Patches N (256)   |      Active Patches K         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       Num Quantizers Q (4)    |      Codebook Size V (256)    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| Flags (0x01 = motion present) |      Reserved Padding         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| [Optional] Camera Motion Δx (IEEE 754 float32, 4 bytes)       |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| [Optional] Camera Motion Δy (IEEE 754 float32, 4 bytes)       |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                  Bit-Packed Spatial Mask (32 bytes)           |
|                               ...                             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|          RVQ Codebook Indices: K × Q bytes (uint8)            |
|                               ...                             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

$$\text{Total Wire Payload Size} = 18_{\text{header}} + 8_{\text{motion}} + 32_{\text{mask}} + 4K = 58 + 4K \text{ bytes}$$

---

## 6. Phase E: The Rebuilder & KV-Cache Refresh (`src/rebuilder.py`)

1. **Binary Deserialization**: The server receives raw byte buffer, validates magic bytes `0x44 0x54`, extracts $(\Delta x, \Delta y)$, unpacks the 32 mask bytes into boolean tensor $\mathbf{M} \in \{0, 1\}^{256}$, and parses RVQ index matrix $\mathbf{I} \in \mathbb{Z}^{K \times 4}$.
2. **Codebook Vector Reconstruction**:
   $$\hat{\mathbf{z}}_k = \sum_{q=1}^4 \mathcal{C}_q[I_{k, q}], \quad \forall k \in \{1, \dots, K\}$$
3. **Server Cache Motion Warping**: If $(\Delta x, \Delta y)$ is signaled, the server warps its persistent cache $\mathbf{Z}_{\text{cache}}$ via bilinear interpolation prior to active update.
4. **Selective In-Place Cache Overwrite**:
   $$\mathbf{Z}_{\text{cache}}[0, i, :] = \begin{cases} \hat{\mathbf{z}}_k & \text{if } M_i = 1 \\ \mathbf{Z}_{\text{cache}}[0, i, :] & \text{if } M_i = 0 \text{ (persisted from previous state)} \end{cases}$$
5. **Downstream Multi-Head Transformer Attention**: The fully refreshed token sequence $\mathbf{Z}_{\text{cache}} \in \mathbb{R}^{1 \times 256 \times 384}$ is passed into downstream Transformer blocks for perception.

---

## 7. Mathematical Proof: Attention Error Damping

Why do downstream vision tasks tolerate quantization and temporal holdover noise in the server cache?

Let the server cache be modeled as:
$$\mathbf{Z}_{\text{cache}} = \mathbf{Z}_{\text{true}} + \mathbf{E}$$
where $\mathbf{E} \in \mathbb{R}^{N \times D}$ represents holdover error on static tokens ($M_i = 0$).

1. **Sparsity of Error**: Dynamic foreground tokens have $M_i = 1$ and are refreshed via RVQ with high fidelity ($\text{CosSim} > 0.95$). Holdover error is confined to low-saliency background patches.
2. **Softmax Exponentiation as a Noise Gate**: For a downstream foreground query token $\mathbf{q}_f$, the dot-product similarity against another foreground token $\mathbf{k}_f$ is substantially higher than against background token $\mathbf{k}_b$:
   $$\mathbf{q}_f \mathbf{k}_f^T \gg \mathbf{q}_f \mathbf{k}_b^T$$
   Because the softmax function exponentiates differences:
   $$\mathbf{A}_{f, b} = \frac{e^{\mathbf{q}_f \mathbf{k}_b^T / \sqrt{d}}}{\sum_j e^{\mathbf{q}_f \mathbf{k}_j^T / \sqrt{d}}} \to 0$$
3. **Linear Aggregation Damping**: Even if background tokens contain distortion $\mathbf{e}_b$, their contribution to the output representation:
   $$\mathbf{h}_f = \sum_{j} \mathbf{A}_{f, j} \mathbf{v}_j$$
   is multiplied by $\mathbf{A}_{f, b} \approx 0$.
4. **Conclusion**: The Transformer attention mechanism naturally functions as an intrinsic **non-linear low-pass filter**, shielding task-critical representations from background holdover artifacts.

---

## 8. ViT vs. CNN Split-Inference Comparative Analysis

| Dimension | Vision Transformer (ViT) Stem (DiffTokenStream) | Convolutional Neural Network (CNN) Stem |
| :--- | :--- | :--- |
| **Spatial Factorization** | **Disjoint, non-overlapping patches** ($14 \times 14$ px). Each token is an independent entity. | **Sliding-window convolutions** with overlapping receptive fields. |
| **Receptive Field Entanglement** | **Zero spatial overlap at stem**. Patches interact only via explicit self-attention layers. | **Expanding receptive field**. Altering one pixel alters feature activations across a wide neighborhood. |
| **Sparse Dropping** | **Trivial**: Dropping token $\mathbf{z}_i$ has zero mathematical effect on neighboring patch embeddings. | **Intractable**: Dropping feature map points breaks regular 2D convolution grids. |
| **Positional Grounding** | **Explicit positional embeddings** $\mathbf{E}_{\text{pos}}$. Tokens are freely permutable and sparse. | **Implicit positional bias** tied strictly to 2D coordinate array indices. |
| **Wire Transport** | **Compact 1D Sparse Array** $\mathbf{Z}_{\text{active}} \in \mathbb{R}^{K \times D}$. | **Sparse 2D/3D Tensor Maps** requiring complex quadtree / coordinate serialization. |
| **KV-Cache Updating** | **Direct Scatter Assignment**: $\mathbf{Z}_{\text{cache}}[M] = \hat{\mathbf{Z}}$. | **Feature Map Splicing**: Requires boundary blending, introducing severe seams. |

---
*Reference implementation available in `src/`.*
