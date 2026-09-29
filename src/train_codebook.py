"""Independent Codebook Training Module for Residual Vector Quantization (RVQ).

Trains a frozen, versioned 4-stage RVQ codebook on disjoint DAVIS 2016 training sequences.
Strictly excludes all test sequences ('blackswan', 'bmx-trees', 'breakdance', 'boat')
to prevent data leakage.
"""

import argparse
import hashlib
from pathlib import Path
import sys
import time
from typing import List, Optional, Set, Tuple

# Ensure repository root is on sys.path
root_dir_path = str(Path(__file__).resolve().parent.parent)
if root_dir_path not in sys.path:
    sys.path.insert(0, root_dir_path)

import torch
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ

from src.davis_loader import DAVISSequenceLoader
from src.slicer import DINOv2Slicer

# Sequences strictly reserved for held-out evaluation
EXCLUDED_TEST_SEQUENCES: Set[str] = {
    "blackswan",
    "bmx-trees",
    "breakdance",
    "boat",
}

DEFAULT_OUTPUT_MODEL_PATH = "models/rvq_codebook_davis_train.pt"


def get_training_sequences(
    split_file: str = "data/DAVIS/ImageSets/480p/train.txt",
    root_dir: str = "data/DAVIS",
    excluded: Optional[Set[str]] = None,
) -> List[str]:
    """Parses DAVIS training sequences, strictly excluding held-out test sequences."""
    if excluded is None:
        excluded = EXCLUDED_TEST_SEQUENCES

    path = Path(split_file)
    if not path.exists():
        raise FileNotFoundError(f"Training split file not found: {split_file}")

    lines = path.read_text().splitlines()
    raw_seqs = sorted(list(set(line.split()[0].split("/")[-2] for line in lines if line.strip())))

    valid_seqs = []
    images_root = Path(root_dir) / "JPEGImages" / "480p"
    for seq in raw_seqs:
        if seq not in excluded and (images_root / seq).exists():
            valid_seqs.append(seq)

    return sorted(valid_seqs)


def compute_codebook_sha256_and_id(weights_path: str) -> Tuple[str, int]:
    """Computes sha256 checksum and 4-byte integer codebook ID from saved weights file."""
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Codebook weights not found: {weights_path}")

    data = path.read_bytes()
    sha256_hash = hashlib.sha256(data).hexdigest()
    # First 4 bytes as 32-bit unsigned integer ID for transmission packet header
    codebook_id = int(sha256_hash[:8], 16)
    return sha256_hash, codebook_id


def train_rvq_codebook(
    target_tokens: int = 60000,
    dim: int = 384,
    num_quantizers: int = 4,
    codebook_size: int = 256,
    kmeans_iters: int = 15,
    split_file: str = "data/DAVIS/ImageSets/480p/train.txt",
    root_dir: str = "data/DAVIS",
    output_path: str = DEFAULT_OUTPUT_MODEL_PATH,
    device: Optional[torch.device] = None,
) -> Tuple[ResidualVQ, str, int]:
    """Extracts DINOv2 tokens from training sequences and trains an independent RVQ codebook."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_seqs = get_training_sequences(split_file=split_file, root_dir=root_dir)
    print("\n" + "=" * 80)
    print("  INDEPENDENT CODEBOOK TRAINING: DAVIS TRAIN SPLIT")
    print(f"  Training Sequences Count : {len(train_seqs)}")
    print(f"  Excluded Test Sequences  : {sorted(list(EXCLUDED_TEST_SEQUENCES))}")
    print(f"  Target Tokens Count      : ~{target_tokens:,}")
    print(f"  Device                   : {device}")
    print("=" * 80 + "\n")

    # Initialize Slicer
    slicer = DINOv2Slicer(model_name="dinov2_vits14", device=device)

    # Calculate sampling stride to achieve target tokens evenly across all sequences
    tokens_per_frame = 256
    target_frames = max(1, target_tokens // tokens_per_frame)
    frames_per_seq = max(1, target_frames // len(train_seqs))

    print(f"Extracting ~{frames_per_seq} frames per training sequence...")
    collected_tokens: List[torch.Tensor] = []
    t_start = time.time()

    for seq in train_seqs:
        loader = DAVISSequenceLoader(sequence=seq, root_dir=root_dir)
        n_available = len(loader)
        if n_available == 0:
            continue

        stride = max(1, n_available // frames_per_seq)
        sample_indices = list(range(0, n_available, stride))[:frames_per_seq]

        for idx in sample_indices:
            frame_tensor = loader[idx].frame.unsqueeze(0).to(device)
            with torch.no_grad():
                out = slicer(frame_tensor)
                # DINOv2 tokens shape [1, 256, 384]
                collected_tokens.append(out.tokens.cpu())

    all_tokens = torch.cat(collected_tokens, dim=1)  # [1, Total_Tokens, Dim]
    total_extracted = all_tokens.shape[1]
    extraction_time = time.time() - t_start
    print(f"Extraction complete: {total_extracted:,} tokens collected across {len(collected_tokens)} frames in {extraction_time:.2f}s.")

    # Normalize tokens for unit-sphere codebook clustering
    print("Normalizing token representations...")
    norm_tokens = F.normalize(all_tokens.to(device), dim=-1)

    # Initialize Residual Vector Quantizer with k-means initialization
    print(f"Fitting {num_quantizers}-stage RVQ (codebook_size={codebook_size}, dim={dim}, iters={kmeans_iters})...")
    rvq = ResidualVQ(
        dim=dim,
        num_quantizers=num_quantizers,
        codebook_size=codebook_size,
        kmeans_init=True,
        kmeans_iters=kmeans_iters,
    ).to(device)

    # Train k-means codebooks on normalized training tokens
    t_train = time.time()
    rvq.train()
    with torch.no_grad():
        quantized, indices, commit_loss = rvq(norm_tokens)
    rvq.eval()
    fit_time = time.time() - t_train

    # Verification on training set
    with torch.no_grad():
        cos_train = F.cosine_similarity(norm_tokens, quantized, dim=-1).mean().item()
        mse_train = F.mse_loss(norm_tokens, quantized).item()

    print(f"Codebook fitting finished in {fit_time:.2f}s.")
    print(f"Training Token Cosine Fidelity: {cos_train * 100:.2f}% | Latent MSE: {mse_train:.6f}")

    # Ensure output directory exists
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Save state dict and metadata
    save_dict = {
        "state_dict": rvq.state_dict(),
        "dim": dim,
        "num_quantizers": num_quantizers,
        "codebook_size": codebook_size,
        "total_tokens_trained": total_extracted,
        "train_sequences": train_seqs,
        "train_cosine_fidelity": cos_train,
        "train_latent_mse": mse_train,
    }
    torch.save(save_dict, out_file)
    print(f"Saved trained codebook to: {out_file.resolve()}")

    sha256_hash, codebook_id = compute_codebook_sha256_and_id(str(out_file))
    print(f"Codebook SHA256 Checksum : {sha256_hash}")
    print(f"Codebook Protocol ID     : 0x{codebook_id:08X} ({codebook_id})")

    return rvq, sha256_hash, codebook_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Independent RVQ Codebook on DAVIS 2016 Train Split")
    parser.add_argument("--target_tokens", type=int, default=60000, help="Target number of tokens to train on")
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_MODEL_PATH, help="Path to save weights")
    parser.add_argument("--kmeans_iters", type=int, default=15, help="Number of k-means iterations")
    parser.add_argument("--root_dir", type=str, default="data/DAVIS", help="Root directory of DAVIS dataset")
    args = parser.parse_args()

    train_rvq_codebook(
        target_tokens=args.target_tokens,
        output_path=args.output_path,
        kmeans_iters=args.kmeans_iters,
        root_dir=args.root_dir,
    )


if __name__ == "__main__":
    main()
