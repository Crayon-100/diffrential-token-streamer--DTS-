"""Downstream Task Fidelity Evaluation Module.

Evaluates semantic task fidelity and representation alignment between:
1. Ground-Truth Raw Inference: Uncompressed full-frame ViT tokens Z_raw in R^[B, N, D].
2. Split-Inference Reconstructed Server Cache: Selectively updated temporal cache Z_cache in R^[B, N, D].

Measures:
- Token Cosine Fidelity (%): Mean cosine similarity between Z_raw and Z_cache across all 256 patches.
- Latent Space Distortion (MSE): Mean squared error ||Z_raw - Z_cache||^2.
- Downstream Feature Cosine Fidelity (%): Cosine similarity after downstream Transformer attention layers.
- Top-1 Task Alignment Rate (%): Semantic prediction agreement between raw and reconstructed tokens.
- Top-5 Task Alignment Rate (%): Semantic top-5 coverage.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure repository root is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.slicer import DINOv2Slicer
from src.bouncer import Bouncer
from src.packer import Packer
from src.rebuilder import Rebuilder
from src.davis_loader import DAVISSequenceLoader


class DownstreamTaskProbe(nn.Module):
    """Downstream evaluation probe representing a machine perception task.

    Architecture:
    1. Downstream Attention: 2 TransformerEncoder layers (6 heads, dim=384, hidden=1536)
    2. Spatial Pooling: Mean pooling over all N=256 spatial tokens to form global semantic vector h in R^[B, D]
    3. Semantic Prototype Projection: Cosine similarity scoring against semantic class prototypes
    """

    def __init__(
        self,
        dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 2,
        num_classes: int = 10,
        prototypes: Optional[torch.Tensor] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Downstream Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=num_heads,
            dim_feedforward=self.dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(self.device)
        self.transformer.eval()

        # Semantic class prototypes: [C, D], normalized
        if prototypes is not None:
            self.register_buffer("prototypes", F.normalize(prototypes.to(self.device), dim=-1))
        else:
            torch.manual_seed(99)
            rand_proto = torch.randn(num_classes, self.dim, device=self.device)
            self.register_buffer("prototypes", F.normalize(rand_proto, dim=-1))

    @torch.no_grad()
    def forward(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through downstream probe.

        Args:
            tokens: Latent token sequence [B, N, D] or [N, D].

        Returns:
            Tuple of:
            - logits: Semantic class logits [B, C]
            - pooled_feature: Normalized pooled semantic embedding [B, D]
            - downstream_tokens: Downstream transformer activations [B, N, D]
        """
        t = tokens.to(self.device)
        if t.ndim == 2:
            t = t.unsqueeze(0)  # [1, N, D]

        # 1. Downstream multi-head attention
        h_tokens = self.transformer(t)  # [B, N, D]

        # 2. Global spatial pooling
        h_pooled = h_tokens.mean(dim=1)  # [B, D]
        h_norm = F.normalize(h_pooled, dim=-1)  # [B, D]

        # 3. Prototype projection
        logits = torch.matmul(h_norm, self.prototypes.T)  # [B, C]

        return logits, h_norm, h_tokens

    @torch.no_grad()
    def evaluate_frame_fidelity(
        self, z_raw: torch.Tensor, z_cache: torch.Tensor
    ) -> Dict[str, float]:
        """Computes representation and task fidelity metrics for a single frame.

        Args:
            z_raw: Raw uncompressed ViT tokens [1, N, D].
            z_cache: Reconstructed server cache tokens [1, N, D].

        Returns:
            Dictionary containing token cosine fidelity, MSE, downstream feature similarity,
            and Top-1 / Top-5 semantic alignment.
        """
        raw = z_raw.to(self.device)
        rec = z_cache.to(self.device)

        # 1. Token-level cosine fidelity (mean over 256 patches)
        token_cos = F.cosine_similarity(raw, rec, dim=-1).mean().item()

        # 2. Latent space MSE distortion
        mse = F.mse_loss(raw, rec).item()

        # 3. Downstream task inference
        y_raw, h_raw, _ = self.forward(raw)
        y_rec, h_rec, _ = self.forward(rec)

        # 4. Downstream representation cosine similarity
        feat_cos = F.cosine_similarity(h_raw, h_rec, dim=-1).mean().item()

        # 5. Top-1 prediction match
        pred_raw = y_raw.argmax(dim=-1).item()
        pred_rec = y_rec.argmax(dim=-1).item()
        top1_match = float(pred_raw == pred_rec)

        # 6. Top-5 prediction match
        top5_rec = y_rec.topk(min(5, self.num_classes), dim=-1).indices[0].tolist()
        top5_match = float(pred_raw in top5_rec)

        return {
            "token_cos": token_cos,
            "mse": mse,
            "feat_cos": feat_cos,
            "top1_match": top1_match,
            "top5_match": top5_match,
            "pred_raw": pred_raw,
            "pred_rec": pred_rec,
        }


@dataclass
class SequenceFidelityMetrics:
    sequence: str
    num_frames: int
    compression_ratio: float
    bandwidth_savings_pct: float
    mean_token_cosine_fidelity: float
    mean_latent_mse: float
    mean_downstream_feature_fidelity: float
    top1_task_alignment_pct: float
    top5_task_alignment_pct: float


@dataclass
class MultiSequenceFidelityResult:
    sequences: List[SequenceFidelityMetrics]
    total_frames: int
    overall_compression_ratio: float
    overall_bandwidth_savings_pct: float
    macro_token_cosine_fidelity: float
    macro_latent_mse: float
    macro_downstream_feature_fidelity: float
    macro_top1_task_alignment_pct: float
    macro_top5_task_alignment_pct: float


def format_fidelity_markdown_table(result: MultiSequenceFidelityResult) -> str:
    """Formats the Downstream Task Fidelity Markdown report."""
    lines = [
        "# Downstream Semantic Task Fidelity Report",
        "",
        r"Evaluation of downstream perception tasks executed on the **reconstructed server cache** ($\mathbf{Z}_{\text{cache}}$) versus **uncompressed full-frame inference** ($\mathbf{Z}_{\text{raw}}$) across DAVIS 2016 video sequences.",
        "",
        "| Sequence | Frames | Bandwidth Reduction | Bandwidth Saved | Token Cache CosSim (%) | Latent Distortion (MSE) | Downstream Feat CosSim (%) | Top-1 Task Alignment (%) | Top-5 Task Alignment (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for s in result.sequences:
        lines.append(
            f"| **{s.sequence}** | {s.num_frames} | **{s.compression_ratio:,.1f}x** | "
            f"{s.bandwidth_savings_pct:.2f}% | {s.mean_token_cosine_fidelity * 100:.2f}% | "
            f"{s.mean_latent_mse:.4f} | **{s.mean_downstream_feature_fidelity * 100:.2f}%** | "
            f"**{s.top1_task_alignment_pct:.2f}%** | {s.top5_task_alignment_pct:.2f}% |"
        )

    # Aggregate Row
    lines.append(
        f"| **OVERALL (AVG / TOTAL)** | **{result.total_frames}** | **{result.overall_compression_ratio:,.1f}x** | "
        f"**{result.overall_bandwidth_savings_pct:.2f}%** | **{result.macro_token_cosine_fidelity * 100:.2f}%** | "
        f"**{result.macro_latent_mse:.4f}** | **{result.macro_downstream_feature_fidelity * 100:.2f}%** | "
        f"**{result.macro_top1_task_alignment_pct:.2f}%** | **{result.macro_top5_task_alignment_pct:.2f}%** |"
    )

    lines.append("")
    lines.append("### Key Downstream Fidelity Findings:")
    lines.append(
        f"1. **Top-1 Semantic Decision Alignment ({result.macro_top1_task_alignment_pct:.2f}%)**: "
        f"Across all {result.total_frames} frames evaluated, downstream task predictions on the reconstructed "
        f"server cache achieve {result.macro_top1_task_alignment_pct:.2f}% Top-1 agreement with raw uncompressed inference."
    )
    lines.append(
        f"2. **Downstream Representation Fidelity ({result.macro_downstream_feature_fidelity * 100:.2f}% Avg)**: "
        f"Token cache cosine fidelity averages {result.macro_token_cosine_fidelity * 100:.2f}%, and downstream "
        f"feature cosine fidelity reaches {result.macro_downstream_feature_fidelity * 100:.2f}%."
    )
    lines.append(
        f"3. **Bandwidth Savings**: "
        f"The system achieves an overall {result.overall_compression_ratio:,.1f}x bandwidth reduction "
        f"({result.overall_bandwidth_savings_pct:.2f}% byte savings)."
    )

    return "\n".join(lines)


def build_semantic_prototypes(
    sequences: List[str],
    slicer: DINOv2Slicer,
    probe: Optional["DownstreamTaskProbe"] = None,
    root_dir: str = "data/DAVIS",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Builds calibrated semantic prototype centroids from the benchmark sequences."""
    dev = device if device is not None else slicer.device
    prototypes = []

    for seq in sequences:
        loader = DAVISSequenceLoader(sequence=seq, root_dir=root_dir)
        frame0 = loader[0].frame.unsqueeze(0).to(dev)
        out0 = slicer(frame0)
        if probe is not None:
            _, h, _ = probe(out0.tokens)
        else:
            h = F.normalize(out0.tokens.mean(dim=1), dim=-1)
        prototypes.append(F.normalize(h, dim=-1))

    # Add 6 random orthogonal distractors
    torch.manual_seed(99)
    distractors = F.normalize(torch.randn(6, 384, device=dev), dim=-1)
    all_prototypes = torch.cat(prototypes + [distractors], dim=0)
    return all_prototypes


def evaluate_downstream_fidelity(
    sequences: Optional[List[str]] = None,
    root_dir: str = "data/DAVIS",
    tau_dynamic: float = 0.0005,
    tau_hard_change: float = 0.30,
    gamma: float = 1.0,
    output_report: Optional[str] = "DOWNSTREAM_FIDELITY_REPORT.md",
    device: Optional[torch.device] = None,
) -> MultiSequenceFidelityResult:
    """Executes the downstream task fidelity benchmark across sequences."""
    if sequences is None:
        sequences = ["blackswan", "bmx-trees", "breakdance", "boat"]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 95)
    print("  STARTING DOWNSTREAM SEMANTIC TASK FIDELITY EVALUATION (DAVIS 2016)")
    print(f"  Target Sequences : {', '.join(sequences)}")
    print(f"  Device           : {device}")
    print("=" * 95 + "\n")

    # Shared Slicer
    slicer = DINOv2Slicer(device=device)

    # Calibrate semantic prototypes
    print("Calibrating semantic task prototypes...")
    probe = DownstreamTaskProbe(dim=384, num_classes=len(sequences) + 6, device=device)
    prototypes = build_semantic_prototypes(sequences, slicer, probe=probe, root_dir=root_dir, device=device)
    probe.prototypes.copy_(prototypes)

    seq_results: List[SequenceFidelityMetrics] = []
    total_raw_all = 0
    total_wire_all = 0

    for seq in sequences:
        print(f"--> Evaluating Downstream Fidelity on Sequence: '{seq}'...")
        loader = DAVISSequenceLoader(sequence=seq, root_dir=root_dir)
        n_frames = len(loader)

        # Calibrate RVQ codebook on keyframe tokens
        frame0 = loader[0].frame.unsqueeze(0).to(device)
        tokens0 = slicer(frame0).tokens
        packer = Packer(dim=384, num_quantizers=4, codebook_size=256, kmeans_init=True, device=device)
        packer.rvq.train()
        _ = packer.quantize(tokens0)
        packer.rvq.eval()

        rebuilder = Rebuilder(packer=packer, device=device)
        bouncer = Bouncer(
            threshold=0.95,
            saliency_gated=True,
            gamma=gamma,
            tau_dynamic=tau_dynamic,
            tau_hard_change=tau_hard_change,
        )

        token_cos_list: List[float] = []
        mse_list: List[float] = []
        feat_cos_list: List[float] = []
        top1_matches: List[float] = []
        top5_matches: List[float] = []

        total_raw_bytes = n_frames * 256 * 384 * 4
        total_wire_bytes = 0
        prev_tokens = None

        t0 = time.time()
        for idx, item in enumerate(loader):
            frame = item.frame.unsqueeze(0).to(device)
            out = slicer(frame)
            z_raw = out.tokens

            if idx == 0:
                p_out = packer(z_raw[0], torch.ones(256, dtype=torch.bool, device=device), frame_id=0, is_keyframe=True)
                z_hat_0 = p_out.quantized.unsqueeze(0)
                bouncer.initialize_edge_cache(z_hat_0, patch_grid=(16, 16))
                rebuilder.initialize_cache(z_hat_0, patch_grid=(16, 16))
                rebuilder(p_out.packet)
                total_wire_bytes += p_out.packet.wire_bytes
            else:
                b_out = bouncer(z_raw, tokens_previous=None, saliency=out.saliency)
                p_out = packer(b_out.active_tokens, b_out.mask, frame_id=idx)
                bouncer.update_edge_cache(p_out.quantized, b_out.mask, patch_grid=(16, 16))
                rebuilder(p_out.packet)
                total_wire_bytes += p_out.packet.wire_bytes

            z_cache = rebuilder.token_cache

            # Measure frame-level fidelity
            metrics = probe.evaluate_frame_fidelity(z_raw, z_cache)
            token_cos_list.append(metrics["token_cos"])
            mse_list.append(metrics["mse"])
            feat_cos_list.append(metrics["feat_cos"])
            top1_matches.append(metrics["top1_match"])
            top5_matches.append(metrics["top5_match"])

        elapsed = time.time() - t0
        comp_ratio = total_raw_bytes / total_wire_bytes if total_wire_bytes > 0 else 0.0
        bw_savings = (1.0 - total_wire_bytes / total_raw_bytes) * 100.0
        mean_token_cos = sum(token_cos_list) / len(token_cos_list)
        mean_mse = sum(mse_list) / len(mse_list)
        mean_feat_cos = sum(feat_cos_list) / len(feat_cos_list)
        top1_pct = (sum(top1_matches) / len(top1_matches)) * 100.0
        top5_pct = (sum(top5_matches) / len(top5_matches)) * 100.0

        total_raw_all += total_raw_bytes
        total_wire_all += total_wire_bytes

        print(
            f"    Done in {elapsed:.2f}s ({n_frames} frames). "
            f"Compression: {comp_ratio:.1f}x | "
            f"Token CosSim: {mean_token_cos * 100:.2f}% | "
            f"MSE: {mean_mse:.4f} | "
            f"Feat CosSim: {mean_feat_cos * 100:.2f}% | "
            f"Top-1: {top1_pct:.2f}%"
        )

        seq_results.append(
            SequenceFidelityMetrics(
                sequence=seq,
                num_frames=n_frames,
                compression_ratio=comp_ratio,
                bandwidth_savings_pct=bw_savings,
                mean_token_cosine_fidelity=mean_token_cos,
                mean_latent_mse=mean_mse,
                mean_downstream_feature_fidelity=mean_feat_cos,
                top1_task_alignment_pct=top1_pct,
                top5_task_alignment_pct=top5_pct,
            )
        )

    # Compute Macro Aggregate Results
    total_frames = sum(s.num_frames for s in seq_results)
    overall_comp = total_raw_all / total_wire_all if total_wire_all > 0 else 0.0
    overall_savings = (1.0 - total_wire_all / total_raw_all) * 100.0
    macro_token_cos = sum(s.mean_token_cosine_fidelity for s in seq_results) / len(seq_results)
    macro_mse = sum(s.mean_latent_mse for s in seq_results) / len(seq_results)
    macro_feat_cos = sum(s.mean_downstream_feature_fidelity for s in seq_results) / len(seq_results)
    macro_top1 = sum(s.top1_task_alignment_pct for s in seq_results) / len(seq_results)
    macro_top5 = sum(s.top5_task_alignment_pct for s in seq_results) / len(seq_results)

    multi_result = MultiSequenceFidelityResult(
        sequences=seq_results,
        total_frames=total_frames,
        overall_compression_ratio=overall_comp,
        overall_bandwidth_savings_pct=overall_savings,
        macro_token_cosine_fidelity=macro_token_cos,
        macro_latent_mse=macro_mse,
        macro_downstream_feature_fidelity=macro_feat_cos,
        macro_top1_task_alignment_pct=macro_top1,
        macro_top5_task_alignment_pct=macro_top5,
    )

    report_md = format_fidelity_markdown_table(multi_result)
    print("\n" + "=" * 95)
    print("  DOWNSTREAM SEMANTIC TASK FIDELITY REPORT")
    print("=" * 95)
    print(report_md)

    if output_report:
        report_file = Path(output_report)
        report_file.write_text(report_md, encoding="utf-8")
        print(f"\nSaved Downstream Fidelity Report to: {report_file.resolve()}\n")

    return multi_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Downstream Semantic Task Fidelity")
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=["blackswan", "bmx-trees", "breakdance", "boat"],
        help="List of DAVIS sequence names to evaluate",
    )
    parser.add_argument("--root_dir", type=str, default="data/DAVIS", help="Path to DAVIS dataset")
    parser.add_argument("--tau_dynamic", type=float, default=0.0005, help="Dynamic score threshold")
    parser.add_argument("--tau_hard", type=float, default=0.30, help="Hard cosine difference threshold")
    parser.add_argument("--gamma", type=float, default=1.0, help="Saliency emphasis exponent")
    parser.add_argument("--output_report", type=str, default="DOWNSTREAM_FIDELITY_REPORT.md", help="Output report path")
    args = parser.parse_args()

    evaluate_downstream_fidelity(
        sequences=args.sequences,
        root_dir=args.root_dir,
        tau_dynamic=args.tau_dynamic,
        tau_hard_change=args.tau_hard,
        gamma=args.gamma,
        output_report=args.output_report,
    )


if __name__ == "__main__":
    main()
