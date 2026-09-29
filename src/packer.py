"""Phase C: The Packer (Residual Vector Quantization & Binary Wire Serialization)

Quantizes continuous active tokens into discrete integer indices and serializes them
into a compact binary wire payload for low-bandwidth transmission.
"""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import struct
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from vector_quantize_pytorch import ResidualVQ


# Binary Wire Protocol Constants
MAGIC_BYTES = b"DT"
PROTOCOL_VERSION = 1
FLAG_KEYFRAME = 0x01
FLAG_MOTION = 0x02
FLAG_TWO_BYTE_INDICES = 0x04
DEFAULT_CODEBOOK_PATH = "models/rvq_codebook_davis_train.pt"
DEFAULT_CODEBOOK_ID = 0xAE15D7B8  # SHA256 prefix of models/rvq_codebook_davis_train.pt

# Binary Header Struct: 18 bytes
# [0:2]   magic: 2s (b"DT")
# [2:3]   flags: B (uint8)
# [3:7]   codebook_id: I (uint32)
# [7:11]  frame_id: I (uint32)
# [11:13] num_patches: H (uint16)
# [13:15] num_active: H (uint16)
# [15:16] num_quantizers: B (uint8)
# [16:17] grid_h: B (uint8)
# [17:18] grid_w: B (uint8)
HEADER_STRUCT = struct.Struct("<2sBIIHHBBB")


def to_bytes(packet: "TransmissionPacket") -> bytes:
    """Serializes a TransmissionPacket into an exact raw binary wire payload.

    Binary Layout:
        - [0:18]  Header (magic, flags, codebook_id, frame_id, N, K, Q, H_p, W_p)
        - [18:26] Optional 8-byte Motion floats (dx, dy as float32) if FLAG_MOTION
        - [Var]   Tightly packed Spatial Mask (ceil(N/8) bytes)
        - [Var]   RVQ Discrete Codebook Indices (K * Q * bytes_per_idx)
    """
    if packet.payload:
        return packet.payload

    flags = 0
    if packet.is_keyframe:
        flags |= FLAG_KEYFRAME
    if packet.motion is not None:
        flags |= FLAG_MOTION
    if packet.codebook_size > 256:
        flags |= FLAG_TWO_BYTE_INDICES

    h_p, w_p = packet.patch_grid
    header = HEADER_STRUCT.pack(
        MAGIC_BYTES,
        flags,
        packet.codebook_id,
        packet.frame_id,
        packet.num_patches,
        packet.num_active,
        packet.num_quantizers,
        h_p,
        w_p,
    )

    parts = [header]
    if packet.motion is not None:
        parts.append(struct.pack("<ff", float(packet.motion[0]), float(packet.motion[1])))

    parts.append(packet.packed_mask)
    parts.append(packet.indices_bytes)

    return b"".join(parts)


def from_bytes(payload: bytes) -> "TransmissionPacket":
    """Deserializes an exact binary wire payload back into a TransmissionPacket."""
    if len(payload) < HEADER_STRUCT.size:
        raise ValueError(
            f"Payload too short: {len(payload)} bytes < header size {HEADER_STRUCT.size}"
        )

    magic, flags, codebook_id, frame_id, num_patches, num_active, num_quantizers, h_p, w_p = (
        HEADER_STRUCT.unpack_from(payload, 0)
    )
    if magic != MAGIC_BYTES:
        raise ValueError(f"Invalid magic bytes: expected {MAGIC_BYTES}, got {magic}")

    offset = HEADER_STRUCT.size

    motion = None
    if flags & FLAG_MOTION:
        if len(payload) < offset + 8:
            raise ValueError(f"Payload truncated: expected 8 motion bytes at offset {offset}")
        dx, dy = struct.unpack_from("<ff", payload, offset)
        motion = (round(float(dx), 5), round(float(dy), 5))
        offset += 8

    mask_len = math.ceil(num_patches / 8)
    if len(payload) < offset + mask_len:
        raise ValueError(f"Payload truncated: expected {mask_len} mask bytes at offset {offset}")
    packed_mask = payload[offset : offset + mask_len]
    offset += mask_len

    bytes_per_idx = 2 if (flags & FLAG_TWO_BYTE_INDICES) else 1
    codebook_size = 65536 if (flags & FLAG_TWO_BYTE_INDICES) else 256
    indices_len = num_active * num_quantizers * bytes_per_idx

    if len(payload) < offset + indices_len:
        raise ValueError(f"Payload truncated: expected {indices_len} index bytes at offset {offset}")
    indices_bytes = payload[offset : offset + indices_len]
    offset += indices_len

    if offset != len(payload):
        raise ValueError(
            f"Payload trailing data: parsed {offset} bytes, total payload {len(payload)} bytes"
        )

    raw_bytes = num_patches * 384 * 4  # Standard float32 token size
    wire_bytes = len(payload)
    comp_ratio = raw_bytes / wire_bytes if wire_bytes > 0 else 0.0
    is_keyframe = bool(flags & FLAG_KEYFRAME)

    return TransmissionPacket(
        frame_id=frame_id,
        num_patches=num_patches,
        num_active=num_active,
        num_quantizers=num_quantizers,
        codebook_size=codebook_size,
        patch_grid=(h_p, w_p),
        packed_mask=packed_mask,
        indices_bytes=indices_bytes,
        raw_bytes=raw_bytes,
        wire_bytes=wire_bytes,
        compression_ratio=comp_ratio,
        motion=motion,
        codebook_id=codebook_id,
        is_keyframe=is_keyframe,
        payload=payload,
    )


@dataclass(frozen=True)
class TransmissionPacket:
    """Wire transmission packet carrying compressed token stream and temporal spatial mask.

    Contains NO PyTorch tensors on the wire. Transmits pure binary bytes.
    """
    frame_id: int
    num_patches: int                    # Total patches N in the frame
    num_active: int                     # Number of active tokens K
    num_quantizers: int                 # Number of RVQ codebooks Q
    codebook_size: int                  # Codebook vocabulary size V
    patch_grid: Tuple[int, int]         # (H_patches, W_patches)
    packed_mask: bytes                  # Bit-packed boolean spatial mask (ceil(N/8) bytes)
    indices_bytes: bytes                # RVQ discrete codebook indices as raw bytes
    raw_bytes: int                      # Uncompressed token sequence size (N * D * 4 bytes)
    wire_bytes: int                     # Exact compressed payload size: len(payload)
    compression_ratio: float            # raw_bytes / wire_bytes
    motion: Optional[Tuple[float, float]] = None # Camera ego-motion vector (dx, dy) in pixels
    codebook_id: int = DEFAULT_CODEBOOK_ID       # 4-byte codebook identifier / hash
    is_keyframe: bool = False           # Flag: True if keyframe, False if delta
    payload: bytes = b""                # Serialized wire bytes object

    def __init__(
        self,
        frame_id: int,
        num_patches: int,
        num_active: int,
        num_quantizers: int,
        codebook_size: int,
        patch_grid: Tuple[int, int],
        packed_mask: bytes,
        indices_bytes: bytes = b"",
        raw_bytes: int = 0,
        wire_bytes: int = 0,
        compression_ratio: float = 0.0,
        motion: Optional[Tuple[float, float]] = None,
        codebook_id: int = DEFAULT_CODEBOOK_ID,
        is_keyframe: bool = False,
        payload: bytes = b"",
        indices: Optional[torch.Tensor] = None,
    ):
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "num_patches", num_patches)
        object.__setattr__(self, "num_active", num_active)
        object.__setattr__(self, "num_quantizers", num_quantizers)
        object.__setattr__(self, "codebook_size", codebook_size)
        object.__setattr__(self, "patch_grid", patch_grid)
        object.__setattr__(self, "packed_mask", packed_mask)

        if not indices_bytes and indices is not None:
            np_dtype = np.uint16 if codebook_size > 256 else np.uint8
            indices_bytes = indices.detach().cpu().numpy().astype(np_dtype).tobytes()
        object.__setattr__(self, "indices_bytes", indices_bytes)

        object.__setattr__(self, "motion", motion)
        object.__setattr__(self, "codebook_id", codebook_id)
        object.__setattr__(self, "is_keyframe", is_keyframe)

        if not payload:
            payload = to_bytes(self)
        object.__setattr__(self, "payload", payload)

        if wire_bytes <= 0:
            wire_bytes = len(payload)
        object.__setattr__(self, "wire_bytes", wire_bytes)

        if raw_bytes <= 0:
            raw_bytes = num_patches * 384 * 4
        object.__setattr__(self, "raw_bytes", raw_bytes)

        if compression_ratio <= 0.0:
            compression_ratio = raw_bytes / wire_bytes if wire_bytes > 0 else 0.0
        object.__setattr__(self, "compression_ratio", compression_ratio)

    def to_bytes(self) -> bytes:
        """Returns the serialized binary wire payload."""
        if self.payload:
            return self.payload
        return to_bytes(self)

    @classmethod
    def from_bytes(cls, payload: bytes) -> "TransmissionPacket":
        """Deserializes a binary wire payload into a TransmissionPacket."""
        return from_bytes(payload)

    def unpack_mask(self) -> torch.Tensor:
        """Unpacks bit-packed bytes back into a boolean tensor [N]."""
        return Packer.unpack_boolean_mask(self.packed_mask, self.num_patches)

    def unpack_indices(self, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """Unpacks discrete RVQ indices from raw bytes into an integer tensor [K, Q]."""
        if self.num_active == 0:
            return torch.empty((0, self.num_quantizers), dtype=torch.int64, device=device)
        np_dtype = np.uint16 if self.codebook_size > 256 else np.uint8
        arr = np.frombuffer(self.indices_bytes, dtype=np_dtype)
        t = torch.from_numpy(arr.copy()).to(dtype=torch.int64)
        if device is not None:
            t = t.to(device)
        return t.view(self.num_active, self.num_quantizers)

    @property
    def indices(self) -> torch.Tensor:
        """Backwards compatibility property: unpacks indices tensor on demand."""
        return self.unpack_indices()


@dataclass(frozen=True)
class PackerOutput:
    """Output container for the Packer module."""
    indices: torch.Tensor               # Discrete codebook indices [K, Q]
    quantized: torch.Tensor             # Quantized approximation Z_hat in R^{K x D}
    commit_loss: torch.Tensor           # Commitment loss for RVQ codebooks
    packet: TransmissionPacket          # Serialized transmission payload


class Packer(nn.Module):
    """Packer module wrapping Residual Vector Quantization (RVQ).

    Architecture Specification:
        - Library: vector-quantize-pytorch
        - Input: Dynamic tokens Z_active in R^{K x D} and spatial mask M in {0, 1}^N
        - Process: Quantizes continuous vectors into discrete integer codebook indices
        - Output: True binary byte-serialized transmission packet ready for network transmission
    """

    def __init__(
        self,
        dim: int = 384,
        num_quantizers: int = 4,
        codebook_size: int = 256,
        kmeans_init: bool = False,
        codebook_id: Optional[int] = None,
        codebook_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        mean_token_norm: float = 47.4,
    ) -> None:
        """Initialize the RVQ Packer.

        Args:
            dim: Latent token dimension (default: 384 for DINOv2-ViT-Small).
            num_quantizers: Number of hierarchical codebook stages Q (default: 4).
            codebook_size: Number of entries V per codebook (default: 256 -> 8-bit index).
            kmeans_init: Whether to run kmeans on initial batch.
            codebook_id: Optional 4-byte identifier / hash for codebook validation.
            codebook_path: Optional path to frozen pretrained codebook checkpoint.
            device: Torch device (defaults to CUDA if available else CPU).
            mean_token_norm: Characteristic ViT token L2 norm for scale restoration.
        """
        super().__init__()
        self.dim = int(dim)
        self.num_quantizers = int(num_quantizers)
        self.codebook_size = int(codebook_size)
        self.mean_token_norm = float(mean_token_norm)
        self.kmeans_init = kmeans_init

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Check for pretrained codebook file
        resolved_path = None
        if codebook_path is not None and Path(codebook_path).exists():
            resolved_path = codebook_path
        elif (
            codebook_path is None
            and not kmeans_init
            and self.dim == 384
            and self.num_quantizers == 4
            and self.codebook_size == 256
            and Path(DEFAULT_CODEBOOK_PATH).exists()
        ):
            resolved_path = DEFAULT_CODEBOOK_PATH

        if resolved_path is not None:
            ckpt = torch.load(resolved_path, map_location=self.device, weights_only=False)
            self.dim = int(ckpt.get("dim", self.dim))
            self.num_quantizers = int(ckpt.get("num_quantizers", self.num_quantizers))
            self.codebook_size = int(ckpt.get("codebook_size", self.codebook_size))
            self.normalize_tokens = True

            # Compute sha256 checksum and derive protocol id
            data = Path(resolved_path).read_bytes()
            sha256_hash = hashlib.sha256(data).hexdigest()
            calc_id = int(sha256_hash[:8], 16)
            self.codebook_id = calc_id if codebook_id is None else int(codebook_id)
            self.codebook_sha256 = sha256_hash

            self.rvq = ResidualVQ(
                dim=self.dim,
                num_quantizers=self.num_quantizers,
                codebook_size=self.codebook_size,
                kmeans_init=False,
            ).to(self.device)
            self.rvq.load_state_dict(ckpt["state_dict"])
            self.rvq.eval()
            self._is_calibrated = True
        else:
            self.normalize_tokens = False
            self.codebook_id = DEFAULT_CODEBOOK_ID if codebook_id is None else int(codebook_id)
            self.codebook_sha256 = ""
            self.rvq = ResidualVQ(
                dim=self.dim,
                num_quantizers=self.num_quantizers,
                codebook_size=self.codebook_size,
                kmeans_init=kmeans_init,
            ).to(self.device)
            self.rvq.eval()

    @property
    def codebook_weights(self) -> torch.Tensor:
        """Returns stacked codebook embeddings [Q, V, D]."""
        return torch.stack([layer._codebook.embed.squeeze(0) for layer in self.rvq.layers], dim=0)

    @classmethod
    def load_pretrained(
        cls,
        path: str = DEFAULT_CODEBOOK_PATH,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "Packer":
        """Loads a frozen, versioned RVQ codebook from disk."""
        return cls(codebook_path=path, device=device)

    @staticmethod
    def pack_boolean_mask(mask: torch.Tensor) -> bytes:
        """Packs a 1D boolean tensor of length N into compact bits (ceil(N/8) bytes)."""
        mask_flat = mask.detach().cpu().flatten().bool()
        n = mask_flat.numel()
        pad_len = (8 - (n % 8)) % 8
        if pad_len > 0:
            padded = torch.cat([mask_flat, torch.zeros(pad_len, dtype=torch.bool)])
        else:
            padded = mask_flat

        padded_u8 = padded.to(dtype=torch.uint8).view(-1, 8)
        shifts = torch.arange(8, dtype=torch.uint8)
        packed_bytes = (padded_u8 << shifts).sum(dim=1).to(dtype=torch.uint8)
        return bytes(packed_bytes.tolist())

    @staticmethod
    def unpack_boolean_mask(packed_mask: bytes, num_patches: int) -> torch.Tensor:
        """Unpacks bit-packed bytes back into a boolean tensor [N]."""
        if not packed_mask:
            return torch.zeros(num_patches, dtype=torch.bool)
        byte_tensor = torch.tensor(list(packed_mask), dtype=torch.uint8)
        shifts = torch.arange(8, dtype=torch.uint8)
        unpacked = (byte_tensor.unsqueeze(1) >> shifts) & 1
        mask = unpacked.flatten()[:num_patches].bool()
        return mask

    def quantize(
        self, z_active: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantizes continuous dynamic tokens Z_active into discrete RVQ integers.

        Args:
            z_active: Tensor of dynamic tokens [K, D] or [B, K, D].

        Returns:
            - quantized: Reconstructed vectors Z_hat [K, D]
            - indices: Codebook indices [K, Q] of type int64
            - commit_loss: Scalar loss tensor
        """
        orig_dim = z_active.ndim
        k = z_active.shape[0] if orig_dim == 2 else z_active.shape[1]

        # Handle edge case: K = 0 (100% static frame)
        if k == 0:
            indices = torch.empty((0, self.num_quantizers), dtype=torch.int64, device=self.device)
            quantized = torch.empty((0, self.dim), dtype=z_active.dtype, device=self.device)
            commit_loss = torch.tensor(0.0, device=self.device)
            return quantized, indices, commit_loss

        # ResidualVQ expects 3D tensor [B, K, D]
        x = z_active.to(self.device)

        if self.normalize_tokens:
            norms = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=1e-6)
            x_in = x / norms
        else:
            norms = None
            x_in = x

        if orig_dim == 2:
            x_in = x_in.unsqueeze(0)  # [1, K, D]

        quantized, indices, commit_loss = self.rvq(x_in)

        if orig_dim == 2:
            quantized = quantized.squeeze(0)  # [K, D]
            indices = indices.squeeze(0)      # [K, Q]

        if self.normalize_tokens and norms is not None:
            quantized = quantized * norms

        return quantized, indices, commit_loss

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstructs continuous vectors Z_hat from RVQ integer indices.

        Args:
            indices: Codebook indices tensor of shape [K, Q] or [B, K, Q].

        Returns:
            Reconstructed token vectors [K, D] or [B, K, D].
        """
        orig_dim = indices.ndim
        k = indices.shape[0] if orig_dim == 2 else indices.shape[1]

        if k == 0:
            return torch.empty((0, self.dim), dtype=torch.float32, device=self.device)

        idx = indices.to(self.device)
        if orig_dim == 2:
            idx = idx.unsqueeze(0)  # [1, K, Q]

        recon = self.rvq.get_output_from_indices(idx)

        if orig_dim == 2:
            recon = recon.squeeze(0)  # [K, D]

        if self.normalize_tokens:
            recon = recon * self.mean_token_norm

        return recon

    def forward(
        self,
        z_active: torch.Tensor,
        mask: torch.Tensor,
        frame_id: int = 0,
        patch_grid: Optional[Tuple[int, int]] = None,
        motion: Optional[Tuple[float, float]] = None,
        is_keyframe: bool = False,
    ) -> PackerOutput:
        """Packs active tokens and spatial mask into a true binary transmission packet.

        Args:
            z_active: Dynamic token vectors [K, D].
            mask: Boolean spatial mask [N] where True indicates active tokens.
            frame_id: Frame sequence counter.
            patch_grid: Spatial patch dimensions (H_p, W_p).
            motion: Optional camera translation tuple (dx, dy) in pixels.
            is_keyframe: True if keyframe (all patches active), False if delta frame.

        Returns:
            PackerOutput containing indices, quantized approximation, loss, and TransmissionPacket.
        """
        mask_flat = mask.flatten().bool()
        n_patches = int(mask_flat.numel())
        k_active = int(mask_flat.sum().item())

        if patch_grid is None:
            side = int(math.isqrt(n_patches))
            patch_grid = (side, side)

        # 1. Quantize dynamic tokens
        quantized, indices, commit_loss = self.quantize(z_active)

        # 2. Pack boolean spatial mask into bits
        packed_mask = self.pack_boolean_mask(mask_flat)

        # 3. Convert discrete indices to raw bytes
        if k_active > 0:
            if self.codebook_size <= 256:
                indices_bytes = indices.to(dtype=torch.uint8, device="cpu").numpy().tobytes()
            else:
                indices_bytes = indices.to(dtype=torch.int32, device="cpu").numpy().astype(np.uint16).tobytes()
        else:
            indices_bytes = b""

        # 4. Raw uncompressed bytes: N * D * 4 bytes (float32)
        raw_bytes = n_patches * self.dim * 4

        # 5. True Binary Serialization: no theoretical formulas!
        temp_packet = TransmissionPacket(
            frame_id=frame_id,
            num_patches=n_patches,
            num_active=k_active,
            num_quantizers=self.num_quantizers,
            codebook_size=self.codebook_size,
            patch_grid=patch_grid,
            packed_mask=packed_mask,
            indices_bytes=indices_bytes,
            raw_bytes=raw_bytes,
            wire_bytes=0,
            compression_ratio=0.0,
            motion=motion,
            codebook_id=self.codebook_id,
            is_keyframe=is_keyframe,
            payload=b"",
        )
        payload = to_bytes(temp_packet)
        wire_bytes = len(payload)
        compression_ratio = raw_bytes / wire_bytes if wire_bytes > 0 else 0.0

        packet = TransmissionPacket(
            frame_id=frame_id,
            num_patches=n_patches,
            num_active=k_active,
            num_quantizers=self.num_quantizers,
            codebook_size=self.codebook_size,
            patch_grid=patch_grid,
            packed_mask=packed_mask,
            indices_bytes=indices_bytes,
            raw_bytes=raw_bytes,
            wire_bytes=wire_bytes,
            compression_ratio=compression_ratio,
            motion=motion,
            codebook_id=self.codebook_id,
            is_keyframe=is_keyframe,
            payload=payload,
        )

        return PackerOutput(
            indices=indices,
            quantized=quantized,
            commit_loss=commit_loss,
            packet=packet,
        )
