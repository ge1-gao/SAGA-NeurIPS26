"""4D Morton (Z-order) keys. Used to order anchors for (future) entropy coding."""
from __future__ import annotations

import numpy as np


def morton_encode_4d(q: np.ndarray, bits: int) -> np.ndarray:
    """q: [N, 4] non-negative ints < 2**bits  ->  [N] uint64 keys (bit-interleaved, dim 0 lowest)."""
    assert 4 * bits <= 64
    q = q.astype(np.uint64)
    key = np.zeros(len(q), dtype=np.uint64)
    for b in range(bits):
        for d in range(4):
            key |= ((q[:, d] >> np.uint64(b)) & np.uint64(1)) << np.uint64(4 * b + d)
    return key


def morton_decode_4d(key: np.ndarray, bits: int) -> np.ndarray:
    key = key.astype(np.uint64)
    q = np.zeros((len(key), 4), dtype=np.uint64)
    for b in range(bits):
        for d in range(4):
            q[:, d] |= ((key >> np.uint64(4 * b + d)) & np.uint64(1)) << np.uint64(b)
    return q.astype(np.int64)


def hierarchical_keys(coarse_key: np.ndarray, parent: np.ndarray, local_offset: np.ndarray, bits_f: int) -> np.ndarray:
    """κ_mi = (κ_c[m] << 4B_f) | κ_f(ō_mi); ō in [-1, 1]^4 is quantized to a B_f-bit local grid."""
    q = np.clip(((local_offset + 1) / 2 * (2 ** bits_f)).astype(np.int64), 0, 2 ** bits_f - 1)
    return (coarse_key[parent].astype(np.uint64) << np.uint64(4 * bits_f)) | morton_encode_4d(q, bits_f)
