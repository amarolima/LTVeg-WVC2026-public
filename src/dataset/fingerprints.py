"""Image fingerprints used to check a regenerated dataset against the original one.

Three levels, from strictest to most tolerant:
  sha256_file   -- SHA-256 of the PNG bytes. Changes with any re-encoding.
  sha256_pixels -- SHA-256 of the decoded RGB pixel array (alpha dropped; it is a
                   constant 255 in this dataset). Survives lossless re-encoding.
  phash, dhash  -- 64-bit perceptual hashes. Compared by Hamming distance; they
                   tolerate compression, small shifts and brightness changes, and
                   tell whether a regenerated image shows the same scene.
"""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dct


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_pixels(rgb):
    h = hashlib.sha256()
    h.update(f"{rgb.shape[0]}x{rgb.shape[1]}x{rgb.shape[2]}".encode())
    h.update(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
    return h.hexdigest()


def _gray(rgb, size):
    return np.asarray(Image.fromarray(rgb).convert("L").resize(size, Image.Resampling.LANCZOS),
                      dtype=np.float64)


def _bits_to_hex(bits):
    return f"{int(''.join('1' if b else '0' for b in bits.ravel()), 2):016x}"


def phash(rgb):
    """DCT perceptual hash: 32x32 grayscale, 8x8 lowest frequencies vs. their median."""
    g = _gray(rgb, (32, 32))
    c = dct(dct(g, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    return _bits_to_hex(c > np.median(c.ravel()[1:]))


def dhash(rgb):
    """Difference hash: 9x8 grayscale, sign of the horizontal gradient."""
    g = _gray(rgb, (9, 8))
    return _bits_to_hex(g[:, 1:] > g[:, :-1])


def hamming(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def fingerprint(path):
    rgb = load_rgb(path)
    return dict(sha256_file=sha256_file(path), sha256_pixels=sha256_pixels(rgb),
                phash=phash(rgb), dhash=dhash(rgb))
