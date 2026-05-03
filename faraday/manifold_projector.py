"""
faraday.manifold_projector — Barcode to Manifold Embedding

Converts persistent homology barcodes into fixed-length vector embeddings
suitable for similarity search and learning.

The embedding is the "Hilbert coefficients" representation of the barcode:
  N(t) = Σ t^{birth_i} - Σ t^{death_i}

This polynomial encodes the ENTIRE topological structure as a vector.
The God Tensor operates on these embeddings, not raw barcodes.
"""

from __future__ import annotations

import math

import numpy as np

from faraday.logging import get_logger

log = get_logger(__name__)


def barcode_to_coefficients(
    barcode: list[tuple[float, float]], degree: int = 50
) -> np.ndarray:
    """
    Convert a barcode (birth-death pairs) to Hilbert series coefficients.

    The barcode is a list of (birth, death) pairs.
    Hilbert series numerator: N(t) = Σ t^{birth} - Σ t^{death}
    Coeff[i] = number of bars born at i - number that died at i.

    Args:
        barcode: list of (birth, death) tuples
        degree: polynomial degree (embedding dimension)

    Returns:
        coeffs: (degree,) numpy array of Hilbert coefficients
    """
    coeffs = np.zeros(degree)
    for b, d in barcode:
        ib = math.floor(b * degree)
        if 0 <= ib < degree:
            coeffs[ib] += 1.0
        if d != float("inf"):
            id_ = math.floor(d * degree)
            if 0 <= id_ < degree:
                coeffs[id_] -= 1.0
    return coeffs


def barcode_from_field(
    field: np.ndarray, homology_dim: int = 0, threshold: float = 0.1
) -> list[tuple[float, float]]:
    """
    Extract barcode directly from a 2D field.
    Used for quick signature extraction without full fingerprint.
    """
    from .barcode import compute_barcodes, field_to_pointcloud

    points = field_to_pointcloud(field, threshold)
    if len(points) < 10:
        return []
    result = compute_barcodes(points, max_dim=homology_dim)
    diagrams = result.get("diagrams", [])
    if homology_dim < len(diagrams):
        return [(float(b), float(d)) for b, d in diagrams[homology_dim]]
    return []


def embed_barcode(barcode: list[tuple[float, float]], dim: int = 50) -> np.ndarray:
    """
    Convert barcode to fixed-length embedding vector.

    Args:
        barcode: list of (birth, death) tuples
        dim: embedding dimension (default 50)

    Returns:
        embedding: (dim,) numpy array, L2-normalized
    """
    raw = barcode_to_coefficients(barcode, degree=dim)
    norm = np.linalg.norm(raw)
    if norm > 1e-10:
        raw = raw / norm
    return raw


def embed_fingerprint(fp: dict, dim: int = 50) -> np.ndarray:
    """
    Convert a topological_fingerprint dict to a fixed-length embedding.

    Uses:
      - H0 and H1 lifetimes statistics
      - Betti numbers
      - Bar counts
      - Field statistics
      - Coupling metrics

    Args:
        fp: topological_fingerprint dict from barcode.py
        dim: embedding dimension

    Returns:
        embedding: (dim,) numpy array
    """
    vec = np.zeros(dim)

    # H0 lifetime stats (indices 0-9)
    h0_lt = fp.get("h0_lifetimes", [])
    if h0_lt:
        stats = [
            np.mean(h0_lt),
            np.std(h0_lt),
            np.max(h0_lt),
            np.min(h0_lt),
            len(h0_lt),
        ]
    else:
        stats = [0.0] * 5
    for i, s in enumerate(stats[:5]):
        vec[i] = s

    # H1 lifetime stats (indices 10-19)
    h1_lt = fp.get("h1_lifetimes", [])
    if h1_lt:
        stats = [
            np.mean(h1_lt),
            np.std(h1_lt),
            np.max(h1_lt),
            np.min(h1_lt),
            len(h1_lt),
        ]
    else:
        stats = [0.0] * 5
    for i, s in enumerate(stats[:5]):
        vec[10 + i] = s

    # Betti numbers and bar counts (indices 20-29)
    vec[20] = fp.get("betti_0", 0)
    vec[21] = fp.get("betti_1", 0)
    vec[22] = fp.get("h0_bars", 0)
    vec[23] = fp.get("h1_bars", 0)
    vec[24] = fp.get("topological_score", 0.0)
    vec[25] = fp.get("confinement_ratio", 0.0)
    vec[26] = fp.get("field_max", 0.0)
    vec[27] = fp.get("field_mean", 0.0)
    vec[28] = fp.get("field_std", 0.0)
    vec[29] = fp.get("num_grid_points", 0)

    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 1e-10:
        vec = vec / norm
    return vec


class ManifoldProjector:
    """
    Learns a non-linear projection from barcode embeddings to a latent manifold.

    Uses a simple autoencoder:
      - Encoder: barcode_embedding -> latent (dim < input dim)
      - Decoder: latent -> barcode_embedding

    Training finds the manifold structure underlying the topological space.
    The God Tensor refines this further by finding the E<->H fixed point
    on this manifold.
    """

    def __init__(self, input_dim: int = 50, latent_dim: int = 16):
        """
        Args:
            input_dim: dimension of barcode embeddings (default 50)
            latent_dim: dimension of latent space (default 16)
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_weights: np.ndarray | None = None
        self.decoder_weights: np.ndarray | None = None
        self.is_trained = False

    def _init_weights(self):
        """Initialize encoder and decoder weight matrices."""
        rng = np.random.default_rng(42)
        # Xavier-like initialization
        scale = math.sqrt(2.0 / (self.input_dim + self.latent_dim))
        self.encoder_weights = rng.normal(0, scale, (self.latent_dim, self.input_dim))
        self.decoder_weights = rng.normal(0, scale, (self.input_dim, self.latent_dim))

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Project embedding to latent space via encoder."""
        if not self.is_trained:
            self._init_weights()
        # ReLU activation
        h = x @ self.encoder_weights.T
        h = np.maximum(h, 0)
        return h

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Project latent vector back to embedding space via decoder."""
        if not self.is_trained:
            self._init_weights()
        # ReLU activation
        out = z @ self.decoder_weights.T
        out = np.maximum(out, 0)
        return out

    def autoencode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Full autoencoder pass: encode then decode."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return z, x_recon

    def reconstruct(self, x: np.ndarray) -> float:
        """
        Compute reconstruction error (MSE) for a single embedding.
        Also trains weights if not yet trained.
        """
        if not self.is_trained:
            self._init_weights()

        _z, x_recon = self.autoencode(x)
        mse = float(np.mean((x - x_recon) ** 2))
        return mse

    def project(self, barcode_or_fingerprint) -> np.ndarray:
        """
        Project raw barcode or fingerprint to latent space.

        Args:
            barcode_or_fingerprint: either a barcode list or a fingerprint dict

        Returns:
            latent: (latent_dim,) embedding in latent space
        """
        if isinstance(barcode_or_fingerprint, dict):
            emb = embed_fingerprint(barcode_or_fingerprint, dim=self.input_dim)
        else:
            emb = embed_barcode(barcode_or_fingerprint, dim=self.input_dim)
        return self.encode(emb)

    def fit(
        self,
        embeddings: list[np.ndarray],
        lr: float = 0.01,
        epochs: int = 100,
        batch_size: int = 8,
        verbose: bool = False,
    ) -> list[float]:
        """
        Train the autoencoder on a collection of barcode embeddings.

        Args:
            embeddings: list of (input_dim,) arrays
            lr: learning rate
            epochs: training epochs
            batch_size: mini-batch size
            verbose: print progress

        Returns:
            losses: list of MSE loss per epoch
        """
        self._init_weights()
        X = np.array(embeddings)
        losses = []

        rng = np.random.default_rng(42)

        for epoch in range(epochs):
            # Mini-batch SGD
            indices = rng.permutation(len(X))
            epoch_loss = 0.0

            for start in range(0, len(X), batch_size):
                batch = X[indices[start : start + batch_size]]

                # Forward: encode
                h = np.maximum(batch @ self.encoder_weights.T, 0)
                # Decode
                recon = np.maximum(h @ self.decoder_weights.T, 0)

                # MSE loss
                loss = np.mean((batch - recon) ** 2)
                epoch_loss += loss

                # Gradient: dL/dW_enc = dL/dRecon @ dRecon/dh @ dh/dW_enc
                # Simplified: use reconstruction error to update weights
                d_recon = 2 * (recon - batch) / len(batch)
                d_h = d_recon @ self.decoder_weights
                d_h[h == 0] = 0  # ReLU gradient

                # dL/dW_enc = (dL/dh).T @ x  →  (16, n) @ (n, 50) = (16, 50)
                self.encoder_weights -= lr * (d_h.T @ batch) / len(batch)
                # dL/dW_dec = (dL/dRecon).T @ h  →  (50, n) @ (n, 16) = (50, 16)
                self.decoder_weights -= lr * (d_recon.T @ h) / len(batch)

            losses.append(float(epoch_loss))
            if verbose and (epoch + 1) % 20 == 0:
                log.info("manifold_projector_fit_progress", epoch=epoch + 1, epochs=epochs, loss=losses[-1])

        self.is_trained = True
        return losses
