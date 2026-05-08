# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
faraday.manifold_projector — Barcodes → Hilbert-Series Embeddings → Latent Manifold.

The pipeline has two layers:

1. **Hilbert-series encoding** (closed-form): a barcode
   :math:`B = \\{(b_i, d_i)\\}` is mapped to its Hilbert-series numerator,
   sampled at integer powers,

   .. math::

      N_B(t) = \\sum_i t^{b_i} - \\sum_i t^{d_i},

   discretised onto a fixed-degree polynomial basis.  This gives a
   permutation-invariant, fixed-length vector representation
   (Zomorodian–Carlsson, 2005, *Discrete Comp. Geom.* 33, 249–274).

2. **Non-linear projector** (learned): a single-hidden-layer ReLU
   autoencoder with **mathematically correct** mini-batch gradient
   descent on the squared-error loss

   .. math::

      \\mathcal{L}(W_e, W_d) = \\frac{1}{N}
        \\sum_{n=1}^{N} \\|x_n - W_d^\\top \\sigma(W_e x_n)\\|_2^2,

   where :math:`\\sigma(z) = \\max(z, 0)`.  The standard chain-rule
   gradients are

   .. math::

      \\nabla_{W_d} \\mathcal{L} \\;\\propto\\;
        \\delta_{\\text{rec}}^\\top \\, h, \\qquad
      \\nabla_{W_e} \\mathcal{L} \\;\\propto\\;
        (\\delta_{\\text{rec}} W_d \\odot \\mathbb{1}[h>0])^\\top \\, x,

   with :math:`\\delta_{\\text{rec}} = 2 (\\hat x - x) / N`.  These are the
   formulas implemented in :meth:`ManifoldProjector.fit`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from faraday.exceptions import ConfigError
from faraday.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Hilbert-series encoding
# ---------------------------------------------------------------------------


def barcode_to_coefficients(
    barcode: list[tuple[float, float]], degree: int = 50
) -> np.ndarray:
    """Encode a barcode as Hilbert-series numerator coefficients.

    For a barcode :math:`B = \\{(b_i, d_i)\\}` and a polynomial degree
    :math:`D` we return a vector :math:`c \\in \\mathbb{R}^{D}` with

    .. math::

       c_k = |\\{i : \\lfloor D \\, b_i \\rfloor = k\\}|
            - |\\{i : d_i < \\infty \\wedge \\lfloor D \\, d_i \\rfloor = k\\}|.

    Parameters
    ----------
    barcode : list of (birth, death) pairs
    degree : int, optional
        Polynomial degree / embedding dimension.

    Returns
    -------
    np.ndarray
        ``(degree,)`` array of Hilbert coefficients.
    """
    if degree <= 0:
        raise ConfigError("degree must be positive", degree=degree)
    coeffs = np.zeros(degree, dtype=np.float64)
    for b, d in barcode:
        if math.isfinite(b):
            ib = int(math.floor(b * degree))
            if 0 <= ib < degree:
                coeffs[ib] += 1.0
        if math.isfinite(d):
            id_ = int(math.floor(d * degree))
            if 0 <= id_ < degree:
                coeffs[id_] -= 1.0
    return coeffs


def barcode_from_field(
    field: np.ndarray, homology_dim: int = 0, threshold: float = 0.1
) -> list[tuple[float, float]]:
    """Extract a barcode (birth–death pairs) from a 2-D field.

    Convenience wrapper around :func:`compute_barcodes` for callers who
    only want the raw barcode at a single homology dimension.
    """
    from faraday.barcode import compute_barcodes, field_to_pointcloud

    points = field_to_pointcloud(field, threshold)
    if len(points) < 10:
        return []
    result = compute_barcodes(points, max_dim=homology_dim)
    diagrams = result.get("diagrams", [])
    if homology_dim < len(diagrams):
        return [(float(b), float(d)) for b, d in diagrams[homology_dim]]
    return []


def embed_barcode(
    barcode: list[tuple[float, float]], dim: int = 50
) -> np.ndarray:
    """Map a barcode to an L2-normalised Hilbert-series embedding."""
    raw = barcode_to_coefficients(barcode, degree=dim)
    norm = float(np.linalg.norm(raw))
    if norm > 1e-12:
        raw = raw / norm
    return raw


def embed_fingerprint(fp: dict[str, Any], dim: int = 50) -> np.ndarray:
    """Map a fingerprint dict to a fixed-length L2-normalised embedding.

    The first 30 dimensions encode lifetime statistics, Betti numbers,
    bar counts, and field statistics.  Remaining dimensions are zero
    (kept for forward-compatible expansion).
    """
    if dim < 30:
        raise ConfigError("embed_fingerprint requires dim >= 30", dim=dim)
    vec = np.zeros(dim, dtype=np.float64)

    def _stats(values: list[float]) -> list[float]:
        if not values:
            return [0.0] * 5
        a = np.asarray(values, dtype=np.float64)
        return [
            float(a.mean()),
            float(a.std()),
            float(a.max()),
            float(a.min()),
            float(a.size),
        ]

    h0 = _stats(fp.get("h0_lifetimes", []))
    h1 = _stats(fp.get("h1_lifetimes", []))
    vec[0:5] = h0
    vec[10:15] = h1

    vec[20] = float(fp.get("betti_0", 0))
    vec[21] = float(fp.get("betti_1", 0))
    vec[22] = float(fp.get("h0_bars", 0))
    vec[23] = float(fp.get("h1_bars", 0))
    vec[24] = float(fp.get("topological_score", 0.0))
    vec[25] = float(fp.get("confinement_ratio", 0.0))
    vec[26] = float(fp.get("field_max", 0.0))
    vec[27] = float(fp.get("field_mean", 0.0))
    vec[28] = float(fp.get("field_std", 0.0))
    vec[29] = float(fp.get("num_grid_points", 0))

    norm = float(np.linalg.norm(vec))
    if norm > 1e-12:
        vec = vec / norm
    return vec


# ---------------------------------------------------------------------------
# Non-linear manifold projector (proper autoencoder)
# ---------------------------------------------------------------------------


class ManifoldProjector:
    """Single-hidden-layer ReLU autoencoder.

    The encoder/decoder are linear maps :math:`W_e \\in \\mathbb{R}^{L \\times D}`
    and :math:`W_d \\in \\mathbb{R}^{D \\times L}` with a ReLU non-linearity
    on the latent.  The decoder is *linear* (no ReLU on the output) so that
    L2-normalised inputs can be reconstructed faithfully even when they
    contain negative coefficients.

    Parameters
    ----------
    input_dim : int, default 50
        Dimension :math:`D` of the input embedding.
    latent_dim : int, default 16
        Dimension :math:`L` of the latent code.
    seed : int, default 42
        Seed for weight initialisation.
    """

    def __init__(
        self,
        input_dim: int = 50,
        latent_dim: int = 16,
        seed: int = 42,
    ) -> None:
        if input_dim < latent_dim:
            raise ConfigError(
                "latent_dim must be <= input_dim",
                input_dim=input_dim,
                latent_dim=latent_dim,
            )
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seed = seed
        self.encoder_weights: np.ndarray | None = None
        self.decoder_weights: np.ndarray | None = None
        self.is_trained = False

    # ---------------- internal helpers ----------------

    def _init_weights(self) -> None:
        rng = np.random.default_rng(self.seed)
        scale = math.sqrt(2.0 / (self.input_dim + self.latent_dim))
        self.encoder_weights = rng.normal(
            0, scale, (self.latent_dim, self.input_dim)
        )
        self.decoder_weights = rng.normal(
            0, scale, (self.input_dim, self.latent_dim)
        )
        self.is_trained = True

    def _ensure_initialised(self) -> tuple[np.ndarray, np.ndarray]:
        if self.encoder_weights is None or self.decoder_weights is None:
            self._init_weights()
        assert self.encoder_weights is not None
        assert self.decoder_weights is not None
        return self.encoder_weights, self.decoder_weights

    # ---------------- public API ----------------

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Project an embedding into the latent space."""
        We, _ = self._ensure_initialised()
        x = np.asarray(x, dtype=np.float64)
        return np.maximum(x @ We.T, 0.0)

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Lift a latent code back into embedding space.

        The decoder is linear (no ReLU on output) so that L2-normalised
        embeddings — which can have signed entries — can be reconstructed.
        """
        _, Wd = self._ensure_initialised()
        z = np.asarray(z, dtype=np.float64)
        return z @ Wd.T

    def autoencode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode then decode."""
        z = self.encode(x)
        return z, self.decode(z)

    def reconstruct(self, x: np.ndarray) -> float:
        """Mean-squared reconstruction error for a single embedding."""
        _, x_recon = self.autoencode(x)
        return float(np.mean((x - x_recon) ** 2))

    def project(self, barcode_or_fingerprint: Any) -> np.ndarray:
        """Project a barcode or fingerprint dict to latent space."""
        if isinstance(barcode_or_fingerprint, dict):
            emb = embed_fingerprint(barcode_or_fingerprint, dim=self.input_dim)
        else:
            emb = embed_barcode(barcode_or_fingerprint, dim=self.input_dim)
        return self.encode(emb)

    def fit(
        self,
        embeddings: list[np.ndarray] | np.ndarray,
        lr: float = 0.01,
        epochs: int = 100,
        batch_size: int = 8,
        verbose: bool = False,
    ) -> list[float]:
        """Train the autoencoder by mini-batch SGD on MSE loss.

        Implements the chain-rule gradients

        .. math::

           \\nabla_{W_d} \\mathcal{L}_b
              = (h_b)^\\top (2(\\hat x_b - x_b) / |b|), \\\\
           \\nabla_{W_e} \\mathcal{L}_b
              = ((2(\\hat x_b - x_b) / |b|) W_d \\odot \\mathbb{1}[h_b>0])^\\top x_b,

        for each mini-batch :math:`b`.

        Parameters
        ----------
        embeddings : list[np.ndarray] | np.ndarray
            Training embeddings, each shape ``(input_dim,)``.
        lr : float
            SGD learning rate.
        epochs : int
            Number of full passes over the data.
        batch_size : int
            Mini-batch size.
        verbose : bool
            Log progress every 20 epochs.

        Returns
        -------
        list[float]
            Mean per-epoch MSE losses.
        """
        if epochs < 1 or batch_size < 1:
            raise ConfigError(
                "epochs and batch_size must be >=1",
                epochs=epochs,
                batch_size=batch_size,
            )
        self._init_weights()
        We, Wd = self._ensure_initialised()
        X = np.asarray(embeddings, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ConfigError(
                "embeddings must be (N, input_dim)",
                shape=tuple(X.shape),
                input_dim=self.input_dim,
            )

        rng = np.random.default_rng(self.seed)
        losses: list[float] = []
        for epoch in range(epochs):
            perm = rng.permutation(len(X))
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(X), batch_size):
                batch = X[perm[start : start + batch_size]]
                B = batch.shape[0]

                # Forward
                pre_h = batch @ We.T
                h = np.maximum(pre_h, 0.0)
                recon = h @ Wd.T

                # Loss
                resid = recon - batch
                epoch_loss += float(np.mean(resid * resid))
                n_batches += 1

                # Backward
                d_recon = 2.0 * resid / B
                grad_Wd = d_recon.T @ h  # (D, L)
                d_h = (d_recon @ Wd) * (pre_h > 0)
                grad_We = d_h.T @ batch  # (L, D)

                # SGD step
                Wd -= lr * grad_Wd
                We -= lr * grad_We

            losses.append(epoch_loss / max(n_batches, 1))
            if verbose and (epoch + 1) % 20 == 0:
                log.info(
                    "manifold_projector_fit_progress",
                    epoch=epoch + 1,
                    epochs=epochs,
                    loss=losses[-1],
                )

        # Persist the in-place updates back to attributes
        self.encoder_weights = We
        self.decoder_weights = Wd
        self.is_trained = True
        return losses
