Perron-Frobenius Theorem and Convergence
========================================

The iterative scheme used to find the God Tensor is a concrete application of
the **power iteration** method, whose convergence is guaranteed by the
**Perron-Frobenius theorem** and related spectral gap theory.

Power Iteration as Spectral Convergence
---------------------------------------

Faraday's iteration is:

.. math::

   f(\mathbf{x}) = \text{normalize}(T \, \mathbf{x})

On the unit sphere, the operator :math:`\mathbf{x} \mapsto T\mathbf{x}`
has a well-defined dominant eigenvector — the **Perron-Frobenius eigenvector** —
which is the fixed point of the un-normalized iteration.

Normalization maps the iteration back to the unit sphere :math:`S^{d-1}`.
On :math:`S^{d-1}`, power iteration converges geometrically:

.. math::

   \|\mathbf{x}_n - \mathbf{x}^*\| \leq C \cdot |\lambda_2 / \lambda_1|^n

where :math:`\lambda_1` and :math:`\lambda_2` are the largest and second-largest
eigenvalues of :math:`T` by magnitude. The convergence rate is determined by the
**spectral gap** :math:`|\lambda_2/\lambda_1|`.

Spectral Properties of T
~~~~~~~~~~~~~~~~~~~~~~~~

- :math:`T` is learned from data; its eigenvalues depend on the geometry set
- A **large spectral gap** → fast convergence (few iterations needed)
- A **small spectral gap** → slow convergence (many iterations needed)
- A rank-deficient :math:`T` (fewer samples than dimensions) has :math:`\lambda_i = 0`
  for :math:`i > \text{rank}(T)`, making convergence impossible

Faraday requires :math:`n_{\text{geometries}} \geq d = 16` for reliable convergence.

Convergence Criteria in Faraday
-------------------------------

Eigenvectors are defined only up to a sign (the spectrum is invariant
under :math:`x \mapsto -x`).  Faraday therefore measures the
**sign-corrected** residual

.. math::

   r_n = \bigl\| \mathbf{x}_{n+1}
        - \mathrm{sign}\!\bigl(\langle \mathbf{x}_{n+1}, \mathbf{x}_n\rangle\bigr)
              \mathbf{x}_n\bigr\|_2.

Two criteria are used in :meth:`GodTensor.find_fixed_point`:

1. **Sign-corrected residual** (primary):

   .. math::

      r_n < \epsilon_{\text{tol}}

   with default :math:`\epsilon_{\text{tol}} = 10^{-7}`.

2. **Iteration limit** (safety cap):
   :math:`n > n_{\text{max}}` with default :math:`n_{\text{max}} = 500`.

If the iteration hits the safety cap before reaching tolerance, the
``GodTensor`` retains its current dominant-eigenvector estimate and
``fixed_point_converged`` is left ``False``.  Empirically, for
well-conditioned :math:`T` (spectral gap > 0.1) the residual drops to
machine epsilon ($\approx 10^{-16}$) within a few hundred iterations,
which is verified end-to-end in
``tests/test_spectral_fixed_point.py``.

Why Normalization?
~~~~~~~~~~~~~~~~~

Without normalization, :math:`\mathbf{x}_{n+1} = T\mathbf{x}_n` grows or decays
geometrically as :math:`|\lambda_1|^n`. Normalization constrains the iterates to the
unit sphere, making the iteration **scale-invariant** — independent of the
absolute magnitude of :math:`T`.

Connection to Google PageRank
-----------------------------

The power iteration method is the same algorithm used in Google's original
PageRank. There, the Google matrix :math:`G` has a dominant eigenvalue 1
(the PageRank vector), found by iterating :math:`\mathbf{r}_{n+1} = G\mathbf{r}_n`
starting from a uniform distribution. The God Tensor iteration is the same
algorithm applied to the electromagnetic coupling operator.

References
----------

- O. Perron. "Zur Theorie der Matrices." *Mathematische Annalen*, 64(2):248–263, 1907.
- G. Frobenius. "Ueber Matrizen aus nicht negativen Elementen." *Sitzungsberichte der Königlich Preussischen Akademie der Wissenschaften*, 456–477, 1912.
- D. Kincaid, E. Cheney. *Numerical Analysis*, 3rd ed., Brooks/Cole, 2002.
  Chapter 9: Iterative Methods for Solving Linear Systems.
- A. N. Langville, C. D. Meyer. *Google's PageRank and Beyond*. Johns Hopkins
  University Press, 2006.
