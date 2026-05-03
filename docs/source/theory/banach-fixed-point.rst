Banach Fixed-Point Theorem and Convergence
=========================================

The iterative scheme used to find the God Tensor is a concrete application of
the **Banach fixed-point theorem** (also called the contraction mapping theorem),
the foundational result in metric fixed-point theory.

Contraction Mapping
-------------------

A mapping :math:`f: X \to X` on a complete metric space :math:`(X, d)` is a
**contraction** if there exists a constant :math:`0 \leq q < 1` such that:

.. math::

   d\big(f(x), f(y)\big) \leq q \cdot d(x, y) \quad \forall x, y \in X

The constant :math:`q` is called the **Lipschitz constant**.

Banach's Theorem
----------------

If :math:`f` is a contraction on a complete metric space, then:

1. :math:`f` has a **unique fixed point** :math:`x^* \in X`
2. For any initial point :math:`x_0`, the iteration

   .. math::

      x_{n+1} = f(x_n)

   converges to :math:`x^*` at rate :math:`O(q^n)`

Power Iteration as Contraction
-----------------------------

Faraday's iteration is:

.. math::

   f(\mathbf{x}) = \text{normalize}(T \, \mathbf{x})

This is **not** a strict contraction on the unit sphere under Euclidean distance.
However, on the unit sphere, the operator :math:`\mathbf{x} \mapsto T\mathbf{x}`
has a well-defined dominant eigenvector — the **Perron-Frobenius eigenvector** —
which is the fixed point of the un-normalized iteration.

Normalization maps the iteration back to the unit sphere :math:`S^{d-1}`.
On :math:`S^{d-1}`, power iteration converges geometrically:

.. math::

   \|\mathbf{x}_n - \mathbf{x}^*\| \leq C \cdot |\lambda_1 / \lambda_2|^n

where :math:`\lambda_1` and :math:`\lambda_2` are the largest and second-largest
singular values of :math:`T`. The convergence rate is determined by the
**spectral gap** :math:`|\lambda_1/\lambda_2|`.

Spectral Properties of T
~~~~~~~~~~~~~~~~~~~~~~~~

- :math:`T` is learned from data; its singular values depend on the geometry set
- A **large spectral gap** → fast convergence (few iterations needed)
- A **small spectral gap** → slow convergence (many iterations needed)
- A rank-deficient :math:`T` (fewer samples than dimensions) has :math:`\lambda_i = 0`
  for :math:`i > \text{rank}(T)`, making convergence impossible

Faraday requires :math:`n_{\text{geometries}} \geq d = 16` for reliable convergence.

Convergence Criteria in Faraday
------------------------------

Faraday uses two convergence criteria in ``GodTensor.find_fixed_point``:

1. **Fixed-point residual** (primary):

   .. math::

      \|\mathbf{x}_{n+1} - \mathbf{x}_n\| < \epsilon_{\text{tol}}

   with :math:`\epsilon_{\text{tol}} = 10^{-7}`.

2. **Iteration limit** (safety cap):

   .. math::

      n > n_{\text{max}}

   with :math:`n_{\text{max}} = 500` to prevent infinite loops.

If neither criterion is met, the solver raises ``ConvergenceError``.

Why Normalization?
~~~~~~~~~~~~~~~~~

Without normalization, :math:`\mathbf{x}_{n+1} = T\mathbf{x}_n` grows or decays
geometrically as :math:`\|T\|^n`. Normalization constrains the iterates to the
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

- S. Banach. "Sur les opérations dans les ensembles abstraits et leur
  application aux équations intégrales." *Fundamenta Mathematicae*, 3:133–181, 1922.
- D. Kincaid, E. Cheney. *Numerical Analysis*, 3rd ed., Brooks/Cole, 2002.
  Chapter 9: Iterative Methods for Solving Linear Systems.
- A. N. Langville, C. D. Meyer. *Google's PageRank and Beyond*. Johns Hopkins
  University Press, 2006.
