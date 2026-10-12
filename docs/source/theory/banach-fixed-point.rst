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

Shifted Inverse Iteration as Contraction
----------------------------------------

The God Tensor is the invariant of the learned coupling, :math:`T\mathbf{g} = \mathbf{g}`:
the eigenvector of :math:`T` whose eigenvalue :math:`\lambda^*` is nearest 1.

Plain power iteration, :math:`f(\mathbf{x}) = \text{normalize}(T\mathbf{x})`,
cannot find it in general. Every real eigenvector of :math:`T` is a fixed point
of :math:`f`, but only the one with the largest :math:`|\lambda|` is
*attracting*; all others are repelling. A learned :math:`T` need not have
:math:`\lambda_{\max} = 1` (on the default training set :math:`\lambda_{\max} \approx 1.25`
while :math:`\lambda^* = 1.000`), so power iteration converges to the wrong vector.

Faraday therefore iterates the shifted inverse

.. math::

   G(\mathbf{x}) = \frac{(T - \mu I)^{-1}\mathbf{x}}{\|(T - \mu I)^{-1}\mathbf{x}\|},
   \qquad \mu = 1 + \varepsilon

The eigenvalues of :math:`(T - \mu I)^{-1}` are :math:`1/(\lambda_j - \mu)`, so the
eigenvector with :math:`\lambda_j` nearest :math:`\mu` dominates and :math:`G`
contracts onto it (projectively, i.e. up to sign) with Lipschitz constant

.. math::

   q = \frac{|\lambda^* - \mu|}{|\lambda_2 - \mu|} < 1

where :math:`\lambda_2` is the next-nearest eigenvalue to :math:`\mu`. The small
offset :math:`\varepsilon = 10^{-6}\|T\|_2` keeps :math:`T - \mu I` invertible
when :math:`\lambda^* = 1` exactly; it only makes :math:`q` smaller. Iterates satisfy

.. math::

   \|\mathbf{x}_n - \mathbf{x}^*\| \leq C \cdot q^n

If the eigenvalue nearest 1 is a complex pair, :math:`q = 1`: there is no real
fixed point and the iteration does not converge.

Convergence Criteria in Faraday
------------------------------

``GodTensor.find_fixed_point`` starts from a generic (non-eigenvector) unit
vector, so convergence is earned rather than assumed, and stops when:

1. **Fixed-point residual** (primary):

   .. math::

      \|\mathbf{x}_{n+1} - \mathbf{x}_n\| < \epsilon_{\text{tol}}

   with :math:`\epsilon_{\text{tol}} = 10^{-7}` (sign-invariant).

2. **Iteration limit** (safety cap): :math:`n > n_{\text{max}} = 500`.

If the cap is hit, a warning is logged and ``fixed_point_converged`` stays
``False``. The result is certified by ``fixed_point_eigenvalue``
(:math:`\lambda^*`), ``contraction_factor`` (:math:`q`), the eigen-residual
:math:`\|T\mathbf{x} - \lambda^*\mathbf{x}\|` and the fixed-point defect
:math:`\|T\mathbf{x} - \mathbf{x}\| = |\lambda^* - 1|`.

Why Normalization?
~~~~~~~~~~~~~~~~~

Without normalization, :math:`\mathbf{x}_{n+1} = (T - \mu I)^{-1}\mathbf{x}_n`
grows geometrically as :math:`|\lambda^* - \mu|^{-n}`. Normalization constrains the
iterates to the unit sphere, making the iteration **scale-invariant**: only the
direction of :math:`\mathbf{x}` matters, which is all an eigenvector defines.

Connection to Google PageRank
-----------------------------

PageRank also seeks the eigenvalue-1 eigenvector of an operator, but there the
Google matrix is column-stochastic, so 1 is guaranteed to be its *dominant*
eigenvalue and plain power iteration suffices. A learned :math:`T` carries no
such guarantee, which is why Faraday uses the shifted inverse.

References
----------

- S. Banach. "Sur les opérations dans les ensembles abstraits et leur
  application aux équations intégrales." *Fundamenta Mathematicae*, 3:133–181, 1922.
- D. Kincaid, E. Cheney. *Numerical Analysis*, 3rd ed., Brooks/Cole, 2002.
  Chapter 9: Iterative Methods for Solving Linear Systems.
- A. N. Langville, C. D. Meyer. *Google's PageRank and Beyond*. Johns Hopkins
  University Press, 2006.
