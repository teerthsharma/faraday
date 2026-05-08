Maxwell's Equations
====================

Faraday uses the frequency-domain Maxwell's equations as the physical foundation
for computing electromagnetic cavity modes. These equations govern how electric
(:math:`\mathbf{E}`) and magnetic (:math:`\mathbf{H}`) fields interact inside
a resonant cavity.

The Curl Equations
------------------

In a source-free, time-harmonic region (:math:`e^{-i\omega t}` convention):

.. math::

   \nabla \times \mathbf{E} &= i\omega\mu\;\mathbf{H} \tag{1a}

   \nabla \times \mathbf{H} &= -i\omega\epsilon\;\mathbf{E} \tag{1b}

Taking the curl of (1a) and substituting (1b) gives the **vector Helmholtz equation**:

.. math::

   \nabla^2\mathbf{E} + k^2\mathbf{E} = 0, \qquad
   k = \omega\sqrt{\mu\epsilon}

For a rectangular PEC cavity with perfectly conducting walls, the tangential
electric field vanishes on the boundary (:math:`\mathbf{\hat{n}}\times\mathbf{E}=0`),
yielding analytical eigenmodes. Faraday uses a **finite-difference frequency-domain
(FDFD)** discretization to handle arbitrary geometries.

FDFD Discretization
-------------------

Faraday discretizes the Helmholtz equation on a regular grid using a
**5-point finite-difference stencil**:

.. math::

   \frac{E_{i+1,j} - 2E_{i,j} + E_{i-1,j}}{\Delta x^2}
   + \frac{E_{i,j+1} - 2E_{i,j} + E_{i,j-1}}{\Delta y^2}
   + k^2 E_{i,j} = 0

This leads to a **sparse matrix eigenvalue problem**:

.. math::

   \mathbf{L}\;\mathbf{e} = k^2\;\mathbf{e}

where :math:`\mathbf{L}` is the discretized Laplacian (negative-semidefinite)
and :math:`k^2` are the squared resonant wave numbers.

Eigenvalue Spectrum
~~~~~~~~~~~~~~~~~~~

- :math:`L` is negative-semidefinite, so its eigenvalues
  :math:`\lambda = -k^2 \le 0`.
- The fundamental cavity mode is the eigenpair with the *smallest*
  :math:`k`, i.e. the eigenvalue of :math:`L` closest to zero.
- Higher modes have larger :math:`|k|` and increasingly oscillatory
  patterns.

Faraday's solver uses :func:`scipy.sparse.linalg.eigsh` in
**shift-invert mode** with ``sigma=0`` and ``which="LM"`` to target the
eigenvalues nearest the origin.  Shift-invert internally inverts
:math:`L - \sigma I` once and runs Lanczos on the inverse, which is
both faster and more numerically stable than the previous
``which="SM"`` (smallest-magnitude) iteration on :math:`L` itself.

Analytic reference
~~~~~~~~~~~~~~~~~~

For a rectangular PEC cavity of width :math:`w` and height :math:`h`
the closed-form eigenvalues are

.. math::

   k_{mn}^2 = \left(\frac{m\pi}{w}\right)^2
            + \left(\frac{n\pi}{h}\right)^2,
   \qquad m, n = 1, 2, 3, \dots

This identity is verified end-to-end in
``tests/test_em_solver_analytic.py`` against the FDFD output to within
:math:`5 \times 10^{-3}` relative error at :math:`80 \times 80`
resolution.

H-Field Derivation
------------------

Once the E-field eigenmode is known, the H-field follows from Maxwell's curl equation:

.. math::

   \mathbf{H} = \frac{i}{\omega\mu}\nabla\times\mathbf{E}

In 2D (transverse electric, :math:`E_z` only):

.. math::

   H_x = \frac{i}{\omega\mu}\frac{\partial E_z}{\partial y}, \qquad
   H_y = -\frac{i}{\omega\mu}\frac{\partial E_z}{\partial x}

The **coupled E/H pair** is then passed to persistent homology analysis.

PEC Boundary Conditions
-----------------------

A Perfect Electric Conductor (PEC) imposes:

.. math::

   \mathbf{\hat{n}}\times\mathbf{E} = \mathbf{0} \quad \text{on } \partial\Omega

In the FDFD grid, this means setting the tangential field components at the
boundary to zero before applying the stencil.

Physical Units
--------------

Faraday works in **normalized units** (:math:`\mu = \epsilon = 1`) to focus on
topological structure rather than absolute field values. The coupling analysis
is unit-invariant — only the relative distribution of field energy matters.

References
----------

- T. A. M. (Taflove), *Computational Electrodynamics*, Artech House, 1995
- J. D. Jackson, *Classical Electrodynamics*, 3rd ed., Wiley, 1998
- M. N. O. Sadiku, *Numerical Techniques in Electromagnetics*, CRC Press, 2000
