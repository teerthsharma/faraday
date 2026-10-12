# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
Correctness tests: each one pins a mathematical or physical invariant that a
buggy implementation violates (analytic cavity eigenvalues, Maxwell energy
equipartition, nodal-domain counts, exact gradients, Banach contraction).
"""

import numpy as np
import pytest

from faraday import CavityGeometry, CavityShape, solve_cavity_modes

# Zeros of the Bessel functions: TM01 → j_{0,1}, TM11 → j_{1,1}
J01 = 2.404825557695773
J11 = 3.831705970207512


def _rect(w, h, nx, ny, num_modes, seed=0):
    geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(w, h))
    return solve_cavity_modes(geom, nx=nx, ny=ny, num_modes=num_modes, seed=seed)


def _analytic_rect_k(w, h, n):
    ks = sorted(
        np.pi * np.hypot(m / w, p / h) for m in range(1, 8) for p in range(1, 8)
    )
    return np.array(ks[:n])


# ---------------------------------------------------------------------------
# FDFD solver
# ---------------------------------------------------------------------------


class TestCavityPhysics:
    def test_rectangular_eigenvalues_match_analytic(self):
        """TM_mn: k = π √((m/w)² + (n/h)²), lowest first, within O(h²) error."""
        res = _rect(2.0, 1.0, 41, 41, 4)
        k = np.array(res["k_values"])
        np.testing.assert_allclose(k, _analytic_rect_k(2.0, 1.0, 4), rtol=5e-3)

    def test_modes_sorted_fundamental_first(self):
        res = _rect(1.7, 1.1, 30, 30, 6)
        k = np.array(res["k_values"])
        assert np.all(np.diff(k) >= -1e-9)
        assert res["e_modes"]["mode_0"]["k"] == pytest.approx(k[0])

    def test_eigenvalue_error_shrinks_with_resolution(self):
        """Second-order scheme: halving the grid spacing cuts the error ~4x."""
        exact = np.pi * np.hypot(1 / 2.0, 1 / 1.0)
        err_coarse = abs(_rect(2.0, 1.0, 21, 21, 1)["k_values"][0] - exact)
        err_fine = abs(_rect(2.0, 1.0, 41, 41, 1)["k_values"][0] - exact)
        assert err_fine < err_coarse / 3

    def test_circular_eigenvalues_match_bessel_zeros(self):
        """TM01: k = j01 / r; TM11: k = j11 / r and is doubly degenerate."""
        r = 1.0
        geom = CavityGeometry(shape=CavityShape.CIRCULAR, dims=(r,))
        k = solve_cavity_modes(geom, nx=61, num_modes=3, seed=0)["k_values"]
        # Staircase boundary → first-order geometric error, a few percent
        assert k[0] == pytest.approx(J01 / r, rel=0.03)
        assert k[1] == pytest.approx(J11 / r, rel=0.03)
        assert k[2] == pytest.approx(k[1], rel=1e-6)

    def test_circular_ignores_ny(self):
        geom = CavityGeometry(shape=CavityShape.CIRCULAR, dims=(1.0,))
        res = solve_cavity_modes(geom, nx=21, ny=33, num_modes=2, seed=0)
        assert np.array(res["e_modes"]["mode_0"]["field"]).shape == (21, 21)
        assert res["ny"] == 21

    def test_laplacian_symmetric_negative_definite(self):
        from faraday.em_solver import build_laplacian_2d, make_rectangular_grid

        geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(2.0, 1.0))
        X, Y = make_rectangular_grid(2.0, 1.0, 12, 9)
        interior = geom.contains(X, Y)
        L = build_laplacian_2d(12, 9, 2.0 / 11, 1.0 / 8, interior).toarray()
        assert L.shape == (interior.sum(), interior.sum())
        np.testing.assert_array_equal(L, L.T)
        assert np.linalg.eigvalsh(L).max() < 0

    def test_laplacian_rejects_mismatched_mask(self):
        from faraday.em_solver import build_laplacian_2d

        with pytest.raises(ValueError, match="interior mask shape"):
            build_laplacian_2d(5, 4, 1.0, 1.0, np.ones((5, 5), dtype=bool))

    def test_too_small_grid_raises_solver_error(self):
        from faraday import SolverError

        with pytest.raises(SolverError):
            _rect(1.0, 1.0, 3, 3, 2)

    def test_fields_vanish_on_pec_and_are_unit_norm(self):
        res = _rect(1.5, 1.0, 25, 25, 3)
        interior = np.array(res["interior"])
        for key, mode in res["e_modes"].items():
            e = np.array(mode["field"])
            assert np.all(e[~interior] == 0), key
            assert np.linalg.norm(e) == pytest.approx(1.0)

    def test_electric_magnetic_energy_equipartition(self):
        """At resonance ∫|H|² = ∫|E|² (c = μ = ε = 1, H = ∇×E / k)."""
        res = _rect(2.0, 1.0, 61, 61, 3)
        for key in res["e_modes"]:
            e = np.array(res["e_modes"][key]["field"])
            hmag = np.array(res["h_modes"][key]["field"])
            ratio = np.sum(hmag**2) / np.sum(e**2)
            assert ratio == pytest.approx(1.0, rel=0.05), key

    def test_h_components_follow_curl(self):
        """H_x = ∂E/∂y / k, H_y = -∂E/∂x / k at an interior point."""
        res = _rect(2.0, 1.0, 41, 41, 1)
        e = np.array(res["e_modes"]["mode_0"]["field"])
        hx = np.array(res["h_modes"]["mode_0"]["hx"])
        hy = np.array(res["h_modes"]["mode_0"]["hy"])
        k = res["k_values"][0]
        dx, dy = 2.0 / 40, 1.0 / 40
        j, i = 10, 13
        assert hx[j, i] == pytest.approx((e[j + 1, i] - e[j - 1, i]) / (2 * dy) / k)
        assert hy[j, i] == pytest.approx(-(e[j, i + 1] - e[j, i - 1]) / (2 * dx) / k)

    def test_seed_reproducible(self):
        a = _rect(1.3, 0.9, 30, 30, 3, seed=7)
        b = _rect(1.3, 0.9, 30, 30, 3, seed=7)
        assert a["k_values"] == b["k_values"]


# ---------------------------------------------------------------------------
# Persistent homology
# ---------------------------------------------------------------------------


def _blob(cx, cy, n=40, s=0.08):
    y, x = np.mgrid[0:1:n * 1j, 0:1:n * 1j]
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s**2))


class TestTopology:
    def test_single_blob_has_one_component(self):
        from faraday import topological_fingerprint

        fp = topological_fingerprint(_blob(0.5, 0.5), threshold=0.1)
        assert fp["betti_0"] == 1

    def test_two_separated_blobs_have_two_components(self):
        from faraday import topological_fingerprint

        field = _blob(0.25, 0.5) + _blob(0.75, 0.5)
        assert topological_fingerprint(field, threshold=0.1)["betti_0"] == 2

    def test_annulus_has_one_loop(self):
        from faraday import topological_fingerprint

        y, x = np.mgrid[0:1:50j, 0:1:50j]
        rho = np.hypot(x - 0.5, y - 0.5)
        ring = np.exp(-((rho - 0.3) ** 2) / (2 * 0.04**2))
        fp = topological_fingerprint(ring, threshold=0.3)
        assert fp["betti_0"] == 1
        assert fp["betti_1"] == 1

    def test_e_field_betti0_counts_nodal_domains(self):
        """TM_mn has m·n nodal domains of alternating sign."""
        from faraday import topological_fingerprint

        res = _rect(2.0, 1.0, 40, 40, 3)
        # For a 2×1 box the three lowest modes are TM11, TM21, TM31
        for idx, domains in enumerate([1, 2, 3]):
            e = np.array(res["e_modes"][f"mode_{idx}"]["field"])
            assert topological_fingerprint(e)["betti_0"] == domains

    def test_topological_score_finite_without_h1(self):
        from faraday import topological_fingerprint

        fp = topological_fingerprint(_blob(0.5, 0.5), threshold=0.1)
        assert np.isfinite(fp["topological_score"])

    def test_identical_fields_are_perfectly_coupled(self):
        from faraday import coupled_fingerprint

        field = _blob(0.5, 0.5)
        out = coupled_fingerprint(field, field)
        assert out["emd_S"] == pytest.approx(0.0, abs=1e-12)
        assert out["coupling_strength"] == pytest.approx(1.0)

    def test_tiny_field_reports_error(self):
        from faraday import topological_fingerprint

        field = np.zeros((10, 10))
        field[5, 5] = 1.0
        assert "error" in topological_fingerprint(field)


# ---------------------------------------------------------------------------
# Autoencoder gradient
# ---------------------------------------------------------------------------


def test_manifold_projector_gradient_is_exact():
    """One full-batch SGD step equals -lr × finite-difference gradient."""
    from faraday.manifold_projector import ManifoldProjector

    rng = np.random.default_rng(3)
    X = rng.random((6, 10))
    mp = ManifoldProjector(input_dim=10, latent_dim=4)
    mp._init_weights()
    W_enc0, W_dec0 = mp.encoder_weights.copy(), mp.decoder_weights.copy()

    def loss(we, wd):
        h = np.maximum(X @ we.T, 0)
        return np.mean((X - np.maximum(h @ wd.T, 0)) ** 2)

    lr = 1e-3
    mp.fit(list(X), lr=lr, epochs=1, batch_size=len(X))
    grad_enc = (W_enc0 - mp.encoder_weights) / lr
    grad_dec = (W_dec0 - mp.decoder_weights) / lr

    eps = 1e-6
    for W, grad, is_enc in ((W_enc0, grad_enc, True), (W_dec0, grad_dec, False)):
        for idx in [(0, 0), (1, 3), (W.shape[0] - 1, W.shape[1] - 1)]:
            Wp, Wm = W.copy(), W.copy()
            Wp[idx] += eps
            Wm[idx] -= eps
            if is_enc:
                fd = (loss(Wp, W_dec0) - loss(Wm, W_dec0)) / (2 * eps)
            else:
                fd = (loss(W_enc0, Wp) - loss(W_enc0, Wm)) / (2 * eps)
            assert grad[idx] == pytest.approx(fd, rel=1e-4, abs=1e-9)


# ---------------------------------------------------------------------------
# Banach fixed point
# ---------------------------------------------------------------------------


def _matrix_with_spectrum(eigs, seed=0):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((len(eigs), len(eigs)))
    return Q @ np.diag(eigs) @ np.linalg.inv(Q), Q


class TestFixedPoint:
    def test_contraction_finds_eigenvalue_one_not_dominant(self):
        """Power iteration would find λ=2.5; the God Tensor is the λ=1 vector."""
        from faraday.god_tensor import banach_step, fixed_point_operator, initial_vector

        T, Q = _matrix_with_spectrum([2.5, 1.0, 0.4, -0.7])
        G, lam, q = fixed_point_operator(T)
        assert lam.real == pytest.approx(1.0)
        assert q < 1e-4

        x = initial_vector(4)
        for _ in range(10):
            x, delta = banach_step(G, x)
        assert delta < 1e-10
        np.testing.assert_allclose(T @ x, x, atol=1e-9)
        v = Q[:, 1] / np.linalg.norm(Q[:, 1])
        assert min(np.linalg.norm(x - v), np.linalg.norm(x + v)) < 1e-9

    def test_contraction_factor_matches_spectral_ratio(self):
        from faraday.god_tensor import fixed_point_operator

        T, _ = _matrix_with_spectrum([3.0, 1.2, 0.5])
        _, lam, q = fixed_point_operator(T)
        mu = 1.0 + 1e-6 * np.linalg.norm(T, 2)
        assert lam.real == pytest.approx(1.2)
        assert q == pytest.approx(abs(1.2 - mu) / abs(0.5 - mu))

    def test_complex_pair_nearest_one_is_not_a_contraction(self):
        from faraday.god_tensor import fixed_point_operator

        rot = np.array([[1.0, -0.1], [0.1, 1.0]])  # eigenvalues 1 ± 0.1i
        T = np.block([[rot, np.zeros((2, 1))], [np.zeros((1, 2)), np.eye(1) * 3]])
        _, lam, q = fixed_point_operator(T)
        assert abs(lam.imag) == pytest.approx(0.1)
        assert q == pytest.approx(1.0)

    def test_negative_eigenvalue_sign_alignment(self):
        from faraday.god_tensor import banach_step

        T = np.diag([-2.0, 0.5])
        x = np.array([0.6, 0.8])
        for _ in range(60):
            x, delta = banach_step(lambda v: T @ v, x)
        assert delta < 1e-12
        assert abs(x[0]) == pytest.approx(1.0)

    def test_sign_invariant_distance(self):
        from faraday.god_tensor import sign_invariant_distance

        g = np.array([1.0, 0.0])
        assert sign_invariant_distance(-g, g) == pytest.approx(0.0)
        assert sign_invariant_distance(np.array([0.0, 1.0]), g) == pytest.approx(
            np.sqrt(2)
        )


@pytest.fixture(scope="module")
def trained_gt():
    from faraday import GodTensor

    gt = GodTensor(n_geometries=12)
    gt.collect_training_data(nx=20, ny=20, num_modes=2, seed=42)
    gt.learn_T()
    gt.find_fixed_point(iters=500, tol=1e-10)
    return gt


class TestTrainedGodTensor:
    def test_fixed_point_is_certified(self, trained_gt):
        gt = trained_gt
        assert gt.fixed_point_converged
        assert gt.contraction_factor < 1
        lam = gt.fixed_point_eigenvalue.real
        x = gt.god_tensor
        assert np.linalg.norm(x) == pytest.approx(1.0)
        np.testing.assert_allclose(gt.T_matrix @ x, lam * x, atol=1e-8)

    def test_converges_from_generic_start(self, trained_gt):
        """History starts with a real step, not a pre-converged eigenvector."""
        assert trained_gt.convergence_history[0]["delta"] > 1e-3

    def test_god_score_sign_invariant(self, trained_gt):
        gt = trained_gt
        s1 = gt.god_score()
        gt.god_tensor = -gt.god_tensor
        s2 = gt.god_score()
        gt.god_tensor = -gt.god_tensor
        assert s1 == pytest.approx(s2)
        assert 0 <= s1 <= 1

    def test_summary_reports_certificate(self, trained_gt):
        summary = trained_gt.summary()
        assert summary["converged"] is True
        assert summary["contraction_factor"] == trained_gt.contraction_factor
        assert summary["fixed_point_eigenvalue"] == trained_gt.fixed_point_eigenvalue

    def test_h_to_e_uses_pseudo_inverse(self, trained_gt):
        gt = trained_gt
        h_emb = gt.samples[0].h_embedding
        z = gt.projector_h.encode(h_emb)
        expected = np.linalg.pinv(gt.T_matrix) @ z
        expected /= np.linalg.norm(expected) + 1e-10
        np.testing.assert_allclose(gt.get_h_to_e_map(h_emb), expected)

    def test_maps_require_training(self):
        from faraday import GodTensor

        gt = GodTensor()
        with pytest.raises(ValueError):
            gt.get_e_to_h_map(np.zeros(50))
        with pytest.raises(ValueError):
            gt.get_h_to_e_map(np.zeros(50))
        assert gt.god_score() == 0.0

    def test_predict_returns_h0_barcode_with_essential_class(self, trained_gt):
        barcode = trained_gt.predict(w=2.0, h=1.0)
        assert len(barcode) > 0
        assert sum(np.isinf(d) for _, d in barcode) == 1

    def test_save_load_roundtrip(self, trained_gt, tmp_path):
        from faraday import GodTensor

        path = tmp_path / "gt.pkl"
        trained_gt.save(str(path))
        loaded = GodTensor.load(str(path))
        np.testing.assert_array_equal(loaded.god_tensor, trained_gt.god_tensor)
        np.testing.assert_array_equal(loaded.T_matrix, trained_gt.T_matrix)

    def test_checkpoint_roundtrip(self, trained_gt, tmp_path):
        from faraday import GodTensor

        path = str(tmp_path / "ckpt.npz")
        state = np.random.default_rng(1).bit_generator.state
        trained_gt.save_checkpoint(path, epoch=7, rng_state=state)
        god, epoch, rng_state = GodTensor.load_checkpoint(path)
        np.testing.assert_array_equal(god, trained_gt.god_tensor)
        assert epoch == 7
        assert rng_state == state

    def test_training_sample_roundtrip(self, trained_gt):
        from faraday.god_tensor import TrainingSample

        s = trained_gt.samples[0]
        back = TrainingSample.from_dict(s.to_dict())
        np.testing.assert_array_equal(back.e_embedding, s.e_embedding)
        assert back.geometry_params == tuple(s.geometry_params)


class TestPrediction:
    def test_knn_uses_only_same_shape_neighbours(self, trained_gt):
        from faraday.predict import predict_eh_barcode

        circles = [s for s in trained_gt.samples if len(s.geometry_params) == 1]
        if not circles:
            pytest.skip("no circular geometry in this training draw")
        pred = predict_eh_barcode(trained_gt, (1.0,), "circle")
        expected = np.mean([s.e_fingerprint["betti_0"] for s in circles[:5]])
        assert min(s.e_fingerprint["betti_0"] for s in circles) <= pred[
            "knn_e_fingerprint"
        ]["betti_0"] <= max(s.e_fingerprint["betti_0"] for s in circles)
        assert np.isfinite(expected)

    def test_far_query_does_not_underflow(self, trained_gt):
        from faraday.predict import predict_eh_barcode

        pred = predict_eh_barcode(trained_gt, (100.0, 100.0), "rect")
        assert np.isfinite(pred["knn_e_fingerprint"]["betti_0"])
        assert np.isfinite(pred["god_distance_e"])
        assert 0 <= pred["god_distance_e"] <= np.sqrt(2) + 1e-9

    def test_unknown_shape_falls_back_to_all_samples(self, trained_gt):
        from faraday.predict import predict_eh_barcode

        pred = predict_eh_barcode(trained_gt, (1.0, 1.0, 1.0), "box")
        assert np.isfinite(pred["knn_h_fingerprint"]["betti_0"])

    def test_predict_requires_fixed_point(self, trained_gt):
        from faraday import GodTensor
        from faraday.predict import predict_eh_barcode

        gt = GodTensor(samples=trained_gt.samples, T_matrix=trained_gt.T_matrix)
        with pytest.raises(ValueError):
            predict_eh_barcode(gt, (2.0, 1.0), "rect")

    def test_benchmark_against_fdfd(self, trained_gt):
        from faraday.predict import benchmark

        out = benchmark(
            trained_gt, [(2.0, 1.0), (0.9,)], ["rect", "circle"], nx=20, ny=20
        )
        assert out["n_test"] == 2
        # Fundamental mode: one nodal domain, connected |H|
        assert out["avg_e_betti0_error"] == pytest.approx(0.0)
        assert out["avg_h_betti0_error"] == pytest.approx(0.0)
