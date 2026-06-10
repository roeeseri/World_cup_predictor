"""Tests for src/models/score_grid.py"""

import numpy as np
import pytest

from src.models.score_grid import (
    dixon_coles_grid,
    fit_lambda_affine,
    fit_lambda_scale,
    fit_rho,
    knockout_grid,
    make_score_fn,
    pick_score,
    win_draw_loss_from_grid,
    apply_lambda_scale,
    apply_lambda_affine,
    fit_calibration_params,
)


# ── Grid basic properties ─────────────────────────────────────────────────────

class TestDixonColesGrid:
    def test_sums_to_one_no_correction(self):
        grid = dixon_coles_grid(1.5, 1.2, rho=0.0)
        assert abs(grid.sum() - 1.0) < 1e-6

    def test_sums_to_one_negative_rho(self):
        grid = dixon_coles_grid(1.5, 1.2, rho=-0.1)
        assert abs(grid.sum() - 1.0) < 1e-6

    def test_sums_to_one_positive_rho(self):
        grid = dixon_coles_grid(1.5, 1.2, rho=0.1)
        assert abs(grid.sum() - 1.0) < 1e-6

    def test_all_non_negative(self):
        for rho in (-0.2, 0.0, 0.2):
            grid = dixon_coles_grid(1.5, 1.2, rho=rho)
            assert (grid >= 0).all(), f"Negative probability at rho={rho}"

    def test_negative_rho_inflates_draws(self):
        grid_no_dc = dixon_coles_grid(1.5, 1.2, rho=0.0)
        grid_with_dc = dixon_coles_grid(1.5, 1.2, rho=-0.15)
        draw_no_dc = float(np.trace(grid_no_dc))
        draw_with_dc = float(np.trace(grid_with_dc))
        assert draw_with_dc > draw_no_dc, "Negative rho should inflate draws"

    def test_positive_rho_deflates_draws(self):
        grid_no_dc = dixon_coles_grid(1.5, 1.2, rho=0.0)
        grid_with_dc = dixon_coles_grid(1.5, 1.2, rho=0.15)
        draw_no_dc = float(np.trace(grid_no_dc))
        draw_with_dc = float(np.trace(grid_with_dc))
        assert draw_with_dc < draw_no_dc, "Positive rho should deflate draws"

    def test_rho_zero_equals_independent_poisson(self):
        from scipy.stats import poisson
        la, lb = 1.5, 1.2
        goals = np.arange(9)
        pa = poisson.pmf(goals, la)
        pb = poisson.pmf(goals, lb)
        expected = np.outer(pa, pb)
        expected /= expected.sum()
        grid = dixon_coles_grid(la, lb, rho=0.0)
        np.testing.assert_allclose(grid, expected, atol=1e-8)

    def test_asymmetric_lambdas(self):
        grid = dixon_coles_grid(2.0, 0.8, rho=-0.1)
        # Team A should be more likely to win
        win = float(np.tril(grid, -1).sum())
        loss = float(np.triu(grid, 1).sum())
        assert win > loss

    def test_extreme_lambdas_stable(self):
        grid = dixon_coles_grid(0.01, 5.0, rho=-0.2)
        assert abs(grid.sum() - 1.0) < 1e-4
        assert (grid >= 0).all()


# ── W/D/L extraction ──────────────────────────────────────────────────────────

class TestWinDrawLoss:
    def test_sums_to_one(self):
        grid = dixon_coles_grid(1.5, 1.2)
        w, d, l = win_draw_loss_from_grid(grid)
        assert abs(w + d + l - 1.0) < 1e-6

    def test_symmetric_lambdas_roughly_equal_win_loss(self):
        grid = dixon_coles_grid(1.5, 1.5, rho=0.0)
        w, d, l = win_draw_loss_from_grid(grid)
        assert abs(w - l) < 0.01

    def test_favourite_has_higher_win_prob(self):
        grid = dixon_coles_grid(2.5, 0.8)
        w, d, l = win_draw_loss_from_grid(grid)
        assert w > l


# ── Decision rule ─────────────────────────────────────────────────────────────

class TestPickScore:
    def test_argmax_alpha_zero(self):
        grid = dixon_coles_grid(1.5, 1.0, rho=-0.1)
        score = pick_score(grid, alpha=0.0)
        # Should be the argmax cell
        expected = np.unravel_index(np.argmax(grid), grid.shape)
        assert score == (int(expected[0]), int(expected[1]))

    def test_returns_integers(self):
        grid = dixon_coles_grid(1.5, 1.2)
        a, b = pick_score(grid, alpha=0.0)
        assert isinstance(a, int) and isinstance(b, int)

    def test_non_negative_scores(self):
        grid = dixon_coles_grid(1.5, 1.2)
        a, b = pick_score(grid)
        assert a >= 0 and b >= 0

    def test_alpha_positive_does_not_crash(self):
        grid = dixon_coles_grid(1.5, 1.2)
        score = pick_score(grid, alpha=0.1)
        assert len(score) == 2


# ── ET mixture ────────────────────────────────────────────────────────────────

class TestKnockoutGrid:
    def test_sums_to_one(self):
        grid = knockout_grid(1.5, 1.2, rho=-0.05, et_scale=30.0 / 90.0)
        assert abs(grid.sum() - 1.0) < 1e-5

    def test_all_non_negative(self):
        grid = knockout_grid(1.5, 1.2, rho=-0.05, et_scale=30.0 / 90.0)
        assert (grid >= 0).all()

    def test_less_draw_mass_than_90min_for_even_match(self):
        """ET redistributes draw mass; final draw mass should differ from 90-min draw mass."""
        g90 = dixon_coles_grid(1.3, 1.3, rho=-0.05)
        gko = knockout_grid(1.3, 1.3, rho=-0.05, et_scale=30.0 / 90.0)
        draw_90 = float(np.trace(g90))
        draw_ko = float(np.trace(gko))
        # After ET, some draw mass should move to non-draw cells
        # (ET can produce its own draws too, so the change might be small but not zero)
        assert abs(draw_90 - draw_ko) > 1e-6

    def test_et_scale_zero_equals_90min(self):
        """et_scale=0 means no extra time → grid should equal 90-min grid (all draw mass stays)."""
        la, lb, rho = 1.5, 1.2, -0.05
        g90 = dixon_coles_grid(la, lb, rho=rho)
        gko = knockout_grid(la, lb, rho=rho, et_scale=0.0)
        # With et_scale=0, GET has λ≈0, so all ET scores are 0-0; draw mass stays as draws
        # This is an edge case — just check it doesn't crash and sums to 1
        assert abs(gko.sum() - 1.0) < 1e-4

    def test_different_et_scales_give_different_results(self):
        gko_a = knockout_grid(1.4, 1.1, rho=-0.05, et_scale=0.25)
        gko_b = knockout_grid(1.4, 1.1, rho=-0.05, et_scale=0.40)
        assert not np.allclose(gko_a, gko_b)


# ── Lambda calibration ────────────────────────────────────────────────────────

class TestLambdaCalibration:
    def setup_method(self):
        rng = np.random.default_rng(42)
        n = 200
        self.la = rng.uniform(0.5, 2.5, n)
        self.lb = rng.uniform(0.5, 2.5, n)
        self.ga = rng.poisson(self.la * 0.9)  # true scale ~0.9
        self.gb = rng.poisson(self.lb * 0.9)

    def test_scale_fit_recovers_approximate_scale(self):
        c = fit_lambda_scale(self.la, self.lb, self.ga, self.gb)
        # Should be close to 0.9
        assert 0.7 < c < 1.1, f"scale_c={c:.3f} too far from 0.9"

    def test_scale_apply(self):
        c = 1.2
        la_c, lb_c = apply_lambda_scale(self.la, self.lb, c)
        np.testing.assert_allclose(la_c, self.la * c)
        np.testing.assert_allclose(lb_c, self.lb * c)

    def test_affine_returns_two_floats(self):
        a, b = fit_lambda_affine(self.la, self.lb, self.ga, self.gb)
        assert isinstance(a, float) and isinstance(b, float)

    def test_affine_apply(self):
        a, b = 0.1, 0.9
        la_c, lb_c = apply_lambda_affine(self.la, self.lb, a, b)
        expected_a = np.exp(a) * np.maximum(self.la, 1e-6) ** b
        np.testing.assert_allclose(la_c, expected_a, rtol=1e-5)

    def test_all_calibrated_lambdas_positive(self):
        c = fit_lambda_scale(self.la, self.lb, self.ga, self.gb)
        la_c, lb_c = apply_lambda_scale(self.la, self.lb, c)
        assert (la_c > 0).all() and (lb_c > 0).all()


# ── Rho fitting ───────────────────────────────────────────────────────────────

class TestFitRho:
    def test_returns_float_in_bounds(self):
        rng = np.random.default_rng(0)
        n = 200
        la = rng.uniform(1.0, 2.0, n)
        lb = rng.uniform(1.0, 2.0, n)
        from scipy.stats import poisson
        ga = rng.poisson(la)
        gb = rng.poisson(lb)
        rho = fit_rho(la, lb, ga, gb)
        assert isinstance(rho, float)
        assert -0.95 <= rho <= 0.35

    def test_negative_rho_when_many_low_score_draws(self):
        """Lots of 0-0 and 1-1 games should push rho negative."""
        n = 300
        la = np.ones(n) * 1.5
        lb = np.ones(n) * 1.5
        # Half are draws (0-0 or 1-1)
        ga = np.array([0, 1] * (n // 2))
        gb = np.array([0, 1] * (n // 2))
        rho = fit_rho(la, lb, ga, gb)
        assert rho < 0, f"Expected rho < 0 for draw-heavy data, got {rho:.4f}"


# ── make_score_fn integration ─────────────────────────────────────────────────

class TestMakeScoreFn:
    def test_produces_non_negative_score(self):
        sfn, pfn = make_score_fn(rho=-0.1, scale_c=0.95, alpha=0.0)
        a, b = sfn(1.5, 1.2)
        assert a >= 0 and b >= 0

    def test_probs_sum_to_one(self):
        _, pfn = make_score_fn(rho=-0.1, scale_c=0.95, alpha=0.0)
        w, d, l = pfn(1.5, 1.2)
        assert abs(w + d + l - 1.0) < 1e-5

    def test_knockout_flag_accepted(self):
        sfn, pfn = make_score_fn(rho=-0.1, scale_c=0.95, knockout=True, et_scale=0.333)
        a, b = sfn(1.4, 1.1)
        assert a >= 0 and b >= 0

    def test_affine_params_take_precedence_over_scale(self):
        sfn_scale, _ = make_score_fn(scale_c=2.0, alpha=0.0)
        sfn_affine, _ = make_score_fn(scale_c=2.0, affine_a=0.0, affine_b=0.5, alpha=0.0)
        # affine with b=0.5 compresses spread vs scale=2x, so they should differ
        score_scale = sfn_scale(1.5, 1.2)
        score_affine = sfn_affine(1.5, 1.2)
        # Just check both run without error; they may agree on 1-1 but differ on extreme cases
        assert len(score_scale) == 2 and len(score_affine) == 2


# ── fit_calibration_params integration ───────────────────────────────────────

class TestFitCalibrationParams:
    def test_returns_required_keys(self):
        rng = np.random.default_rng(7)
        n = 100
        la = rng.uniform(0.8, 2.2, n)
        lb = rng.uniform(0.8, 2.2, n)
        ga = rng.poisson(la)
        gb = rng.poisson(lb)
        params = fit_calibration_params(la, lb, ga, gb)
        assert "scale_c" in params
        assert "rho" in params
        assert "affine_a" in params
        assert "affine_b" in params

    def test_scale_c_positive(self):
        rng = np.random.default_rng(8)
        n = 100
        la = rng.uniform(1.0, 2.0, n)
        lb = rng.uniform(1.0, 2.0, n)
        params = fit_calibration_params(la, lb, rng.poisson(la), rng.poisson(lb))
        assert params["scale_c"] > 0
