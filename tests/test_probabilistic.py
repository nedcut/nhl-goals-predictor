import numpy as np


def test_poisson_pmf_matrix_rows_sum_to_one():
    from src.probabilistic import poisson_pmf_matrix

    mu = np.array([3.5, 6.2, 0.8])
    pmf = poisson_pmf_matrix(mu, max_goals=25)
    assert pmf.shape == (3, 26)
    assert np.all(pmf >= 0)
    assert np.allclose(pmf.sum(axis=1), 1.0, atol=1e-8)


def test_nb2_pmf_matrix_rows_sum_to_one():
    from src.probabilistic import nb2_pmf_matrix

    mu = np.array([3.5, 6.2, 0.8])
    pmf = nb2_pmf_matrix(mu, alpha=0.4, max_goals=30)
    assert pmf.shape == (3, 31)
    assert np.all(pmf >= 0)
    assert np.allclose(pmf.sum(axis=1), 1.0, atol=1e-8)


def test_nb2_alpha_zero_matches_poisson():
    from src.probabilistic import nb2_pmf_matrix, poisson_pmf_matrix

    mu = np.array([4.0, 7.0])
    pmf_p = poisson_pmf_matrix(mu, max_goals=25)
    pmf_nb0 = nb2_pmf_matrix(mu, alpha=0.0, max_goals=25)
    assert np.allclose(pmf_nb0, pmf_p, atol=1e-10)


def test_prob_over_from_pmf_threshold_65():
    from src.probabilistic import poisson_pmf_matrix, prob_over_from_pmf

    mu = np.array([6.0])
    pmf = poisson_pmf_matrix(mu, max_goals=30)
    p_over_65 = prob_over_from_pmf(pmf, threshold=6.5)[0]
    # P(X >= 7) must be between 0 and 1
    assert 0.0 <= p_over_65 <= 1.0


def test_reliability_curve_bins_sum_counts():
    from src.probabilistic import reliability_curve

    p = np.linspace(0.01, 0.99, 100)
    y = (p > 0.5).astype(int)
    bins = reliability_curve(p, y, n_bins=10)
    assert sum(b.count for b in bins) == 100

