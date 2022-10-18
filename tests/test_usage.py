#!/usr/bin/env python3

import pytest
import numpy as np
from mftma.alldata_dimension_analysis import alldata_dimension_analysis
from mftma.manifold_analysis_correlation import manifold_analysis_corr


@pytest.fixture()
def random_data():
    np.random.seed(0)
    X = [np.random.randn(5000, 50) for i in range(100)]
    return X


def test_mftma(random_data):
    kappa = 0
    n_t = 200
    capacity_all, radius_all, dimension_all, center_correlation, K = manifold_analysis_corr(random_data, kappa, n_t)
    avg_capacity = 1/np.mean(1/capacity_all)
    avg_radius = np.mean(radius_all)
    avg_dimension = np.mean(dimension_all)
    assert pytest.approx(avg_capacity, 0.04)
    assert pytest.approx(avg_radius, 1.47)
    assert pytest.approx(avg_dimension, 36.17)
    assert pytest.approx(center_correlation, 0.01)
    assert pytest.approx(K, 1)


def test_total_dimension_analysis(random_data):
    percentage = 0.90
    D_participation_ratio, D_explained_variance, D_feature = alldata_dimension_analysis(random_data, perc=percentage)
    assert pytest.approx(D_participation_ratio, 2499)
    assert pytest.approx(D_explained_variance, 2548)
    assert pytest.approx(D_feature, 5000)
