#! /usr/bin/env python3
"""Tests for the metrics module."""

import unittest
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from newbie import estimators, metrics
from newbie.metrics import MAPEMeasure, RMSEMeasure, RSquaredMeasure


def load_idata(rootdir):
    """Load inference data from json file."""
    data = az.from_json(Path(rootdir, 'test-data', 'inference_data.json'))
    return data


def load_truth(rootdir):
    """Load the true parameter values to an xarray dataset."""
    df = pd.read_csv(Path(rootdir, 'test-data', 'synthetic_truth.csv'),
                     index_col=0)
    df = df.loc['data'].to_dict()
    return xr.Dataset(df)


def test_calculate_distance(rootdir):
    """Test the calculate distance method."""
    idata = load_idata(rootdir)
    truth = load_truth(rootdir)
    for m in ['peak', 'mean', 'mode']:
        metric = metrics.Metric(m, idata, truth)
        metric.calculate_distance()
        metric.calculate_distance(normalize='max')
        metric.calculate_distance(normalize='truth')
        metric.calculate_distance(normalize='estimator')
        metric.calculate_distance(normalize='range')
        metric.calculate_distance(normalize='abssum')
        metric.calculate_distance(absolute=True)
        assert True
    for m in ['peak', 'mean', 'mode']:
        metric = metrics.Metric(m,
                                idata,
                                truth,
                                data_vars=['burnupA', 'powerA'])
        metric.calculate_distance()
        metric.calculate_distance(normalize='max')
        metric.calculate_distance(normalize='truth')
        metric.calculate_distance(absolute=True)
        assert True


def test_calculate_metric(rootdir):
    """Test the calculate distance method."""
    idata = load_idata(rootdir)
    truth = load_truth(rootdir)
    for m in ['peak', 'mean', 'mode']:
        metric = metrics.Metric('peak', idata, truth)
        metric.calculate_metric_norm()
        metric.calculate_metric_norm(unit=True)
        assert True


def test_metric_set_distances(rootdir):
    """Test the MetricSet class."""
    idata = load_idata(rootdir)
    truth = load_truth(rootdir)
    ms = metrics.MetricSet(idata, truth)
    ms.calculate_distances()
    assert True


def test_metric_set_metrics(rootdir):
    """Test the MetricSet class."""
    idata = load_idata(rootdir)
    truth = load_truth(rootdir)
    ms = metrics.MetricSet(idata, truth)
    ms.calculate_metric_norms()
    assert True


class AccuracyMeasureTests(unittest.TestCase):

    def test_r_squared_measure(self):
        truth = np.array([1, 2, 3, 4, 5])
        prediction = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
        measure = RSquaredMeasure()
        accuracy = measure(truth, prediction)
        self.assertAlmostEqual(accuracy, 0.981, places=4)

    def test_mape_measure(self):
        truth = np.array([1, 2, 3, 4, 5])
        prediction = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
        measure = MAPEMeasure()
        accuracy = measure(truth, prediction)
        self.assertAlmostEqual(accuracy, 7.0333, places=4)

    def test_rmse_measure(self):
        truth = np.array([1, 2, 3, 4, 5])
        prediction = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
        measure = RMSEMeasure()
        accuracy = measure(truth, prediction)
        self.assertAlmostEqual(accuracy, 0.1949358, places=6)
