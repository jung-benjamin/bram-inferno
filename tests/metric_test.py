#! /usr/bin/env python3
"""Tests for the metrics module."""

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from newbie import metrics
from newbie.inferencedata import InferenceDataSet
from newbie.metrics import (MAPEMeasure, MetricDataSet, RMSEMeasure,
                            RSquaredMeasure)


@pytest.fixture
def inference_data(rootdir):
    """Load inference data from json file."""
    data = az.from_json(Path(rootdir, 'test-data', 'inference_data.json'))
    return data


@pytest.fixture
def inference_data_2(rootdir):
    """Load second inference data from a json file."""
    data = az.from_json(rootdir / 'test-data' / 'inference_data_2.json')
    return data


@pytest.fixture
def inference_dataset(inference_data, inference_data_2):
    """Create an InferenceDataSet for testing"""
    return InferenceDataSet({
        'data': inference_data,
        'data_2': inference_data_2
    })


@pytest.fixture
def truth(rootdir):
    """Load the true parameter values to an xarray dataset."""
    df = pd.read_csv(Path(rootdir, 'test-data', 'synthetic_truth.csv'),
                     index_col=0)
    df = df.loc['data'].to_dict()
    return xr.Dataset(df)


@pytest.fixture
def truth_dataset(rootdir):
    """Load the true parameter values to an xarray dataset."""
    df = pd.read_csv(Path(rootdir, 'test-data', 'synthetic_truth.csv'),
                     index_col=0)
    return xr.Dataset(df).rename({'dim_0': 'ID'})


@pytest.fixture
def metric_data_set(inference_dataset, truth_dataset):
    """Load a MetricDataSet for testing."""
    return MetricDataSet(inference_dataset, truth_dataset)


@pytest.fixture
def priors():
    return {
        'alphaA': {
            'lower': 0,
            'upper': 1
        },
        'burnupA': {
            'lower': 0.1,
            'upper': 60
        },
        'powerA': {
            'lower': 30,
            'upper': 160
        },
        'coolingA': {
            'lower': 0,
            'upper': 10000
        },
        'enrichmentA': {
            'lower': 1,
            'upper': 5
        },
        'alphaB': {
            'lower': 0,
            'upper': 1
        },
        'burnupB': {
            'lower': 0.1,
            'upper': 8
        },
        'powerB': {
            'lower': 1,
            'upper': 20
        },
        'coolingB': {
            'lower': 0,
            'upper': 10000
        },
        'enrichmentB': {
            'lower': 0.72,
            'upper': 1.5
        }
    }


def test_calculate_distance(inference_data, truth):
    """Test the calculate distance method."""
    for m in ['peak', 'mean', 'mode']:
        metric = metrics.Metric(m, inference_data, truth)
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
                                inference_data,
                                truth,
                                data_vars=['burnupA', 'powerA'])
        metric.calculate_distance()
        metric.calculate_distance(normalize='max')
        metric.calculate_distance(normalize='truth')
        metric.calculate_distance(absolute=True)
        assert True


def test_calculate_metric(inference_data, truth):
    """Test the calculate distance method."""
    for m in ['peak', 'mean', 'mode']:
        metric = metrics.Metric('peak', inference_data, truth)
        metric.calculate_metric_norm()
        metric.calculate_metric_norm(unit=True)
        assert True


def test_metric_set_distances(inference_data, truth):
    """Test the MetricSet class."""
    ms = metrics.MetricSet(inference_data, truth)
    ms.calculate_distances()
    assert True


def test_metric_set_metrics(inference_data, truth):
    """Test the MetricSet class."""
    ms = metrics.MetricSet(inference_data, truth)
    ms.calculate_metric_norms()
    assert True


class TestAccuracyMeasures:

    def test_r_squared_measure(self):
        truth = xr.DataArray([1, 2, 3, 4, 5], dims='ID')
        prediction = xr.DataArray([1.1, 2.2, 2.8, 4.1, 5.3], dims='ID')
        measure = RSquaredMeasure()
        accuracy = measure(truth, prediction)
        assert pytest.approx(accuracy, abs=1e-4) == 0.9810

    def test_mape_measure(self):
        truth = xr.DataArray([1, 2, 3, 4, 5], dims='ID')
        prediction = xr.DataArray([1.1, 2.2, 2.8, 4.1, 5.3], dims='ID')
        measure = MAPEMeasure()
        accuracy = measure(truth, prediction)
        assert pytest.approx(accuracy, abs=0.1) == 7.0333

    def test_rmse_measure(self):
        truth = xr.DataArray([1, 2, 3, 4, 5], dims='ID')
        prediction = xr.DataArray([1.1, 2.2, 2.8, 4.1, 5.3], dims='ID')
        measure = RMSEMeasure()
        accuracy = measure(truth, prediction)
        assert pytest.approx(accuracy, abs=1e-4) == 0.1949


def test_metric_data_set(metric_data_set):
    """Test the MetricDataSet class"""
    assert metric_data_set


def test_metric_data_set_calc_measure(metric_data_set, truth_dataset):
    """Test for the calc_measure method of MetricDataSet"""
    mds = metric_data_set
    assert mds.calc_measure('r_squared', 'mean')
    assert mds.calc_measure('mape', 'mode')
    assert mds.calc_measure('rmse', 'mean')


def test_metric_data_set_calc_hdi(metric_data_set):
    """Test for the calc_measure method of MetricDataSet"""
    mds = metric_data_set
    assert mds.calc_hdi()


def test_metric_data_set_compare_hdi_prior(metric_data_set, priors):
    """Test for the compare_hdi_prior method."""
    mds = metric_data_set
    assert mds.compare_hdi_prior(priors)


def test_truth_in_hdi(metric_data_set):
    """Test for the truth_in_hdi method."""
    mds = metric_data_set
    assert mds.truth_in_hdi()


def test_dataset_distance(metric_data_set):
    """Test for the calculate_distance method of MetricDataSet."""
    mds = metric_data_set
    assert mds.calculate_distance('mode')
    assert mds.calculate_distance('mean')
    assert mds.calculate_distance('mode', normalize='truth')
    assert mds.calculate_distance('mode', normalize='predicted')
    assert mds.calculate_distance('mode', normalize='max')
    assert mds.calculate_distance('mode', normalize='abssum')


def test_dataset_distance_scan(metric_data_set):
    """Test for the distance_scan method of MetricDataSet."""
    mds = metric_data_set
    assert mds.distance_scan(estimator_types=['mean', 'mode'],
                             normalize=['abssum', 'max', 'truth', 'predicted'],
                             absolute=[False, True])
