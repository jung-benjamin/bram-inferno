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


def load_idata(rootdir):
    """Load inference data from json file."""
    data = az.from_json(Path(rootdir, 'test-data', 'inference_data.json'))
    return data


def load_idata_file(rootdir, fname):
    """Load inference data from a json file."""
    data = az.from_json(rootdir / 'test-data' / fname)
    return data


def load_truth(rootdir):
    """Load the true parameter values to an xarray dataset."""
    df = pd.read_csv(Path(rootdir, 'test-data', 'synthetic_truth.csv'),
                     index_col=0)
    df = df.loc['data'].to_dict()
    return xr.Dataset(df)


def load_truth_set(rootdir):
    """Load the true parameter values to an xarray dataset."""
    df = pd.read_csv(Path(rootdir, 'test-data', 'synthetic_truth.csv'),
                     index_col=0)
    return xr.Dataset(df).rename_dims({'dim_0': 'ID'})


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


def test_metric_data_set(rootdir):
    """Test the MetricDataSet class"""
    idata1 = load_idata_file(rootdir, 'inference_data.json')
    idata2 = load_idata_file(rootdir, 'inference_data_2.json')
    ids = InferenceDataSet({'data': idata1, 'data_2': idata2})
    assert MetricDataSet(ids, load_truth(rootdir))


def test_metric_data_set_calc_measure(rootdir):
    """Test for the calc_measure method of MetricDataSet"""
    idata1 = load_idata_file(rootdir, 'inference_data.json')
    idata2 = load_idata_file(rootdir, 'inference_data_2.json')
    ids = InferenceDataSet({'data': idata1, 'data_2': idata2})
    mds = MetricDataSet(ids, load_truth_set(rootdir))
    assert mds.calc_measure('r_squared', 'mean')
    assert mds.calc_measure('mape', 'mode')
    assert mds.calc_measure('rmse', 'mean')