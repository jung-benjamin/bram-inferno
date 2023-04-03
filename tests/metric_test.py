#! /usr/bin/env python3
"""Tests for the metrics module."""

from pathlib import Path

import arviz as az
import pandas as pd
import xarray as xr

from newbie import estimators
from newbie import metrics


def load_idata(rootdir):
    """Load inference data from json file."""
    data = az.from_json(Path(rootdir, 'test-data', 'inference_data.json'))
    return data


def load_truth(rootdir):
    """Load the true parameter values to an xarray dataset."""
    df = pd.read_csv(Path(rootdir, 'test-data', 'synthetic_truth.csv'),
                     index_col=0)
    df = df.loc['c64p_0704_059_yb4p_0871'].to_dict()
    return xr.Dataset(df)


def test_calculate_distance(rootdir):
    """Test the calculate distance method."""
    idata = load_idata(rootdir)
    truth = load_truth(rootdir)
    for m in ['peak', 'mean', 'mode']:
        metric = metrics.Metric('peak', idata, truth)
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
        metric.calculate_metric()
        metric.calculate_metric(unit=True)
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
    ms.calculate_metrics()
    assert True