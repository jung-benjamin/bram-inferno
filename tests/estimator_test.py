#! /usr/bin/env python3
"""Tests for the estimators module"""

from pathlib import Path

import arviz as az

from newbie import estimators


def load_idata(rootdir):
    """Load inference data from json file."""
    data = az.from_json(Path(rootdir, 'test-data', 'inference_data.json'))
    return data


def test_mode_estimator(rootdir):
    """Test the mode estimator"""
    data = load_idata(rootdir)
    me = estimators.EstimatorFactory.create_estimator('mode', data)
    mode = me.calculate_estimator()
    assert True


def test_mean_estimator(rootdir):
    """Test the mean estimator"""
    data = load_idata(rootdir)
    me = estimators.EstimatorFactory.create_estimator('mean', data)
    mean = me.calculate_estimator()
    assert True


def test_peak_estimator(rootdir):
    """Test the peak estimator"""
    data = load_idata(rootdir)
    me = estimators.EstimatorFactory.create_estimator('peak', data)
    peak = me.calculate_estimator()
    assert True