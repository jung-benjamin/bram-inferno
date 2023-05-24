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
    e1 = estimators.EstimatorFactory.create_estimator('mode', data)
    m1 = e1.calculate_estimator()
    m1A = e1.calculate_estimator('burnupA')
    e2 = estimators.EstimatorFactory.create_estimator('mode')
    m2 = e2.calculate_estimator(inference_data=data)
    m2A = e2.calculate_estimator('burnupA', inference_data=data)
    assert m1 == m2
    assert m1A == m2A


def test_mean_estimator(rootdir):
    """Test the mean estimator"""
    data = load_idata(rootdir)
    e1 = estimators.EstimatorFactory.create_estimator('mean', data)
    m1 = e1.calculate_estimator()
    m1A = e1.calculate_estimator('burnupA')
    e2 = estimators.EstimatorFactory.create_estimator('mean')
    m2 = e2.calculate_estimator(inference_data=data)
    m2A = e2.calculate_estimator('burnupA', inference_data=data)
    assert m1 == m2
    assert m1A == m2A


def test_peak_estimator(rootdir):
    """Test the peak estimator"""
    data = load_idata(rootdir)
    e1 = estimators.EstimatorFactory.create_estimator('peak', data)
    m1 = e1.calculate_estimator()
    m1A = e1.calculate_estimator('burnupA')
    e2 = estimators.EstimatorFactory.create_estimator('peak')
    m2 = e2.calculate_estimator(inference_data=data)
    m2A = e2.calculate_estimator('burnupA', inference_data=data)
    assert m1 == m2
    assert m1A == m2A