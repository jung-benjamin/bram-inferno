#! /usr/bin/env python3
"""Tests for the analysis module."""

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from newbie import inferencedata


def load_idata(rootdir, fname):
    """Load inference data from json file."""
    data = az.from_json(Path(rootdir, 'test-data', fname))
    return data


def classification_results(rootdir):
    """Create an instance of ClassificationResults for use in tests."""
    cr = inferencedata.ClassificationResults.from_json(
        class_var='cat',
        filepath=rootdir / 'test-data' / 'classification' /
        'idata_categorical_label1_0239.json')
    return cr


def test_class_results_from_json(rootdir):
    """Test for the ClassificationResults.from_json classmethod."""
    cr = inferencedata.ClassificationResults.from_json(
        class_var='cat',
        filepath=rootdir / 'test-data' / 'classification' /
        'idata_categorical_label1_0239.json')
    assert cr.batch_map == {'A': 0, 'B': 1}


def test_class_posterior(rootdir):
    """Test for the _calc_class_posterior method."""
    cr = classification_results(rootdir)
    cr._calc_class_posterior()
    assert cr.class_posterior == ['A'] * 4000
    cr.reactor_map = {'label1': 0, 'label2': 1}
    cr._calc_class_posterior()
    assert cr.class_posterior == ['label1'] * 4000


def test_get_class_results(rootdir):
    """Test for the get_class_results method of ClassificationResults."""
    cr = classification_results(rootdir)
    cr.get_class_results()
    assert cr.class_results == 'A'
    cr.reactor_map = {'label1': 0, 'label2': 1}
    cr.get_class_results()
    assert cr.class_results == 'label1'


def test_sort_posteriors_by_batch(rootdir):
    """Test for the sort_posteriors_by_batch method."""
    cr = classification_results(rootdir)
    cr.sort_posteriors_by_batch()
    assert cr.batch_posteriors


def test_classification_results_from_idata(rootdir):
    """Test for the from_inferencedata method of ClassificationResults."""
    idata = load_idata(rootdir,
                       'classification/idata_categorical_label1_0239.json')
    cr = inferencedata.ClassificationResults.from_inferencedata(
        class_var='cat', inference_data=idata)
    assert cr