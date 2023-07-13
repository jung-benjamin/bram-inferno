#! /usr/bin/env python3
"""Tests for the analysis module."""

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from newbie import inferencedata


def load_idata(rootdir, fname):
    """Load inference data from json file."""
    data = az.from_json(Path(rootdir, 'test-data', fname))
    return data


@pytest.fixture
def classification_results(rootdir):
    """Create an instance of ClassificationResults for use in tests."""
    cr = inferencedata.ClassificationResults.from_json(
        class_var='cat',
        filepath=rootdir / 'test-data' / 'classification' /
        'idata_categorical_label1_0239.json')
    return cr


def test_infernce_data_from_json(rootdir):
    """Test for the InferenceData.from_json classmethod."""
    assert inferencedata.InferenceData.from_json(
        filepath=rootdir / 'test-data' / 'inference_data.json')


def test_inference_data_from_inferencedata(inference_data):
    """Test for the InferenceData.from_inferencedata classmethod."""
    assert inferencedata.InferenceData.from_inferencedata(inference_data)


def test_inference_data_estimator(inference_data):
    """Test for the InferenceData.calculate_estimator method."""
    idata = inferencedata.InferenceData.from_inferencedata(inference_data)
    assert idata.calculate_estimator('mean')


def test_class_results_from_json(rootdir):
    """Test for the ClassificationResults.from_json classmethod."""
    cr = inferencedata.ClassificationResults.from_json(
        class_var='cat',
        filepath=rootdir / 'test-data' / 'classification' /
        'idata_categorical_label1_0239.json')
    assert cr.batch_map == {'A': 0, 'B': 1}


def test_class_posterior(classification_results):
    """Test for the _calc_class_posterior method."""
    classification_results._calc_class_posterior()
    assert classification_results.class_posterior == ['A'] * 4000
    classification_results.reactor_map = {'label1': 0, 'label2': 1}
    classification_results._calc_class_posterior()
    assert classification_results.class_posterior == ['label1'] * 4000


def test_get_class_results(classification_results):
    """Test for the get_class_results method of ClassificationResults."""
    classification_results.get_class_results()
    assert classification_results.class_results == 'A'
    classification_results.reactor_map = {'label1': 0, 'label2': 1}
    classification_results.get_class_results()
    assert classification_results.class_results == 'label1'


def test_sort_posteriors_by_batch(classification_results):
    """Test for the sort_posteriors_by_batch method."""
    classification_results.sort_posteriors_by_batch()
    assert classification_results.batch_posteriors


def test_classification_results_from_idata(rootdir):
    """Test for the from_inferencedata method of ClassificationResults."""
    idata = load_idata(rootdir,
                       'classification/idata_categorical_label1_0239.json')
    cr = inferencedata.ClassificationResults.from_inferencedata(
        class_var='cat', inference_data=idata)
    assert cr


def test_classification_hide_posteriors(classification_results):
    """Test for teh hide_non_posteriors method."""
    classification_results.hide_non_posteriors()
    assert list(classification_results.posterior.data_vars) == [
        'burnup', 'power', 'cooling', 'enrichment'
    ]


def test_inference_data_set(inference_data, inference_data_2):
    """Test for the __init__ of InferenceDataSet."""
    idata1 = inference_data
    idata2 = inference_data_2
    assert inferencedata.InferenceDataSet({'i1': idata1, 'i2': idata2})
    assert inferencedata.InferenceDataSet([('i1', idata1), ('i2', idata2)])
    assert inferencedata.InferenceDataSet([idata1, idata2])


def test_inference_data_set_from_json(rootdir):
    """Test for the from_json classmethod of InferenceDataSet."""
    fdir = rootdir / 'test-data' / 'classification'
    i1 = inferencedata.InferenceDataSet.from_json(list(fdir.iterdir()))
    assert isinstance(i1, inferencedata.InferenceDataSet)
    assert isinstance(list(i1.data.values())[0], inferencedata.InferenceData)
    i2 = inferencedata.InferenceDataSet.from_json(list(fdir.iterdir()),
                                                  class_var='cat')
    assert isinstance(
        list(i2.data.values())[0], inferencedata.ClassificationResults)
    assert isinstance(i2, inferencedata.InferenceDataSet)
    assert inferencedata.InferenceDataSet.from_json(list(fdir.iterdir()),
                                                    fmt='')


def test_inference_data_set_get_variables(rootdir):
    """Test for the get_variable method of InferenceDataSet."""
    fdir = rootdir / 'test-data' / 'classification'
    idataset = inferencedata.InferenceDataSet.from_json(list(fdir.iterdir()),
                                                        class_var='cat')
    burnupA = idataset.get_variables('burnupA')
    assert len(burnupA) == len(list(fdir.iterdir()))
    burnupAB = idataset.get_variables(['burnupA', 'burnupB'])
    assert len(burnupAB) == len(list(fdir.iterdir()))


def test_inference_data_set_calculate_estimator(inference_dataset):
    """Test for the __init__ of InferenceDataSet."""
    assert inference_dataset.calculate_estimator('mean')
    assert inference_dataset.calculate_estimator('mode')
    assert list(inference_dataset.estimators['Estimator']) == ['mean', 'mode']


def test_inference_data_get_data_attributes_dict(inference_dataset):
    """Test for the gets_data_attributes_dict of InferenceDataSet."""
    assert inference_dataset.get_data_attributes_dict('calculate_estimator',
                                                      'mean')
    assert inference_dataset.get_data_attributes_dict('posterior')


def test_inference_data_set_apply_func(inference_dataset):
    """Test for the apply_func method of InferenceDataSet."""
    assert inference_dataset.apply_func(az.hdi)