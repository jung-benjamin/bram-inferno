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


def inference_data(rootdir):
    """Create instance of InferenceData for use in tests."""
    return inferencedata.InferenceData.from_json(
        filepath=rootdir / 'test-data' / 'inference_data.json')


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


def test_inference_data_from_inferencedata(rootdir):
    """Test for the InferenceData.from_inferencedata classmethod."""
    idata = az.from_json(rootdir / 'test-data' / 'inference_data.json')
    assert inferencedata.InferenceData.from_inferencedata(idata)


def test_inference_data_estimator(rootdir):
    """Test for the InferenceData.calculate_estimator method."""
    idata = inferencedata.InferenceData.from_json(rootdir / 'test-data' /
                                                  'inference_data.json')
    assert idata.calculate_estimator('mean')


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


def test_classification_hide_posteriors(rootdir):
    """Test for teh hide_non_posteriors method."""
    idata = load_idata(rootdir,
                       'classification/idata_categorical_label1_0239.json')
    cr = inferencedata.ClassificationResults.from_inferencedata(
        class_var='cat', inference_data=idata)
    cr.hide_non_posteriors()
    assert list(cr.posterior.data_vars) == [
        'burnup', 'power', 'cooling', 'enrichment'
    ]


def test_inference_data_set(rootdir):
    """Test for the __init__ of InferenceDataSet."""
    idata1 = load_idata(rootdir, 'inference_data.json')
    idata2 = load_idata(rootdir, 'inference_data_2.json')
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


def test_inference_data_set_calculate_estimator(rootdir):
    """Test for the __init__ of InferenceDataSet."""
    idata1 = load_idata(rootdir, 'inference_data.json')
    idata2 = load_idata(rootdir, 'inference_data_2.json')
    ids = inferencedata.InferenceDataSet({'i1': idata1, 'i2': idata2})
    assert ids.calculate_estimator('mean')
    assert ids.calculate_estimator('mode')
    assert list(ids.estimators['Estimator']) == ['mean', 'mode']


def test_inference_data_get_data_attributes_dict(rootdir):
    """Test for the gets_data_attributes_dict of InferenceDataSet."""
    idata1 = load_idata(rootdir, 'inference_data.json')
    idata2 = load_idata(rootdir, 'inference_data_2.json')
    ids = inferencedata.InferenceDataSet({'i1': idata1, 'i2': idata2})
    assert ids.get_data_attributes_dict('calculate_estimator', 'mean')