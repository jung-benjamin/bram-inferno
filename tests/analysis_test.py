#! /usr/bin/env python3
"""Tests for the analysis module."""

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from newbie import analysis

CONFUSION_MATRIX = np.array([[10, 0], [0, 10]])


def load_idata(rootdir, fname):
    """Load inference data from json file."""
    data = az.from_json(Path(rootdir, 'test-data', fname))
    return data


def load_truth(rootdir, key):
    """Load the true parameter values to an xarray dataset."""
    df = pd.read_csv(Path(rootdir, 'test-data', 'synthetic_truth.csv'),
                     index_col=0)
    df = df.loc[key].to_dict()
    return xr.Dataset(df)


def confusion_analyzer(rootdir):
    """Create an instance of the ConfusionAnalyzer for use in tests."""
    ca = analysis.ConfusionAnalyzer.from_json(list(
        (rootdir / 'test-data' / 'classification').iterdir()),
                                              class_var='cat')
    ca.reactor_map = {'label1': 0, 'label2': 1}
    return ca


def test_posterior_analysis(rootdir):
    """Test for the posterior analysis class."""
    keys = ['data', 'data_2']
    idata = [
        load_idata(rootdir, 'inference_data.json'),
        load_idata(rootdir, 'inference_data_2.json')
    ]
    truth = list(map(lambda x: load_truth(rootdir, x), keys))
    pa = analysis.PosteriorAnalysis((keys[0], idata[0], truth[0]),
                                    (keys[1], idata[1], truth[1]))
    assert pa.calculate_distances()


def test_posterior_analysis_from_files(rootdir):
    """Test for the from_filepaths classmethod of PosteriorAnalysis."""
    keys = ['data', 'data_2']
    truth = Path(rootdir, 'test-data', 'synthetic_truth.csv')
    idata = {
        'data': Path(rootdir, 'test-data', 'inference_data.json'),
        'data_2': Path(rootdir, 'test-data', 'inference_data_2.json')
    }
    assert analysis.PosteriorAnalysis.from_filepaths(idata, truth, keys)


def test_confusion_analyzer_from_json(rootdir):
    """Test for the from_json method of ConfusionAnalyzer."""
    ca = analysis.ConfusionAnalyzer.from_json(
        list((rootdir / 'test-data' / 'classification').iterdir()))
    assert ca.info
    assert ca.idata


def test_confusion_matrix(rootdir):
    """Test for the calc_confusion_matrix method of ConfusionAnalyzer."""
    ca = confusion_analyzer(rootdir)
    ca.calc_confusion_matrix()
    np.testing.assert_equal(CONFUSION_MATRIX, ca.confusion_matrix)


def test_confusion_plot(rootdir):
    """Test for the plot_confusion_matrix method of ConfusionAalyzer."""
    ca = confusion_analyzer(rootdir)
    ca.plot_confusion_matrix()
    assert True