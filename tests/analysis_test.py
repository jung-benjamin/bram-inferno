#! /usr/bin/env python3
"""Tests for the analysis module."""

from pathlib import Path

import arviz as az
import pandas as pd
import xarray as xr

from newbie import analysis


def load_idata(rootdir, fname):
    """Load inference data from json file."""
    data = az.from_json(Path(rootdir, 'test-data', fname))
    return data


def load_truth(rootdir, key):
    """Load the true parameter values to an xarray dataset."""
    df = pd.read_csv(Path(rootdir, 'test-data', 'synthetic_truth.csv'),
                     index_col=0)
    df = df.loc[k].to_dict()
    return xr.Dataset(df)


def posterior_analysis_test(rootdir):
    """Test for the posterior analysis class."""
    keys = ['data', 'data_2']
    idata = [
        load_idata(rootdir, 'inference_data.json'),
        load_idata(rootdir, 'inference_data_2.json')
    ]
    truth = list(map(lambda x: load_truth(rootdir, x)), keys)
    pa = analysis.PosteriorAnalysis((keys[0], idata[0], truth[0]),
                                    (keys[1], idata[1], truth[1]))
    pa.calculate_distances()
    assert True


def posterior_analysis_from_files_test(rootdir):
    """Test for the from_filepaths classmethod of PosteriorAnalysis."""
    keys = ['data', 'data_2']
    truth = Path(rootdir, 'tests', 'test-data', 'synthetic_truth.csv')
    idata = {
        'data':
        Path(rootdir, 'tests', 'test-data', 'inference_data.json'),
        'data_2':
        Path(rootdir, 'tests', 'test-data', 'inference_data_2.json')
    }
    pa = analysis.PosteriorAnalysis.from_filepaths(idata, truth, keys)
    assert True