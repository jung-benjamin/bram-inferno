#! /usr/bin/env python3
"""Configure the pytest unit tests"""

from pathlib import Path

import arviz as az
import pytest

from newbie.inferencedata import InferenceDataSet


@pytest.fixture
def rootdir():
    return Path(__file__).resolve().parent


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