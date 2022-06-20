#! /usr/bin/env python3

"""Tests for the kernels module"""

import json
from newbie import kernels, wastebin

def model_dict():
    """Dictionary of model attributes"""
    d = {
    'Params': [33.90938176807465, 8.014395700596914, 13.602488333804923, 1e-10],
    'LAMBDA': [[0.12477547120932896, 0.0], [0.0, 0.07351596086392508]],
    'alpha_': [9495.266495455951, 899.6204311144278,
               -4344.025299121975, 9857.440586445668,
               1399.307539288503, 12619.751314661358,
               -16391.488934019377, -2434.233532291585,
               -5138.175102070208, -5963.440700278297],
    'x_train': [[1318.463134765625, 1981.8115234375],
                [368.463134765625, 6981.8115234375],
                [843.463134765625, 9481.8115234375],
                [1793.463134765625, 4481.8115234375],
                [130.963134765625, 3231.8115234375],
                [1080.963134765625, 8231.8115234375],
                [1555.963134765625, 5731.8115234375],
                [605.963134765625, 731.8115234375],
                [724.713134765625, 6356.8115234375],
                [1674.713134765625, 1356.8115234375]],
    'y_train': [8038470000000000.0, 2078810000000000.0, 4963040000000000.0,
                1.12809e+16, 721153000000000.0, 6478440000000000.0,
                9640250000000000.0, 3495480000000000.5, 4223540000000000.0,
                1.0455700000000002e+16],
    'y_trafo': ['StandardNormalize', [6137578300000000.0, 3635217363511814.5]],
    'x_trafo': ['Normalize', [1793.463134765625, 9481.8115234375]],
    'kernel': 'AnisotropicSquaredExponential'
    }
    return d

def store_model(temp):
    """Store model_dict to a json file"""
    with open(temp, 'w') as f:
        json.dump(model_dict(), f)

def test_load_model(tmp_path):
    """Test for load_model method"""
    p = tmp_path / 't.json'
    store_model(p)
    m1 = wastebin.WasteBin(kernels.ASQEKernelPredictor, filepaths = {'t': p},
                           labels=None, evidence=None)
    m1.load_models(['t'])
    assert True
    m2 = wastebin.WasteBin(kernels.ASQEKernelPredictor, filepaths = {'t': p},
                            labels=None, evidence=None)
    m2.model_ratios = False
    m2.load_models(['t/t'])
    assert True

def test_make_filepaths(tmp_path):
    """Test for make_filepaths staticmethod"""
    fp = tmp_path / 't.json'
    d = wastebin.WasteBin.make_filepaths(['t'], tmp_path, '{}.json')
    assert d['t'] == str(fp)
