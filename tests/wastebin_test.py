#! /usr/bin/env python3

"""Tests for the kernels module"""

import json

import pymc3 as pm

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

LIMITS = {
    'Uptime': {'lower': 100, 'upper': 2000},
    'Downtime': {'lower': 0, 'upper': 10000},
}

MXT_LIMITS = {
    'alpha1': {'lower': 0, 'upper': 5},
    'Uptime1': {'lower': 100, 'upper': 2000},
    'Downtime1': {'lower': 0, 'upper': 10000},
    'alpha2': {'lower': 0, 'upper': 5},
    'Uptime2': {'lower': 100, 'upper': 2000},
    'Downtime2': {'lower': 0, 'upper': 10000},
}

LABELS = ['Uptime', 'Downtime']

MXT_LABELS={
    'A': ['alpha1', 'Uptime1', 'Downtime1',],
    'B': ['alpha2', 'Uptime2', 'Downtime2',],
}

EVIDENCE = {'t': 1., 't/t': 1., 't1/t1': 1., 't2/t2': 1.}

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


def test_make_priors():
    """Test for the _make_priors method"""
    m = wastebin.WasteBin(
        kernels.ASQEKernelPredictor, labels=None, evidence=None
    )
    with pm.Model():
        m._make_priors(LABELS, LIMITS, None)
        assert True
    with pm.Model():
        m._make_priors(
            LABELS,
            {'Uptime': {'lower': 100, 'upper': 2000}},
            {'Downtime': 1000}
        )
        assert True

def test_make_distributions(tmp_path):
    """Test for the _make_distributions method"""
    p = tmp_path / 't.json'
    store_model(p)
    m = wastebin.WasteBin(kernels.ASQEKernelPredictor, filepaths = {'t': p},
                            labels=None, evidence=EVIDENCE)
    m.model_ratios = False
    with pm.Model():
        m.load_models(['t/t'])
        m._make_priors(LABELS, LIMITS, None)
        m._make_distributions(['t/t'], 0.1)
        assert True

def test_joint_probability(tmp_path):
    """Test for the _joint_probability method"""
    p1 = tmp_path / 't1.json'
    p2 = tmp_path / 't2.json'
    store_model(p1)
    store_model(p2)
    m = wastebin.WasteBin(
        kernels.ASQEKernelPredictor,
        filepaths={'t1': p1, 't2': p2},
        labels=None,
        evidence=EVIDENCE
    )
    m.model_ratios = False
    with pm.Model():
        m.load_models(['t1/t1', 't2/t2'])
        m._make_priors(LABELS, LIMITS, None)
        distr = m._make_distributions(['t1/t1', 't2/t2'], 0.1)
        m._joint_probability(['t1/t1', 't2/t2'], distr)
        assert True

def test_mixture_load_model(tmp_path):
    """Test for load models method of WasteBinMixture"""
    p = tmp_path / 't.json'
    store_model(p)
    m2 = wastebin.WasteBinMixture(
        {'A': kernels.ASQEKernelPredictor, 'B': kernels.ASQEKernelPredictor},
        filepaths = {'A': {'t1': p}, 'B': {'t1': p}},
        labels=MXT_LABELS,
        evidence=None
    )
    m2.load_models(['t1/t1'])
    assert True

def test_mixture_make_priors(tmp_path):
    """Test for _make_priors method of WasteBinMixture"""
    p = tmp_path / 't.json'
    store_model(p)
    m2 = wastebin.WasteBinMixture(
        {'A': kernels.ASQEKernelPredictor, 'B': kernels.ASQEKernelPredictor},
        filepaths = {'A': {'t1': p}, 'B': {'t1': p}},
        labels=MXT_LABELS,
        evidence=None
    )
    with pm.Model():
        m2.load_models(['t1/t1'])
        m2._make_priors(
            labels=MXT_LABELS,
            limits=MXT_LIMITS,
            fallback=None
        )
    assert True

def test_mixture_model_building(tmp_path):
    """Test for building the model in WasteBinMixture"""
    p = tmp_path / 't.json'
    store_model(p)
    m2 = wastebin.WasteBinMixture(
        {'A': kernels.ASQEKernelPredictor, 'B': kernels.ASQEKernelPredictor},
        filepaths = {'A': {'t1': p, 't2': p}, 'B': {'t1': p, 't2':p}},
        labels=MXT_LABELS,
        evidence=EVIDENCE
    )
    with pm.Model():
        m2.load_models(['t1/t1', 't2/t2'])
        m2._make_priors(
            labels=MXT_LABELS,
            limits=MXT_LIMITS,
            fallback=None
        )
        dist = m2._make_distributions(['t1/t1', 't2/t2'], 0.1)
        m2._joint_probability(['t1/t1', 't2/t2',], dist)
    m3 = wastebin.WasteBinMixture(
        {'A': kernels.ASQEKernelPredictor, 'B': kernels.ASQEKernelPredictor},
        filepaths = {'A': {'t1': p, 't2': p}, 'B': {'t1': p, 't2':p}},
        labels=MXT_LABELS,
        evidence=EVIDENCE,
        combination='LinearCombination'
    )
    with pm.Model():
        m3.load_models(['t1/t1', 't2/t2'])
        m3._make_priors(
            labels=MXT_LABELS,
            limits=MXT_LIMITS,
            fallback=None
        )
        dist = m3._make_distributions(['t1/t1', 't2/t2'], 0.1)
        m3._joint_probability(['t1/t1', 't2/t2',], dist)
    assert True
