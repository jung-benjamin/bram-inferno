#! /usr/bin/env python3

"""Tests for the evidence module"""

import pandas as pd

from newbie.evidence import Evidence, Mixture, SyntheticEvidence


def series():
    """Series with nuclide measurements"""
    return pd.Series([0.99, 0.007, 0.003], index=['U-238', 'U-235', 'U-234'])


def dataframe():
    """Dataframe with nuclide measurements"""
    df = pd.concat([series(), series(), series()], keys=['A', 'B', 'C'], axis=1)
    return df


def param_series():
    """Series with parameter values"""
    return pd.Series([2, 200], index=['Burnup', 'Cooling'])

def parameters():
    """DataFrame with parameter values"""
    df = pd.concat(
        [param_series(), param_series(), param_series()],
        keys=['A', 'B', 'C'],
        axis=0
    )
    return df


def test_evidence_dict():
    """Test for create_dict method of Evidence"""
    ratios = ['U-238/U-235', 'U-235/U-234']
    target = {'U-238/U-235': 0.99/0.007, 'U-235/U-234': 0.007/0.003}
    ev1 = Evidence(series())
    assert target == ev1.create_dict(ratios)
    ev2 = Evidence(dataframe())
    assert target == ev2.create_dict(ratios, 'A')


def test_iter_batches():
    """Test for iter_batches method"""
    ratios = ['U-238/U-235', 'U-235/U-234']
    target = {'U-238/U-235': 0.99/0.007, 'U-235/U-234': 0.007/0.003}
    ev1 = Evidence(series())
    for b, d in ev1.iter_batches(ratios):
        assert target == d
    ev2 = Evidence(dataframe())
    for b, d in ev2.iter_batches(ratios):
        assert target == d


def test_synthetic_evidence():
    """Test for the SyntheticEvidence class"""
    synth = SyntheticEvidence(dataframe(), parameters())
    assert (synth.true_parameters('A') == parameters().loc['A',:]).all()


def test_mixture():
    """Test for the Mixture class"""
    mix = Mixture(dataframe(), [['A', 'B']], [[1., 1.]])
    test1 = dataframe()['A'] + dataframe()['B']
    test1.name = '1.0A+1.0B'
    test1 = pd.concat([test1], axis=1)
    assert (mix.isotopes == test1).all().all()
    test2 = dataframe()['A'] + dataframe()['C']
    test2.name = '1.0A+1.0C'
    test2 = pd.concat([test2], axis=1)
    mix2 = Mixture(dataframe(), [['A', 'B'], ['A', 'C']], [[1., 1.], [1., 1.]])
    test_comb = pd.concat([test1, test2], axis=1)
    assert (mix2.isotopes == test_comb).all().all()
