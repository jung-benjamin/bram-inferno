#! /usr/bin/env python3

"""Tests for the evidence module"""

import pandas as pd

from newbie.evidence import Evidence, Mixture, SyntheticEvidence, SyntheticMixture


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
        axis=1
    ).T
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


def test_mixture_makeup():
    """Test for the get_mixture_makeup method"""
    mix = Mixture(dataframe(), [['A', 'B']], [[1., 1.]])
    test = {'1.0A+1.0B': (['A', 'B'], [1., 1.])}
    assert mix.get_mixture_makeup() == test


def test_synthetic_mixture():
    """Test for the SyntheticMixture class"""
    synth = SyntheticMixture(
        dataframe(),
        parameters(),
        [['A', 'B'], ['A', 'C']],
        [[1., 1.], [1., 1.]]
    )
    test1 = {'alpha_A': 1., 'Burnup_A': 2, 'Cooling_A': 200,
             'alpha_B': 1., 'Burnup_B': 2, 'Cooling_B': 200
             }
    test1 = pd.Series(test1, name='1.0A+1.0B')
    assert (test1 == synth.true_parameters('1.0A+1.0B')).all()


def test_sort_params():
    """Test for the sort_params method"""
    synth = SyntheticMixture(
        dataframe(),
        parameters(),
        [['A', 'B'], ['A', 'C']],
        [[1., 1.], [1., 1.]]
    )
    group = {
        'A': ['alpha_A', 'Burnup_A', 'Cooling_A'],
        'B': ['alpha_B', 'Burnup_B', 'Cooling_B']
    }
    assert synth.sort_params('1.0A+1.0B') == group


def test_group_labels():
    """Test for the group_labels methdo of SyntheticMixture"""
    synth = SyntheticMixture(
        dataframe(),
        parameters(),
        [['A', 'B'], ['A', 'C']],
        [[1., 1.], [1., 1.]]
    )
    group = {
        'A': ['alpha_A':, 'Burnup_A', 'Cooling_A'],
        'B': ['alpha_B':, 'Burnup_B', 'Cooling_B']
    }
    assert synth.group_labels()['1.0A+1.0B'] == group
