#! /usr/bin/env python3

"""Tests for the evidence module"""

import pandas as pd

from newbie.evidence import Evidence


def series():
    """Series with nuclide measurements"""
    return pd.Series([0.99, 0.007, 0.003], index=['U-238', 'U-235', 'U-234'])


def dataframe():
    """Dataframe with nuclide measurements"""
    df = pd.concat([series(), series(), series()], keys=['A', 'B', 'C'], axis=1)
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
