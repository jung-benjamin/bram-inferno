#! /usr/bin/env python3

"""Handle measured evidence for inference

A class that handles the measured evidence (isotopic ratios) for
the inference.

@author: jung-benjamin
"""

import os

import pandas as pd


class Evidence():
    """Evidence for Bayesian inference

    A class for handling isotopic ratio measurement data
    as evidence for Bayesian inference.
    """

    def __init__(self, data):
        """Create an instance of the class

        Parameters
        ----------
        data : pd.DataFrame or pd.Series
            Isotopic concentration or nuclide density data. The index
            contains the nuclide ids and the columns can contain ids
            for different batches of waste.
        """
        self.isotopes = data
        if isinstance(data, pd.DataFrame):
            self.batches = list(data.columns)
        else:
            self.batches = []

    def _iter_ratiolist(self, ratiolist):
        """Iterate over list of ratios

        Yields the ratio-id and the composing isotopes.
        """
        for r in ratiolist:
            i, j = r.split('/')
            yield r, i, j

    def create_dict(self, ratiolist, batch=None):
        """Return a dictionary of isotopic ratios

        Parameters
        ----------
        ratiolist : list of str
            List of isotopic ratios to create from the nuclide data.
        batch : str (optional, default is None)
            Identifier for the column in the nuclide data.

        Returns
        -------
        evidence : dict
            Dictionary with ratios as keys and the corresponding
            values as entries.
        """
        evidence = {}
        for r, i, j in self._iter_ratiolist(ratiolist):
            if self.batches:
                evidence[r] = (self.isotopes.loc[i, batch]
                               / self.isotopes.loc[j, batch])
            else:
                evidence[r] = self.isotopes[i] / self.isotopes[j]
        return evidence

    def iter_batches(self, ratiolist):
        """Yield evidence dicts for all batches"""
        if not self.batches:
            yield self.create_dict(ratiolist)
        else:
            for b in self.batches:
                yield self.create_dict(ratiolist, batch=b)
