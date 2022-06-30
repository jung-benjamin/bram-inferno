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

    @classmethod
    def from_csv(cls, filepath):
        """Create the class from a csv file"""
        data = pd.read_csv(filepath, index_col=0)
        return cls(data)

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
            yield 'Batch', self.create_dict(ratiolist)
        else:
            for b in self.batches:
                yield b, self.create_dict(ratiolist, batch=b)


class SyntheticEvidence(Evidence):
    """Simulated evidence for Bayesian inference

    Handles simulated evidence along with the corresponding
    true parameters for the Bayesian inference and associated
    analysis.
    """

    def __init__(self, data, parameters):
        """Create an instance of the class

        Parameters
        ----------
        data : pd.DataFrame or pd.Series
            Isotopic concentration or nuclide density data. The index
            contains the nuclide ids and the columns can contain ids
            for different batches of waste.
        parameters : pd.DataFrame of pd.Series
            True parameter values corresponding to the simulated
            evidence. Index contains identifiers for each batch of
            evidence and the columns are the parameter labels.
        """
        self.isotopes = data
        self.parameters = parameters
        if isinstance(data, pd.DataFrame):
            self.batches = list(data.columns)
        else:
            self.batches = []

    @classmethod
    def from_csv(cls, datapath, parameterpath):
        """Create the class from two csv files"""
        data = pd.read_csv(datapath, index_col=0)
        parameters = pd.read_csv(parameterpath, index_col=0)
        return cls(data, parameters)

    def true_parameters(self, batch):
        """Select parameters of a batch of synthetic evidence."""
        return self.parameters.loc[batch,:]


class Mixture(Evidence):
    """Isotopic composition of mixtures of batches"""

    def __init__(self, data, mixing_ids, mixing_ratios):
        """Create mixtures of isotopic compositions

        Parameters
        ----------
        data : pd.DataFrame
            Isotopic composition data for at least two batches
            of waste. Ids need to be in columns.
        mixing_ids : list of list of str
            Ids of the batches to mix. Must be contained in
            the columns of `data`.
        mixing_ratios : list of list of float
            Ratios for mixing the batches respectively.
        """
        self.isotopes = pd.concat(
            [self._mix(data, i, r) for i, r in zip(mixing_ids, mixing_ratios)],
            axis=1,
        )

    def _mix(self, data, mixing_ids, mixing_ratios):
        """Mix two or more isotopic compositions

        Parameters
        ----------
        data : pd.DataFrame
            Isotopic composition data for at least two batches
            of waste. Ids need to be in columns.
        mixing_ids : list of str
            Ids of the batches to mix. Must be contained in
            the columns of `data`.
        mixing_ratios : list of float
            Ratios for mixing the batches respectively.
        """
        conc = [a * data[n] for a, n in zip(mixing_ratios, mixing_ids)]
        mix = pd.concat(conc, axis=1).sum(axis=1)
        key = '+'.join([f'{a}{n}' for a, n in zip(mixing_ratios, mixing_ids)])
        mix.name = key
        return mix
