#! /usr/bin/env python3

"""PyMC models for nuclear waste

Bayesian inference models for reconstructing reactor
operating histories from nuclear waste compositions.

@author jung-benjamin
"""

import os
import pymc3 as pm

class WasteBin():
    """Bayesian inference with nuclear waste

    A class for building a surrogate model of nuclear for
    reconstructing operating history parameters of reactors.
    """

    def __init__(self, model_type, filepaths):
        """Set model type and filepaths for loading models

        Parameters
        ----------
        model
            Surrogate model for predicting the value of an
            isotope concentration or ratio from the operating
            parameters.
        filepaths : dict
            Paths to each isotope's or each ratio's stored
            model parameters and data. The file must be
            loadable by the model.from_file classmethod.
            Keys are identifiers of isotope or ratio.
        """
        self.model_type = model_type
        self.filepaths = filepaths
        self.models = {}

    @staticmethod
    def make_filepaths(ids, base, form):
        """Create dictionary of filepaths

        Parameters
        ----------
        ids : list
            List of string identifiers of the isotopes or
            isotopic ratios.
        base : str, path-like
            Path to the directory where the model files
            are stored.
        form : str
            String for formatting the filename.
            E.g.: 'kernel{}.json'

        Returns
        -------
        filepaths : dict
            Dictionary of filepaths, with model identifiers
            as keys.
        """
        p = os.path.join(base, form)
        filepaths = {i: p.format(i) for i in ids}
        return filepaths

    def load_models(self, ids):
        """Load the surrogate models from files

        Parameters
        ----------
        ids : list
            String identifiers of the isotopes or isotopic
            ratios.
        """
        for i in ids:
            self.models[i] = self.model_type.from_file(self.filepaths[i])
