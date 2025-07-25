#! /usr/bin/env python3
"""PyMC models for nuclear waste

Bayesian inference models for reconstructing reactor
operating histories from nuclear waste compositions.

@author jung-benjamin
"""

import json
import logging
import os
import re
from pathlib import Path, PurePath

import numpy as np
import pandas as pd

try:
    import pymc3 as pm
except ImportError:
    import pymc as pm
try:
    import theano.tensor as tt
except ImportError:
    import pytensor.tensor as tt

from itertools import chain

from . import kernels

ISOTOPE_REGEX = re.compile(r"""([A-Za-z]+)(-*)(\d+)(m|\*)*""")
RATIO_REGEX = re.compile(
    r"""
    ([A-Za-z]+)(-*)(\d+)(m|\*)*
    /
    ([A-Za-z]+)(-*)(\d+)(m|\*)*""", re.X)


def get_distribution(name, dist="Uniform", **kwargs):
    """Create a prior from a dictionary of limits."""
    return getattr(pm, dist)(name, **kwargs)


def get_deterministic(name, value):
    """Create a constant prior from a value."""
    return pm.Deterministic(name, tt.cast(value, 'float64'))


class WasteBin():
    """Bayesian inference with nuclear waste

    A class for building a surrogate model of nuclear for
    reconstructing operating history parameters of reactors.
    """

    def __init__(self,
                 model_type,
                 labels,
                 evidence,
                 filepaths=None,
                 model_ratios=True):
        """Set model type and filepaths for loading models

        Parameters
        ----------
        model_type
            Surrogate model for predicting the value of an
            isotope concentration or ratio from the operating
            parameters.
        filepaths : dict (optional, default is None)
            Paths to each isotope's or each ratio's stored
            model parameters and data. The file must be
            loadable by the model.from_file classmethod.
            Keys are identifiers of isotope or ratio.
        labels : list
            Strings, names of the parameters in the model.
        evidence : dict-like
            Isotopic evidence, from which the parameters are
            to be inferred. Keys are identifiers of isotope
            or ratio.
        model_ratios : bool (default is True)
            Select whether GPR models are trained on ratios or
            isotopic concentrations.
        """
        self.model_type = model_type
        self.filepaths = filepaths
        self.labels = labels
        self.evidence = evidence
        self.models = {}
        self.model_ratios = model_ratios

    def load_filepaths(self, ids, modelfile, prefix=None):
        """Load the filepath dict from a json file"""
        filepaths = {}
        with open(modelfile, 'r') as f:
            paths = json.load(f)
        if prefix is None:
            prefix = str(PurePath(modelfile).parent)
        for i in ids:
            filepaths[i] = os.path.join(prefix, *paths[i])
        self.filepaths = filepaths

    def load_models(self, ids):
        """Load the surrogate models from files

        Parameters
        ----------
        ids : list
            String identifiers of the isotopes or isotopic
            ratios.
        """
        if self.model_ratios:
            for i in ids:
                self.models[i] = self.model_type.from_file(self.filepaths[i])
        else:
            for i in ids:
                iso1, iso2 = i.split('/')
                self.models[i] = kernels.Quotient.from_file(
                    (self.filepaths[iso1], self.model_type),
                    (self.filepaths[iso2], self.model_type))

    def _make_priors(self, p, limits, fallback):
        """Turn parameter limits into prior distribution

        Needs to be called within the pm.Model context.
        """
        priors = []
        for param in p:
            if param in limits:
                priors.append(get_distribution(param, **limits[param]))
            else:
                priors.append(get_deterministic(param, fallback[param]))
        self.priors = priors
        return priors

    def _limits_from_file(self, filepath):
        """Load limits for priors from a json file"""
        with open(filepath, 'r') as f:
            limits = json.load(f)
        const = {}
        for p, l in limits.copy().items():
            if isinstance(l, dict):
                pass
            elif isinstance(l, (float, int)):
                const[p] = limits.pop(p)
        return limits, const

    def _make_distributions(self, ids, uncertainty):
        """Create a normal distribution for each isotpic ratio

        The priors need to be created and the models need to be
        loaded before calling this method.
        """
        evidence = [self.evidence[i] for i in ids]
        if isinstance(uncertainty, float):
            sigma = [uncertainty * e for e in evidence]
        else:
            sigma = [self.evidence[i] * uncertainty[i] for i in ids]
        models = [self.models[i].predict(self.priors) for i in ids]
        try:
            distrib = [
                pm.Normal(i, mu=m, sd=s, observed=o)
                for i, m, s, o in zip(ids, models, sigma, evidence)
            ]
        except TypeError:
            distrib = [
                pm.Normal(i, mu=m, sigma=s, observed=o)
                for i, m, s, o in zip(ids, models, sigma, evidence)
            ]
        self.probabilities = distrib
        return distrib

    def _joint_probability(self, ids, dists):
        """Calculate the join probability distribution

        Parameters
        ----------
        ids : list of str
            Identifiers of isotopes or isotopic ratios.
        dists : list
            Probability density of each surrogate model.

        Returns
        l : pymc.DensityDist
            Joint probability density distribution of the
            surrogate models.
        """
        if len(ids) == 1:
            l = dists
        else:

            def joint(**kwargs):
                return np.product([n for i, n in kwargs.items()])

        try:
            l = pm.DensityDist('L', joint, observed=dict(zip(ids, dists)))
        except TypeError:

            l = dists
        return l

    def inference(
        self,
        ids,
        limits,
        uncertainty=0.1,
        const=None,
        plot=True,
        load=None,
        debug=False,
        bias=None,
        **kwargs,
    ):
        """Run bayesian inference with pymc uniform priors

        Creates the Model context manager of pymc and runs bayesian
        inference with the specified parameters.
        Parameters that are also variable in the surrogate model, but
        are not intended for inference are set to constant values.

        Parameters
        ----------
        ids : list of str
            Identifiers of isotopes or isotopic ratios to be used in the
            inference.
        limits : dict or path-like str
            Dictionary of limits or a filepath to a json file
            containing the dictionary.
            Keys are the labels of the parameters that are to be
            reconstructed. Each entry is a dict with keys 'lower'
            and 'upper', specifying the lower and upper limits of
            the uniform prior distribution of the parameter.
        uncertainty : float or dict-like, optional (default is 0.1)
            Relative uncertainty assumed for the evidence. If uncertainty
            is a float, the same uncertainty is assumed for each ratio.
            If a dict-like object is passed, it must kontain keys that
            correspond to elements in `ids`.
        const : dict floats, optional (default is None)
            Specifies the constant values of the parameters that
            are not varied. Keys are parameter labels. All parameter
            labels must be either in `limits` or `const`.
        plot : bool, optional (default is True)
            If True, plots the trace after running the sampler.
        load : str, optional (default is None)
            Path to a directory, where a trace was stored by a previous
            sampling process. If the path is specified, the model is
            loaded from the trace and the sampler is not run.
        bias: dict(float), optional (default is None)
            Dictionary of biases to be applied for correcting the predictions
            of the surrogate models. The keys are the model identifiers
            (usually nuclides).
        **kwargs
            Keyword arguments for the pymc.sample method.

        Returns
        -------
        summaries : dict
            Key properties of the reconstructed distributions.
        trace : pymc trace object
            Trace of the sampling algorithm.
        """

        with pm.Model() as model:
            ## Needs to be called inside the context manager
            ## Otherwise models don't work
            self.load_models(ids)
            if bias is not None:
                for n, it in self.models.items():
                    if self.model_ratios:
                        raise NotImplementedError(
                            "Bias correction not implemented for ratios.")
                    i1, i2 = n.split('/')
                    if i1 in bias:
                        it.surrogates[0] *= kernels.Constant(1 - bias[i1])
                    if i2 in bias:
                        it.surrogates[1] *= kernels.Constant(1 - bias[i2])
            labels = self.labels
            logging.debug(f'labels: {self.labels}')
            if isinstance(limits, (str, Path, PurePath)):
                bounds, constants = self._limits_from_file(limits)
                limits = bounds
                if const:
                    const = constants.update(const)
            priors = self._make_priors(labels, limits, const)
            logging.debug(f'priors: {priors}')
            distrib = self._make_distributions(ids, uncertainty)
            self._joint_probability(ids, distrib)

            if 'step' in kwargs and 'target_accept' in kwargs:
                step = kwargs['step']
                kw = {'target_accept': kwargs.pop('target_accept')}
                kwargs.update({'step': getattr(pm, step)(**kw)})
            elif 'step' in kwargs:
                step = kwargs['step']
                kwargs.update({'step': getattr(pm, step)()})
            elif 'target_accept' in kwargs:
                kwargs.update(
                    {'nuts': {
                        'target_accept': kwargs.pop('target_accept')
                    }})

            if load is None:
                ## Silence the deprecation warning
                if 'return_inferencedata' not in kwargs:
                    kwargs['return_inferencedata'] = False
                if debug:
                    print(model.debug())
                trace = pm.sample(**kwargs)
            else:
                trace = pm.load_trace(load)

            if plot:
                pm.plot_trace(trace, lines=list(limits.keys()))

            if kwargs['return_inferencedata']:
                return trace
            else:
                summaries = {n: pm.summary(trace, n).T for n in limits}
                return summaries, trace


class WasteBinMixture(WasteBin):
    """Bayesian inference with a mixture of nuclear wastes.

    The GPR models are trained on concentrations of single
    isotope, but the inference is done with ratios. Thus
    several GPR models are combined to predict one ratio.
    """

    def __init__(
        self,
        model_type,
        labels,
        evidence,
        filepaths=None,
        model_ratios=False,
        combination='PredictorSum2',
        mixing_type='relative',
    ):
        """Set model type and filepaths for loading models

        Parameters
        ----------
        model_type : dict
            Dictionary associating each batch component
            of the mixture with a surrogate model class from
            the kernels module.
        labels : dict
            Dictionary associating each batch component of
            the mixture with a list of parameter names.
        evidence : dict-like
            Isotopic evidence, from which the parameters are
            to be inferred. Keys are identifiers of isotope
            or ratio.
        filepaths : dict
            Dictionary of filepath dictionaries. Associates
            each batch component of the miture with a set of
            filepaths from which the trained models of the
            isotopes can be loaded.
        model_ratios : bool (optional, default is False)
            Set to True if GP-models predict isotopic ratios
            directly. This will not lead to a physically
            meaningful mixture of isotopes.
        combination : str (optional, default is PredictorSum2)
            Specify the class used for adding models to form
            the mixture.
        mixing_type : str, (optional, default is 'relative')
            Select between 'absolute', 'relative', normalize or
            'categorical'. Set to 'relative' to enforce inferring
            mixing ratios relative to the first batch. This option 
            overrides a any prior given in a limits file or dictionary
            for the first mixing ratio parameter.
            Set to 'categorical' to set mixing ratios as categorical
            variables that are either 0 or 1.
            Set to normalize to normalize the mixing ratio so that
            their sum equals 1.
        """
        super().__init__(model_type, labels, evidence, filepaths, model_ratios)
        self.batches = list(model_type.keys())
        self.combination = getattr(kernels, combination)
        self.mixing_type = mixing_type

    def load_filepaths(self, ids, modelfile, prefix=None):
        """Load the filepath dict from a json file."""
        if not self.model_ratios:
            if all(map(RATIO_REGEX.fullmatch, ids)):
                isolist = []
                for r in ids:
                    isolist.extend(r.split('/'))
                ids = isolist
            else:
                pass
        filepaths = {}
        if isinstance(modelfile, dict):
            paths = {}
            for b, mfile in modelfile.items():
                with open(mfile, 'r') as f:
                    paths[b] = json.load(f)
        else:
            with open(modelfile, 'r') as f:
                paths = json.load(f)
        for b in self.batches:
            fp = {}
            if prefix is None:
                if isinstance(modelfile, dict):
                    prefix = str(PurePath(modelfile[b]).parent)
                else:
                    prefix = str(PurePath(modelfile).parent)
            for i in ids:
                if isinstance(modelfile, dict):
                    try:
                        fp[i] = os.path.join(prefix, *paths[b][i])
                    except KeyError:
                        msg = f'No model for {i} in batch {b}.'
                        raise KeyError(msg)
                else:
                    fp[i] = os.path.join(prefix, *paths[i])
            filepaths[b] = fp
        self.filepaths = filepaths

    def load_models(self, ids):
        """Load the surrogate models of the isotopes

        Takes a list of isotope identifiers and creates a
        composite surrogate model for the isotope value of
        the mixture from the surrogate models of each batch.

        Parameters
        ----------
        ids : list
            String identifiers of the isotopes or isotopic
            ratios.
        combination : str, optional (default is PredictorSum2)
            Specifies the class that is used to combine the
            surrogate models.
        """
        m = self.combination
        if self.model_ratios:
            for i in ids:
                if isinstance(m, kernels.PredictorSum2):
                    args = list(
                        chain(*[[self.filepaths[j][i], self.model_type[j]]
                                for j in self.batches]))
                else:
                    args = list([self.filepaths[j][i], self.model_type[j]]
                                for j in self.batches)
                self.models[i] = m.from_file(*args)
        else:
            for r in ids:
                i, j = r.split('/')
                if m is kernels.PredictorSum2:
                    args_i = list(
                        chain(*[[self.filepaths[b][i], self.model_type[b]]
                                for b in self.batches]))
                    mi = m.from_file(*args_i)
                    args_j = list(
                        chain(*[[self.filepaths[b][j], self.model_type[b]]
                                for b in self.batches]))
                    mj = m.from_file(*args_j)
                    self.models[r] = kernels.PredictorQuotient(mi, mj)
                else:
                    args_i = list([self.filepaths[b][i], self.model_type[b]]
                                  for b in self.batches)
                    mi = m.from_file(*args_i)
                    args_j = list([self.filepaths[b][j], self.model_type[b]]
                                  for b in self.batches)
                    mj = m.from_file(*args_j)
                    self.models[r] = kernels.Quotient([mi, mj])

    def _make_priors(self, labels, limits, fallback):
        """Turn parameter limits into prior distribution

        Overrides limits of first mixing ratio if self.mixing_type
        is set to True.
        """

        def prior_generator(par, lim, fall):
            for param in par:
                if param in lim:
                    if isinstance(lim[param], float) or isinstance(
                            lim[param], int):
                        logging.debug(
                            f'Generating {param} prior from const:{lim[param]}.'
                        )
                        yield get_deterministic(param, lim[param])
                    else:
                        logging.debug(
                            f'Generating {param} prior from {lim[param]}.')
                        yield get_distribution(param, **lim[param])
                else:
                    logging.debug(f'Using {fall[param]} for {param}.')
                    yield get_deterministic(param, fall[param])

        priors = []
        n_batches = len(labels)
        mixing_ids = []
        for i, (b, l) in enumerate(labels.items()):
            a, p = l[0], l[1:]
            logging.debug(f'Adding priors for {b}')
            if self.mixing_type == 'relative' and i == 0:
                priors.extend(list(prior_generator([a], {}, {a: 1.})))
            elif self.mixing_type == 'categorical':
                if i == 0:
                    d = {
                        'cat': {
                            'p': np.ones(n_batches) / n_batches,
                            "dist": "Categorical"
                        }
                    }
                    base = list(prior_generator(['cat'], d, {}))
                ## Two options to map i to 0 or 1
                ## Option 2 is probably faster
                # f = tt.round(tt.exp(-(base[0] - i)**2))
                f = 0**((base[0] - i)**2)
                priors.extend([pm.Deterministic(a, f)])
            elif self.mixing_type == 'normalize':
                if i == 0:
                    mix_id_1 = a
                else:
                    uni = list(
                        prior_generator([a], {a: {
                            'lower': 0,
                            'upper': 1
                        }}, None))
                    mixing_ids.extend(uni)
                    priors.extend(uni)
            else:
                priors.extend(list(prior_generator([a], limits, fallback)))
            priors.extend(list(list(prior_generator(p, limits, fallback))))
        if self.mixing_type == 'normalize':
            priors.insert(0, pm.Deterministic(mix_id_1, 1 - sum(mixing_ids)))
        self.priors = priors
        return priors
