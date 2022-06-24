#! /usr/bin/env python3

"""PyMC models for nuclear waste

Bayesian inference models for reconstructing reactor
operating histories from nuclear waste compositions.

@author jung-benjamin
"""

import os
import json
import numpy as np
import pymc3 as pm
import theano.tensor as tt

from itertools import chain

from . import kernels


class WasteBin():
    """Bayesian inference with nuclear waste

    A class for building a surrogate model of nuclear for
    reconstructing operating history parameters of reactors.
    """

    def __init__(
        self, model_type, labels, evidence, filepaths=None, model_ratios=True
        ):
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

    @staticmethod
    def make_filepaths(ids, base, form, join='-'):
        """Create dictionary of filepaths

        If ids are isotopic ratios separated by a /,
        the / is replaced by a -.

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
        join : str, optional (default is "-")
            Character with which isotopes are separated in
            the names of the kernel files.

        Returns
        -------
        filepaths : dict
            Dictionary of filepaths, with model identifiers
            as keys.
        """
        p = os.path.join(base, form)
        filepaths = {i: p.format(i.replace('/', join)) for i in ids}
        return filepaths

    def load_filepaths(self, ids, modelfile, prefix):
        """Load the filepath dict from a json file"""
        filepaths = {}
        with open(modelfile, 'r') as f:
            paths = json.load(f)
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
                self.models[i] = kernels.PredictorQuotient.from_file(
                    self.filepaths[iso1],
                    self.model_type,
                    self.filepaths[iso2],
                    self.model_type
                )

    def _make_priors(self, p, limits, fallback):
        """Turn parameter limits into prior distribution

        Needs to be called within the pm.Model context.
        """
        priors = []
        for param in p:
            if param in limits:
                priors.append(pm.Uniform(param, **limits[param]))
            else:
                priors.append(tt.cast(fallback[param], 'float64'))
        self.priors = priors
        return priors

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
        distrib = [pm.Normal(i, mu=m, sd=s, observed=o)
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
            l = pm.DensityDist('L', joint, observed=dict(zip(ids, dists)))
        return l

    def inference(self, ids, limits, uncertainty=0.1, const=None,
                  plot=True, load=None, **kwargs):
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
        limits : dict
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
        **kwargs
            Keyword arguments for the pymc.sample method.

        Returns
        -------
        summaries : dict
            Key properties of the reconstructed distributions.
        trace : pymc trace object
            Trace of the sampling algorithm.
        """

        with pm.Model():
            ## Needs to be called inside the context manager
            ## Otherwise models don't work
            self.load_models(ids)

            labels = self.labels
            ## Create priors
            priors = self._make_priors(labels, limits, const)

            distrib = self._make_distributions(ids, uncertainty)
            self._joint_probability(ids, distrib)

            if 'step' in kwargs:
                step = kwargs['step']
                kwargs.update({'step': getattr(pm, step)()})

            if load is None:
                ## Silence the deprecation warning
                if 'return_inferencedata' not in kwargs:
                    kwargs['return_inferencedata'] = False
                trace = pm.sample(**kwargs)
            else:
                trace = pm.load_trace(load)

            if plot:
                pm.plot_trace(trace, lines=list(limits.keys()))

            summaries = {n: pm.summary(trace, n).T for n in limits}
            return summaries, trace


class WasteBinMixture(WasteBin):
    """Bayesian inference with a mixture of nuclear wastes.

    The GPR models are trained on concentrations of single
    isotope, but the inference is done with ratios. Thus
    several GPR models are combined to predict one ratio.
    """

    def __init__(
        self, model_types, filepaths, labels, evidence, model_ratios=False
    ):
        """Set model type and filepaths for loading models

        Parameters
        ----------
        model_type : dict
            Dictionary associating each batch component
            of the mixture with a surrogate model class from
            the kernels module.
        filepaths : dict
            Dictionary of filepath dictionaries. Associates
            each batch component of the miture with a set of
            filepaths from which the trained models of the
            isotopes can be loaded.
        labels : dict
            Dictionary associating each batch component of
            the mixture with a list of parameter names.
        evidence : dict-like
            Isotopic evidence, from which the parameters are
            to be inferred. Keys are identifiers of isotope
            or ratio.
        """
        ## Verify that each dict has the same keys
        assert set(model_types.keys()) == set(filepaths.keys())
        assert set(filepaths.keys()) == set(labels.keys())
        self.batches = list(model_types.keys())
        self.model_types = model_types
        self.filepaths = filepaths
        self.labels = labels
        self.evidence = evidence
        self.models = {}
        self.model_ratios = model_ratios

    def load_models(self, ids, combination='PredictorSum2'):
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
        m = getattr(kernels, combination)
        if self.model_ratios:
            for i in ids:
                args = list(chain(*[
                    [self.filepaths[j][i], self.model_types[j]]
                    for j in self.batches
                ]))
                self.models[i] = m.from_file(*args)
        else:
            for r in ids:
                i, j = r.split('/')
                args_i = list(chain(*[
                    [self.filepaths[b][i], self.model_types[b]]
                    for b in self.batches
                ]))
                mi = m.from_file(*args_i)
                args_j = list(chain(*[
                    [self.filepaths[b][j], self.model_types[b]]
                    for b in self.batches
                ]))
                mj = m.from_file(*args_j)
                self.models[r] = kernels.PredictorQuotient(mi, mj)

    def _make_priors(self, labels, limits, fallback):
        """Turn parameter limits into prior distribution"""
        def prior_generator(par, lim, fall):
            for param in par:
                if param in lim:
                    yield pm.Uniform(param, **lim[param])
                else:
                    yield fall[param]

        priors = []
        for b, l in labels.items():
            a, p = l[0], l[1:]
            priors.extend(list(prior_generator([a], limits, fallback)))
            priors.extend(list(list(prior_generator(p, limits, fallback))))
        self.priors = priors
        return priors

    def inference(self, ids, limits, combination='PredictorSum2',
                  uncertainty=0.1, const=None, plot=True, load=None, **kwargs):
        """Run bayesian inference with pymc uniform priors

        Creates the Model context manager of pymc and runs bayesian
        inference with the specified parameters. Isotopic ratios
        of the mixture are calculated by summing the GPR models of
        the isotope ratios and dividing to obtain the ratios.
        Parameters that are also variable in the surrogate model, but
        are not intended for inference are set to constant values.

        Currently only intended for a mixture of two batches with the
        PredictorSum2 kernel.

        Parameters
        ----------
        ids : list of str
            Identifiers of isotopic ratios to be used in the inference.
        limits : dict
            Keys are the labels of the parameters that are to be
            reconstructed. Each entry is a dict with keys 'lower'
            and 'upper', specifying the lower and upper limits of
            the uniform prior distribution of the parameter.
        combination : str, optional (default is PredictorSum2)
            Specify the class to be used for combining the surrogate
            models of each batch into a mixture.
        uncertainty : float or dict-like, optional (default is 0.1)
            Relative uncertainty assumed for the evidence. If uncertainty
            is a float, the same uncertainty is assumed for each ratio.
            If a dict-like object is passed, it must kontain keys that
            correspond to elements in `ids`.
        const : dict of theano floats, optional (default is None)
            Must be specified if not all elements of `reconstruct` are
            True. Specifies the constant values of the parameters that
            are not varied. Keys are parameter labels. All parameter
            labels must be either in `limits` or `const`.
        plot : bool, optional (default is True)
            If True, plots the trace after running the sampler.
        load : str, optional (default is None)
            Path to a directory, where a trace was stored by a previous
            sampling process. If the path is specified, the model is
            loaded from the trace and the sampler is not run.
        **kwargs
            Keyword arguments for the pymc.sample method.

        Returns
        -------
        summaries : dict
            Key properties of the reconstructed distributions.
        trace : pymc trace object
            Trace of the sampling algorithm.
        """

        with pm.Model():
            ## Needs to be called inside the context manager
            ## Otherwise models don't work
            iso_ids = list(set(list(
                chain(*list(map(lambda x: x.split('/'), ids)))
            )))
            self.load_models(ids, combination)

            labels = list(chain(*[d for i, d in self.labels.items()]))
            ## Create priors
            priors = self._make_priors(labels, limits, const)

            evidence = [self.evidence[i] for i in ids]
            if isinstance(uncertainty, float):
                sigma = [uncertainty * e for e in evidence]
            else:
                sigma = [self.evidence[i] * uncertainty[i] for i in ids]
            models = []
            for i in ids:
                models.append(
                    self.models[j].predict(
                        priors[0], priors[1:3], priors[3], priors[4:6]
                    )
                )
            distrib = [pm.Normal(i, mu=m, sd=s, observed=o)
                       for i, m, s, o in zip(ids, models, sigma, evidence)
                      ]
            self._joint_probability(ids, distrib)

            if 'step' in kwargs:
                step = kwargs['step']
                kwargs.update({'step': getattr(pm, step)()})

            if load is None:
                ## Silence the deprecation warning
                if 'return_inferencedata' not in kwargs:
                    kwargs['return_inferencedata'] = False
                trace = pm.sample(**kwargs)
            else:
                trace = pm.load_trace(load)

            if plot:
                pm.plot_trace(trace, lines=list(limits.keys()))

            summaries = {n: pm.summary(trace, n).T for n in limits}
            return summaries, trace
