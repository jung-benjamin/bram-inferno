#! /usr/bin/env python3
"""Gaussian Process Kernels in Theano

Contains kernels for surrogate modelling of isotopic ratios in
nuclear waste with gaussian processes implemented in theano.

The kernels are intended to be used with hyperparameters that have
already been optimized. The focus is on good compatibility for
bayesian inference with pymc.

@author jung-benjamin
"""

import json
import logging
from abc import ABC, abstractmethod

import numpy as np

try:
    import theano.tensor as tt
    from theano import scan
except ImportError:
    import pytensor.tensor as tt
    from pytensor import scan

OLDKEYS = [
    'Params',
    'LAMBDA',
    'alpha_',
    'x_train',
    'y_train',
    'y_trafo',
    'x_trafo',
]
KEYS = [
    'parameters',
    'lambda_inv',
    'alpha',
    'xtrain',
    'ytrain',
    'ytrafo',
    'xtrafo',
]
KEYMAP = dict(zip(OLDKEYS, KEYS))


class Surrogate(ABC):
    """A base class for surrogate models"""

    @classmethod
    @abstractmethod
    def from_file(self, ):
        """Load the model from a file"""
        pass

    @property
    def n_args(self):
        """The number of arguments accepted by the model."""
        return self._n_args

    @n_args.setter
    def n_args(self, x):
        """Set the number of arguments."""
        self._n_args = x

    @abstractmethod
    def predict(self, x):
        """Calculate model predictions for a point."""
        pass

    def __add__(self, other):
        """Add two surrogate models together

        Returns a new surrogate model that is the sum of the
        two models.
        """
        if other == 0 or other == 0.0:
            return self
        else:
            return Sum([self, other])

    def __radd__(self, other):
        """Add two surrogate models together

        Returns a new surrogate model that is the sum of the
        two models.
        """
        if other == 0 or other == 0.0:
            return self
        else:
            return Sum([other, self])

    def __mul__(self, other):
        """Multiply two surrogate models together

        Returns a new surrogate model that is the product of the
        two models.
        """
        if other == 1 or other == 1.0:
            return self
        else:
            return Product([self, other])

    def __rmul__(self, other):
        """Multiply two surrogate models together

        Returns a new surrogate model that is the product of the
        two models.
        """
        if other == 1 or other == 1.0:
            return self
        else:
            return Product([other, self])

    def predict_many(self, x, eval=False):
        """Calculate the posterior predictive for a vector x

        Uses the theano.scan method for faster computation of
        the loop.

        Parameters
        ----------
        x : list of float
            Values for which the posterior predictive is to be
            evaluated.
        eval : bool, optional (default is False)
            If True, the eval() method of the theano object is
            called before returning the predictions.
            This significantly increases the runtime.

        Returns
        -------
        posterior
            Posterior predictive for each point in x.
        """
        xtt = tt.cast(x, 'float64')
        posterior, updates = scan(self.predict, xtt)
        if eval:
            return posterior.eval()
        return posterior


class Combination(Surrogate):
    """Combine two surrogate models into one object"""

    def __init__(self, surrogates):
        """Set a list of composing models.

        Parameters
        ----------
        surrogates : list of Surrogate
            List of the surrogate models to be combined to
            a new model.
        """
        self.surrogates = surrogates

    @classmethod
    def from_file(cls, *args):
        """Load the model from a file

        Parameters
        ----------
        args : tuples (filepath, class)
            Each tuple contains a filepath and a class. The
            class needs to be able to use its from_file method
            to instantiate itself from the respective filepath.

        Returns:
        k : Combination
            Instance of the class.
        """
        surrogates = [c.from_file(f) for (f, c) in args]
        return cls(surrogates)


class LinearCombination(Combination):
    """Create a linear combination of surrogate models"""

    def _split_arglist(self, arglist):
        """Yield arguments of a given length from a list.

        Each chunk is divided into its first element and
        a list of the remaining arguments.
        """
        num = len(self.surrogates)
        logging.debug(f'Num. surrogates in linear combination: {num}.')
        chunk_len = [x.n_args + 1 for x in self.surrogates]  ## + 1 for mixing
        logging.debug(f'Num. arguments of each surrogate: {chunk_len}.')
        upper = list(np.cumsum(chunk_len))
        lower = [0] + upper[:-1]
        slices = map(slice, lower, upper)
        for s in slices:
            y = arglist[s]
            yield y[0], y[1:]

    def predict(self, x):
        """Calculate the posterior predictive"""
        iter = zip(self.surrogates, self._split_arglist(x))
        return sum(i * m.predict(j) for m, (i, j) in iter)


class Quotient(Combination):
    """Calcualte the quotient of two surrogate models"""

    def predict(self, x1):
        """Calculate the quotient of the posterior predictives

        The first entry in the surrogates list is the numerator and
        the second entry is the denominator. Any further entries are
        ignored.
        """
        return self.surrogates[0].predict(x1) / self.surrogates[1].predict(x1)


class Constant(Surrogate):
    """A constant surrogate model

    This model always returns the same value, which is
    set during initialization.
    """

    def __init__(self, value):
        """Set the constant value"""
        self.value = tt.cast(value, 'float64')

    def predict(self, x):
        """Return the constant value for any input"""
        return self.value

    @classmethod
    def from_file(cls, filepath):
        """Load the model from a file

        Parameters
        ----------
        filepath : str, path-like
            Path to the file containing model data.
        """
        if str(filepath).endswith('.json'):
            with open(filepath, 'r') as f:
                d = json.load(f)
        elif str(filepath).endswith('.npy'):
            d = np.load(filepath, allow_pickle=True).item()
        else:
            msg = 'File format is not supported by the model.'
            print(msg)
        return cls(d['Constant'])


class Sum(Combination):
    """Calculate the sum of two surrogate models

    The first entry in the surrogates list is added to the
    second entry. Any further entries are ignored.
    """

    def predict(self, x):
        """Calculate the posterior predictive"""
        return self.surrogates[0].predict(x) + self.surrogates[1].predict(x)


class Product(Combination):
    """Calculate the product of two surrogate models

    The first entry in the surrogates list is multiplied with
    the second entry. Any further entries are ignored.
    """

    def predict(self, x):
        """Calculate the posterior predictive"""
        return (self.surrogates[0].predict(x) * self.surrogates[1].predict(x))


class AnisotropicSquaredExponentialKernel:
    """Anisotropic squared exponential kernel

    This kernel is used for Gaussian processes and is defined
    by a set of hyperparameters.
    """

    def __init__(self, parameters, xtrain, ytrain, lambda_inv, alpha, xtrafo,
                 ytrafo):
        """Cast the data to theano objects"""
        self.parameters = tt.cast(parameters, 'float64')
        self.constant = self.parameters[0]
        self.noise = self.parameters[-1]
        self.lengthscales = self.parameters[1:-1]
        self.xtrain = tt.cast(xtrain, 'float64')
        self.ytrain = tt.cast(ytrain, 'float64')
        self.lambda_inv = tt.cast(lambda_inv, 'float64')
        self.alpha = tt.cast(alpha, 'float64')
        if isinstance(xtrafo[0], str):
            self.xtrafo = tt.cast(xtrafo[1], 'float64')
        else:
            self.xtrafo = tt.cast(xtrafo, 'float64')
        if isinstance(ytrafo[0], str):
            self.ytrafo = tt.cast(ytrafo[1], 'float64')
        else:
            self.ytrafo = tt.cast(ytrafo, 'float64')
        self.n_args = len(parameters) - 2

    @classmethod
    def from_dict(cls, d):
        """Load the model from a file

        Parameters
        ----------
        d : dict
            Dictionary containing model data. The keys should
            match the keys defined in KEYS or OLDKEYS.
        Returns
        -------
        model : AnisotropicSquaredExponentialKernel
            Instance of the class.
        """
        arg_dict = {}
        for n, it in d.items():
            if n in OLDKEYS:
                arg_dict[KEYMAP[n]] = it
            elif n in KEYS:
                arg_dict[n] = it
            else:
                pass
        return cls(arg_dict['parameters'], arg_dict['xtrain'],
                   arg_dict['ytrain'], arg_dict['lambda_inv'],
                   arg_dict['alpha'], arg_dict['xtrafo'], arg_dict['ytrafo'])

    @classmethod
    def from_file(cls, filepath):
        """Load the model from a file

        Parameters
        ----------
        filepath : str, path-like
            Path to the file containing model data.

        Returns:
        -------
        model : AnisotropicSquaredExponentialKernel
            Instance of the class.
        """
        if str(filepath).endswith('.json'):
            with open(filepath, 'r') as f:
                d = json.load(f)
        elif str(filepath).endswith('.npy'):
            d = np.load(filepath, allow_pickle=True).item()
        else:
            msg = 'File format is not supported by the model.'
            print(msg)
        return cls.from_dict(d)

    def transform_x(self, x):
        """Transform the x data

        The xtrafo is the maximum of the training data.
        """
        return x / self.xtrafo

    def untransform_y(self, y):
        """Untransform the y predictions

        The ytrafo are the mean and std of a normal distribution.
        """
        return y * self.ytrafo[1] + self.ytrafo[0]

    def __call__(self, x):
        """Calculate the posterior predictive for x"""
        xtt = self.transform_x(tt.cast(x, 'float64'))
        distance = self.transform_x(self.xtrain) - xtt
        distance_sq = tt.dot(tt.dot(distance, self.lambda_inv**2),
                             distance.T).diagonal()
        ktrans = self.constant**2 * tt.exp(-distance_sq / 2)
        noise_diag = tt.cast(np.ones(self.xtrain.shape.eval()[0]),
                             'float64') * self.noise**2
        ktrans += noise_diag
        y = self.untransform_y(tt.dot(ktrans, self.alpha))
        return tt.max([y, 0])


class GaussianProcessModel(Surrogate):
    """Gaussian process-based surrogate model.
    
    Instances of this class are the building blocks for more
    complex surrogate models.
    """

    _kernels = {
        'AnisotropicSquaredExponential': AnisotropicSquaredExponentialKernel,
    }

    def __init__(self, kernel):
        """Set the kernel for the model

        Parameters
        ----------
        kernel : Kernel
            The kernel to be used for the Gaussian process.
        """
        self.kernel = kernel

    @classmethod
    def from_file(cls, filepath):
        """Load the model from a file

        Parameters
        ----------
        filepath : str, path-like
            Path to the file containing model data.
        """
        if str(filepath).endswith('.json'):
            with open(filepath, 'r') as f:
                d = json.load(f)
        elif str(filepath).endswith('.npy'):
            d = np.load(filepath, allow_pickle=True).item()
        else:
            msg = 'File format is not supported by the model.'
            print(msg)
        kernel_type = d.get('kernel')
        if kernel_type is None:
            raise ValueError('Kernel type not specified in the file.')
        kernel_class = cls._kernels.get(kernel_type)
        if kernel_class is None:
            raise ValueError(f'Unknown kernel type: {kernel_type}')
        return cls(kernel_class.from_dict(d))

    def predict(self, x):
        """Calculate the posterior predictive for a point x

        This method is intended to be overridden by subclasses.
        """
        return self.kernel(x)


class ASQEKernelPredictor(Surrogate):
    """Posterior predictive of an ASQE kernel

    Works with pre-computed matrices to facilitate
    faster predictions.
    """

    def __init__(self, parameters, xtrain, ytrain, lambda_inv, alpha, xtrafo,
                 ytrafo):
        """Cast the data to theano objects"""
        warnings.warn(
            'ASQEKernelPredictor is deprecated. Use AnisotropicSquaredExponentialKernel instead.',
            DeprecationWarning,
            stacklevel=2)
        self.parameters = tt.cast(parameters, 'float64')
        self.constant = self.parameters[0]
        self.noise = self.parameters[-1]
        self.lengthscales = self.parameters[1:-1]
        self.xtrain = tt.cast(xtrain, 'float64')
        self.ytrain = tt.cast(ytrain, 'float64')
        self.lambda_inv = tt.cast(lambda_inv, 'float64')
        self.alpha = tt.cast(alpha, 'float64')
        if isinstance(xtrafo[0], str):
            self.xtrafo = tt.cast(xtrafo[1], 'float64')
        else:
            self.xtrafo = tt.cast(xtrafo, 'float64')
        if isinstance(ytrafo[0], str):
            self.ytrafo = tt.cast(ytrafo[1], 'float64')
        else:
            self.ytrafo = tt.cast(ytrafo, 'float64')
        self.n_args = len(parameters) - 2

    @classmethod
    def from_file(cls, filepath):
        """Load the model from a file

        Parameters
        ----------
        filepath : str, path-like
            Path to the file containing model data.
        """
        if str(filepath).endswith('.json'):
            with open(filepath, 'r') as f:
                d = json.load(f)
        elif str(filepath).endswith('.npy'):
            d = np.load(filepath, allow_pickle=True).item()
        else:
            msg = 'File format is not supported by the model.'
            print(msg)
        arg_dict = {}
        for n, it in d.items():
            if n in OLDKEYS:
                arg_dict[KEYMAP[n]] = it
            elif n in KEYS:
                arg_dict[n] = it
            else:
                pass
        return cls(arg_dict['parameters'], arg_dict['xtrain'],
                   arg_dict['ytrain'], arg_dict['lambda_inv'],
                   arg_dict['alpha'], arg_dict['xtrafo'], arg_dict['ytrafo'])

    def transform_x(self, x):
        """Transform the x data

        The xtrafo is the maximum of the training data.
        """
        return x / self.xtrafo

    def untransform_y(self, y):
        """Untransform the y predictions

        The ytrafo are the mean and std of a normal distribution.
        """
        return y * self.ytrafo[1] + self.ytrafo[0]

    def predict(self, x):
        """Calculate the posterior predictive for x"""
        xtt = self.transform_x(tt.cast(x, 'float64'))
        distance = self.transform_x(self.xtrain) - xtt
        distance_sq = tt.dot(tt.dot(distance, self.lambda_inv**2),
                             distance.T).diagonal()
        ktrans = self.constant**2 * tt.exp(-distance_sq / 2)
        noise_diag = tt.cast(np.ones(self.xtrain.shape.eval()[0]),
                             'float64') * self.noise**2
        ktrans += noise_diag
        y = self.untransform_y(tt.dot(ktrans, self.alpha))
        return tt.max([y, 0])


class PredictorSum2():
    """Sum of two predictor kernels

    A composite class of two kernel predictor classes that
    calculates the sum of the two composites.
    """

    def __init__(self, k1, k2):
        """Set the two kernel predictor instances"""
        self.k1 = k1
        self.k2 = k2

    @classmethod
    def from_file(cls, filepath1, modeltype1, filepath2, modeltype2):
        """Load the model from a file

        Parameters
        ----------
        filepath1, filepath2 : str, path-like
            Path to the file containing model data.
        modeltype1, modeltype2
            Model types intended for each kernel. Both must have
            a `from_file` classmethod.

        Returns:
        k : PredictorSum2 object
            Instance of the class.
        """
        k1 = modeltype1.from_file(filepath1)
        k2 = modeltype2.from_file(filepath2)
        k = cls(k1, k2)
        return k

    def predict(self, p):
        """Calculate the posterior predictive of the sum"""
        return (p[0] * self.k1.predict([p[1], p[2]]) +
                p[3] * self.k2.predict([p[4], p[5]]))


class PredictorQuotient():
    """Quotient of two predictor kernels

    A composite class of two kernel predictors that calculates the
    quotient of the predictions. The predictor kernels take the same
    input parameters.
    """

    def __init__(self, k1, k2):
        """Set the two kernel predictor instances

        k1 is the numerator and k2 is the denominator.
        """
        self.k1 = k1
        self.k2 = k2

    @classmethod
    def from_file(cls, filepath1, modeltype1, filepath2, modeltype2):
        """Load the model from a file

        Parameters
        ----------
        filepath1, filepath2 : str, path-like
            Path to the file containing model data.
        modeltype1, modeltype2
            Model types intended for each kernel. Both must have
            a `from_file` classmethod.

        Returns:
        k : PredictorSum2 object
            Instance of the class.
        """
        k1 = modeltype1.from_file(filepath1)
        k2 = modeltype2.from_file(filepath2)
        k = cls(k1, k2)
        return k

    def predict(self, x1):
        """Calculate the quotient of the posterior predictives"""
        return self.k1.predict(x1) / self.k2.predict(x1)

    def predict_many(self, x, eval=False):
        """Calculate the posterior predictive for a vector x

        Uses the theano.scan method for faster computation of
        the loop.

        Parameters
        ----------
        x : list of float
            Values for which the posterior predictive is to be
            evaluated.
        eval : bool, optional (default is False)
            If True, the eval() method of the theano object is
            called before returning the predictions.
            This significantly increases the runtime.

        Returns
        -------
        posterior
            Posterior predictive for each point in x.
        """
        xtt = tt.cast(x, 'float64')
        posterior, updates = scan(self.predict, xtt)
        if eval:
            return posterior.eval()
        return posterior
