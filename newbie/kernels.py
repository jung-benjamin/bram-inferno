#! /usr/bin/env python3

"""Gaussian Process Kernels in Theano

Contains kernels for surrogate modelling of isotopic ratios in
nuclear waste with gaussian processes implemented in theano.

The kernels are intended to be used with hyperparameters that have
already been optimized. The focus is on good compatibility for
bayesian inference with pymc.

@author jung-benjamin
"""

import numpy as np
import theano.tensor as tt


class ASQEKernelPredictor():
    """Posterior predictive of an ASQE kernel

    Works with pre-computed matrices to facilitate
    faster predictions.
    """

    def __init__(self, parameters, xtrain, ytrain,
                 lambda_inv, alpha, xtrafo, ytrafo):
        """Cast the data to theano objects"""
        self.parameters = tt.cast(parameters, 'float64')
        self.constant = self.parameters[0]
        self.noise = self.parameters[-1]
        self.lengthscales = self.parameters[1:-1]
        self.xtrain = tt.cast(xtrain, 'float64')
        self.ytrain = tt.cast(ytrain, 'float64')
        self.lambda_inv = tt.cast(lambda_inv, 'float64')
        self.alpha = tt.cast(alpha, 'float64')
        self.xtrafo = tt.cast(xtrafo, 'float64')
        self.ytrafo = tt.cast(ytrafo, 'float64')

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
        distance_sq = tt.dot(tt.dot(distance, self.lambda_inv), distance.T).diagonal()
        ktrans = self.constant**2 * tt.exp(-distance_sq / 2)
        noise_diag = tt.cast(np.ones(self.xtrain.shape.eval()[0]), 'float64') * self.noise **2
        ktrans += noise_diag
        y = self.untransform_y(tt.dot(ktrans, self.alpha))
        return y
