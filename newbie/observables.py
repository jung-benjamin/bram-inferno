#! /usr/bin/env python3
"""Predict the observable quantities for the inference model.

The observables are isotopic ratios. The models for predicting these
ratios either predict the ratios directly or predict the
numerator and denominator separately.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from .kernels import GaussianProcessModel, Quotient

RATIO_PATTERN = re.compile(
    r"[A-Za-z]+-?\d+(\*|_?m\d?)/[A-Za-z]+-?\d+(\*|_?m\d?)")


def is_ratio(name):
    """Check if the name is a ratio."""
    return bool(RATIO_PATTERN.fullmatch(name))


class GaussianProcessCollection(dict):
    """A collection of gaussian process models."""

    def __init__(self, *args, ratio_models=False, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialize the collection of GP models."""
        self.ratio_models = ratio_models

    def set_filepaths(self, file_dict, prefix=""):
        """Set the filepaths that contain stored GP models."""
        filepaths = {}
        for k, it in file_dict.items():
            filepaths[k] = Path(prefix, *it)
        self.filepaths = filepaths

    def load_filepaths(self, fp, prefix=None):
        """Load the filepaths from a json file."""
        fp = Path(fp)
        with fp.open("r") as f:
            file_dict = json.load(f)
        if prefix is None:
            prefix = fp.parent
        self.set_filepaths(file_dict, prefix=prefix)

    def load_models(self, ids):
        """Load models for the ratios from the filepaths."""
        for id_ in ids:
            if self.ratio_models:
                self[id_] = GaussianProcessModel.from_file(self.filepaths[id_])
            else:
                if not is_ratio(id_):
                    self[id_] = GaussianProcessModel.from_file(
                        self.filepaths[id_])
                else:
                    i, j = id_.split("/")
                    self[id_] = Quotient.from_file(self.filepaths[i],
                                                   self.filepaths[j])
