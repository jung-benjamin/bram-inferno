#! /usr/bin/env python3
"""Configure the pytest unit tests"""

from pathlib import Path
import pytest


@pytest.fixture
def rootdir():
    return Path(__file__).resolve().parent