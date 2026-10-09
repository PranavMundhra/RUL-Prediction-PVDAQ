# SPDX-License-Identifier: Apache-2.0
import pytest

from pvdaq_rul import Config
from tests.fake_pvdaq import metric_names


@pytest.fixture(scope="session")
def cfg():
    return Config(exclude_periods=())


@pytest.fixture(scope="session")
def names():
    return metric_names()
