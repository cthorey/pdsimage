"""Shared fixtures."""

from __future__ import annotations

import pytest

from helpers import make_lola_table


@pytest.fixture
def lola_table(tmp_path):
    return make_lola_table(tmp_path)
