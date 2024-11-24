from __future__ import annotations

import importlib.metadata

import patato as m


def test_version():
    assert importlib.metadata.version("patato") == m.__version__
