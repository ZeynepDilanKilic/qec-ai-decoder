"""
_truba_fix.py
=============
TRUBA'daki torch surumune ozgu `torch._dynamo` dairesel import hatasi
icin stub modul. Egitim/degerlendirme scriptlerinde torch'tan ONCE
import edilir:

    import _truba_fix  # noqa: F401

Mekanizma: torch._dynamo normal import edilebiliyorsa HICBIR SEY yapmaz
(normal makinelerde no-op). Import patliyorsa sys.modules'a sahte bir
`torch._dynamo` modulu koyar; torch.compile() calismaz ama normal
egitim/cikarim etkilenmez.
"""
from __future__ import annotations

import sys
import types


def _install_stub():
    if "torch._dynamo" in sys.modules:
        return
    stub = types.ModuleType("torch._dynamo")

    def _noop(*args, **kwargs):
        return None

    stub.disable = _noop
    stub.reset = _noop
    stub.optimize = lambda *a, **k: (lambda f: f)
    stub.config = types.SimpleNamespace()
    sys.modules["torch._dynamo"] = stub


try:  # normal ortam: gercek modul calisiyor, dokunma
    import torch._dynamo  # noqa: F401
except Exception:  # TRUBA: dairesel import -> stub devreye girer
    _install_stub()
