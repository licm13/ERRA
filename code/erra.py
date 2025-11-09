"""Legacy entry point that re-exports the authoritative ERRA implementation.

旧版入口模块，直接转发至权威的 :mod:`src.erra.erra_core` 实现。
"""

from __future__ import annotations

from erra.erra_core import ERRAResult, erra

__all__ = ["ERRAResult", "erra"]
