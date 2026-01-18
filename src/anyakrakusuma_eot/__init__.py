"""
anyakrakusuma: 2D Schrödinger Bridge Solver via Entropic Optimal Transport

A high-performance Python solver for the Schrödinger Bridge Problem using
log-domain Sinkhorn-Knopp algorithm with Numba acceleration.
"""

__version__ = "0.0.1"
__author__ = "Sandy H. S. Herho, Iwan P. Anwar, Faruq Khadami, Rusmawan Suwarman, Dasapta E. Irawan"
__license__ = "MIT"

from .core.solver import SchrodingerBridgeSolver
from .core.distributions import (
    generate_circle,
    generate_spiral,
    generate_two_moons,
    generate_gaussian_mixture,
    generate_lissajous,
    generate_trefoil
)
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler

__all__ = [
    "SchrodingerBridgeSolver",
    "generate_circle",
    "generate_spiral",
    "generate_two_moons",
    "generate_gaussian_mixture",
    "generate_lissajous",
    "generate_trefoil",
    "ConfigManager",
    "DataHandler"
]
