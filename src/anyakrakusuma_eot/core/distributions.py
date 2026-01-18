"""
Point cloud generators for Schrödinger Bridge experiments.

Provides various 2D distribution generators:
- Circle: Points on a circle
- Spiral: Archimedean spiral
- Two Moons: Interleaving crescents
- Gaussian Mixture: Multi-modal clusters
- Lissajous: Parametric curve
- Trefoil: Knot projection
"""

import numpy as np
from typing import List, Optional


def generate_circle(
    n: int, 
    radius: float = 1.0, 
    noise: float = 0.0,
    center: tuple = (0.0, 0.0),
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate points uniformly distributed on a circle.
    
    Parameters
    ----------
    n : int - number of points
    radius : float - circle radius
    noise : float - Gaussian noise std
    center : tuple - center coordinates
    seed : int - random seed
    
    Returns
    -------
    X : (n, 2) array of points
    """
    if seed is not None:
        np.random.seed(seed)
    
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    theta += np.random.uniform(0, 2 * np.pi / n)  # Random phase
    
    X = np.column_stack([
        center[0] + radius * np.cos(theta), 
        center[1] + radius * np.sin(theta)
    ])
    
    if noise > 0:
        X += np.random.randn(n, 2) * noise
    
    return X.astype(np.float64)


def generate_spiral(
    n: int, 
    turns: float = 2.0, 
    noise: float = 0.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate Archimedean spiral.
    
    Parameters
    ----------
    n : int - number of points
    turns : float - number of spiral turns
    noise : float - Gaussian noise std
    seed : int - random seed
    
    Returns
    -------
    X : (n, 2) array of points
    """
    if seed is not None:
        np.random.seed(seed)
    
    theta = np.linspace(0.5, turns * 2 * np.pi, n)
    r = theta / (turns * 2 * np.pi) * 1.5
    
    X = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    if noise > 0:
        X += np.random.randn(n, 2) * noise
    
    return X.astype(np.float64)


def generate_two_moons(
    n: int, 
    noise: float = 0.05,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate two interleaving half-circles (moons).
    
    Parameters
    ----------
    n : int - number of points
    noise : float - Gaussian noise std
    seed : int - random seed
    
    Returns
    -------
    X : (n, 2) array of points
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_half = n // 2
    theta1 = np.linspace(0, np.pi, n_half)
    theta2 = np.linspace(0, np.pi, n - n_half)
    
    moon1 = np.column_stack([np.cos(theta1), np.sin(theta1)])
    moon2 = np.column_stack([1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5])
    
    X = np.vstack([moon1, moon2])
    X += np.random.randn(n, 2) * noise
    
    return X.astype(np.float64)


def generate_two_moons_rotated(
    n: int, 
    noise: float = 0.05,
    rotation_angle: float = 90.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate two moons with rotation.
    
    Parameters
    ----------
    n : int - number of points
    noise : float - Gaussian noise std
    rotation_angle : float - rotation in degrees
    seed : int - random seed
    
    Returns
    -------
    X : (n, 2) array of points
    """
    X = generate_two_moons(n, noise, seed)
    
    # Rotation matrix
    theta = np.radians(rotation_angle)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # Center, rotate, uncenter
    center = X.mean(axis=0)
    X_centered = X - center
    X_rotated = X_centered @ R.T
    
    return X_rotated.astype(np.float64)


def generate_gaussian_mixture(
    n: int, 
    n_clusters: int = 4,
    std: float = 0.15,
    spread: float = 1.5,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate Gaussian mixture.
    
    Parameters
    ----------
    n : int - total number of points
    n_clusters : int - number of Gaussian components
    std : float - per-cluster standard deviation
    spread : float - cluster center spread
    seed : int - random seed
    
    Returns
    -------
    X : (n, 2) array of points
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate cluster centers evenly on a circle
    angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    centers = np.column_stack([
        spread * np.cos(angles),
        spread * np.sin(angles)
    ])
    
    n_per_cluster = n // n_clusters
    X = []
    
    for i, c in enumerate(centers):
        n_i = n_per_cluster if i < n_clusters - 1 else n - len(np.vstack(X) if X else [])
        X.append(c + np.random.randn(n_i, 2) * std)
    
    return np.vstack(X).astype(np.float64)


def generate_lissajous(
    n: int, 
    a: int = 3, 
    b: int = 2, 
    delta: float = np.pi / 2,
    scale: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate Lissajous curve.
    
    Parametric form: (sin(a·t + δ), sin(b·t))
    
    Parameters
    ----------
    n : int - number of points
    a, b : int - frequency ratio
    delta : float - phase shift
    scale : float - scaling factor
    seed : int - random seed
    
    Returns
    -------
    X : (n, 2) array of points
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X = np.column_stack([
        scale * np.sin(a * t + delta), 
        scale * np.sin(b * t)
    ])
    
    return X.astype(np.float64)


def generate_trefoil(
    n: int,
    scale: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate 2D projection of trefoil knot.
    
    Parametric form:
        x = sin(t) + 2·sin(2t)
        y = cos(t) - 2·cos(2t)
    
    Parameters
    ----------
    n : int - number of points
    scale : float - scaling factor
    seed : int - random seed
    
    Returns
    -------
    X : (n, 2) array of points
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    
    X = np.column_stack([x, y]) / 3.0 * scale
    
    return X.astype(np.float64)


def generate_custom_distribution(
    n: int,
    distribution_type: str,
    **kwargs
) -> np.ndarray:
    """
    Factory function to generate distributions by name.
    
    Parameters
    ----------
    n : int - number of points
    distribution_type : str - one of: circle, spiral, two_moons, 
                                      two_moons_rotated, gaussian_mixture,
                                      lissajous, trefoil
    **kwargs : additional arguments for specific generator
    
    Returns
    -------
    X : (n, 2) array of points
    """
    generators = {
        'circle': generate_circle,
        'spiral': generate_spiral,
        'two_moons': generate_two_moons,
        'two_moons_rotated': generate_two_moons_rotated,
        'gaussian_mixture': generate_gaussian_mixture,
        'lissajous': generate_lissajous,
        'trefoil': generate_trefoil
    }
    
    if distribution_type not in generators:
        raise ValueError(f"Unknown distribution type: {distribution_type}. "
                        f"Available: {list(generators.keys())}")
    
    return generators[distribution_type](n, **kwargs)
