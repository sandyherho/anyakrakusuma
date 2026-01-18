"""
anyakrakusuma: Schrödinger Bridge Solver via Entropic Optimal Transport

Mathematical Formulation
------------------------
The Schrödinger Bridge Problem (SBP) seeks the most likely evolution of a 
diffusion process X_t constrained to have marginals ρ₀ at t=0 and ρ₁ at t=T.

The optimal coupling has the form π*ᵢⱼ = uᵢ Kᵢⱼ vⱼ where:
    - K = exp(-C/ε) is the Gibbs kernel
    - u, v are the Sinkhorn potentials found by iterative scaling

Log-Domain Stability
--------------------
We work in log-domain: f = ε log(u), g = ε log(v) to prevent over/underflow.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
import warnings
import os

try:
    from numba import njit, prange, set_num_threads, float64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    float64 = None
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    prange = range
    def set_num_threads(n):
        pass

warnings.filterwarnings('ignore')


# =============================================================================
# NUMERICAL PRIMITIVES (Log-Domain Stable)
# =============================================================================

if NUMBA_AVAILABLE:
    @njit(float64(float64[:]), cache=True, fastmath=True)
    def logsumexp_1d(x):
        """
        Numerically stable log-sum-exp: log(Σᵢ exp(xᵢ))
        Uses the max-trick: log(Σ exp(xᵢ)) = m + log(Σ exp(xᵢ - m))
        """
        m = np.max(x)
        if np.isinf(m):
            return m
        s = 0.0
        for i in range(len(x)):
            s += np.exp(x[i] - m)
        return m + np.log(s)
else:
    def logsumexp_1d(x):
        m = np.max(x)
        if np.isinf(m):
            return m
        return m + np.log(np.sum(np.exp(x - m)))


@njit(parallel=True, cache=True, fastmath=True)
def compute_cost_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute pairwise squared Euclidean cost matrix C[i,j] = ||xᵢ - yⱼ||²
    
    Parameters
    ----------
    X : (n, d) source points
    Y : (m, d) target points  
    
    Returns
    -------
    C : (n, m) cost matrix
    """
    n, d = X.shape
    m = Y.shape[0]
    C = np.empty((n, m), dtype=np.float64)
    
    for i in prange(n):
        for j in range(m):
            dist_sq = 0.0
            for k in range(d):
                diff = X[i, k] - Y[j, k]
                dist_sq += diff * diff
            C[i, j] = dist_sq
    return C


@njit(parallel=True, cache=True, fastmath=True)
def sinkhorn_log_iteration(
    f: np.ndarray, 
    g: np.ndarray, 
    C: np.ndarray, 
    log_a: np.ndarray, 
    log_b: np.ndarray, 
    epsilon: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single Sinkhorn iteration in log-domain for numerical stability.
    
    Updates:
        f ← ε log(a) - ε·LSE_j((g - C)/ε)
        g ← ε log(b) - ε·LSE_i((f - Cᵀ)/ε)
    """
    n, m = C.shape
    f_new = np.empty(n, dtype=np.float64)
    g_new = np.empty(m, dtype=np.float64)
    inv_eps = 1.0 / epsilon
    
    # Update f
    for i in prange(n):
        tmp = np.empty(m, dtype=np.float64)
        for j in range(m):
            tmp[j] = (g[j] - C[i, j]) * inv_eps
        f_new[i] = epsilon * log_a[i] - epsilon * logsumexp_1d(tmp)
    
    # Update g
    for j in prange(m):
        tmp = np.empty(n, dtype=np.float64)
        for i in range(n):
            tmp[i] = (f_new[i] - C[i, j]) * inv_eps
        g_new[j] = epsilon * log_b[j] - epsilon * logsumexp_1d(tmp)
    
    return f_new, g_new


@njit(cache=True, fastmath=True)
def compute_marginal_error(
    f: np.ndarray, 
    g: np.ndarray, 
    C: np.ndarray, 
    log_a: np.ndarray, 
    epsilon: float
) -> float:
    """
    Compute L¹ marginal constraint violation: ||π1 - a||₁
    """
    n, m = C.shape
    inv_eps = 1.0 / epsilon
    error = 0.0
    
    for i in range(n):
        tmp = np.empty(m, dtype=np.float64)
        for j in range(m):
            tmp[j] = (g[j] - C[i, j]) * inv_eps
        log_marginal = f[i] * inv_eps + logsumexp_1d(tmp)
        error += np.abs(np.exp(log_marginal) - np.exp(log_a[i]))
    
    return error


@njit(parallel=True, cache=True, fastmath=True)
def compute_transport_plan(
    f: np.ndarray, 
    g: np.ndarray, 
    C: np.ndarray, 
    epsilon: float
) -> np.ndarray:
    """
    Compute optimal transport plan: πᵢⱼ = exp((fᵢ + gⱼ - Cᵢⱼ)/ε)
    """
    n, m = C.shape
    inv_eps = 1.0 / epsilon
    pi = np.empty((n, m), dtype=np.float64)
    
    for i in prange(n):
        for j in range(m):
            pi[i, j] = np.exp((f[i] + g[j] - C[i, j]) * inv_eps)
    
    return pi


@njit(parallel=True, cache=True, fastmath=True)
def sample_bridge_interpolation(
    X: np.ndarray,
    Y: np.ndarray,
    pi: np.ndarray,
    t: float,
    epsilon: float,
    seed: int = 42
) -> np.ndarray:
    """
    Sample from Schrödinger bridge marginal ρₜ at time t ∈ [0, 1].
    
    The entropic displacement interpolation:
        Xₜ = (1-t)X₀ + t·X₁ + √(ε·t·(1-t))·Z
    
    where (X₀, X₁) ~ π* and Z ~ N(0, I).
    """
    np.random.seed(seed)
    n, d = X.shape
    m = Y.shape[0]
    Xt = np.empty((n, d), dtype=np.float64)
    
    # Diffusion coefficient: σ(t) = √(ε·t·(1-t))
    sigma_t = np.sqrt(epsilon * t * (1.0 - t))
    
    for i in prange(n):
        # Normalize row to get conditional distribution P(Y|X=xᵢ)
        row_sum = 0.0
        for j in range(m):
            row_sum += pi[i, j]
        
        if row_sum < 1e-300:
            for k in range(d):
                Xt[i, k] = X[i, k]
            continue
        
        # Sample target index from conditional
        u = np.random.random()
        cumsum = 0.0
        j_sample = m - 1
        for j in range(m):
            cumsum += pi[i, j] / row_sum
            if u <= cumsum:
                j_sample = j
                break
        
        # Entropic interpolation with Brownian bridge correction
        for k in range(d):
            mean_t = (1.0 - t) * X[i, k] + t * Y[j_sample, k]
            noise = np.random.randn() * sigma_t
            Xt[i, k] = mean_t + noise
    
    return Xt


# =============================================================================
# SCHRÖDINGER BRIDGE SOLVER CLASS
# =============================================================================

class SchrodingerBridgeSolver:
    """
    High-performance Schrödinger Bridge solver via Entropic Optimal Transport.
    
    Solves the entropic optimal transport problem:
        min_{π∈Π(μ,ν)} ⟨C, π⟩ + εH(π|μ⊗ν)
    
    using the log-domain Sinkhorn-Knopp algorithm for numerical stability.
    
    Key features:
    - Log-domain stability for small ε
    - Numba JIT acceleration with parallel processing
    - Comprehensive diagnostic metrics
    """
    
    def __init__(self, n_cores: Optional[int] = None, verbose: bool = True,
                 logger: Optional[Any] = None):
        """
        Initialize Schrödinger Bridge solver.
        
        Args:
            n_cores: Number of CPU cores (None = all available)
            verbose: Print progress messages
            logger: Optional logger instance
        """
        self.verbose = verbose
        self.logger = logger
        
        # Setup parallel processing
        if n_cores is None or n_cores == 0:
            n_cores = os.cpu_count()
        self.n_cores = n_cores
        
        if NUMBA_AVAILABLE:
            set_num_threads(self.n_cores)
        
        if verbose:
            print(f"  CPU cores: {self.n_cores}")
            print(f"  Numba: {'ENABLED' if NUMBA_AVAILABLE else 'DISABLED'}")
    
    def solve(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        a: np.ndarray = None,
        b: np.ndarray = None,
        epsilon: float = 0.01,
        max_iter: int = 2000,
        tol: float = 1e-9
    ) -> Dict[str, Any]:
        """
        Solve entropic optimal transport via log-domain Sinkhorn algorithm.
        
        Parameters
        ----------
        X : (n, d) array - source point cloud
        Y : (m, d) array - target point cloud
        a : (n,) array - source weights (uniform if None)
        b : (m,) array - target weights (uniform if None)
        epsilon : float - entropic regularization (diffusivity)
        max_iter : int - maximum Sinkhorn iterations
        tol : float - convergence tolerance on marginal error
        
        Returns
        -------
        result : dict with keys:
            'f', 'g' : optimal dual potentials
            'pi' : optimal transport plan
            'cost' : transport cost ⟨C, π⟩
            'converged' : bool
            'n_iter' : iterations used
            'marginal_errors' : convergence history
            'metrics' : diagnostic metrics dictionary
        """
        n, d = X.shape
        m = Y.shape[0]
        
        if self.verbose:
            print(f"\n  Solving Schrödinger Bridge...")
            print(f"    Source points: {n}")
            print(f"    Target points: {m}")
            print(f"    Dimension: {d}")
            print(f"    ε = {epsilon} (entropic regularization)")
        
        # Default uniform marginals
        if a is None:
            a = np.ones(n, dtype=np.float64) / n
        if b is None:
            b = np.ones(m, dtype=np.float64) / m
        
        # Ensure normalization
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        a = a / a.sum()
        b = b / b.sum()
        
        # Precompute
        log_a = np.log(a + 1e-300)
        log_b = np.log(b + 1e-300)
        
        if self.verbose:
            print(f"    Computing cost matrix...")
        C = compute_cost_matrix(X, Y)
        
        # Initialize potentials
        f = np.zeros(n, dtype=np.float64)
        g = np.zeros(m, dtype=np.float64)
        
        # Sinkhorn iterations
        marginal_errors = []
        converged = False
        
        if self.verbose:
            pbar = tqdm(
                total=max_iter,
                desc="    Sinkhorn",
                unit="iter",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]'
            )
        
        for iteration in range(max_iter):
            f, g = sinkhorn_log_iteration(f, g, C, log_a, log_b, epsilon)
            
            # Check convergence every 10 iterations
            if iteration % 10 == 0:
                error = compute_marginal_error(f, g, C, log_a, epsilon)
                marginal_errors.append(error)
                
                if self.verbose:
                    pbar.set_postfix({'error': f'{error:.2e}'})
                
                if error < tol:
                    converged = True
                    if self.verbose:
                        pbar.update(max_iter - iteration)
                    break
            
            if self.verbose:
                pbar.update(1)
        
        if self.verbose:
            pbar.close()
            if converged:
                print(f"    ✓ Converged at iteration {iteration}")
            else:
                print(f"    ⚠ Did not converge in {max_iter} iterations")
        
        # Compute final transport plan
        pi = compute_transport_plan(f, g, C, epsilon)
        
        # Transport cost
        cost = np.sum(pi * C)
        
        # Compute diagnostic metrics
        metrics = self._compute_metrics(pi, C, epsilon, marginal_errors, iteration + 1, tol)
        
        if self.verbose:
            print(f"\n    Results:")
            print(f"      Transport cost: {cost:.6f}")
            print(f"      Plan entropy: {metrics['plan_entropy']:.4f}")
            print(f"      Effective sparsity: {metrics['effective_sparsity']:.4f}")
            print(f"      Marginal fidelity: {metrics['marginal_fidelity']:.6f}")
        
        return {
            'X': X,
            'Y': Y,
            'f': f,
            'g': g,
            'pi': pi,
            'C': C,
            'cost': cost,
            'converged': converged,
            'n_iter': iteration + 1,
            'marginal_errors': np.array(marginal_errors),
            'epsilon': epsilon,
            'metrics': metrics,
            'params': {
                'epsilon': epsilon,
                'max_iter': max_iter,
                'tol': tol,
                'n_source': n,
                'n_target': m,
                'dimension': d,
                'n_cores': self.n_cores,
                'numba_enabled': NUMBA_AVAILABLE
            }
        }
    
    def _compute_metrics(
        self, 
        pi: np.ndarray, 
        C: np.ndarray, 
        epsilon: float,
        marginal_errors: list,
        n_iter: int,
        tol: float
    ) -> Dict[str, float]:
        """
        Compute comprehensive diagnostic metrics.
        
        Metrics:
        - plan_entropy: H(π) = -Σᵢⱼ πᵢⱼ log(πᵢⱼ)
        - effective_sparsity: exp(H) / (n×m) - normalized effective support
        - wasserstein_cost: ⟨C, π⟩
        - marginal_fidelity: 1 - ||π1 - a||₁
        - bridge_diffusivity: ε × E[t(1-t)] = ε/6 for uniform t
        - iteration_efficiency: -log(tol) / n_iter
        """
        n, m = pi.shape
        
        # Plan entropy (avoid log(0))
        pi_flat = pi.flatten()
        pi_nonzero = pi_flat[pi_flat > 1e-300]
        plan_entropy = -np.sum(pi_nonzero * np.log(pi_nonzero))
        
        # Effective sparsity (normalized)
        effective_support = np.exp(plan_entropy)
        effective_sparsity = effective_support / (n * m)
        
        # Wasserstein cost
        wasserstein_cost = np.sum(pi * C)
        
        # Marginal fidelity
        final_error = marginal_errors[-1] if marginal_errors else 1.0
        marginal_fidelity = 1.0 - final_error
        
        # Bridge diffusivity (theoretical average)
        # For t ~ U[0,1], E[t(1-t)] = 1/6
        bridge_diffusivity = epsilon / 6.0
        
        # Iteration efficiency
        iteration_efficiency = -np.log10(tol) / n_iter if n_iter > 0 else 0.0
        
        # Row/column marginal errors
        row_sums = pi.sum(axis=1)
        col_sums = pi.sum(axis=0)
        a_uniform = np.ones(n) / n
        b_uniform = np.ones(m) / m
        row_error = np.abs(row_sums - a_uniform).max()
        col_error = np.abs(col_sums - b_uniform).max()
        
        return {
            'plan_entropy': plan_entropy,
            'effective_sparsity': effective_sparsity,
            'wasserstein_cost': wasserstein_cost,
            'marginal_fidelity': marginal_fidelity,
            'bridge_diffusivity': bridge_diffusivity,
            'iteration_efficiency': iteration_efficiency,
            'row_marginal_error': row_error,
            'col_marginal_error': col_error,
            'plan_mass': pi.sum()
        }
    
    def generate_trajectory(
        self,
        result: Dict[str, Any],
        n_frames: int = 100,
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate full Schrödinger bridge trajectory.
        
        Parameters
        ----------
        result : output from solve()
        n_frames : number of time steps
        seed : random seed for reproducibility
        
        Returns
        -------
        trajectory : (n_frames, n, d) array of interpolated point clouds
        times : (n_frames,) array of time values
        """
        X = result['X']
        Y = result['Y']
        pi = result['pi']
        epsilon = result['epsilon']
        
        n, d = X.shape
        trajectory = np.empty((n_frames, n, d), dtype=np.float64)
        times = np.linspace(0, 1, n_frames)
        
        if self.verbose:
            print(f"\n  Generating bridge trajectory ({n_frames} frames)...")
        
        for frame, t in enumerate(times):
            if t == 0:
                trajectory[frame] = X.copy()
            elif t == 1:
                trajectory[frame] = Y.copy()
            else:
                trajectory[frame] = sample_bridge_interpolation(
                    X, Y, pi, t, epsilon, seed=seed + frame
                )
        
        if self.verbose:
            print(f"    ✓ Trajectory generated: shape {trajectory.shape}")
        
        return trajectory, times
