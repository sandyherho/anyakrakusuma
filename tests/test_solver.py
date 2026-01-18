"""
Unit tests for Schrödinger Bridge solver.

Tests cover:
- Solver initialization
- Cost matrix computation
- Sinkhorn convergence
- Marginal constraints
- Transport plan properties
- Trajectory generation
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from anyakrakusuma_eot.core.solver import (
    SchrodingerBridgeSolver,
    compute_cost_matrix,
    compute_transport_plan,
    logsumexp_1d
)
from anyakrakusuma_eot.core.distributions import generate_circle, generate_spiral


class TestNumericalPrimitives:
    """Test low-level numerical functions."""
    
    def test_logsumexp_stability(self):
        """Test log-sum-exp numerical stability with large values."""
        # Large positive values
        x = np.array([1000.0, 1001.0, 1002.0])
        result = logsumexp_1d(x)
        expected = 1002.0 + np.log(np.exp(-2) + np.exp(-1) + 1)
        assert np.isclose(result, expected, rtol=1e-10)
    
    def test_logsumexp_small_values(self):
        """Test log-sum-exp with small values."""
        x = np.array([-1000.0, -999.0, -998.0])
        result = logsumexp_1d(x)
        expected = -998.0 + np.log(np.exp(-2) + np.exp(-1) + 1)
        assert np.isclose(result, expected, rtol=1e-10)
    
    def test_cost_matrix_shape(self):
        """Test cost matrix has correct shape."""
        n, m, d = 50, 60, 2
        X = np.random.randn(n, d)
        Y = np.random.randn(m, d)
        C = compute_cost_matrix(X, Y)
        assert C.shape == (n, m)
    
    def test_cost_matrix_symmetry(self):
        """Test cost matrix symmetry when X=Y."""
        n, d = 30, 2
        X = np.random.randn(n, d)
        C = compute_cost_matrix(X, X)
        assert np.allclose(C, C.T, rtol=1e-10)
    
    def test_cost_matrix_diagonal(self):
        """Test cost matrix diagonal is zero when X=Y."""
        n, d = 30, 2
        X = np.random.randn(n, d)
        C = compute_cost_matrix(X, X)
        assert np.allclose(np.diag(C), 0, atol=1e-10)
    
    def test_cost_matrix_nonnegative(self):
        """Test cost matrix is non-negative."""
        n, m, d = 50, 60, 2
        X = np.random.randn(n, d)
        Y = np.random.randn(m, d)
        C = compute_cost_matrix(X, Y)
        assert np.all(C >= 0)


class TestSchrodingerBridgeSolver:
    """Test the main solver class."""
    
    @pytest.fixture
    def solver(self):
        """Create solver instance for testing."""
        return SchrodingerBridgeSolver(n_cores=2, verbose=False)
    
    @pytest.fixture
    def simple_data(self):
        """Create simple test data."""
        np.random.seed(42)
        n = 100
        X = generate_circle(n, radius=1.0, seed=42)
        Y = generate_circle(n, radius=1.5, seed=43)
        return X, Y
    
    def test_solver_initialization(self, solver):
        """Test solver initializes correctly."""
        assert solver.n_cores >= 1
        assert solver.verbose == False
    
    def test_solve_returns_dict(self, solver, simple_data):
        """Test solve returns dictionary with expected keys."""
        X, Y = simple_data
        result = solver.solve(X, Y, epsilon=0.1, max_iter=100, tol=1e-6)
        
        required_keys = ['X', 'Y', 'f', 'g', 'pi', 'C', 'cost', 
                        'converged', 'n_iter', 'marginal_errors', 
                        'epsilon', 'metrics', 'params']
        
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_transport_plan_shape(self, solver, simple_data):
        """Test transport plan has correct shape."""
        X, Y = simple_data
        result = solver.solve(X, Y, epsilon=0.1, max_iter=100, tol=1e-6)
        
        n = X.shape[0]
        m = Y.shape[0]
        assert result['pi'].shape == (n, m)
    
    def test_transport_plan_nonnegative(self, solver, simple_data):
        """Test transport plan is non-negative."""
        X, Y = simple_data
        result = solver.solve(X, Y, epsilon=0.1, max_iter=100, tol=1e-6)
        
        assert np.all(result['pi'] >= 0)
    
    def test_transport_plan_mass(self, solver, simple_data):
        """Test transport plan has unit mass."""
        X, Y = simple_data
        result = solver.solve(X, Y, epsilon=0.1, max_iter=100, tol=1e-6)
        
        assert np.isclose(result['pi'].sum(), 1.0, rtol=1e-6)
    
    def test_row_marginal_constraint(self, solver, simple_data):
        """Test row marginals sum to source distribution."""
        X, Y = simple_data
        result = solver.solve(X, Y, epsilon=0.1, max_iter=500, tol=1e-8)
        
        n = X.shape[0]
        a_expected = np.ones(n) / n
        row_sums = result['pi'].sum(axis=1)
        
        assert np.allclose(row_sums, a_expected, rtol=1e-4)
    
    def test_col_marginal_constraint(self, solver, simple_data):
        """Test column marginals sum to target distribution."""
        X, Y = simple_data
        result = solver.solve(X, Y, epsilon=0.1, max_iter=500, tol=1e-8)
        
        m = Y.shape[0]
        b_expected = np.ones(m) / m
        col_sums = result['pi'].sum(axis=0)
        
        assert np.allclose(col_sums, b_expected, rtol=1e-4)
    
    def test_convergence_with_sufficient_iterations(self, solver, simple_data):
        """Test solver converges with sufficient iterations."""
        X, Y = simple_data
        result = solver.solve(X, Y, epsilon=0.1, max_iter=2000, tol=1e-8)
        
        assert result['converged'] == True
    
    def test_cost_positive(self, solver, simple_data):
        """Test transport cost is positive."""
        X, Y = simple_data
        result = solver.solve(X, Y, epsilon=0.1, max_iter=100, tol=1e-6)
        
        assert result['cost'] > 0
    
    def test_metrics_computed(self, solver, simple_data):
        """Test diagnostic metrics are computed."""
        X, Y = simple_data
        result = solver.solve(X, Y, epsilon=0.1, max_iter=100, tol=1e-6)
        
        metrics = result['metrics']
        required_metrics = ['plan_entropy', 'effective_sparsity', 
                          'wasserstein_cost', 'marginal_fidelity']
        
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert not np.isnan(metrics[metric]), f"NaN metric: {metric}"
    
    def test_plan_entropy_positive(self, solver, simple_data):
        """Test plan entropy is positive."""
        X, Y = simple_data
        result = solver.solve(X, Y, epsilon=0.1, max_iter=100, tol=1e-6)
        
        assert result['metrics']['plan_entropy'] > 0
    
    def test_effective_sparsity_bounded(self, solver, simple_data):
        """Test effective sparsity is in [0, 1]."""
        X, Y = simple_data
        result = solver.solve(X, Y, epsilon=0.1, max_iter=100, tol=1e-6)
        
        sparsity = result['metrics']['effective_sparsity']
        assert 0 <= sparsity <= 1


class TestTrajectoryGeneration:
    """Test bridge trajectory generation."""
    
    @pytest.fixture
    def solved_problem(self):
        """Create a solved problem for trajectory testing."""
        np.random.seed(42)
        n = 50
        X = generate_circle(n, radius=1.0, seed=42)
        Y = generate_circle(n, radius=2.0, seed=43)
        
        solver = SchrodingerBridgeSolver(n_cores=2, verbose=False)
        result = solver.solve(X, Y, epsilon=0.1, max_iter=500, tol=1e-8)
        
        return solver, result
    
    def test_trajectory_shape(self, solved_problem):
        """Test trajectory has correct shape."""
        solver, result = solved_problem
        n_frames = 20
        
        trajectory, times = solver.generate_trajectory(result, n_frames=n_frames)
        
        n = result['X'].shape[0]
        d = result['X'].shape[1]
        
        assert trajectory.shape == (n_frames, n, d)
        assert times.shape == (n_frames,)
    
    def test_trajectory_boundary_source(self, solved_problem):
        """Test trajectory starts at source distribution."""
        solver, result = solved_problem
        
        trajectory, times = solver.generate_trajectory(result, n_frames=20)
        
        assert np.allclose(trajectory[0], result['X'])
    
    def test_trajectory_boundary_target(self, solved_problem):
        """Test trajectory ends at target distribution."""
        solver, result = solved_problem
        
        trajectory, times = solver.generate_trajectory(result, n_frames=20)
        
        assert np.allclose(trajectory[-1], result['Y'])
    
    def test_times_monotonic(self, solved_problem):
        """Test time values are monotonically increasing."""
        solver, result = solved_problem
        
        trajectory, times = solver.generate_trajectory(result, n_frames=20)
        
        assert np.all(np.diff(times) > 0)
    
    def test_times_boundary(self, solved_problem):
        """Test time values span [0, 1]."""
        solver, result = solved_problem
        
        trajectory, times = solver.generate_trajectory(result, n_frames=20)
        
        assert times[0] == 0.0
        assert times[-1] == 1.0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_identical_distributions(self):
        """Test solver handles identical source and target."""
        np.random.seed(42)
        n = 50
        X = generate_circle(n, radius=1.0, seed=42)
        Y = X.copy()  # Identical
        
        solver = SchrodingerBridgeSolver(n_cores=2, verbose=False)
        result = solver.solve(X, Y, epsilon=0.1, max_iter=100, tol=1e-6)
        
        # Cost should be relatively small for identical distributions
        # Note: Entropic OT has non-zero cost due to ε·H(π) regularization term
        # The cost scales with ε, so we use a more relaxed threshold
        assert result['cost'] < 0.1
        
        # More importantly, the transport plan should be close to identity
        # (diagonal-dominant for identical distributions)
        pi = result['pi']
        diag_mass = np.trace(pi * n)  # Sum of diagonal * n (since uniform weights = 1/n)
        assert diag_mass > 0.5, "Transport plan should favor diagonal for identical distributions"
    
    def test_small_epsilon(self):
        """Test solver handles small epsilon (sharper transport)."""
        np.random.seed(42)
        n = 50
        X = generate_circle(n, radius=1.0, seed=42)
        Y = generate_circle(n, radius=1.5, seed=43)
        
        solver = SchrodingerBridgeSolver(n_cores=2, verbose=False)
        result = solver.solve(X, Y, epsilon=0.01, max_iter=2000, tol=1e-8)
        
        # Should still produce valid transport plan
        assert result['pi'].sum() > 0.99
        assert np.all(result['pi'] >= 0)
    
    def test_large_epsilon(self):
        """Test solver handles large epsilon (diffuse transport)."""
        np.random.seed(42)
        n = 50
        X = generate_circle(n, radius=1.0, seed=42)
        Y = generate_circle(n, radius=1.5, seed=43)
        
        solver = SchrodingerBridgeSolver(n_cores=2, verbose=False)
        result = solver.solve(X, Y, epsilon=1.0, max_iter=100, tol=1e-6)
        
        # Large epsilon should give more uniform transport plan
        assert result['metrics']['effective_sparsity'] > 0.1
    
    def test_different_sizes(self):
        """Test solver handles different source/target sizes."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        Y = np.random.randn(60, 2)
        
        solver = SchrodingerBridgeSolver(n_cores=2, verbose=False)
        result = solver.solve(X, Y, epsilon=0.1, max_iter=100, tol=1e-6)
        
        assert result['pi'].shape == (50, 60)
        assert np.isclose(result['pi'].sum(), 1.0, rtol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
