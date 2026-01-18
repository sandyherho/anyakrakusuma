"""Data handler for Schrödinger Bridge simulation results."""

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class DataHandler:
    """NetCDF and CSV output handler for Schrödinger Bridge simulations."""
    
    @staticmethod
    def save_netcdf(
        filename: str, 
        result: Dict[str, Any], 
        trajectory: np.ndarray,
        times: np.ndarray,
        metadata: Dict[str, Any],
        output_dir: str = "outputs"
    ) -> str:
        """
        Save Schrödinger Bridge results to NetCDF file.
        
        Structure:
            dimensions: particle, time, dim
            variables:
                X_source(particle, dim): Source distribution
                X_target(particle, dim): Target distribution
                trajectory(time, particle, dim): Bridge trajectory
                time(time): Interpolation times
                transport_plan(particle, particle): Optimal coupling
                marginal_errors(convergence_step): Sinkhorn convergence
            
            attributes:
                Physical parameters (ε, cost)
                Diagnostic metrics
                Simulation metadata
        
        Parameters
        ----------
        filename : str - output filename
        result : dict - results from solver.solve()
        trajectory : (n_frames, n, d) - bridge trajectory
        times : (n_frames,) - time values
        metadata : dict - configuration dictionary
        output_dir : str - output directory
        
        Returns
        -------
        filepath : str - full path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        with Dataset(filepath, 'w', format='NETCDF4') as nc:
            
            # Get dimensions
            n_particles = result['X'].shape[0]
            n_dim = result['X'].shape[1]
            n_frames = trajectory.shape[0]
            n_convergence = len(result['marginal_errors'])
            
            # Create dimensions
            nc.createDimension('particle', n_particles)
            nc.createDimension('dim', n_dim)
            nc.createDimension('time', n_frames)
            nc.createDimension('convergence_step', n_convergence)
            
            # Coordinates
            nc_time = nc.createVariable('time', 'f4', ('time',), zlib=True, complevel=4)
            nc_time[:] = times
            nc_time.units = "dimensionless"
            nc_time.long_name = "interpolation_time"
            nc_time.description = "Bridge interpolation time t ∈ [0, 1]"
            
            # Source distribution
            nc_source = nc.createVariable('X_source', 'f4', ('particle', 'dim'), 
                                         zlib=True, complevel=4)
            nc_source[:] = result['X']
            nc_source.units = "dimensionless"
            nc_source.long_name = "source_distribution"
            nc_source.description = "Source point cloud μ"
            
            # Target distribution
            nc_target = nc.createVariable('X_target', 'f4', ('particle', 'dim'), 
                                         zlib=True, complevel=4)
            nc_target[:] = result['Y']
            nc_target.units = "dimensionless"
            nc_target.long_name = "target_distribution"
            nc_target.description = "Target point cloud ν"
            
            # Bridge trajectory
            nc_traj = nc.createVariable('trajectory', 'f4', ('time', 'particle', 'dim'), 
                                       zlib=True, complevel=5)
            nc_traj[:] = trajectory
            nc_traj.units = "dimensionless"
            nc_traj.long_name = "bridge_trajectory"
            nc_traj.description = "Schrödinger bridge displacement interpolation Xₜ"
            
            # Transport plan (subsampled if too large)
            max_plan_size = 500
            if n_particles <= max_plan_size:
                nc.createDimension('plan_i', n_particles)
                nc.createDimension('plan_j', n_particles)
                nc_plan = nc.createVariable('transport_plan', 'f4', ('plan_i', 'plan_j'), 
                                           zlib=True, complevel=5)
                nc_plan[:] = result['pi']
                nc_plan.description = "Optimal transport plan π*"
            else:
                # Subsample
                idx = np.linspace(0, n_particles-1, max_plan_size, dtype=int)
                nc.createDimension('plan_i', max_plan_size)
                nc.createDimension('plan_j', max_plan_size)
                nc_plan = nc.createVariable('transport_plan', 'f4', ('plan_i', 'plan_j'), 
                                           zlib=True, complevel=5)
                nc_plan[:] = result['pi'][np.ix_(idx, idx)]
                nc_plan.description = f"Optimal transport plan π* (subsampled to {max_plan_size})"
            
            # Convergence history
            nc_errors = nc.createVariable('marginal_errors', 'f4', ('convergence_step',), 
                                         zlib=True, complevel=4)
            nc_errors[:] = result['marginal_errors']
            nc_errors.units = "dimensionless"
            nc_errors.long_name = "sinkhorn_convergence"
            nc_errors.description = "Marginal constraint violation ||π1 - a||₁"
            
            # Physical parameters
            nc.epsilon = float(result['epsilon'])
            nc.epsilon_description = "Entropic regularization (diffusivity)"
            
            nc.transport_cost = float(result['cost'])
            nc.transport_cost_description = "Wasserstein cost ⟨C, π⟩"
            
            nc.converged = int(result['converged'])
            nc.n_iterations = int(result['n_iter'])
            
            # Diagnostic metrics
            metrics = result['metrics']
            nc.plan_entropy = float(metrics['plan_entropy'])
            nc.plan_entropy_description = "Plan entropy H(π)"
            
            nc.effective_sparsity = float(metrics['effective_sparsity'])
            nc.effective_sparsity_description = "Normalized effective support exp(H)/(n×m)"
            
            nc.marginal_fidelity = float(metrics['marginal_fidelity'])
            nc.marginal_fidelity_description = "Marginal constraint satisfaction"
            
            nc.bridge_diffusivity = float(metrics['bridge_diffusivity'])
            nc.bridge_diffusivity_description = "Average stochastic spread ε/6"
            
            nc.iteration_efficiency = float(metrics['iteration_efficiency'])
            nc.iteration_efficiency_description = "Convergence rate per iteration"
            
            # Simulation parameters
            params = result['params']
            nc.n_source = int(params['n_source'])
            nc.n_target = int(params['n_target'])
            nc.dimension = int(params['dimension'])
            nc.max_iter = int(params['max_iter'])
            nc.tolerance = float(params['tol'])
            nc.n_cores = int(params['n_cores'])
            nc.numba_enabled = int(params['numba_enabled'])
            
            # Scenario info
            nc.scenario = metadata.get('scenario_name', 'unknown')
            nc.source_type = metadata.get('source_type', 'unknown')
            nc.target_type = metadata.get('target_type', 'unknown')
            
            # Provenance
            nc.created = datetime.now().isoformat()
            nc.software = "anyakrakusuma"
            nc.version = "0.0.1"
            nc.method = "log_domain_sinkhorn"
            nc.method_description = "Entropic Optimal Transport via log-domain Sinkhorn-Knopp"
            
            nc.Conventions = "CF-1.8"
            nc.title = f"Schrödinger Bridge: {metadata.get('scenario_name', 'unknown')}"
            nc.institution = "FITB ITB"
            nc.license = "MIT"
            nc.history = f"Created {datetime.now().isoformat()}"
        
        return str(filepath)
    
    @staticmethod
    def save_csv(
        filename: str,
        result: Dict[str, Any],
        metadata: Dict[str, Any],
        output_dir: str = "outputs"
    ) -> str:
        """
        Save diagnostic metrics to CSV file.
        
        Parameters
        ----------
        filename : str - output filename
        result : dict - results from solver.solve()
        metadata : dict - configuration dictionary
        output_dir : str - output directory
        
        Returns
        -------
        filepath : str - full path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        metrics = result['metrics']
        params = result['params']
        
        # Compile all metrics into dataframe
        data = {
            'scenario': [metadata.get('scenario_name', 'unknown')],
            'source_type': [metadata.get('source_type', 'unknown')],
            'target_type': [metadata.get('target_type', 'unknown')],
            'n_particles': [params['n_source']],
            'epsilon': [result['epsilon']],
            'transport_cost': [result['cost']],
            'converged': [result['converged']],
            'n_iterations': [result['n_iter']],
            'plan_entropy': [metrics['plan_entropy']],
            'effective_sparsity': [metrics['effective_sparsity']],
            'marginal_fidelity': [metrics['marginal_fidelity']],
            'bridge_diffusivity': [metrics['bridge_diffusivity']],
            'iteration_efficiency': [metrics['iteration_efficiency']],
            'row_marginal_error': [metrics['row_marginal_error']],
            'col_marginal_error': [metrics['col_marginal_error']],
            'plan_mass': [metrics['plan_mass']]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        return str(filepath)
    
    @staticmethod
    def append_comparison_csv(
        filename: str,
        result: Dict[str, Any],
        metadata: Dict[str, Any],
        output_dir: str = "outputs"
    ) -> str:
        """
        Append metrics to comparison CSV (for multi-case analysis).
        
        Parameters
        ----------
        filename : str - output filename
        result : dict - results from solver.solve()
        metadata : dict - configuration dictionary
        output_dir : str - output directory
        
        Returns
        -------
        filepath : str - full path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        metrics = result['metrics']
        params = result['params']
        
        data = {
            'scenario': metadata.get('scenario_name', 'unknown'),
            'source_type': metadata.get('source_type', 'unknown'),
            'target_type': metadata.get('target_type', 'unknown'),
            'n_particles': params['n_source'],
            'epsilon': result['epsilon'],
            'transport_cost': result['cost'],
            'converged': result['converged'],
            'n_iterations': result['n_iter'],
            'plan_entropy': metrics['plan_entropy'],
            'effective_sparsity': metrics['effective_sparsity'],
            'marginal_fidelity': metrics['marginal_fidelity'],
            'iteration_efficiency': metrics['iteration_efficiency']
        }
        
        df_new = pd.DataFrame([data])
        
        if filepath.exists():
            df_existing = pd.read_csv(filepath)
            df = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df = df_new
        
        df.to_csv(filepath, index=False)
        
        return str(filepath)
