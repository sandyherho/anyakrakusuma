"""Simulation logger for Schrödinger Bridge solver."""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class SimulationLogger:
    """Enhanced logger for Schrödinger Bridge simulations."""
    
    def __init__(self, scenario_name: str, log_dir: str = "logs",
                 verbose: bool = True):
        """Initialize simulation logger."""
        self.scenario_name = scenario_name
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{scenario_name}.log"
        
        self.logger = self._setup_logger()
        self.warnings = []
        self.errors = []
    
    def _setup_logger(self):
        """Configure Python logging."""
        logger = logging.getLogger(f"eot_{self.scenario_name}")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        
        handler = logging.FileHandler(self.log_file, mode='w')
        handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        return logger
    
    def info(self, msg: str):
        """Log informational message."""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)
        self.warnings.append(msg)
        
        if self.verbose:
            print(f"  WARNING: {msg}")
    
    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)
        self.errors.append(msg)
        
        if self.verbose:
            print(f"  ERROR: {msg}")
    
    def log_parameters(self, params: Dict[str, Any]):
        """Log all simulation parameters."""
        self.info("=" * 60)
        self.info(f"PARAMETERS - {params.get('scenario_name', 'Unknown')}")
        self.info("=" * 60)
        
        for key, value in sorted(params.items()):
            self.info(f"  {key}: {value}")
        
        self.info("=" * 60)
    
    def log_timing(self, timing: Dict[str, float]):
        """Log timing breakdown."""
        self.info("=" * 60)
        self.info("TIMING BREAKDOWN")
        self.info("=" * 60)
        
        for key, value in sorted(timing.items()):
            self.info(f"  {key}: {value:.3f} s")
        
        self.info("=" * 60)
    
    def log_results(self, results: Dict[str, Any]):
        """Log simulation results with metrics."""
        self.info("=" * 60)
        self.info("SIMULATION RESULTS")
        self.info("=" * 60)
        
        params = results['params']
        metrics = results['metrics']
        
        self.info(f"  Physical parameters:")
        self.info(f"    ε = {results['epsilon']}")
        self.info(f"    n_particles = {params['n_source']}")
        
        self.info(f"  Transport results:")
        self.info(f"    Cost = {results['cost']:.6f}")
        self.info(f"    Converged = {results['converged']}")
        self.info(f"    Iterations = {results['n_iter']}")
        
        self.info(f"  Diagnostic metrics:")
        self.info(f"    Plan entropy = {metrics['plan_entropy']:.4f}")
        self.info(f"    Effective sparsity = {metrics['effective_sparsity']:.6f}")
        self.info(f"    Marginal fidelity = {metrics['marginal_fidelity']:.8f}")
        
        # Quality assessment
        if metrics['marginal_fidelity'] > 0.99999:
            quality = "EXCELLENT"
        elif metrics['marginal_fidelity'] > 0.9999:
            quality = "GOOD"
        elif metrics['marginal_fidelity'] > 0.999:
            quality = "ACCEPTABLE"
        else:
            quality = "POOR"
            self.warning(f"Low marginal fidelity: {metrics['marginal_fidelity']:.6f}")
        
        self.info(f"  Solution quality: {quality}")
        self.info("=" * 60)
    
    def finalize(self):
        """Write final summary."""
        self.info("=" * 60)
        self.info("SIMULATION SUMMARY")
        self.info("=" * 60)
        
        if self.errors:
            self.info(f"  ERRORS: {len(self.errors)}")
            for i, err in enumerate(self.errors, 1):
                self.info(f"    {i}. {err}")
        else:
            self.info("  ERRORS: None")
        
        if self.warnings:
            self.info(f"  WARNINGS: {len(self.warnings)}")
            for i, warn in enumerate(self.warnings, 1):
                self.info(f"    {i}. {warn}")
        else:
            self.info("  WARNINGS: None")
        
        self.info(f"  Log file: {self.log_file}")
        self.info("=" * 60)
        self.info(f"Simulation completed: {self.scenario_name}")
        self.info("=" * 60)
