#!/usr/bin/env python
"""
Command Line Interface for anyakrakusuma Schrödinger Bridge Solver

Four test cases with increasing complexity for comparative analysis.
"""

import argparse
import sys
import numpy as np
from pathlib import Path

from .core.solver import SchrodingerBridgeSolver
from .core.distributions import (
    generate_circle, generate_spiral, generate_two_moons,
    generate_two_moons_rotated, generate_gaussian_mixture,
    generate_lissajous, generate_trefoil
)
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler
from .visualization.animator import Animator
from .utils.logger import SimulationLogger
from .utils.timer import Timer


def print_header():
    """Print ASCII art header."""
    print("\n" + "=" * 70)
    print(" " * 10 + "anyakrakusuma: 2D Schrödinger Bridge Solver")
    print(" " * 15 + "Entropic Optimal Transport")
    print(" " * 22 + "Version 0.0.1")
    print("=" * 70)
    print("\n  Log-Domain Sinkhorn-Knopp Algorithm")
    print("  Numba JIT Acceleration + Parallel Processing")
    print("\n  License: MIT")
    print("=" * 70 + "\n")


def normalize_scenario_name(scenario_name: str) -> str:
    """Convert scenario name to clean filename format."""
    clean = scenario_name.lower()
    clean = clean.replace(' - ', '_')
    clean = clean.replace('-', '_')
    clean = clean.replace(' ', '_')
    
    while '__' in clean:
        clean = clean.replace('__', '_')
    
    if clean.startswith('case_'):
        parts = clean.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            case_num = parts[1]
            rest = '_'.join(parts[2:])
            clean = f"case{case_num}_{rest}"
    
    clean = clean.rstrip('_')
    return clean


def create_distributions(config: dict) -> tuple:
    """Create source and target distributions based on configuration."""
    n = config.get('n_particles', 1000)
    seed = config.get('seed', 42)
    
    source_type = config.get('source_type', 'circle')
    target_type = config.get('target_type', 'circle')
    
    # Source distribution
    if source_type == 'circle':
        X_source = generate_circle(
            n, 
            radius=config.get('source_radius', 1.0),
            noise=config.get('source_noise', 0.0),
            seed=seed
        )
    elif source_type == 'spiral':
        X_source = generate_spiral(
            n,
            turns=config.get('source_turns', 2.0),
            noise=config.get('source_noise', 0.0),
            seed=seed
        )
    elif source_type == 'two_moons':
        X_source = generate_two_moons(
            n,
            noise=config.get('source_noise', 0.05),
            seed=seed
        )
    elif source_type == 'lissajous':
        X_source = generate_lissajous(
            n,
            a=config.get('lissajous_a', 3),
            b=config.get('lissajous_b', 2),
            delta=config.get('lissajous_delta', np.pi/2),
            scale=config.get('scale_factor', 1.5),
            seed=seed
        )
    else:
        raise ValueError(f"Unknown source type: {source_type}")
    
    # Target distribution
    if target_type == 'circle':
        X_target = generate_circle(
            n,
            radius=config.get('target_radius', 2.0),
            noise=config.get('target_noise', 0.0),
            seed=seed + 1000
        )
    elif target_type == 'spiral':
        X_target = generate_spiral(
            n,
            turns=config.get('target_turns', 2.0),
            noise=config.get('target_noise', 0.0),
            seed=seed + 1000
        )
    elif target_type == 'gaussian_mixture':
        X_target = generate_gaussian_mixture(
            n,
            n_clusters=config.get('target_n_clusters', 4),
            std=config.get('target_std', 0.15),
            seed=seed + 1000
        )
    elif target_type == 'two_moons':
        X_target = generate_two_moons(
            n,
            noise=config.get('target_noise', 0.05),
            seed=seed + 1000
        )
    elif target_type == 'two_moons_rotated':
        X_target = generate_two_moons_rotated(
            n,
            noise=config.get('target_noise', 0.05),
            rotation_angle=config.get('rotation_angle', 90.0),
            seed=seed + 1000
        )
    elif target_type == 'trefoil':
        X_target = generate_trefoil(
            n,
            scale=config.get('scale_factor', 1.5),
            seed=seed + 1000
        )
    else:
        raise ValueError(f"Unknown target type: {target_type}")
    
    return X_source, X_target


def run_scenario(config: dict, output_dir: str = "outputs",
                verbose: bool = True, n_cores: int = None):
    """Run a complete Schrödinger Bridge simulation scenario."""
    
    # Validate config
    config = ConfigManager.validate_config(config)
    
    scenario_name = config.get('scenario_name', 'simulation')
    clean_name = normalize_scenario_name(scenario_name)
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'=' * 60}")
    
    logger = SimulationLogger(clean_name, "logs", verbose)
    timer = Timer()
    timer.start("total")
    
    try:
        logger.log_parameters(config)
        
        # [1/5] Initialize solver
        with timer.time_section("solver_init"):
            if verbose:
                print("\n[1/5] Initializing solver...")
            
            cores = n_cores if n_cores else config.get('n_cores', 0)
            solver = SchrodingerBridgeSolver(
                n_cores=cores if cores > 0 else None,
                verbose=verbose,
                logger=logger
            )
        
        # [2/5] Create distributions
        with timer.time_section("distributions"):
            if verbose:
                print("\n[2/5] Creating distributions...")
            
            X_source, X_target = create_distributions(config)
            
            if verbose:
                print(f"      Source: {config.get('source_type')} ({X_source.shape[0]} particles)")
                print(f"      Target: {config.get('target_type')} ({X_target.shape[0]} particles)")
        
        # [3/5] Solve Schrödinger Bridge
        with timer.time_section("solve"):
            if verbose:
                print("\n[3/5] Solving Schrödinger Bridge...")
            
            result = solver.solve(
                X_source, X_target,
                epsilon=config.get('epsilon', 0.05),
                max_iter=config.get('max_iter', 2000),
                tol=config.get('tol', 1e-9)
            )
            
            logger.log_results(result)
        
        # Generate trajectory
        with timer.time_section("trajectory"):
            trajectory, times = solver.generate_trajectory(
                result,
                n_frames=config.get('n_frames', 120),
                seed=config.get('seed', 42)
            )
        
        # [4/5] Save data
        if config.get('save_netcdf', True) or config.get('save_csv', True):
            with timer.time_section("save_data"):
                if verbose:
                    print("\n[4/5] Saving data...")
                
                if config.get('save_netcdf', True):
                    nc_file = f"{clean_name}.nc"
                    DataHandler.save_netcdf(
                        nc_file, result, trajectory, times, config, output_dir
                    )
                    if verbose:
                        print(f"      Saved: {output_dir}/{nc_file}")
                
                if config.get('save_csv', True):
                    csv_file = f"{clean_name}_metrics.csv"
                    DataHandler.save_csv(csv_file, result, config, output_dir)
                    if verbose:
                        print(f"      Saved: {output_dir}/{csv_file}")
                    
                    # Also append to comparison file
                    DataHandler.append_comparison_csv(
                        "comparison_metrics.csv", result, config, output_dir
                    )
        
        # [5/5] Create visualizations
        if config.get('save_animation', True) or config.get('save_diagnostics', True):
            with timer.time_section("visualization"):
                if verbose:
                    print("\n[5/5] Creating visualizations...")
                
                if config.get('save_diagnostics', True):
                    diag_file = f"{clean_name}_diagnostics.png"
                    Animator.create_diagnostics(
                        result, X_source, X_target,
                        diag_file, output_dir, scenario_name,
                        dpi=config.get('dpi', 150)
                    )
                
                if config.get('save_animation', True):
                    gif_file = f"{clean_name}.gif"
                    Animator.create_gif(
                        trajectory, times, X_source, X_target,
                        gif_file, output_dir, scenario_name,
                        fps=config.get('fps', 30),
                        dpi=config.get('dpi', 150),
                        colormap=config.get('colormap', 'viridis'),
                        marker_size=config.get('marker_size', 8),
                        alpha=config.get('alpha', 0.8)
                    )
        
        timer.stop("total")
        logger.log_timing(timer.get_times())
        
        if verbose:
            print(f"\n{'=' * 60}")
            print("SIMULATION COMPLETED SUCCESSFULLY")
            print(f"  Solve time: {timer.times.get('solve', 0):.2f} s")
            print(f"  Visualization time: {timer.times.get('visualization', 0):.2f} s")
            print(f"  Total time: {timer.times.get('total', 0):.2f} s")
            
            if logger.warnings:
                print(f"  Warnings: {len(logger.warnings)}")
            if logger.errors:
                print(f"  Errors: {len(logger.errors)}")
            
            print(f"{'=' * 60}\n")
        
        return result
    
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"SIMULATION FAILED")
            print(f"  Error: {str(e)}")
            print(f"{'=' * 60}\n")
        
        raise
    
    finally:
        logger.finalize()


def main():
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description='anyakrakusuma: 2D Schrödinger Bridge Solver via Entropic OT',
        epilog='Example: anyakrakusuma case1 --cores 8'
    )
    
    parser.add_argument(
        'case',
        nargs='?',
        choices=['case1', 'case2', 'case3', 'case4'],
        help='Test case to run (case1-4 with increasing complexity)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to custom configuration file'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run all test cases sequentially'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='outputs',
        help='Output directory for results (default: outputs)'
    )
    
    parser.add_argument(
        '--cores',
        type=int,
        default=None,
        help='Number of CPU cores to use (default: all available)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode (minimal output)'
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    if verbose:
        print_header()
    
    # Custom config
    if args.config:
        config = ConfigManager.load(args.config)
        run_scenario(config, args.output_dir, verbose, args.cores)
    
    # All cases
    elif args.all:
        configs_dir = Path(__file__).parent.parent.parent / 'configs'
        config_files = sorted(configs_dir.glob('case*.txt'))
        
        if not config_files:
            print("ERROR: No configuration files found in configs/")
            sys.exit(1)
        
        results = []
        for i, cfg_file in enumerate(config_files, 1):
            if verbose:
                print(f"\n[Case {i}/{len(config_files)}] Running {cfg_file.stem}...")
            
            config = ConfigManager.load(str(cfg_file))
            result = run_scenario(config, args.output_dir, verbose, args.cores)
            results.append(result)
        
        # Print comparison summary
        if verbose and len(results) > 1:
            print("\n" + "=" * 70)
            print("COMPARISON SUMMARY")
            print("=" * 70)
            print(f"{'Case':<30} {'Cost':<12} {'Iter':<8} {'Entropy':<10} {'Sparsity':<10}")
            print("-" * 70)
            for i, r in enumerate(results, 1):
                m = r['metrics']
                print(f"Case {i:<26} {r['cost']:<12.4f} {r['n_iter']:<8} "
                      f"{m['plan_entropy']:<10.4f} {m['effective_sparsity']:<10.6f}")
            print("=" * 70)
            print(f"\nComparison CSV saved: {args.output_dir}/comparison_metrics.csv")
    
    # Single case
    elif args.case:
        case_map = {
            'case1': 'case1_circle_to_circle',
            'case2': 'case2_spiral_to_gaussian',
            'case3': 'case3_moons_to_moons',
            'case4': 'case4_lissajous_to_trefoil'
        }
        
        cfg_name = case_map[args.case]
        configs_dir = Path(__file__).parent.parent.parent / 'configs'
        cfg_file = configs_dir / f'{cfg_name}.txt'
        
        if cfg_file.exists():
            config = ConfigManager.load(str(cfg_file))
            run_scenario(config, args.output_dir, verbose, args.cores)
        else:
            print(f"ERROR: Configuration file not found: {cfg_file}")
            sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == '__main__':
    main()
