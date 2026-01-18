"""
2D Visualization for Schrödinger Bridge.

Creates:
- Animated GIFs of particle transport
- Static diagnostic plots (convergence, transport plan)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional
import matplotlib as mpl

# Professional styling
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.linewidth'] = 1.2


class Animator:
    """2D visualization for Schrödinger Bridge simulations."""
    
    @staticmethod
    def create_custom_cmap(name: str = 'bridge') -> LinearSegmentedColormap:
        """Create aesthetic colormap for visualization."""
        colors = ['#0d1b2a', '#1b263b', '#415a77', '#778da9', '#e0e1dd']
        return LinearSegmentedColormap.from_list(name, colors)
    
    @staticmethod
    def create_gif(
        trajectory: np.ndarray,
        times: np.ndarray,
        X_source: np.ndarray,
        X_target: np.ndarray,
        filename: str,
        output_dir: str = "outputs",
        title: str = "Schrödinger Bridge",
        fps: int = 30,
        dpi: int = 150,
        colormap: str = "viridis",
        marker_size: float = 8,
        alpha: float = 0.8,
        figsize: tuple = (10, 10)
    ) -> str:
        """
        Create animated GIF of Schrödinger bridge particle transport.
        
        Parameters
        ----------
        trajectory : (n_frames, n, 2) - interpolated points
        times : (n_frames,) - time values
        X_source, X_target : source and target clouds for reference
        filename : output filename
        output_dir : output directory
        title : animation title
        fps : frames per second
        dpi : resolution
        colormap : matplotlib colormap name
        marker_size : particle marker size
        alpha : marker transparency
        figsize : figure dimensions
        
        Returns
        -------
        filepath : str - full path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        n_frames, n_points, _ = trajectory.shape
        
        # Compute bounds with padding
        all_points = np.vstack([trajectory.reshape(-1, 2), X_source, X_target])
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        padding = 0.15 * max(x_max - x_min, y_max - y_min)
        
        # Setup figure with dark theme
        fig, ax = plt.subplots(figsize=figsize, facecolor='#0d1b2a')
        ax.set_facecolor('#0d1b2a')
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Title
        ax.set_title(
            f'{title}: 2D Entropic Optimal Transport',
            color='#e0e1dd', fontsize=14, fontweight='bold', pad=15
        )
        
        # Time indicator
        time_text = ax.text(
            0.02, 0.98, '', transform=ax.transAxes,
            color='#778da9', fontsize=14, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#0d1b2a', 
                     edgecolor='#415a77', alpha=0.9)
        )
        
        # Ghost source and target
        ax.scatter(X_source[:, 0], X_source[:, 1], 
                  c='#415a77', s=marker_size*0.5, alpha=0.15, zorder=1,
                  label='Source')
        ax.scatter(X_target[:, 0], X_target[:, 1], 
                  c='#778da9', s=marker_size*0.5, alpha=0.15, zorder=1,
                  label='Target')
        
        # Get colormap
        cmap = plt.get_cmap(colormap)
        
        # Color interpolation endpoints
        color_start = np.array(cmap(0.2))
        color_end = np.array(cmap(0.8))
        
        # Main scatter
        init_colors = np.tile(color_start, (n_points, 1))
        init_colors[:, 3] = alpha
        scatter = ax.scatter(
            trajectory[0, :, 0], trajectory[0, :, 1], 
            s=marker_size, c=init_colors, zorder=3
        )
        
        def init():
            time_text.set_text('')
            return scatter, time_text
        
        def update(frame):
            t = times[frame]
            points = trajectory[frame]
            
            # Interpolate colors
            colors = (1 - t) * color_start + t * color_end
            colors = np.tile(colors, (n_points, 1))
            
            # Add slight alpha variation for depth
            alpha_variation = (alpha - 0.1) + 0.2 * np.random.rand(n_points)
            colors[:, 3] = np.clip(alpha_variation, 0.3, 1.0)
            
            scatter.set_offsets(points)
            scatter.set_color(colors)
            
            # Update time display
            time_text.set_text(f't = {t:.3f}')
            
            return scatter, time_text
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, update, init_func=init,
            frames=n_frames, interval=1000/fps, blit=True
        )
        
        # Save with progress bar
        print(f"    Saving animation ({n_frames} frames)...")
        writer = animation.PillowWriter(fps=fps)
        
        with tqdm(total=n_frames, desc="    Rendering", unit="frame") as pbar:
            def progress_callback(current_frame, total_frames):
                pbar.n = current_frame + 1
                pbar.refresh()
            
            anim.save(filepath, writer=writer, dpi=dpi,
                     progress_callback=progress_callback)
        
        plt.close(fig)
        print(f"    ✓ Animation saved: {filepath}")
        
        return str(filepath)
    
    @staticmethod
    def create_diagnostics(
        result: Dict[str, Any],
        X_source: np.ndarray,
        X_target: np.ndarray,
        filename: str,
        output_dir: str = "outputs",
        title: str = "Schrödinger Bridge",
        dpi: int = 150
    ) -> str:
        """
        Create static diagnostic plots.
        
        Includes:
        - Source/Target distributions
        - Sinkhorn convergence curve
        - Transport plan heatmap
        - Metrics summary
        
        Parameters
        ----------
        result : dict - results from solver.solve()
        X_source, X_target : distributions
        filename : output filename
        output_dir : output directory
        title : plot title
        dpi : resolution
        
        Returns
        -------
        filepath : str - full path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'{title}: Diagnostics', fontsize=16, fontweight='bold')
        
        # [0,0] Source and Target distributions
        ax1 = axes[0, 0]
        ax1.scatter(X_source[:, 0], X_source[:, 1], 
                   c='#3498db', s=8, alpha=0.6, label='Source μ')
        ax1.scatter(X_target[:, 0], X_target[:, 1], 
                   c='#e74c3c', s=8, alpha=0.6, label='Target ν')
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('y', fontsize=12)
        ax1.set_title('Source and Target Distributions', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # [0,1] Sinkhorn convergence
        ax2 = axes[0, 1]
        iterations = np.arange(0, len(result['marginal_errors'])) * 10
        ax2.semilogy(iterations, result['marginal_errors'], 
                    'b-', linewidth=2, label='Marginal error')
        ax2.axhline(result['params']['tol'], color='r', linestyle='--', 
                   linewidth=1.5, label=f"Tolerance ({result['params']['tol']:.0e})")
        ax2.set_xlabel('Sinkhorn Iteration', fontsize=12)
        ax2.set_ylabel('Marginal Error $\\|\\pi\\mathbf{1} - \\mathbf{a}\\|_1$', fontsize=12)
        ax2.set_title('Sinkhorn Convergence', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # [1,0] Transport plan heatmap
        ax3 = axes[1, 0]
        pi = result['pi']
        n = pi.shape[0]
        # Subsample for visualization
        n_show = min(100, n)
        idx = np.linspace(0, n-1, n_show, dtype=int)
        pi_sub = pi[np.ix_(idx, idx)]
        
        im = ax3.imshow(np.log10(pi_sub + 1e-20), 
                       cmap='viridis', aspect='auto',
                       vmin=-10, vmax=np.log10(pi_sub.max()))
        ax3.set_xlabel('Target Index', fontsize=12)
        ax3.set_ylabel('Source Index', fontsize=12)
        ax3.set_title(f'Transport Plan $\\log_{{10}}(\\pi^*)$ (n={n_show})', 
                     fontsize=13, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax3, label='$\\log_{10}(\\pi_{ij})$')
        
        # [1,1] Metrics summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        metrics = result['metrics']
        params = result['params']
        
        summary_text = f"""
╔══════════════════════════════════════════════════════════╗
║                    DIAGNOSTIC METRICS                    ║
╠══════════════════════════════════════════════════════════╣
║  Physical Parameters                                     ║
║  ├─ Entropic regularization (ε):  {result['epsilon']:.4f}                   ║
║  ├─ Number of particles:          {params['n_source']}                     ║
║  └─ Dimension:                    {params['dimension']}                       ║
╠══════════════════════════════════════════════════════════╣
║  Transport Results                                        ║
║  ├─ Wasserstein cost ⟨C,π⟩:       {result['cost']:.6f}               ║
║  ├─ Converged:                    {'Yes' if result['converged'] else 'No'}                     ║
║  └─ Iterations:                   {result['n_iter']}                     ║
╠══════════════════════════════════════════════════════════╣
║  Diagnostic Metrics                                       ║
║  ├─ Plan entropy H(π):            {metrics['plan_entropy']:.4f}                  ║
║  ├─ Effective sparsity:           {metrics['effective_sparsity']:.6f}               ║
║  ├─ Marginal fidelity:            {metrics['marginal_fidelity']:.8f}             ║
║  ├─ Bridge diffusivity (ε/6):     {metrics['bridge_diffusivity']:.6f}               ║
║  └─ Iteration efficiency:         {metrics['iteration_efficiency']:.4f}                  ║
╠══════════════════════════════════════════════════════════╣
║  Verification                                             ║
║  ├─ Row marginal error:           {metrics['row_marginal_error']:.2e}               ║
║  ├─ Column marginal error:        {metrics['col_marginal_error']:.2e}               ║
║  └─ Plan mass (should≈1):         {metrics['plan_mass']:.10f}           ║
╚══════════════════════════════════════════════════════════╝
"""
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=10, fontfamily='monospace',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#f8f9fa', 
                         edgecolor='#dee2e6', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"    ✓ Diagnostics saved: {filepath}")
        
        return str(filepath)
