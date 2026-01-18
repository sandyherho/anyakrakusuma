# `anyakrakusuma`: 2D Schrödinger Bridge Solver via Entropic Optimal Transport

[![DOI](https://zenodo.org/badge/1136688443.svg)](https://doi.org/10.5281/zenodo.18287395)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/anyakrakusuma.svg)](https://pypi.org/project/anyakrakusuma/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?logo=scipy&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![netCDF4](https://img.shields.io/badge/netCDF4-%23004B87.svg)](https://unidata.github.io/netcdf4-python/)
[![Numba](https://img.shields.io/badge/Numba-%2300A3E0.svg?logo=numba&logoColor=white)](https://numba.pydata.org/)
[![Pillow](https://img.shields.io/badge/Pillow-%23000000.svg)](https://python-pillow.org/)
[![tqdm](https://img.shields.io/badge/tqdm-%23FFC107.svg)](https://tqdm.github.io/)


> The nomenclature `anyakrakusuma` is directly derived from the regnal titulature of **Sultan Agung of Mataram** (r. 1613–1645), the third and most formidable ruler of the Mataram Sultanate in Java. Born as **Raden Mas Jatmika** (also known as *Raden Mas Rangsang*), he initially adopted the title **Susuhunan Anyakrakusuma** (or *Prabu Pandita Anyakrakusuma*) upon his accession in 1613, a period marked by his aggressive consolidation of power across Central and East Java. His reign is historically distinguished by his staunch military campaigns against the Dutch East India Company (VOC) in Batavia (1628–1629) and his cultural synthesis of Javanese and Islamic traditions, most notably the creation of the Javanese calendar. In 1641, following an investiture by the envoy of Sharif Zaid ibnu Muhsin Al Hasyimi of Mecca, he consecrated his authority with the supreme title **Sultan Abdullah Muhammad Maulana Matarani al-Jawi**, yet he remains immortalized in Javanese historiography by his syncretic honorific: **Sultan Agung Adi Prabu Anyakrakusuma Senapati ing Ngalaga Abdurrahman Sayyidin Panatagama Khalifatullah Tanah Jawi**, identifying him as the "Great Sultan," the supreme ruler, and the commander of the faithful in the land of Java.


## Overview

`anyakrakusuma` is a high-performance Python solver for the **Schrödinger Bridge Problem (SBP)** via **Entropic Optimal Transport (EOT)**. The Schrödinger Bridge computes the most likely stochastic evolution between two probability distributions, combining the geometric structure of optimal transport with the diffusive dynamics of Brownian motion.

This solver implements the **log-domain Sinkhorn-Knopp algorithm** for numerical stability, achieving robust convergence even with small regularization parameters. The implementation leverages **Numba JIT compilation** with parallel processing for significant performance gains (10-100× speedup) on large-scale particle systems.

**Key Applications:**
- Generative modeling and distribution morphing
- Stochastic process interpolation
- Computational biology (cell trajectory inference)
- Machine learning (diffusion models, score matching)
- Statistical physics (entropy-regularized transport)

**Note:** This package provides **2D visualization** of particle transport dynamics.

## Physics

### The Schrödinger Bridge Problem

The Schrödinger Bridge Problem seeks the most likely evolution of a diffusion process $X_t$ constrained to have marginals $\rho_0$ at $t=0$ and $\rho_1$ at $t=T$. 

The solution satisfies the **Schrödinger system** (coupled via Feynman-Kac):

$$\partial_t \varphi(x,t) = \frac{\varepsilon}{2} \Delta \varphi(x,t) \quad \text{(forward heat equation)}$$

$$\partial_t \psi(x,t) = -\frac{\varepsilon}{2} \Delta \psi(x,t) \quad \text{(backward heat equation)}$$

with boundary conditions:
- $\varphi(x,0)\psi(x,0) = \rho_0(x)$
- $\varphi(x,T)\psi(x,T) = \rho_1(x)$

The marginal at time $t$ is: $\rho_t(x) = \varphi(x,t)\psi(x,t)$

### Entropic Optimal Transport Formulation

For discrete measures $\mu = \sum_i a_i \delta_{x_i}$ and $\nu = \sum_j b_j \delta_{y_j}$, the entropic optimal transport problem is:

$$\min_{\pi \in \Pi(\mu,\nu)} \langle C, \pi \rangle + \varepsilon H(\pi | \mu \otimes \nu)$$

where:
- $C_{ij} = \|x_i - y_j\|^2$ is the squared Euclidean cost
- $H(\pi | \mu \otimes \nu)$ is the Kullback-Leibler divergence
- $\varepsilon > 0$ is the entropic regularization (diffusivity)
- $\Pi(\mu,\nu)$ is the set of couplings with marginals $\mu$ and $\nu$

The optimal coupling has the form $\pi^*_{ij} = u_i K_{ij} v_j$ where:
- $K = \exp(-C/\varepsilon)$ is the **Gibbs kernel** (heat kernel for squared distance)
- $u, v$ are the **Sinkhorn potentials** found by iterative scaling

### Schrödinger Bridge Displacement Interpolation

The Schrödinger bridge displacement interpolation at time $t \in [0,1]$:

$$X_t = (1-t)X_0 + t X_1 + \sqrt{\varepsilon \cdot t(1-t)} \cdot Z, \quad Z \sim \mathcal{N}(0, I)$$

where $(X_0, X_1)$ are coupled via the optimal entropic plan $\pi^*$.

**Physical Interpretation:**
- **Deterministic drift**: $(1-t)X_0 + tX_1$ — linear interpolation (McCann geodesic)
- **Stochastic diffusion**: $\sqrt{\varepsilon \cdot t(1-t)} \cdot Z$ — Brownian bridge correction
- The diffusion term vanishes at $t=0$ and $t=1$, ensuring exact boundary conditions

### Key Theoretical Properties

1. **Entropy Production**: The entropy of $\rho_t$ follows a parabolic profile
2. **Cost Decomposition**: Transport cost = Wasserstein cost + Entropic regularization
3. **Marginal Constraints**: $\sum_j \pi_{ij} = a_i$ and $\sum_i \pi_{ij} = b_j$
4. **Plan Positivity**: $\pi_{ij} > 0$ for all $i,j$ (full support)

## Numerical Methods

### Log-Domain Sinkhorn Algorithm

The Sinkhorn iterations in log-domain for numerical stability:

$$f^{n+1} = \varepsilon \log(a) - \varepsilon \cdot \text{LSE}_j\left(\frac{g^n - C}{\varepsilon}\right)$$

$$g^{n+1} = \varepsilon \log(b) - \varepsilon \cdot \text{LSE}_i\left(\frac{f^{n+1} - C^\top}{\varepsilon}\right)$$

where $\text{LSE}$ is the log-sum-exp operator with max-trick for stability:

$$\text{LSE}(x) = \max(x) + \log\left(\sum_i \exp(x_i - \max(x))\right)$$

### Convergence Criterion

Marginal constraint violation:

$$\text{error} = \left\|\pi \mathbf{1} - a\right\|_1$$

### Novel Diagnostic Metrics

| Metric | Formula | Physical Meaning |
|--------|---------|------------------|
| **Wasserstein Cost** | $W = \langle C, \pi^* \rangle$ | Total transport work |
| **Plan Entropy** | $H = -\sum_{ij} \pi_{ij} \log(\pi_{ij})$ | Coupling uncertainty |
| **Effective Sparsity** | $S_{\text{eff}} = \exp(H) / nm$ | Normalized plan support |
| **Bridge Diffusivity** | $D_{\text{bridge}} = \varepsilon \cdot \mathbb{E}[t(1-t)]$ | Average stochastic spread |
| **Marginal Fidelity** | $1 - \|\pi\mathbf{1} - a\|_1$ | Constraint satisfaction |
| **Iteration Efficiency** | $\eta = -\log(\text{tol})/n_{\text{iter}}$ | Convergence rate per iteration |

## Features

- **Log-domain stability**: Robust convergence for small $\varepsilon$
- **Adaptive convergence**: Automatic iteration based on marginal error
- **High-performance computing**: Numba JIT compilation (10-100× speedup)
- **Parallel processing**: Multi-core CPU utilization via `prange`
- **Professional 2D visualization**: Animated particle transport
- **Comprehensive diagnostics**: Static PNG with convergence curves
- **Scientific data formats**: NetCDF4 (CF-1.8) and CSV output
- **Comparable test cases**: Four scenarios with increasing complexity

## Directory Structure

```
anyakrakusuma/
├── configs/                          # Simulation configuration files
│   ├── case1_circle_to_circle.txt   # Baseline: concentric circles
│   ├── case2_spiral_to_gaussian.txt # Topology change
│   ├── case3_moons_to_moons.txt     # Symmetric rotation
│   └── case4_lissajous_to_trefoil.txt # Complex curves
│
├── src/anyakrakusuma_eot/           # Main package source
│   ├── __init__.py                  # Package initialization
│   ├── cli.py                       # Command-line interface
│   │
│   ├── core/                        # Core numerical algorithms
│   │   ├── __init__.py
│   │   ├── solver.py                # Sinkhorn solver (log-domain)
│   │   └── distributions.py         # Point cloud generators
│   │
│   ├── io/                          # Input/output handlers
│   │   ├── __init__.py
│   │   ├── config_manager.py        # Configuration file parser
│   │   └── data_handler.py          # NetCDF/CSV writer
│   │
│   ├── visualization/               # Animation and plotting
│   │   ├── __init__.py
│   │   └── animator.py              # 2D GIF generator + diagnostics
│   │
│   └── utils/                       # Utilities
│       ├── __init__.py
│       ├── logger.py                # Simulation logging
│       └── timer.py                 # Performance profiling
│
├── tests/                           # Unit tests
│   ├── __init__.py
│   ├── test_solver.py               # Solver correctness tests
│   └── test_distributions.py        # Distribution generator tests
│
├── outputs/                         # Generated results (created at runtime)
│   ├── *.nc                         # NetCDF data files
│   ├── *.csv                        # CSV metric summaries
│   ├── *_diagnostics.png            # Static convergence plots
│   └── *.gif                        # Animation files
│
├── logs/                            # Simulation logs (created at runtime)
│   └── *.log
│
├── pyproject.toml                   # Poetry build configuration
├── README.md                        # This file
├── LICENSE                          # MIT license
└── .gitignore                       # Git ignore patterns
```

## Installation

**From PyPI:**
```bash
pip install anyakrakusuma
```

**From source:**
```bash
git clone https://github.com/sandyherho/anyakrakusuma.git
cd anyakrakusuma
pip install -e .
```

## Quick Start

**Command line:**
```bash
anyakrakusuma case1              # Circle to circle
anyakrakusuma case4 --cores 8    # Complex case with 8 cores
anyakrakusuma --all              # Run all cases
```

**Python API:**
```python
from anyakrakusuma_eot import SchrodingerBridgeSolver, generate_circle, generate_spiral

# Initialize solver
solver = SchrodingerBridgeSolver(n_cores=8, verbose=True)

# Generate distributions
X_source = generate_circle(n=1000, radius=1.0)
X_target = generate_spiral(n=1000, turns=2.0)

# Solve Schrödinger Bridge
result = solver.solve(
    X_source, X_target,
    epsilon=0.05,
    max_iter=2000,
    tol=1e-9
)

# Check convergence
print(f"Transport cost: {result['cost']:.6f}")
print(f"Converged: {result['converged']}")
```

## Test Cases (Comparable)

All test cases use the same number of particles (n=1000) and similar computational settings to enable direct comparison of transport characteristics.

| Case | Source → Target | $\varepsilon$ | Key Insight |
|------|-----------------|---------------|-------------|
| 1 | Circle (r=1) → Circle (r=2) | 0.02 | Baseline radial scaling |
| 2 | Spiral → Gaussian mixture | 0.05 | Topology dissolution |
| 3 | Two moons → Two moons (rotated) | 0.03 | Symmetric interchange |
| 4 | Lissajous (3:2) → Trefoil knot | 0.04 | Complex curve morphing |

**Comparison Metrics (per case):**
- Wasserstein cost (transport effort)
- Sinkhorn iterations (convergence difficulty)
- Plan entropy (coupling complexity)
- Effective sparsity (plan structure)


## Citation

If you use this software in your study, please cite:

```bibtex
@software{herho2026anyakrakusuma,
  title   = {{anyakrakusuma: 2D Schrödinger Bridge Solver via Entropic Optimal Transport}},
  author  = {Herho, Sandy H. S. and Anwar, Iwan P. and Khadami, Faruq and Suwarman, Rusmawan and Irawan, Dasapta E.},
  year    = {2026},
  version = {0.0.1},
  url     = {https://github.com/sandyherho/anyakrakusuma}
}
```

## Authors

- Sandy H. S. Herho (sandy.herho@ronininstitute.org)
- Iwan P. Anwar
- Faruq Khadami
- Rusmawan Suwarman
- Dasapta E. Irawan

## License

MIT License - See [LICENSE](LICENSE) for details.
