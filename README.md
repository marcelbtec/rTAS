# Tangential Action Spaces (TAS) and Reflective TAS (rTAS): Reference Implementation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy->2.0-orange.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository provides the reference implementation and figure-generation code for the manuscript:

> **"Tangential Action Spaces: Geometry, Memory and Cost in Holonomic and Nonholonomic Agents"**
> 
> Authors: [Author Names]  
> Preprint: [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

The codebase implements the theoretical framework of Tangential Action Spaces (TAS) and its reflective extension (rTAS), demonstrating fundamental principles of embodied agency through geometric constructions, energy-memory trade-offs, and multi-agent dynamics.

## Table of Contents

1. [Key Contributions](#key-contributions)
2. [Mathematical Framework](#mathematical-framework)
3. [Repository Structure](#repository-structure)
4. [Installation](#installation)
5. [Reproducing Results](#reproducing-results)
6. [API Documentation](#api-documentation)
7. [Citation](#citation)
8. [License](#license)

## Key Contributions

This implementation provides:

1. **Geometric Formalism**: Complete implementation of the TAS hierarchy (P → C → I) with various lift operators
2. **Energy-Memory Trade-offs**: Numerical verification of the cost-memory duality theorem
3. **Holonomy Classification**: Four archetypal examples demonstrating different memory mechanisms
4. **Reflective Extension**: rTAS framework with learnable model manifolds
5. **Multi-Agent Dynamics**: Cooperative, competitive, and phase transition phenomena

## Mathematical Framework

### Core Concepts

#### 1. TAS Stack and Lift Operators

The implementation realizes agents as hierarchies of manifolds with projections:
- **Physical manifold** P (base space)
- **Cognitive manifold** C (task space) 
- **Intentional manifold** I (goal space)

Two primary lift operators are implemented:

**Geometric Lift** (diffeomorphic case):
```
ů_geom = DΦ⁻¹ċ
```
Properties: Unique, norm-preserving, holonomy-free (Proposition 1, §1.3.1)

**Metric Lift** (fibration case):
```
ů_metric = G⁻¹DΦᵀ(DΦG⁻¹DΦᵀ)⁻¹ċ
```
Properties: Energy-minimizing, admits Pythagorean decomposition (Proposition 2, §1.3.2)

#### 2. Cost-Memory Duality

The fundamental theorem (Theorem 1) states that holonomic memory requires excess energy:
```
ΔE ≥ ΔE_metric ∝ (Δh)² + O(|γ|³)
```
For small loops, excess scales quadratically with memory magnitude (Proposition 3, §3.2).

#### 3. Holonomy Classification

Four archetypal mechanisms are implemented:
1. **Intrinsically conservative**: Diffeomorphism + geometric lift
2. **Conditionally conservative**: Flat fibration (zero curvature)
3. **Geometrically non-conservative**: Curved connection
4. **Dynamically non-conservative**: Prescribed vertical dynamics

#### 4. Reflective TAS (rTAS)

Extends TAS with model manifold M and weighted block metric:
```
Ĝ = G ⊕ λH
```

The reflective metric lift solves:
```
[ů*]   = Ĝ⁻¹Aᵀ(AĜ⁻¹Aᵀ)⁻¹ċ
[ṁ*]
```
where A = [DΦₘ ∂ₘΦₘ] (Definition 7, Lemma 3, §8.2).

### Canonical Examples

1. **Strip-Sine** (§6.1): Φ(u,v,h) = (u, v + κsin(u)), ḣ = α(c₁ċ₂ - c₂ċ₁)
2. **Helical Fibration** (§6.2): Connection ω = dz - α(ydx - xdy)
3. **Cylindrical Fibration** (§6.3): Flat connection, non-trivial topology
4. **Twisted Fibration** (§6.4): Spatially varying curvature F = 2(α + βcos(θ))dx∧dy

## Repository Structure

```
rTAS/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── scripts/
│   ├── tas_figures.py          # Single-agent TAS experiments (§6)
│   ├── rtas_figures.py         # Reflective TAS visualizations (§8)
│   └── rtas_multiagent.py      # Multi-agent dynamics (§§8.6, 9)
├── tests/
│   ├── test_tas_simple.py      # TAS property tests
│   └── test_rtas_simple.py     # rTAS property tests
└── [Generated directories]
    ├── figures/                 # TAS figures (created on first run)
    ├── rtas_figures/           # rTAS visualizations
    └── rtas_multiagent_results/# Multi-agent outputs
```

## Installation

### Requirements

- Python ≥ 3.9 (uses modern type hints)
- NumPy > 2.0
- Matplotlib ≥ 3.6
- SciPy ≥ 1.8 (for multi-agent experiments only)

### Setup

```bash
# Clone repository
git clone https://github.com/[username]/rTAS.git
cd rTAS

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install numpy matplotlib scipy

# Optional: Set backend for headless execution
export MPLBACKEND=Agg
```

## Testing

The repository includes minimal, focused tests that verify key mathematical properties using the actual implementation.

### Running Tests

```bash
# Test TAS properties (strip-sine system)
python tests/test_tas_simple.py

# Test rTAS properties (reflective lift)
python tests/test_rtas_simple.py
```

### What's Tested

**TAS Tests** (`test_tas_simple.py`):
- Strip-sine area law (§6.1): Δh = 2α·Area
- Cost-memory trade-off (Theorem 1): Prescribed dynamics require excess energy
- Small-loop quadratic law (Proposition 3): E_excess ∝ (Δh)²
- Geometric lift produces zero holonomy

**rTAS Tests** (`test_rtas_simple.py`):
- Projection constraint (Definition 7): Reflective lift satisfies c = Φ_m(u,v)
- Lambda effect: Higher λ reduces model adaptation

All tests use the actual functions exposed by the implementation and verify the specific mathematical claims from the manuscript.

## Reproducing Results

### TAS Figures (Single-Agent, §6)

Generate all canonical example figures:
```bash
python scripts/tas_figures.py
```

Outputs (saved to repository root):
- `figure_strip_sine_detailed.*` — Dynamically non-conservative (§6.1)
- `figure_flat_detailed.*` — Flat fibration, zero holonomy (§6.3)
- `figure_cylinder_detailed.*` — Cylindrical base, flat connection (§6.3)
- `figure_helical_detailed.*` — Constant curvature, area law (§6.2)
- `figure_twisted_detailed.*` — Variable curvature (§6.4)

### rTAS Figures (Reflective Extension, §8)

Generate reflective TAS visualizations:
```bash
python scripts/rtas_figures.py
```

Outputs (saved to `./rtas_figures/`):
- `fig_rtas1_reflective_lift.png` — Reflective metric lift & effort split (§8.2)
- `fig_rtas2_block_curvature.png` — Cross-curvature channels Fₚₘ, Fₘₚ (§8.3)
- `fig_rtas3_cost_frontier_and_policy.png` — Small-loop law & cost-aware dynamics (§8.4)
- `fig_rtasA_projective_channel.png` — Projective channel Φₘ(u,v) = (u, v + m·sin(u))
- `fig_rtasB_connection_channel.png` — Reflective helical/twisted connections (§8.5)

### Multi-Agent rTAS Experiments (§§8.6, 9)

Run all experiments:
```bash
python scripts/rtas_multiagent.py
```

Quick test mode (reduced iterations):
```bash
python scripts/rtas_multiagent.py --test
```

Experiments include:
1. **Cooperative Formation** — Figure-8 tracking with lateral offset (Fig. 13)
2. **Pursuit-Evasion** — Information asymmetry & learning lunge (Fig. 14)
3. **Phase Diagram** — Stability regions in (λ₁,λ₂) space (Fig. 15)
4. **Resonance Catastrophe** — Runaway co-adaptation (Fig. 16)
5. **Phase Transition Analysis** — First-order switch in optimal λ*(κ) (Fig. 12)

## API Documentation

### TAS Functions (`scripts/tas_figures.py`)

#### `GeometricTASExplorer`
```python
explorer = GeometricTASExplorer()
```

**Strip-sine methods**:
```python
# Geometric lift (zero holonomy)
uv_path, h_path = explorer.compute_strip_sine_geometric_lift(
    c_path,           # Cognitive trajectory
    kappa=0.5         # Coupling strength
)

# Prescribed dynamics (with holonomy)
uv_path, h_path = explorer.compute_strip_sine_prescribed_dynamics(
    c_path,           # Cognitive trajectory
    kappa=0.5,        # Coupling strength
    alpha=0.3         # Holonomy rate
)
```

### rTAS Functions (`scripts/rtas_figures.py`)

**Reflective lift**:
```python
u, v, m, e_p, e_m = reflective_lift_projective(
    c1, c2,           # Cognitive positions
    dc1, dc2,         # Cognitive velocities
    lam,              # Learning cost λ
    dt,               # Time step
    enforce_projection=True
)
```

### Multi-Agent System (`scripts/rtas_multiagent.py`)

```python
system = MultiAgentRTAS(
    lambda1=1.0,          # Agent 1 learning cost
    lambda2=1.0,          # Agent 2 learning cost
    coupling_strength=0.5, # Inter-agent coupling κ
    info_asymmetry=False  # Enable pursuit-evasion
)

# Step simulation
system.step(dt, c_dot1, c_dot2)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{authors2024tas,
  title={Tangential Action Spaces: Geometry, Memory and Cost in Holonomic and Nonholonomic Agents},
  author={[Author Names]},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

[Funding acknowledgments and institutional affiliations]

---

**Note**: Some docstrings may reference legacy filenames. Use the scripts as provided in this repository.