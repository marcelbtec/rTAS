#!/usr/bin/env python3
"""
Multi-Agent Reflective Tangential Action Spaces (rTAS) Experiments

Publication-ready code for reproducing multi-agent rTAS experiments from the manuscript.

This script explores emergent behaviors when multiple rTAS agents interact through
their model adaptations, creating a dynamically changing environment. The experiments
demonstrate cooperative formation control, pursuit-evasion dynamics, phase transitions,
and resonance catastrophes.

Key concepts:
- Reflective metric lift: Optimal lift minimizing combined physical-model cost
- Block metric: Ĝ = G ⊕ λH where λ controls the learning cost
- Cross-agent coupling through model parameters
- Reflective connection inducing holonomy

Experiments:
1. Cooperative Formation Control: Agents maintain formation while traversing figure-8
2. Pursuit-Evasion: Asymmetric dynamics with information asymmetry
3. Phase Diagram: Stability analysis across (λ₁, λ₂) parameter space
4. Resonance Catastrophe: Runaway co-adaptation under strong coupling

Usage:
  python rtas_multiagent_v2.py

Requirements:
  - numpy
  - matplotlib
  - scipy
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from scipy.integrate import odeint
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for publication-quality figures
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# Professional color palette
COLORS = {
    'agent1': '#2E86AB',  # Blue
    'agent2': '#A23B72',  # Purple
    'coupled': '#F18F01',  # Orange
    'stable': '#73AB84',  # Green
    'unstable': '#C73E1D',  # Red
    'neutral': '#6C757D'    # Gray
}

@dataclass
class AgentState:
    """State of a single rTAS agent"""
    u: np.ndarray      # Physical position (x,y) in base P
    v: np.ndarray      # Reserved for future extensions
    m: float           # Model parameter (scalar)
    z: float           # Fiber coordinate (physical holonomy)
    c: np.ndarray      # Cognitive position Φ_m(u)
    energy: float      # Accumulated reflective energy
    holonomy: float    # Physical holonomy (= z)
    E_phys: float = 0.0    # Accumulated physical energy ∫||u̇||² dt
    E_model: float = 0.0   # Accumulated model energy ∫||ṁ||² dt (unweighted)

class MultiAgentRTAS:
    """
    Multi-agent reflective TAS system.
    
    Implements coupled dynamics between two rTAS agents with configurable
    learning costs, inter-agent coupling, and information asymmetry.
    """
    def __init__(self, lambda1: float = 1.0, lambda2: float = 1.0,
                 coupling_strength: float = 0.1, info_asymmetry: bool = False,
                 alpha0: float = 0.25, alpha1: float = 0.20, alpha_beta: float = 0.8):
        """
        Initialize two-agent rTAS system.

        Parameters
        ----------
        lambda1 : float
            Trade-off parameter for agent 1 (physical vs model effort)
        lambda2 : float
            Trade-off parameter for agent 2
        coupling_strength : float
            Strength of inter-agent coupling
        info_asymmetry : bool
            If True, agent 1 ignores agent 2's model in its lift
        alpha0, alpha1, alpha_beta : float
            Connection strength parameters: α(m) = alpha0 + alpha1*tanh(alpha_beta*m)
        """
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.coupling = float(coupling_strength)
        self.info_asymmetry = bool(info_asymmetry)

        # Metrics
        self.G = np.eye(2)  # Physical metric
        self.H = 1.0        # Model metric

        # Connection parameters
        self.alpha0 = float(alpha0)
        self.alpha1 = float(alpha1)
        self.alpha_beta = float(alpha_beta)

        # Initialize
        self.reset_agents()
        self.history = {'agent1': [], 'agent2': [], 'time': []}

    def projective_map(self, u: np.ndarray, m: float) -> np.ndarray:
        """
        Projective channel map.
        
        Implements: Φ_m(u) = (u_x, u_y + m*sin(u_x))
        
        Parameters
        ----------
        u : np.ndarray
            Physical position [x, y]
        m : float
            Model parameter
            
        Returns
        -------
        np.ndarray
            Cognitive position
        """
        return np.array([u[0], u[1] + m * np.sin(u[0])])

    def projective_jacobian(self, u: np.ndarray, m: float) -> np.ndarray:
        """
        Jacobian of projective map with respect to physical coordinates.
        
        Returns DΦ_m(u) matrix.
        """
        return np.array([
            [1.0, 0.0],
            [m * np.cos(u[0]), 1.0]
        ])

    def model_sensitivity(self, u: np.ndarray) -> np.ndarray:
        """
        Sensitivity of projection to model parameter.
        
        Returns B = ∂_m Φ_m(u).
        """
        return np.array([0.0, np.sin(u[0])])

    def alpha(self, m: float) -> float:
        """
        Connection strength as function of model parameter.
        
        Implements: α(m) = α0 + α1 * tanh(αβ * m)
        """
        return self.alpha0 + self.alpha1 * np.tanh(self.alpha_beta * m)

    def reflective_lift(self, agent: AgentState, c_dot: np.ndarray,
                        other_m: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """
        Compute reflective metric lift for one agent:
            [u_dot; m_dot] = Ghat^{-1} A^T (A Ghat^{-1} A^T)^{-1} c_dot_eff

        c_dot_eff optionally includes a simple cross-agent coupling on the ċ2 channel.
        """
        lam = self.lambda1 if agent is self.agent1 else self.lambda2

        # Build constraint matrix
        D_Phi = self.projective_jacobian(agent.u, agent.m)
        B = self.model_sensitivity(agent.u).reshape(-1, 1)
        A = np.hstack([D_Phi, B])

        # Apply inter-agent coupling if present
        if (other_m is not None) and (self.coupling > 0.0):
            coupling_effect = self.coupling * other_m * np.array([0.0, np.cos(agent.u[0])])
            c_dot_eff = c_dot - coupling_effect
        else:
            c_dot_eff = c_dot

        # Compute reflective lift
        Ghat_inv = np.diag([1.0, 1.0, 1.0 / lam])
        AGA = A @ Ghat_inv @ A.T
        AGA_inv = np.linalg.pinv(AGA)

        lift = Ghat_inv @ A.T @ AGA_inv @ c_dot_eff
        u_dot = lift[:2]
        m_dot = float(lift[2])
        return u_dot, m_dot

    def compute_energy_components(self, u_dot: np.ndarray, m_dot: float) -> tuple[float, float]:
        """
        Compute instantaneous energy rate components.
        
        Returns (||u_dot||^2, ||m_dot||^2) unweighted.
        """
        phys_rate = float(u_dot @ u_dot)
        model_rate = float(m_dot ** 2)
        return phys_rate, model_rate

    def step(self, dt: float, c1_target: np.ndarray, c2_target: np.ndarray):
        """
        Execute one time step of coupled multi-agent dynamics.
        
        Parameters
        ----------
        dt : float
            Time step
        c1_target : np.ndarray
            Desired cognitive velocity for agent 1
        c2_target : np.ndarray
            Desired cognitive velocity for agent 2
        """
        # Compute reflective lifts
        if self.info_asymmetry:
            u1_dot, m1_dot = self.reflective_lift(self.agent1, c1_target, self.agent2.m)
            u2_dot, m2_dot = self.reflective_lift(self.agent2, c2_target, None)
        else:
            u1_dot, m1_dot = self.reflective_lift(self.agent1, c1_target, self.agent2.m)
            u2_dot, m2_dot = self.reflective_lift(self.agent2, c2_target, self.agent1.m)

        # Apply connection constraint
        a1 = self.alpha(self.agent1.m)
        a2 = self.alpha(self.agent2.m)
        z1_dot = a1 * (self.agent1.u[1] * u1_dot[0] - self.agent1.u[0] * u1_dot[1])
        z2_dot = a2 * (self.agent2.u[1] * u2_dot[0] - self.agent2.u[0] * u2_dot[1])

        # Update states
        self.agent1.u = self.agent1.u + u1_dot * dt
        self.agent1.m = self.agent1.m + m1_dot * dt
        self.agent1.z = self.agent1.z + z1_dot * dt
        self.agent1.c = self.projective_map(self.agent1.u, self.agent1.m)

        self.agent2.u = self.agent2.u + u2_dot * dt
        self.agent2.m = self.agent2.m + m2_dot * dt
        self.agent2.z = self.agent2.z + z2_dot * dt
        self.agent2.c = self.projective_map(self.agent2.u, self.agent2.m)

        # Update energies with exact decomposition
        phys1, model1 = self.compute_energy_components(u1_dot, m1_dot)
        phys2, model2 = self.compute_energy_components(u2_dot, m2_dot)
        
        self.agent1.E_phys += phys1 * dt
        self.agent1.E_model += model1 * dt
        self.agent1.energy += (phys1 + self.lambda1 * model1) * dt
        
        self.agent2.E_phys += phys2 * dt
        self.agent2.E_model += model2 * dt
        self.agent2.energy += (phys2 + self.lambda2 * model2) * dt

        # Update holonomies
        self.agent1.holonomy = self.agent1.z
        self.agent2.holonomy = self.agent2.z

        # Record history
        self.history['agent1'].append({
            'u': self.agent1.u.copy(),
            'c': self.agent1.c.copy(),
            'm': float(self.agent1.m),
            'z': float(self.agent1.z),
            'energy': float(self.agent1.energy),
            'holonomy': float(self.agent1.holonomy)
        })
        self.history['agent2'].append({
            'u': self.agent2.u.copy(),
            'c': self.agent2.c.copy(),
            'm': float(self.agent2.m),
            'z': float(self.agent2.z),
            'energy': float(self.agent2.energy),
            'holonomy': float(self.agent2.holonomy)
        })
        self.history['time'].append(self.history['time'][-1] + dt if self.history['time'] else 0.0)

    def reset_agents(self):
        """Reset both agents to initial configuration."""
        self.agent1 = AgentState(
            u=np.array([1.0, 0.0]), v=np.zeros(2), m=0.0, z=0.0,
            c=np.array([1.0, 0.0]), energy=0.0, holonomy=0.0,
            E_phys=0.0, E_model=0.0
        )
        self.agent2 = AgentState(
            u=np.array([-1.0, 0.0]), v=np.zeros(2), m=0.0, z=0.0,
            c=np.array([-1.0, 0.0]), energy=0.0, holonomy=0.0,
            E_phys=0.0, E_model=0.0
        )


def experiment_cooperative_formation(lambdas: List[Tuple[float, float]],
                                    T: float = 10.0, dt: float = 0.01):
    """
    Experiment 1: Cooperative formation control.
    
    Agents maintain formation while traversing a figure-8 trajectory.
    Explores how different learning costs affect coordination.
    
    Parameters
    ----------
    lambdas : List[Tuple[float, float]]
        List of (λ₁, λ₂) parameter pairs to test
    T : float
        Total simulation time
    dt : float
        Time step
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure
    results : list
        Experiment results for each parameter pair
    """
    results = []
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.25, wspace=0.2)

    for idx, (lam1, lam2) in enumerate(lambdas):
        system = MultiAgentRTAS(lambda1=lam1, lambda2=lam2, coupling_strength=0.3,info_asymmetry=False)

        t = 0.0
        while t < T:
            theta = 2 * np.pi * t / T
            # Compute desired velocity for figure-8 trajectory
            theta_dot = 2 * np.pi / T
            c_dot = np.array([np.cos(theta), 2 * np.cos(2 * theta)]) * theta_dot

            system.step(dt, c_dot, c_dot)
            t += dt

        # Compute metrics
        c1_traj = np.array([h['c'] for h in system.history['agent1']])
        c2_traj = np.array([h['c'] for h in system.history['agent2']])
        formation_error = np.mean(np.linalg.norm(c1_traj - c2_traj - np.array([1.0, 0.0]), axis=1))

        results.append({
            'lambda1': lam1,
            'lambda2': lam2,
            'total_energy': system.agent1.energy + system.agent2.energy,
            'model_divergence': abs(system.agent1.m - system.agent2.m),
            'formation_error': formation_error,
            'holonomy_1': system.agent1.holonomy,
            'holonomy_2': system.agent2.holonomy,
            'history': system.history
        })

        # Plot trajectory
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        ax.plot(c1_traj[:, 0], c1_traj[:, 1], color=COLORS['agent1'],
                label=f'Agent 1 (λ={lam1})', alpha=0.8, linewidth=2)
        ax.plot(c2_traj[:, 0], c2_traj[:, 1], color=COLORS['agent2'],
                label=f'Agent 2 (λ={lam2})', alpha=0.8, linewidth=2)
        ax.scatter(*c1_traj[0], color=COLORS['agent1'], s=60, marker='o', zorder=5)
        ax.scatter(*c1_traj[-1], color=COLORS['agent1'], s=60, marker='s', zorder=5)
        ax.scatter(*c2_traj[0], color=COLORS['agent2'], s=60, marker='o', zorder=5)
        ax.scatter(*c2_traj[-1], color=COLORS['agent2'], s=60, marker='s', zorder=5)
        
        # Label only edge subplots
        if idx >= 6:
            ax.set_xlabel('Cognitive x')
        if idx % 3 == 0:
            ax.set_ylabel('Cognitive y')
            
        ax.set_title(f'λ₁={lam1}, λ₂={lam2}\nEnergy: {results[-1]["total_energy"]:.2f}, '
                     f'ΔH₁={results[-1]["holonomy_1"]:.3f}, ΔH₂={results[-1]["holonomy_2"]:.3f}')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-1.5, 2.5)

    plt.suptitle('Experiment 1: Cooperative Formation Control', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, results


def experiment_pursuit_evasion(lambda_pursuer: float = 0.5,
                               lambda_evader: float = 2.0,
                               T: float = 20.0, dt: float = 0.01):
    """
    Experiment 2: Pursuit-evasion dynamics.
    
    Demonstrates asymmetric dynamics with information asymmetry
    where pursuer adapts to evader but not vice versa.
    
    Parameters
    ----------
    lambda_pursuer : float
        Learning cost for pursuer
    lambda_evader : float
        Learning cost for evader
    T : float
        Total simulation time
    dt : float
        Time step
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure
    results : dict
        Experiment metrics
    """
    system = MultiAgentRTAS(lambda1=lambda_pursuer, lambda2=lambda_evader,
                            coupling_strength=0.3, info_asymmetry=True)

    # Initialize agent positions
    system.agent1.u = np.array([0.0, 0.0])  # pursuer
    system.agent2.u = np.array([2.0, 2.0])  # evader
    system.agent1.c = system.projective_map(system.agent1.u, system.agent1.m)
    system.agent2.c = system.projective_map(system.agent2.u, system.agent2.m)

    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    t = 0.0
    capture_time = None
    min_distance = float('inf')

    while t < T:
        # Compute pursuit strategy
        direction = system.agent2.c - system.agent1.c
        distance = np.linalg.norm(direction)
        min_distance = min(min_distance, distance)
        if distance < 0.2 and capture_time is None:
            capture_time = t
        c1_dot = (0.5 * direction / distance) if distance > 1e-3 else np.zeros(2)

        # Compute evasion strategy
        escape = system.agent2.c - system.agent1.c
        if np.linalg.norm(escape) > 1e-3:
            escape /= np.linalg.norm(escape)
            perp = np.array([-escape[1], escape[0]])
            c2_dot = 0.4 * escape + 0.3 * perp * np.sin(5 * t)
        else:
            c2_dot = np.array([0.3, 0.3])

        system.step(dt, c1_dot, c2_dot)
        t += dt

    # Generate visualization panels
    ax1 = fig.add_subplot(gs[0, 0])
    c1_traj = np.array([h['c'] for h in system.history['agent1']])
    c2_traj = np.array([h['c'] for h in system.history['agent2']])
    ax1.plot(c1_traj[:, 0], c1_traj[:, 1], color=COLORS['agent1'], label='Pursuer', linewidth=2)
    ax1.plot(c2_traj[:, 0], c2_traj[:, 1], color=COLORS['agent2'], label='Evader', linewidth=2)
    ax1.scatter(*c1_traj[0], color=COLORS['agent1'], s=80, marker='o')
    ax1.scatter(*c2_traj[0], color=COLORS['agent2'], s=80, marker='o')
    ax1.scatter(*c1_traj[-1], color=COLORS['agent1'], s=80, marker='s')
    ax1.scatter(*c2_traj[-1], color=COLORS['agent2'], s=80, marker='s')
    if capture_time is not None:
        idx = int(capture_time / dt)
        ax1.scatter(*c1_traj[idx], color=COLORS['unstable'], s=160, marker='*',
                    label=f'Capture at t={capture_time:.2f}')
    ax1.set_xlabel('Cognitive x'); ax1.set_ylabel('Cognitive y')
    ax1.set_title('Pursuit–Evasion Trajectories'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    times = system.history['time']
    distances = [np.linalg.norm(h1['c'] - h2['c'])
                 for h1, h2 in zip(system.history['agent1'], system.history['agent2'])]
    ax2.plot(times, distances, color=COLORS['neutral'], linewidth=2)
    ax2.axhline(0.2, color=COLORS['unstable'], linestyle='--', label='Capture threshold')
    if capture_time is not None:
        ax2.axvline(capture_time, color=COLORS['unstable'], linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time'); ax2.set_ylabel('Distance'); ax2.set_title('Inter-agent Distance')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    e1 = [h['energy'] for h in system.history['agent1']]
    e2 = [h['energy'] for h in system.history['agent2']]
    ax3.plot(times, e1, color=COLORS['agent1'], label=f'Pursuer (λ={lambda_pursuer})', linewidth=2)
    ax3.plot(times, e2, color=COLORS['agent2'], label=f'Evader (λ={lambda_evader})', linewidth=2)
    ax3.set_xlabel('Time'); ax3.set_ylabel('Cumulative Energy'); ax3.set_title('Energy Expenditure')
    ax3.legend(); ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 0])
    m1 = [h['m'] for h in system.history['agent1']]
    m2 = [h['m'] for h in system.history['agent2']]
    ax4.plot(times, m1, color=COLORS['agent1'], label='Pursuer model', linewidth=2)
    ax4.plot(times, m2, color=COLORS['agent2'], label='Evader model', linewidth=2)
    ax4.set_xlabel('Time'); ax4.set_ylabel('Model parameter m'); ax4.set_title('Model Adaptation')
    ax4.legend(); ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(m1, m2, color=COLORS['coupled'], linewidth=1.5, alpha=0.8)
    ax5.scatter(m1[0], m2[0], color=COLORS['stable'], s=80, marker='o', label='Start')
    ax5.scatter(m1[-1], m2[-1], color=COLORS['unstable'], s=80, marker='s', label='End')
    ax5.set_xlabel('Pursuer m₁'); ax5.set_ylabel('Evader m₂'); ax5.set_title('Model Space Trajectory')
    ax5.legend(); ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2])
    dm1 = np.abs(np.diff(m1)); dm2 = np.abs(np.diff(m2))
    de1 = np.diff(e1); de2 = np.diff(e2)
    ax6.scatter(dm1, de1, color=COLORS['agent1'], alpha=0.5, s=10, label='Pursuer')
    ax6.scatter(dm2, de2, color=COLORS['agent2'], alpha=0.5, s=10, label='Evader')
    ax6.set_xlabel('|Δm| (rate)'); ax6.set_ylabel('ΔE'); ax6.set_title('Effort–Learning Trade-off')
    ax6.legend(); ax6.grid(True, alpha=0.3)

    plt.suptitle('Experiment 2: Pursuit–Evasion Dynamics', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig, {
        'capture_time': capture_time,
        'min_distance': min_distance,
        'pursuer_energy': e1[-1],
        'evader_energy': e2[-1],
        'model_divergence': abs(m1[-1] - m2[-1]),
        'holonomy_pursuer': system.agent1.holonomy,
        'holonomy_evader': system.agent2.holonomy
    }


def experiment_phase_diagram(lambda_range: np.ndarray = np.logspace(-1, 1, 45),
                             T: float = 15.0, dt: float = 0.01):
    """
    Experiment 3: Phase diagram in (λ₁, λ₂) parameter space.
    
    Explores stability, synchronization, and energy landscapes
    across different learning cost configurations.
    
    Parameters
    ----------
    lambda_range : np.ndarray
        Range of λ values to test
    T : float
        Total simulation time per configuration
    dt : float
        Time step
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure with multiple phase diagrams
    results : np.ndarray
        3D array of metrics for each parameter combination
    """
    results = np.zeros((len(lambda_range), len(lambda_range), 5))

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.25, wspace=0.35)

    for i, lam1 in enumerate(lambda_range):
        for j, lam2 in enumerate(lambda_range):
            system = MultiAgentRTAS(lambda1=lam1, lambda2=lam2, coupling_strength=0.3,info_asymmetry=False)

            t = 0.0
            while t < T:
                theta = 2 * np.pi * t / T
                c_dot = np.array([np.cos(theta), np.sin(theta)]) * (2 * np.pi / T)
                system.step(dt, c_dot, c_dot)
                t += dt

            total_energy = system.agent1.energy + system.agent2.energy
            m1_vals = np.array([h['m'] for h in system.history['agent1']])
            m2_vals = np.array([h['m'] for h in system.history['agent2']])
            model_div = float(np.mean(np.abs(m1_vals - m2_vals)))
            stability = float(np.var(m1_vals) + np.var(m2_vals))
            sync = float(np.corrcoef(m1_vals, m2_vals)[0, 1]) if len(m1_vals) > 1 else 0.0
            mean_holo = float((abs(system.agent1.holonomy) + abs(system.agent2.holonomy)) / 2)

            results[i, j, 0] = total_energy
            results[i, j, 1] = model_div
            results[i, j, 2] = stability
            results[i, j, 3] = sync
            results[i, j, 4] = mean_holo

    titles = ['Total Energy', 'Model Divergence', 'Stability (Variance)',
              'Synchronization', 'Mean |Δz| (Holonomy)']
    cmaps = ['viridis', 'plasma', 'RdYlBu_r', 'coolwarm', 'magma']

    for idx, (title, cmap) in enumerate(zip(titles[:5], cmaps)):
        row = idx // 4
        col = idx % 4
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(results[:, :, idx], extent=[lambda_range[0], lambda_range[-1],
                                                   lambda_range[0], lambda_range[-1]],
                       origin='lower', aspect='auto', cmap=cmap)
        ax.set_xlabel('λ₂ (Agent 2)'); ax.set_ylabel('λ₁ (Agent 1)'); ax.set_title(title)
        ax.set_xscale('log'); ax.set_yscale('log')
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        if idx == 0:
            min_idx = np.unravel_index(np.argmin(results[:, :, 0]), results[:, :, 0].shape)
            ax.scatter(lambda_range[min_idx[1]], lambda_range[min_idx[0]],
                       color='red', s=100, marker='*', label='Energy min')
            ax.legend()

    # Compute bifurcation boundaries
    ax5 = fig.add_subplot(gs[1, 1])
    grad_y, grad_x = np.gradient(results[:, :, 2])
    bifurcation = grad_x**2 + grad_y**2
    im5 = ax5.contourf(lambda_range, lambda_range, bifurcation, levels=20, cmap='hot')
    ax5.set_xlabel('λ₂ (Agent 2)'); ax5.set_ylabel('λ₁ (Agent 1)')
    ax5.set_title('Bifurcation Boundaries'); ax5.set_xscale('log'); ax5.set_yscale('log')
    plt.colorbar(im5, ax=ax5, shrink=0.8, pad=0.02)

    # Find and visualize Nash equilibria
    ax6 = fig.add_subplot(gs[1, 2])
    from scipy.ndimage import minimum_filter, gaussian_filter
    
    # Find local minima at different scales
    local_minima_3 = (results[:, :, 0] == minimum_filter(results[:, :, 0], size=3))
    local_minima_5 = (results[:, :, 0] == minimum_filter(results[:, :, 0], size=5))
    
    local_minima_5_only = local_minima_5 & ~local_minima_3
    
    # Compute gradient field
    grad_y, grad_x = np.gradient(results[:, :, 0])
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    im6 = ax6.contourf(lambda_range, lambda_range, results[:, :, 0], 
                       levels=20, cmap='viridis', alpha=0.7)
    
    ax6.contour(lambda_range, lambda_range, gradient_magnitude, 
                levels=10, colors='white', alpha=0.5, linewidths=0.5)
    
    nash_points_3 = []
    nash_points_5 = []
    
    # Plot 5x5 minima as orange circles
    for i in range(len(lambda_range)):
        for j in range(len(lambda_range)):
            if local_minima_5[i, j]:
                nash_points_5.append((lambda_range[j], lambda_range[i]))
                ax6.scatter(lambda_range[j], lambda_range[i], 
                           color='orange', s=300, marker='o', 
                           edgecolor='white', linewidth=2, zorder=8,
                           alpha=0.8)
    
    # Plot 3x3 minima as red stars on top
    for i in range(len(lambda_range)):
        for j in range(len(lambda_range)):
            if local_minima_3[i, j]:
                nash_points_3.append((lambda_range[j], lambda_range[i]))
                ax6.scatter(lambda_range[j], lambda_range[i], 
                           color='red', s=350, marker='*', 
                           edgecolor='white', linewidth=2.5, zorder=10)
    
    ax6.set_xlabel('λ₂ (Agent 2)'); ax6.set_ylabel('λ₁ (Agent 1)')
    ax6.set_title('Energy Landscape & Nash Equilibria')
    ax6.set_xscale('log'); ax6.set_yscale('log')
    
    # Adjust plot limits with margin for visibility
    if nash_points_3 or nash_points_5:
        all_nash = nash_points_3 + nash_points_5
        nash_x = [p[0] for p in all_nash]
        nash_y = [p[1] for p in all_nash]
        
        x_min, x_max = min(lambda_range), max(lambda_range)
        y_min, y_max = min(lambda_range), max(lambda_range)
        
        if any(x <= x_min * 1.1 for x in nash_x) or any(x >= x_max * 0.9 for x in nash_x):
            log_x_min = np.log10(x_min)
            log_x_max = np.log10(x_max)
            x_margin = (log_x_max - log_x_min) * 0.1
            ax6.set_xlim(10**(log_x_min - x_margin), 10**(log_x_max + x_margin))
        
        if any(y <= y_min * 1.1 for y in nash_y) or any(y >= y_max * 0.9 for y in nash_y):
            log_y_min = np.log10(y_min)
            log_y_max = np.log10(y_max)
            y_margin = (log_y_max - log_y_min) * 0.1
            ax6.set_ylim(10**(log_y_min - y_margin), 10**(log_y_max + y_margin))
    
    # Add symmetry line
    ax6.plot([lambda_range[0], lambda_range[-1]], 
             [lambda_range[0], lambda_range[-1]], 
             'k--', alpha=0.3, linewidth=1)
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
               markersize=15, markeredgecolor='white', markeredgewidth=2,
               label='Nash Eq. (3×3)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
               markersize=10, markeredgecolor='white', markeredgewidth=1.5,
               label='Nash Eq. (5×5)')
    ]
    leg = ax6.legend(handles=legend_elements, loc='upper right', fontsize=8, 
                     frameon=True, facecolor='white', edgecolor='gray')
    leg.get_frame().set_alpha(0.8)
    
    # Annotate boundary Nash points
    if nash_points_3:
        annotated_count = 0
        for i, (x, y) in enumerate(nash_points_3):
            if x <= x_min * 1.2 or x >= x_max * 0.8 or y <= y_min * 1.2 or y >= y_max * 0.8:
                if annotated_count < 2:
                    if x <= x_min * 1.2 and y >= y_max * 0.8:
                        offset_x = 0.4
                        offset_y = -0.2
                    else:
                        offset_x = 0.3 if x > (x_min * x_max)**0.5 else -0.3
                        offset_y = 0.3 if y > (y_min * y_max)**0.5 else -0.3
                    
                    ax6.annotate(f'λ=({x:.2f},{y:.2f})', 
                               xy=(x, y), 
                               xytext=(x * 10**offset_x, y * 10**offset_y),
                               arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                               fontsize=7, color='red',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                       edgecolor='red', alpha=0.8))
                    annotated_count += 1

    plt.suptitle('Experiment 3: Phase Diagram in (λ₁, λ₂) Space', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, results


def experiment_resonance_catastrophe(lambda_low: float = 0.1,
                                     coupling_strengths: List[float] = [0.0, 0.1, 0.3, 0.5,1.0,2.0],
                                     T: float = 10.0, dt: float = 0.01):
    """
    Experiment 4: Resonance catastrophe demonstration.
    
    Shows how low learning costs and strong coupling can lead to
    runaway co-adaptation and system instability.
    
    Parameters
    ----------
    lambda_low : float
        Low learning cost for both agents
    coupling_strengths : List[float]
        Different coupling strengths to test
    T : float
        Total simulation time
    dt : float
        Time step
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure
    results : list
        Catastrophe detection results
    """
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3,2, figure=fig, hspace=0.3, wspace=0.3)
    results = []

    for idx, coupling in enumerate(coupling_strengths):
        system = MultiAgentRTAS(lambda1=lambda_low, lambda2=lambda_low,
                                coupling_strength=coupling)

        t = 0.0; perturbed = False
        while t < T:
            theta = 2 * np.pi * t / T
            c_dot = np.array([np.cos(theta), np.sin(theta)]) * (2 * np.pi / T)
            if (t > 2.0) and (not perturbed):
                system.agent1.m += 0.1
                perturbed = True
            system.step(dt, c_dot, c_dot)
            t += dt

        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        times = system.history['time']
        m1 = [h['m'] for h in system.history['agent1']]
        m2 = [h['m'] for h in system.history['agent2']]
        energy = [h1['energy'] + h2['energy']
                  for h1, h2 in zip(system.history['agent1'], system.history['agent2'])]

        ax2 = ax.twinx()
        l1 = ax.plot(times, m1, color=COLORS['agent1'], label='Agent 1 model', linewidth=2)
        l2 = ax.plot(times, m2, color=COLORS['agent2'], label='Agent 2 model', linewidth=2)
        l3 = ax2.plot(times, energy, color=COLORS['unstable'], label='Total energy',
                      linewidth=2, linestyle='--', alpha=0.7)
        ax.axvline(2.0, color='gray', linestyle=':', alpha=0.6, label='Perturbation')
        ax.set_xlabel('Time'); ax.set_ylabel('m', color='black')
        ax2.set_ylabel('Total Energy', color=COLORS['unstable'])
        ax.set_title(f'Coupling strength = {coupling}')
        lns = l1 + l2 + l3; labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='lower left'); ax.grid(True, alpha=0.3)

        catastrophe = (np.max(np.abs(m1)) > 10) or (np.max(np.abs(m2)) > 10)
        if catastrophe:
            ax.text(0.5, 0.95, 'RESONANCE CATASTROPHE!',
                    transform=ax.transAxes, ha='center', va='top',
                    color=COLORS['unstable'], fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        results.append({
            'coupling': coupling,
            'max_m1': float(np.max(np.abs(m1))),
            'max_m2': float(np.max(np.abs(m2))),
            'final_energy': float(energy[-1]),
            'catastrophe': bool(catastrophe)
        })

    plt.suptitle(f'Experiment 4: Resonance Catastrophe (λ₁=λ₂={lambda_low})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, results


def main(test_mode=False):
    """Run all multi-agent rTAS experiments.
    
    Parameters
    ----------
    test_mode : bool
        If True, run with reduced parameters for quick testing.
    """
    print("=" * 70)
    print("MULTI-AGENT rTAS EXPERIMENTS")
    if test_mode:
        print("[TEST MODE ENABLED]")
    print("=" * 70)

    import os
    os.makedirs('rtas_multiagent_results', exist_ok=True)

    print("\n1. Running Cooperative Formation Control...")
    if test_mode:
        lambdas = [(0.5, 0.5), (1.0, 1.0), (2.0, 2.0)]  # Only 3 pairs for test
    else:
        lambdas = [(0.2, 0.2), (1.0, 1.0), (5.0, 5.0),
                   (0.2, 5.0), (1.0, 0.2), (5.0, 1.0),
                   (0.5, 2.0), (2.0, 0.5), (1.0, 2.0)]
    fig1, results1 = experiment_cooperative_formation(lambdas, T=5.0 if test_mode else 10.0)
    fig1.savefig('rtas_multiagent_results/exp1_cooperative_formation.png', dpi=150, bbox_inches='tight')
    print("   Results summary (first 3):")
    for r in results1[:3]:
        print(f"   λ₁={r['lambda1']}, λ₂={r['lambda2']}: "
              f"Energy={r['total_energy']:.2f}, Formation error={r['formation_error']:.3f}, "
              f"Δz₁={r['holonomy_1']:.3f}, Δz₂={r['holonomy_2']:.3f}")

    print("\n2. Running Pursuit–Evasion Dynamics...")
    pursuit_T = 10.0 if test_mode else 20.0
    fig2, results2 = experiment_pursuit_evasion(lambda_pursuer=0.3, lambda_evader=3.0, T=pursuit_T)
    fig2.savefig('rtas_multiagent_results/exp2_pursuit_evasion.png', dpi=150, bbox_inches='tight')
    print("   Results:")
    if results2['capture_time'] is not None:
        print(f"   Capture time: {results2['capture_time']:.2f}s")
    else:
        print(f"   No capture (min distance: {results2['min_distance']:.3f})")
    print(f"   Pursuer energy: {results2['pursuer_energy']:.2f}")
    print(f"   Evader energy: {results2['evader_energy']:.2f}")
    print(f"   Holonomies: Δz_pursuer={results2['holonomy_pursuer']:.3f}, "
          f"Δz_evader={results2['holonomy_evader']:.3f}")

    print("\n3. Computing Phase Diagram...")
    lambda_range = np.logspace(-0.5, 0.7, 10 if test_mode else 25)
    phase_T = 5.0 if test_mode else 15.0
    fig3, results3 = experiment_phase_diagram(lambda_range, T=phase_T)
    fig3.savefig('rtas_multiagent_results/exp3_phase_diagram.png', dpi=150, bbox_inches='tight')
    min_idx = np.unravel_index(np.argmin(results3[:, :, 0]), results3[:, :, 0].shape)
    opt_lam1 = float(lambda_range[min_idx[0]])
    opt_lam2 = float(lambda_range[min_idx[1]])
    print(f"   Optimal configuration: λ₁={opt_lam1:.3f}, λ₂={opt_lam2:.3f}")
    print(f"   Minimum total energy: {results3[min_idx[0], min_idx[1], 0]:.2f}")

    print("\n4. Testing Resonance Catastrophe...")
    res_couplings = [0.0, 0.5, 1.0] if test_mode else [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
    res_T = 5.0 if test_mode else 10.0
    fig4, results4 = experiment_resonance_catastrophe(lambda_low=0.1, coupling_strengths=res_couplings, T=res_T)
    fig4.savefig('rtas_multiagent_results/exp4_resonance_catastrophe.png', dpi=150, bbox_inches='tight')
    print("   Catastrophe detection:")
    for r in results4:
        status = "CATASTROPHE!" if r['catastrophe'] else "Stable"
        print(f"   Coupling={r['coupling']}: {status} (max |m|={max(r['max_m1'], r['max_m2']):.2f})")

    print("\n5. Running Phase Transition Analysis...")
    print("   This analysis computes min/max energy vs coupling strength")
    print("   and identifies the critical coupling where argmin jumps.")
    if test_mode:
        # Test mode parameters are set inside the function
        fig5, results5 = experiment_phase_transition_analysis(test_mode=True)
    else:
        coupling_range = np.logspace(-2, 0.5, 50)
        lambda_range_transition = np.logspace(-1, 1, 50)
        fig5, results5 = experiment_phase_transition_analysis(
            coupling_range=coupling_range,
            lambda_range=lambda_range_transition,
            T=10.0, dt=0.01, refine_size=5
        )
    fig5.savefig('rtas_multiagent_results/exp5_phase_transition_analysis.png', dpi=200, bbox_inches='tight')
    print(f"\n   Critical coupling κc = {results5['kappa_critical']:.3f}")
    print(f"   Energy range: [{np.min(results5['E_min']):.2f}, {np.max(results5['E_max']):.2f}]")
    print(f"   Optimal λ before transition: λ1*≈{np.mean(results5['argmin_lambda1'][:results5['critical_idx']]):.2f}, "
          f"λ2*≈{np.mean(results5['argmin_lambda2'][:results5['critical_idx']]):.2f}")
    print(f"   Optimal λ after transition: λ1*≈{np.mean(results5['argmin_lambda1'][results5['critical_idx']+1:]):.2f}, "
          f"λ2*≈{np.mean(results5['argmin_lambda2'][results5['critical_idx']+1:]):.2f}")

    print("\n" + "=" * 70)
    print("All experiments completed. Results saved to 'rtas_multiagent_results/'.")
    print("=" * 70)

    plt.show()
    return {
        'cooperative': results1,
        'pursuit': results2,
        'phase_diagram': results3,
        'resonance': results4,
        'phase_transition': results5
    }

def experiment_phase_transition_analysis(coupling_range: np.ndarray = np.logspace(-2, 0.5, 50),
                                         lambda_range: np.ndarray = np.logspace(-1, 1, 50),
                                         T: float = 10.0, dt: float = 0.01,
                                         refine_size: int = 5, seed: Optional[int] = None,
                                         test_mode: bool = False):
    """
    Phase transition analysis for the policy-induced energy landscape under the
    instantaneous reflective split (block pseudoinverse controller).

    Computes E_min/E_max vs coupling, argmin/argmax trajectories, a scale-invariant
    jump metric to estimate critical coupling, and exact energy decomposition at E_min.

    Parameters
    ----------
    coupling_range : np.ndarray
        Log-spaced coupling strengths κ to scan.
    lambda_range : np.ndarray
        Log-spaced λ values to scan for each agent.
    T : float
        Total simulation time (fixed across all runs; one period for the circle).
    dt : float
        Time step.
    refine_size : int
        Local refinement grid size around the best coarse point (per κ).
    seed : int | None
        Random seed (if the dynamics/noise requires it).
    test_mode : bool
        If True, use reduced grids for quick testing.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure with multiple panels
    analysis : dict
        Dict with arrays for E_min/E_max, argmin/argmax locations, κ_c, and decomposed energies.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Override parameters for test mode
    if test_mode:
        print("\n[TEST MODE] Using reduced parameters for quick testing...")
        coupling_range = np.logspace(-1, 0.3, 5)  # Only 5 coupling values
        lambda_range = np.logspace(-0.5, 0.5, 8)  # Only 8x8 lambda grid
        T = 2.0  # Shorter simulation time
        refine_size = 3  # Smaller refinement
        print(f"Test mode: {len(coupling_range)} κ values, {len(lambda_range)}×{len(lambda_range)} λ grid")

    print(f"\nPhase Transition Analysis: {len(coupling_range)} κ × {len(lambda_range)}² λ grid")
    print(f"Initial grid per κ: {len(lambda_range)}×{len(lambda_range)} = {len(lambda_range)**2}; "
          f"refinement: {refine_size}×{refine_size}")
    
    # Storage
    nK = len(coupling_range)
    E_min = np.zeros(nK)
    E_max = np.zeros(nK)
    argmin_lambda1 = np.zeros(nK)
    argmin_lambda2 = np.zeros(nK)
    argmax_lambda1 = np.zeros(nK)
    argmax_lambda2 = np.zeros(nK)
    E_phys_at_min = np.zeros(nK)  # exact ∑ ∫||u̇||²
    E_model_at_min = np.zeros(nK)  # exact ∑ ∫||ṁ||² (unweighted)
    E_model1_at_min = np.zeros(nK)  # ∫||ṁ1||²
    E_model2_at_min = np.zeros(nK)  # ∫||ṁ2||²
    
    # Helper to run a single simulation and return total energy and exact split
    def run_once(lam1, lam2, kappa):
        system = MultiAgentRTAS(lambda1=lam1, lambda2=lam2,
                                coupling_strength=kappa, info_asymmetry=False)
        # simulate exactly one circular loop over [0, T]
        t = 0.0
        omega = 2 * np.pi / T
        while t < T - 1e-12:
            theta = omega * t
            c_dot = np.array([np.cos(theta), np.sin(theta)]) * omega
            system.step(dt, c_dot, c_dot)  # policy computes reflective split internally
            t += dt
        
        # Extract exact decomposition from agent states
        E1_phys = system.agent1.E_phys
        E1_model = system.agent1.E_model
        E2_phys = system.agent2.E_phys
        E2_model = system.agent2.E_model
        
        # policy energy (matches the paper: ∫||u̇||² + λ∫||ṁ||² for each agent)
        E_total = (E1_phys + lam1 * E1_model) + (E2_phys + lam2 * E2_model)
        return E_total, (E1_phys + E2_phys), (E1_model + E2_model), E1_model, E2_model
    
    # Main sweep
    for i, kappa in enumerate(coupling_range):
        print(f"\rProcessing κ = {kappa:.4g}  ({i+1}/{nK})", end='', flush=True)
        energies = np.empty((len(lambda_range), len(lambda_range)))
        energies[:] = np.nan

        # Coarse grid
        for j, lam1 in enumerate(lambda_range):
            for k, lam2 in enumerate(lambda_range):
                E_tot, _, _, _, _ = run_once(lam1, lam2, kappa)
                energies[j, k] = E_tot
        
        # Coarse min/max
        min_idx = np.unravel_index(np.nanargmin(energies), energies.shape)
        max_idx = np.unravel_index(np.nanargmax(energies), energies.shape)

        # Local log-refinement around minimum
        if refine_size > 1:
            j0 = min_idx[0]
            k0 = min_idx[1]
            j_lo = max(0, j0 - 1)
            j_hi = min(len(lambda_range) - 1, j0 + 1)
            k_lo = max(0, k0 - 1)
            k_hi = min(len(lambda_range) - 1, k0 + 1)

            lam1_ref = np.logspace(np.log10(lambda_range[j_lo]),
                                   np.log10(lambda_range[j_hi]), refine_size)
            lam2_ref = np.logspace(np.log10(lambda_range[k_lo]),
                                   np.log10(lambda_range[k_hi]), refine_size)

            best_E = energies[min_idx]
            best_l1 = lambda_range[j0]
            best_l2 = lambda_range[k0]
            best_phys = np.nan
            best_model = np.nan
            best_E1m = np.nan
            best_E2m = np.nan

            for l1 in lam1_ref:
                for l2 in lam2_ref:
                    E_tot, E_phys_sum, E_model_sum, E1m, E2m = run_once(l1, l2, kappa)
                    if E_tot < best_E:
                        best_E = E_tot
                        best_l1 = l1
                        best_l2 = l2
                        best_phys = E_phys_sum
                        best_model = E_model_sum
                        best_E1m = E1m
                        best_E2m = E2m

            E_min[i] = best_E
            argmin_lambda1[i] = best_l1
            argmin_lambda2[i] = best_l2
            # exact decomposition at min
            if not np.isnan(best_phys):
                E_phys_at_min[i] = best_phys
                E_model_at_min[i] = best_model
                E_model1_at_min[i] = best_E1m
                E_model2_at_min[i] = best_E2m
            else:
                # Use coarse min decomposition if refinement did not improve
                E_tot, E_phys_sum, E_model_sum, E1m, E2m = run_once(best_l1, best_l2, kappa)
                E_phys_at_min[i] = E_phys_sum
                E_model_at_min[i] = E_model_sum
                E_model1_at_min[i] = E1m
                E_model2_at_min[i] = E2m
        else:
            E_min[i] = energies[min_idx]
            argmin_lambda1[i] = lambda_range[min_idx[0]]
            argmin_lambda2[i] = lambda_range[min_idx[1]]
            # decomposition at coarse min
            _, E_phys_sum, E_model_sum, E1m, E2m = run_once(argmin_lambda1[i], argmin_lambda2[i], kappa)
            E_phys_at_min[i] = E_phys_sum
            E_model_at_min[i] = E_model_sum
            E_model1_at_min[i] = E1m
            E_model2_at_min[i] = E2m

        E_max[i] = energies[max_idx]
        argmax_lambda1[i] = lambda_range[max_idx[0]]
        argmax_lambda2[i] = lambda_range[max_idx[1]]
    
    print("\nAnalysis complete.")
    
    # Scale-invariant jump detection in log-space
    eps = 1e-12
    d1 = np.diff(np.log10(argmin_lambda1 + eps))
    d2 = np.diff(np.log10(argmin_lambda2 + eps))
    argmin_jumps = np.sqrt(d1**2 + d2**2)
    critical_idx = int(np.argmax(argmin_jumps))
    kappa_critical = np.sqrt(coupling_range[critical_idx] * coupling_range[critical_idx + 1])
    
    # Sanity check (a): Decomposition closure
    # Compute residual: r(κ) = E_min(κ) - (E_phys + λ1* E_model1 + λ2* E_model2)
    residuals = E_min - (E_phys_at_min + argmin_lambda1 * E_model1_at_min + argmin_lambda2 * E_model2_at_min)
    print(f"\nSanity check (a): Decomposition closure")
    print(f"  Max absolute residual: {np.max(np.abs(residuals)):.2e}")
    print(f"  Mean absolute residual: {np.mean(np.abs(residuals)):.2e}")
    
    # Sanity check (b): Boundary refinement at the jump
    if critical_idx > 0 and critical_idx < len(coupling_range) - 2:
        print(f"\nSanity check (b): Boundary refinement around κ_c = {kappa_critical:.4g}")
        
        # Extended lambda range (1.5x in both directions)
        lambda_ext_min = lambda_range[0] / 1.5
        lambda_ext_max = lambda_range[-1] * 1.5
        lambda_extended = np.logspace(np.log10(lambda_ext_min), np.log10(lambda_ext_max), 
                                      len(lambda_range) + 10)
        
        # Check 3 kappa values around critical point
        kappa_check_indices = [max(0, critical_idx-1), critical_idx, min(len(coupling_range)-1, critical_idx+2)]
        for idx in kappa_check_indices:
            kappa = coupling_range[idx]
            print(f"\n  Checking κ = {kappa:.4g} (index {idx}):")
            
            # Run extended grid search
            energies_ext = np.empty((len(lambda_extended), len(lambda_extended)))
            energies_ext[:] = np.nan
            
            for j, lam1 in enumerate(lambda_extended[:10]):  # Quick check on subset
                for k, lam2 in enumerate(lambda_extended[:10]):
                    E_tot, _, _, _, _ = run_once(lam1, lam2, kappa)
                    energies_ext[j, k] = E_tot
            
            min_idx_ext = np.unravel_index(np.nanargmin(energies_ext[:10, :10]), (10, 10))
            lambda1_ext_min = lambda_extended[min_idx_ext[0]]
            lambda2_ext_min = lambda_extended[min_idx_ext[1]]
            
            print(f"    Original grid: λ1* = {argmin_lambda1[idx]:.4g}, λ2* = {argmin_lambda2[idx]:.4g}")
            print(f"    Extended grid: λ1* = {lambda1_ext_min:.4g}, λ2* = {lambda2_ext_min:.4g}")
            
            # Check if minimum is at boundary
            at_boundary = (argmin_lambda1[idx] == lambda_range[0] or 
                          argmin_lambda1[idx] == lambda_range[-1] or
                          argmin_lambda2[idx] == lambda_range[0] or 
                          argmin_lambda2[idx] == lambda_range[-1])
            print(f"    At original grid boundary: {at_boundary}")
    
    print("\nGenerating figures...")
    
    # Create comprehensive figure with better spacing
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)
    
    # Panel 1: Min/Max energy vs coupling
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(coupling_range, E_min, color=COLORS['stable'], linewidth=2.5, 
             label='$E_{\\min}(\\kappa)$', marker='o', markersize=4)
    ax1.plot(coupling_range, E_max, color=COLORS['unstable'], linewidth=2.5, 
             label='$E_{\\max}(\\kappa)$', marker='s', markersize=4)
    ax1.axvline(kappa_critical, color='gray', linestyle='--', alpha=0.7, 
                label=f'$\\kappa_c = {kappa_critical:.3f}$')
    ax1.fill_between(coupling_range, E_min, E_max, alpha=0.1, color='gray')
    ax1.set_xlabel('Coupling strength $\\kappa$')
    ax1.set_ylabel('Total energy $E$')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=np.min(E_min)*0.8)      # > 0 for log scale
    ax1.yaxis.set_major_formatter(mpl.ticker.LogFormatterSciNotation())
    ax1.set_title('Energy Landscape: Min/Max vs Coupling')
    ax1.legend(loc='center left', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Add inset to show E_min dynamic range (positioned next to legend)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # First draw the plot to get legend position
    fig.canvas.draw()
    # Get legend bbox in axis coordinates
    legend = ax1.get_legend()
    legend_bbox = legend.get_window_extent().transformed(ax1.transAxes.inverted())
    # Position inset to the right of legend with small margin
    inset_x = legend_bbox.x1 + 0.1  # right edge of legend + margin
    inset_y = legend_bbox.y0 + (legend_bbox.height - 0.3) / 2  # center vertically with legend
    
    ax1_inset = inset_axes(ax1, width="30%", height="30%", loc='lower left', 
                           bbox_to_anchor=(inset_x, inset_y, 1, 1), bbox_transform=ax1.transAxes)
    ax1_inset.plot(coupling_range, E_min, color=COLORS['stable'], linewidth=2, 
                   marker='o', markersize=3)
    ax1_inset.axvline(kappa_critical, color='gray', linestyle='--', alpha=0.7)
    ax1_inset.set_xlabel('$\\kappa$', fontsize=8)
    ax1_inset.set_ylabel('$E_{\\min}$', fontsize=8)
    ax1_inset.set_xscale('log')
    ax1_inset.tick_params(labelsize=7)
    ax1_inset.grid(True, alpha=0.3)
    ax1_inset.set_title('$E_{\\min}$ detail', fontsize=8)
    # Add a background to make it more visible
    ax1_inset.patch.set_facecolor('white')
    ax1_inset.patch.set_alpha(0.95)
    ax1_inset.patch.set_edgecolor('gray')
    ax1_inset.patch.set_linewidth(1)
    
    # Panel 2: Argmin trajectories
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(coupling_range, argmin_lambda1, color=COLORS['agent1'], 
             linewidth=2, label='$\\lambda_1^*(\\kappa)$', marker='o', markersize=3)
    ax2.plot(coupling_range, argmin_lambda2, color=COLORS['agent2'], 
             linewidth=2, label='$\\lambda_2^*(\\kappa)$', marker='o', markersize=3)
    ax2.axvline(kappa_critical, color='gray', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Coupling strength $\\kappa$')
    ax2.set_ylabel('Optimal $\\lambda_i$')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title('Argmin: Optimal Learning Costs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Argmax trajectories
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(coupling_range, argmax_lambda1, color=COLORS['agent1'], 
             linewidth=2, label='$\\lambda_1^\\dagger(\\kappa)$', marker='s', markersize=3)
    ax3.plot(coupling_range, argmax_lambda2, color=COLORS['agent2'], 
             linewidth=2, label='$\\lambda_2^\\dagger(\\kappa)$', marker='s', markersize=3)
    ax3.axvline(kappa_critical, color='gray', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Coupling strength $\\kappa$')
    ax3.set_ylabel('Worst-case $\\lambda_i$')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_title('Argmax: Worst-Case Learning Costs')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Energy decomposition at optimal point
    ax4 = fig.add_subplot(gs[1, 2])
    # Physical component is just E_phys_at_min
    ax4.plot(coupling_range, E_phys_at_min, color=COLORS['agent1'], 
             linewidth=2, label='Physical: $\\sum_i \\int ||\\dot{u}_i||^2 dt$', marker='o', markersize=3)
    
    # Unweighted model component
    ax4.plot(coupling_range, E_model_at_min, color=COLORS['coupled'], 
             linewidth=2, label='Model (unweighted): $\\sum_i \\int ||\\dot{m}_i||^2 dt$', 
             marker='s', markersize=3, linestyle='--', alpha=0.7)
    
    # Weighted model component - exact contribution to E_min
    E_model_weighted = argmin_lambda1 * E_model1_at_min + argmin_lambda2 * E_model2_at_min
    
    ax4.plot(coupling_range, E_model_weighted, color=COLORS['coupled'], 
             linewidth=2.5, label='Model (weighted): $\\sum_i \\lambda_i^*(\\kappa) \\int ||\\dot{m}_i||^2 dt$', 
             marker='o', markersize=4)
    
    # Add residual as thin gray line (should be near zero)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(coupling_range, residuals, color='gray', linewidth=1, alpha=0.5, 
                  label='Residual $r(\\kappa)$', linestyle=':')
    ax4_twin.set_ylabel('Residual', color='gray', fontsize=9)
    ax4_twin.tick_params(axis='y', labelcolor='gray', labelsize=8)
    ax4_twin.set_ylim(-1e-10, 1e-10)  # Show near machine precision
    
    ax4.axvline(kappa_critical, color='gray', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Coupling strength $\\kappa$')
    ax4.set_ylabel('Energy components')
    ax4.set_xscale('log')
    ax4.set_title('Exact Energy Decomposition at $E_{\\min}$')
    ax4.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Cross-sections at representative κ values
    ax5 = fig.add_subplot(gs[2, 0])
    kappa_vals = [coupling_range[0], kappa_critical, coupling_range[-1]]
    kappa_labels = ['Low $\\kappa$', '$\\kappa_c$', 'High $\\kappa$']
    colors = [COLORS['stable'], COLORS['coupled'], COLORS['unstable']]
    
    for kappa, label, color in zip(kappa_vals, kappa_labels, colors):
        # Find closest κ in our range
        idx = np.argmin(np.abs(coupling_range - kappa))
        energies_slice = np.zeros(len(lambda_range))
        
        # Fix λ2 at its optimal value, vary λ1
        fixed_lam2 = argmin_lambda2[idx]
        for j, lam1 in enumerate(lambda_range):
            E_tot, _, _, _, _ = run_once(lam1, fixed_lam2, coupling_range[idx])
            energies_slice[j] = E_tot
        
        ax5.plot(lambda_range, energies_slice, color=color, linewidth=2, label=label)
    
    ax5.set_xlabel('$\\lambda_1$ (with $\\lambda_2$ fixed at optimal)')
    ax5.set_ylabel('Total energy $E$')
    ax5.set_xscale('log')
    ax5.set_title('Energy Cross-Sections')
    ax5.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Phase diagram in (λ1, λ2) space with coupling as parameter
    ax6 = fig.add_subplot(gs[2, 1])
    scatter = ax6.scatter(argmin_lambda1, argmin_lambda2, c=coupling_range, 
                          cmap='viridis', s=50, alpha=0.8, norm=mpl.colors.LogNorm())
    ax6.plot(argmin_lambda1, argmin_lambda2, 'k-', linewidth=1, alpha=0.5)
    ax6.scatter(argmin_lambda1[critical_idx], argmin_lambda2[critical_idx], 
                color='red', s=200, marker='*', edgecolor='white', linewidth=2,
                label=f'Critical point', zorder=10)
    ax6.set_xlabel('$\\lambda_1^*(\\kappa)$')
    ax6.set_ylabel('$\\lambda_2^*(\\kappa)$')
    ax6.set_xscale('log')
    ax6.set_yscale('log')
    ax6.set_title('Optimal Path in $(\\lambda_1, \\lambda_2)$ Space')
    cbar = plt.colorbar(scatter, ax=ax6, pad=0.1)
    cbar.set_label('Coupling $\\kappa$', rotation=270, labelpad=15)
    ax6.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)
    ax6.grid(True, alpha=0.3)
    
    # Panel 7: Jump magnitude analysis
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.plot(coupling_range[:-1], argmin_jumps, color=COLORS['neutral'], 
             linewidth=2, marker='o', markersize=3)
    ax7.axvline(kappa_critical, color='red', linestyle='--', linewidth=2,
                label=f'Max jump at $\\kappa_c$')
    ax7.set_xlabel('Coupling strength $\\kappa$')
    ax7.set_ylabel('Jump magnitude $||\\Delta\\lambda^*||$')
    ax7.set_xscale('log')
    ax7.set_title('Argmin Jump Detection')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Summary statistics box
    ax8 = fig.add_subplot(gs[0, 2])
    ax8.axis('off')
    summary_text = f"""Phase Transition Analysis Summary
(Policy-induced landscape under reflective split)
    
Critical coupling: κc = {kappa_critical:.3f}
Jump magnitude: {argmin_jumps[critical_idx]:.3f}

Before transition (κ < κc):
  λ1* ≈ {np.mean(argmin_lambda1[:critical_idx]):.2f} ± {np.std(argmin_lambda1[:critical_idx]):.2f}
  λ2* ≈ {np.mean(argmin_lambda2[:critical_idx]):.2f} ± {np.std(argmin_lambda2[:critical_idx]):.2f}
  
After transition (κ > κc):
  λ1* ≈ {np.mean(argmin_lambda1[critical_idx+1:]):.2f} ± {np.std(argmin_lambda1[critical_idx+1:]):.2f}
  λ2* ≈ {np.mean(argmin_lambda2[critical_idx+1:]):.2f} ± {np.std(argmin_lambda2[critical_idx+1:]):.2f}

Energy range:
  Min: {np.min(E_min):.2f} at κ = {coupling_range[np.argmin(E_min)]:.3f}
  Max: {np.max(E_max):.2f} at κ = {coupling_range[np.argmax(E_max)]:.3f}
  
Grid: {len(lambda_range)}×{len(lambda_range)} initial
      {refine_size}×{refine_size} refinement"""
    
    ax8.text(0.02, 0.98, summary_text, transform=ax8.transAxes, 
             fontsize=8.5, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8, edgecolor='gray'))
    
    plt.suptitle('Phase Transition Analysis: Multi-Agent rTAS', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Prepare analysis results
    analysis = {
        'coupling_range': coupling_range,
        'lambda_range': lambda_range,
        'E_min': E_min,
        'E_max': E_max,
        'argmin_lambda1': argmin_lambda1,
        'argmin_lambda2': argmin_lambda2,
        'argmax_lambda1': argmax_lambda1,
        'argmax_lambda2': argmax_lambda2,
        'kappa_critical': kappa_critical,
        'critical_idx': critical_idx,
        'E_phys_at_min': E_phys_at_min,
        'E_model_at_min': E_model_at_min,
        'E_model1_at_min': E_model1_at_min,
        'E_model2_at_min': E_model2_at_min,
        'residuals': residuals,
        'argmin_jumps': argmin_jumps
    }
    
    return fig, analysis


if __name__ == "__main__":
    import sys
    # Check for test mode flag
    test_mode = '--test' in sys.argv or '-t' in sys.argv
    results = main(test_mode=test_mode)
