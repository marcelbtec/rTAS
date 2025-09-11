"""
Reflective Tangential Action Spaces (rTAS) - Figure Generation

Publication-ready code for reproducing figures from the rTAS manuscript.

This script generates publication-quality figures demonstrating the key concepts
of Reflective Tangential Action Spaces (rTAS), including:

1. Reflective metric lift and the effort-learning trade-off
2. Block curvature and cross-coupling effects
3. Quadratic energy law and cost-aware navigation
4. Projective and connection channels

Generated figures:
  - fig_rtas1_reflective_lift.png: Demonstrates the reflective metric lift
  - fig_rtas2_block_curvature.png: Shows block curvature effects
  - fig_rtas3_cost_frontier_and_policy.png: Illustrates cost-memory frontier
  - fig_rtasA_projective_channel.png: Projective channel analysis
  - fig_rtasB_connection_channel.png: Connection channel behavior

Key concepts:
  - Reflective metric lift: Optimal lift minimizing combined physical-model cost
  - Block metric: Ĝ = G ⊕ λH where λ controls the learning cost
  - Instantaneous optimality: arg min ||u̇||²_G + λ||ṁ||²_H subject to DΦ_m(p)u̇ + B(p,m)ṁ = ċ
  - Cross-curvature: Coupling between physical and model degrees of freedom

Usage:
  python rtas_showcase_beautiful2.py

Requirements:
  - numpy
  - matplotlib
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec

# Professional color palette for consistent styling
COLORS = {
    'primary': '#2E86AB',      # Deep blue
    'secondary': '#A23B72',     # Rose
    'tertiary': '#F18F01',      # Orange
    'quaternary': '#C73E1D',    # Red-orange
    'success': '#52B788',       # Green
    'warning': '#F77F00',       # Bright orange
    'info': '#6C757D',         # Gray
    'light': '#F8F9FA',        # Light gray
    'dark': '#212529',         # Dark gray
    'background': '#FFFFFF',    # White
    'grid': '#E9ECEF'          # Light grid
}

def rtas_rcparams():
    """Apply beautiful, consistent styling"""
    mpl.rcParams.update({
        # Font settings
        "font.family": "sans-serif",
        "font.sans-serif": ['Helvetica', 'Arial', 'DejaVu Sans'],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        
        # Figure settings
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "figure.facecolor": COLORS['background'],
        "savefig.facecolor": COLORS['background'],
        "savefig.edgecolor": 'none',
        
        # Axes settings
        "axes.linewidth": 1.0,
        "axes.edgecolor": COLORS['info'],
        "axes.facecolor": COLORS['background'],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        
        # Grid settings
        "grid.alpha": 0.2,
        "grid.linestyle": '-',
        "grid.linewidth": 0.5,
        "grid.color": COLORS['grid'],
        
        # Line settings
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        
        # Patch settings
        "patch.linewidth": 1.0,
        "patch.edgecolor": COLORS['dark'],
    })

def panel(ax, letter):
    """
    Add a panel letter to a subplot.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add the panel letter to
    letter : str
        The letter to display (e.g., 'a', 'b', 'c')
    """
    ax.text(0.02, 0.98, f"({letter})", transform=ax.transAxes,
            va="top", ha="left", weight="bold", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                     edgecolor=COLORS['info'], alpha=0.8))

def circle_in_C(R=1.0, n=800, t0=0.0, t1=2*np.pi, center=(0.0,0.0)):
    """
    Generate a uniform-speed circular trajectory in cognitive space C.
    
    Parameters
    ----------
    R : float
        Radius of the circle
    n : int
        Number of sample points
    t0, t1 : float
        Start and end parameter values
    center : tuple
        Center coordinates (cx, cy)
        
    Returns
    -------
    t : array
        Time/parameter values
    c1, c2 : array
        Trajectory coordinates
    dc1, dc2 : array
        Velocity components
    dt : float
        Time step
    """
    t = np.linspace(t0, t1, n, endpoint=True)
    cx, cy = center
    c1 = cx + R*np.cos(t)
    c2 = cy + R*np.sin(t)
    dt = t[1]-t[0]
    dc1 = np.gradient(c1, dt)
    dc2 = np.gradient(c2, dt)
    return t, c1, c2, dc1, dc2, dt

def trapz_int(y, dt):
    """Compute trapezoidal integral of y with uniform spacing dt."""
    return float(np.trapezoid(y, dx=dt))

def ensure_outdir(path="rtas_figures"):
    """Create output directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def reflective_lift_projective(c1, c2, dc1, dc2, lam, dt, 
                               m0=0.0, u0=None, v0=None, 
                               enforce_projection=False, return_mdot=False):
    """
    Compute the reflective metric lift along a given cognitive trajectory.
    
    This implements the optimal lift that minimizes the combined cost:
    E = ∫(||u̇||²_G + λ||ṁ||²_H)dt subject to DΦ_m(p)u̇ + B(p,m)ṁ = ċ
    
    For the projective channel: Φ_m(u,v) = (u, v + m sin u)
    
    Parameters
    ----------
    c1, c2 : array
        Cognitive trajectory coordinates
    dc1, dc2 : array
        Cognitive velocity components
    lam : float
        Learning cost parameter λ
    dt : float
        Time step
    m0 : float
        Initial model parameter
    u0, v0 : float, optional
        Initial physical coordinates
    enforce_projection : bool
        If True, enforce discrete projection consistency
    return_mdot : bool
        If True, also return ṁ time series
        
    Returns
    -------
    u, v : array
        Physical trajectory
    m : array
        Model parameter trajectory
    e_phys, e_model : array
        Instantaneous physical and model energy densities
    dt : float
        Time step
    mdot : array (if return_mdot=True)
        Model velocity ṁ
    """
    n = len(c1)
    # Initialize u, v to match C via Φ_m(u,v) = (u, v + m sin u)
    u = np.zeros(n); v = np.zeros(n); m = np.zeros(n)
    mdot_series = np.zeros(n)
    if u0 is None:
        u0 = c1[0]
    if v0 is None:
        # pick v0 so that c2[0] = v0 + m0 sin(u0)
        v0 = c2[0] - m0*np.sin(u0)

    u[0], v[0], m[0] = u0, v0, m0

    # Energy (instantaneous, per step)
    e_phys = np.zeros(n)  # ||(u̇,v̇)||^2
    e_model = np.zeros(n) # ||ṁ||^2

    for k in range(n-1):
        uk, mk = u[k], m[k]
        c_dot = np.array([dc1[k], dc2[k]])  # desired cognitive velocity at step k

        # Build constraint matrix A = [DΦ_m | B] and block metric inverse
        DPhi = np.array([[1.0, 0.0],
                         [mk*np.cos(uk), 1.0]])
        B = np.array([[0.0],
                      [np.sin(uk)]])
        A = np.hstack([DPhi, B])
        Ghat_inv = np.diag([1.0, 1.0, 1.0/lam])

        # Apply reflective metric lift formula
        M = A @ Ghat_inv @ A.T
        w = Ghat_inv @ A.T @ np.linalg.inv(M) @ c_dot
        ud, vd, md = w

        # Energies
        e_phys[k]  = ud*ud + vd*vd
        e_model[k] = md*md
        mdot_series[k] = md

        # Euler integration step
        u[k+1] = u[k] + ud*dt
        v[k+1] = v[k] + vd*dt
        m[k+1] = m[k] + md*dt

        if enforce_projection:
            # Enforce projection constraint
            u[k+1] = c1[k+1]
            v[k+1] = c2[k+1] - m[k+1]*np.sin(u[k+1])

    if return_mdot:
        mdot_series[-1] = mdot_series[-2]
        return u, v, m, e_phys, e_model, dt, mdot_series
    else:
        return u, v, m, e_phys, e_model, dt


def gamma_xy(x, y):
    """
    Coupling function for cross-curvature demonstration.
    Using γ = r² gives area-proportional holonomy.
    """
    return x*x + y*y

def cross_curvature_holonomy_case_a(R, lam, n=800):
    """
    Demonstrate cross-curvature effects: F_pp = 0, F_pm ≠ 0.
    
    Physical holonomy arises through model coupling via ż = γ(u,v)ṁ.
    
    Parameters
    ----------
    R : float
        Loop radius
    lam : float
        Learning cost parameter
    n : int
        Number of sample points
        
    Returns
    -------
    dz : float
        Physical holonomy
    dm : float
        Model holonomy
    area : float
        Enclosed area
    E : float
        Total energy cost
    """
    t, c1, c2, dc1, dc2, dt = circle_in_C(R=R, n=n)
    u, v, m, e_phys, e_model, _, mdot = reflective_lift_projective(c1, c2, dc1, dc2, lam=lam, dt=dt, return_mdot=True, enforce_projection=False)

    # Compute cross-channel coupling
    zdot = gamma_xy(u, v) * mdot
    z = np.cumsum(zdot) * dt
    # Compute holonomies
    dz = z[-1] - z[0]
    dm = m[-1] - m[0]
    area = np.pi * R*R

    # Total reflective energy
    E = trapz_int(e_phys + lam*e_model, dt)
    return dz, dm, area, E

def pure_meta_holonomy_case_b(R, delta=0.3, n=800):
    """
    Demonstrate pure meta-holonomy: F_mm ≠ 0, no physical coupling.
    Model holonomy proportional to enclosed area.
    """
    t = np.linspace(0, 2*np.pi, n, endpoint=True)
    theta_dot = 1.0
    r2 = R*R
    mdot = delta * r2
    m = np.cumsum(mdot) * (t[1]-t[0])
    dm = m[-1] - m[0]
    dz = 0.0
    area = np.pi * R*R
    return dz, dm, area


def grad_V_ring(s, R0=1.2, k=1.0):
    """
    Ring attractor potential gradient.
    
    Creates an attractive potential around a ring of radius R0:
    V(s) = 0.5 * k * (||s|| - R0)²
    
    Parameters
    ----------
    s : array
        Position in I-space
    R0 : float
        Target ring radius
    k : float
        Potential strength
    """
    x, y = s
    r = np.hypot(x, y)
    if r < 1e-9:
        return np.array([0.0, 0.0])
    return k * (r - R0) * np.array([x, y]) / r

def E_proxy_cost(s, hill_center=(0.0,0.0), hill_rad=0.7, amp=2.5):
    """
    Synthetic cost field for visualization.
    
    Creates a Gaussian cost hill to demonstrate cost-aware navigation.
    """
    x, y = s
    cx, cy = hill_center
    rr = (x-cx)**2 + (y-cy)**2
    return amp * np.exp(-rr/(2*hill_rad**2))

def grad_E_proxy(s, hill_center=(0.0,0.0), hill_rad=0.7, amp=2.5):
    x, y = s; cx, cy = hill_center
    dx, dy = x-cx, y-cy
    base = E_proxy_cost(s, hill_center, hill_rad, amp)
    g = - base * np.array([dx, dy]) / (hill_rad**2)
    return g

def integrate_I_pushback(s0, etaP=0.8, T=25.0, dt=0.02, R0=1.2, hill_center=(0.0,0.0), hill_rad=0.7, amp=2.5):
    """
    Simulate cost-aware goal dynamics in intentional space.
    
    Integrates: ṡ = -∇V(s) - η_P ∇E_proxy(s)
    
    Parameters
    ----------
    s0 : array
        Initial position
    etaP : float
        Cost awareness parameter (0 = goal-only)
    T : float
        Total simulation time
    dt : float
        Time step
    R0 : float
        Goal ring radius
    hill_center : tuple
        Center of cost hill
    hill_rad : float
        Cost hill radius
    amp : float
        Cost hill amplitude
        
    Returns
    -------
    S : array
        Trajectory in I-space
    """
    steps = int(T/dt)+1
    S = np.zeros((steps, 2))
    S[0] = np.array(s0, dtype=float)
    for k in range(steps-1):
        gV = grad_V_ring(S[k], R0=R0, k=1.0)
        gE = grad_E_proxy(S[k], hill_center=hill_center, hill_rad=hill_rad, amp=amp)
        sdot = -gV - etaP*gE
        S[k+1] = S[k] + dt*sdot
    return S


def fig_rtas1_reflective_lift(outdir):
    """
    Generate Figure rTAS-1: Reflective metric lift demonstration.
    
    Shows how the reflective metric lift trades off physical effort
    and model learning cost as λ varies.
    
    Parameters
    ----------
    outdir : str
        Output directory for saving the figure
    """
    rtas_rcparams()
    fig = plt.figure(figsize=(12, 8), facecolor=COLORS['background'])
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1,1.1,1.0], height_ratios=[1,1], 
                          wspace=0.35, hspace=0.45)

    # One agent, one closed loop in C; vary λ
    R = 0.9
    t, c1, c2, dc1, dc2, dt = circle_in_C(R=R, n=900)

    lambdas = [0.2, 1.0, 1e6]
    cols = [COLORS['primary'], COLORS['tertiary'], COLORS['success']]
    curves = []

    # (a) trajectories in (u,m)
    axA = fig.add_subplot(gs[0,0])
    for lam, col in zip(lambdas, cols):
        u, v, m, e_phys, e_model, _ = reflective_lift_projective(c1, c2, dc1, dc2, lam=lam, dt=dt, enforce_projection=False)
        axA.plot(u, m, lw=2.5, color=col, label=f"λ={('∞' if lam>1e3 else lam)}", alpha=0.9)
        curves.append((u, v, m, e_phys, e_model, lam))
    axA.set_title("Reflective metric lift: trajectories in (u, m)", fontsize=12, fontweight='bold')
    axA.set_xlabel("u", fontsize=10)
    axA.set_ylabel("m", fontsize=10)
    axA.legend(loc="best", frameon=True, fancybox=True, shadow=True)
    axA.grid(True, alpha=0.2)
    panel(axA, "a")

    # (b) instantaneous split ||u̇||_G^2 vs λ||ṁ||_H^2 (show λ=1 as representative)
    axB = fig.add_subplot(gs[0,1])
    # pick λ=1
    for (u, v, m, e_phys, e_model, lam) in curves:
        if 0.8 < lam < 1.2:
            axB.plot(np.linspace(0,1,len(u)), e_phys, lw=2, color=COLORS['primary'], 
                    label=r"$\|\dot u\|_G^2$", alpha=0.9)
            axB.plot(np.linspace(0,1,len(u)), lam*e_model, lw=2, color=COLORS['secondary'],
                    label=r"$\lambda\|\dot m\|_H^2$", alpha=0.9)
            break
    axB.set_title("Instantaneous effort split (λ=1)", fontsize=12, fontweight='bold')
    axB.set_xlabel("loop parameter", fontsize=10)
    axB.set_ylabel("power", fontsize=10)
    axB.legend(frameon=True, fancybox=True, shadow=True)
    axB.grid(True, alpha=0.2)
    panel(axB, "b")

    # (c) total cost Ê vs λ  (sweep λ)
    axC = fig.add_subplot(gs[0,2])
    lam_scan = np.geomspace(0.1, 20.0, 30)
    E_tot = []
    for lam in lam_scan:
        u, v, m, e_phys, e_model, _ = reflective_lift_projective(c1, c2, dc1, dc2, lam=lam, dt=dt, enforce_projection=False)
        E_tot.append(trapz_int(e_phys + lam*e_model, dt))
    axC.plot(lam_scan, E_tot, "o-", lw=2, markersize=6, color=COLORS['quaternary'], alpha=0.9)
    axC.fill_between(lam_scan, min(E_tot), E_tot, alpha=0.2, color=COLORS['quaternary'])
    axC.set_xscale("log")
    axC.set_title("Total reflective cost $\\hat{E}$ vs $\\lambda$\n(Cost decreases with λ: agent avoids building shear)", fontsize=11, fontweight='bold')
    axC.set_xlabel("$\\lambda$", fontsize=10)
    axC.set_ylabel("$\\hat{E}$", fontsize=10)
    axC.grid(True, alpha=0.2, which='both')
    panel(axC, "c")

    # (d) TAS limit (λ→∞) recovery: m ~ const, visible lift equals TAS (geometric)
    axD = fig.add_subplot(gs[1,0])
    (u_inf, v_inf, m_inf, _, _, lam_inf) = curves[-1]
    axD.plot(np.linspace(0,1,len(m_inf)), m_inf - m_inf[0], lw=2.5, color=COLORS['success'], alpha=0.9)
    axD.axhline(0, color=COLORS['dark'], lw=1, linestyle='--', alpha=0.5)
    axD.set_title("TAS limit ($\\lambda\\to\\infty$): $m(t)$ frozen", fontsize=12, fontweight='bold')
    axD.set_xlabel("loop parameter", fontsize=10)
    axD.set_ylabel("$m(t)-m(0)$", fontsize=10)
    axD.grid(True, alpha=0.2)
    panel(axD, "d")

    # (e) visible closing vs λ
    axE = fig.add_subplot(gs[1,1])
    closes = []
    for (u, v, m, e_phys, e_model, lam) in curves:
        closes.append( np.hypot(u[-1]-u[0], v[-1]-v[0]) )
    bars = axE.bar([r"$0.2$", r"$1$", r"$\infty$"], closes, color=cols, alpha=0.9, edgecolor='white', linewidth=2)
    for bar, val in zip(bars, closes):
        axE.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    axE.set_title("Visible path closure error vs $\\lambda$", fontsize=12, fontweight='bold')
    axE.set_ylabel("end-to-start distance in $(u,v)$", fontsize=10)
    axE.grid(True, alpha=0.2, axis='y')
    panel(axE, "e")

    fig.suptitle("rTAS‑1 — Reflective metric lift and effort–learning trade", 
                 fontsize=14, fontweight='bold', y=0.96)
    fig.tight_layout(rect=[0,0,1,0.94])
    fig.savefig(os.path.join(outdir, "fig_rtas1_reflective_lift.png"))
    plt.close(fig)

def fig_rtas2_block_curvature(outdir):
    """
    Generate Figure rTAS-2: Block curvature demonstration.
    
    Illustrates cross-curvature effects between physical and model
    degrees of freedom through holonomy analysis.
    
    Parameters
    ----------
    outdir : str
        Output directory for saving the figure
    """
    rtas_rcparams()
    fig = plt.figure(figsize=(12, 8), facecolor=COLORS['background'])
    gs = fig.add_gridspec(2, 3, width_ratios=[1,1,1], height_ratios=[1,1], 
                          wspace=0.35, hspace=0.45)

    # (a) Case F_pp=0, F_pm≠0  — physical holonomy via model coupling
    axA = fig.add_subplot(gs[0,0])
    Rs = np.linspace(0.15, 0.8, 12)
    lam = 0.8
    dzs = []; dms = []; As = []
    for R in Rs:
        dz, dm, A, E = cross_curvature_holonomy_case_a(R, lam=lam, n=800)
        dzs.append(dz); dms.append(dm); As.append(A)
    axA.plot(As, dzs, "o-", lw=2, markersize=7, color=COLORS['primary'], 
            label=r"$\Delta z$ (physical)", alpha=0.9)
    axA.plot(As, dms, "s--", lw=2, markersize=6, color=COLORS['secondary'],
            label=r"$\Delta m$ (meta)", alpha=0.9)
    axA.set_title("Cross-coupled holonomy\n$F_{pp}=0, F_{pm}≠0$", fontsize=12, fontweight='bold')
    axA.set_xlabel("enclosed area in $C$", fontsize=10)
    axA.set_ylabel("holonomy", fontsize=10)
    axA.legend(frameon=True, fancybox=True, shadow=True)
    axA.grid(True, alpha=0.2)
    panel(axA, "a")

    # (b) Case F_mm≠0 — pure meta-holonomy
    axB = fig.add_subplot(gs[0,1])
    dzsB = []; dmsB = []; AsB = []
    for R in Rs:
        dz, dm, A = pure_meta_holonomy_case_b(R, delta=0.3, n=600)
        dzsB.append(dz); dmsB.append(dm); AsB.append(A)
    axB.plot(AsB, dmsB, "o-", lw=2, markersize=7, color=COLORS['tertiary'],
            label=r"$\Delta m$ (meta)", alpha=0.9)
    axB.axhline(0, color=COLORS['dark'], lw=1, linestyle='--', alpha=0.5)
    axB.set_title("Pure meta-holonomy\n$F_{mm}≠0$", fontsize=12, fontweight='bold')
    axB.set_xlabel("enclosed area in $C$", fontsize=10)
    axB.set_ylabel("holonomy", fontsize=10)
    axB.legend(frameon=True, fancybox=True, shadow=True)
    axB.grid(True, alpha=0.2)
    panel(axB, "b")

    # (c) Linear fits: (Δu_phys, Δm_model) vs area
    axC = fig.add_subplot(gs[0,2])
    # simple linear fits
    A = np.array(As);  Z = np.array(dzs);  M = np.array(dms)
    zfit = np.polyfit(A, Z, 1); mfit = np.polyfit(A, M, 1)
    Aplot = np.linspace(min(A), max(A), 200)
    axC.scatter(A, Z, s=40, color=COLORS['primary'], alpha=0.7, label=r"$\Delta z_{\rm phys}$ data")
    axC.plot(Aplot, np.polyval(zfit, Aplot), "-", color=COLORS['primary'], lw=2, alpha=0.9, label="linear fit")
    axC.scatter(A, M, s=40, marker='s', color=COLORS['secondary'], alpha=0.7, label=r"$\Delta m_{\rm model}$ data")
    axC.plot(Aplot, np.polyval(mfit, Aplot), "--", color=COLORS['secondary'], lw=2, alpha=0.9, label="linear fit (meta)")
    axC.set_title("Linear regime\n$(Δz_{phys}, Δm_{model})$ vs area", fontsize=12, fontweight='bold')
    axC.set_xlabel("area", fontsize=10)
    axC.set_ylabel("holonomy", fontsize=10)
    axC.legend(frameon=True, fancybox=True, shadow=True, loc='upper left')
    axC.grid(True, alpha=0.2)
    panel(axC, "c")

    # (d) Block curvature visualization
    axD = fig.add_subplot(gs[1,0])
    axD.axis("off")
    # Create a visual matrix representation
    from matplotlib.patches import Rectangle, FancyBboxPatch
    
    # Draw matrix background
    matrix_bg = FancyBboxPatch((0.15, 0.3), 0.7, 0.5,
                              boxstyle="round,pad=0.02",
                              facecolor=COLORS['light'],
                              edgecolor=COLORS['dark'],
                              linewidth=2)
    axD.add_patch(matrix_bg)
    
    axD.text(0.5, 0.85, "Block Curvature $\\hat{F}$", 
            ha='center', fontsize=13, fontweight='bold')
    
    # Matrix elements
    axD.text(0.35, 0.65, "$F_{pp}$", ha='center', fontsize=11, color=COLORS['primary'])
    axD.text(0.65, 0.65, "$F_{pm}$", ha='center', fontsize=11, color=COLORS['secondary'])
    axD.text(0.35, 0.45, "$F_{mp}$", ha='center', fontsize=11, color=COLORS['secondary'])
    axD.text(0.65, 0.45, "$F_{mm}$", ha='center', fontsize=11, color=COLORS['tertiary'])
    
    # Add grid lines
    axD.plot([0.5, 0.5], [0.35, 0.75], 'k-', lw=1, alpha=0.3)
    axD.plot([0.2, 0.8], [0.55, 0.55], 'k-', lw=1, alpha=0.3)
    
    axD.text(0.5, 0.2, "Physical ↔ Model coupling", 
            ha='center', fontsize=10, style='italic', color=COLORS['info'])
    
    panel(axD, "d")

    # (e) Holonomy interpretation
    axE = fig.add_subplot(gs[1,1])
    axE.axis("off")
    
    # Create info panel
    info_bg = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                             boxstyle="round,pad=0.03",
                             facecolor='white',
                             edgecolor=COLORS['info'],
                             linewidth=1.5,
                             alpha=0.95)
    axE.add_patch(info_bg)
    
    axE.text(0.5, 0.75, "Holonomy Interpretation", 
            ha='center', fontsize=11, fontweight='bold')
    
    text_lines = [
        "• $F_{pp}=0$: No pure physical curvature",
        "• $F_{pm}≠0$: Cross-coupling generates $Δz$",
        "• $F_{mm}≠0$: Pure model holonomy",
        "• Linear area-holonomy in small loops"
    ]
    
    y_pos = 0.55
    for line in text_lines:
        axE.text(0.1, y_pos, line, fontsize=9, color=COLORS['dark'])
        y_pos -= 0.12
    
    panel(axE, "e")

    # (f) Identifiability note
    axF = fig.add_subplot(gs[1,2])
    axF.axis("off")
    
    # Create highlight box
    highlight_bg = FancyBboxPatch((0.05, 0.2), 0.9, 0.6,
                                  boxstyle="round,pad=0.03",
                                  facecolor=COLORS['warning'],
                                  edgecolor=COLORS['quaternary'],
                                  linewidth=2,
                                  alpha=0.1)
    axF.add_patch(highlight_bg)
    
    axF.text(0.5, 0.65, "Key Insight", 
            ha='center', fontsize=11, fontweight='bold', color=COLORS['quaternary'])
    
    axF.text(0.5, 0.45, "Observing $Δz$ with $F_{pp}≈0$\ndistinguishes rTAS from TAS\n(cross-terms active)",
            ha='center', va='center', fontsize=9, color=COLORS['dark'])
    
    panel(axF, "f")

    fig.suptitle("rTAS‑2 — Cross‑curvature holonomy (block curvature)", 
                 fontsize=14, fontweight='bold', y=0.96)
    fig.tight_layout(rect=[0,0,1,0.94])
    fig.savefig(os.path.join(outdir, "fig_rtas2_block_curvature.png"))
    plt.close(fig)

def _project_Q_to_psd(a, b, c, eps=1e-12):
    """Project symmetric 2x2 [[a,b],[b,c]] to the PSD cone."""
    S = np.array([[a, b], [b, c]], dtype=float)
    w, V = np.linalg.eigh(S)
    w_clipped = np.clip(w, eps, None)
    S_psd = (V * w_clipped) @ V.T
    return float(S_psd[0,0]), float(S_psd[0,1]), float(S_psd[1,1])

def _weighted_least_squares(X, y, w):
    """Solve (X^T W X) beta = X^T W y with W=diag(w). Returns beta."""
    Xw = X * w[:, None]
    yw = y * w
    beta, *_ = np.linalg.lstsq(Xw.T @ X, Xw.T @ y, rcond=None)
    return beta

def fit_quadratic_form_Q(dz, dm, E, areas=None, small_pct=25, use_weights=True, force_psd=True):
    """
    Fit E ≈ β0 + [dz dm]ᵀ Q [dz dm] in the small-loop regime.
    Returns dict with Q, β0, predictions, masks and diagnostics.
    """
    dz = np.asarray(dz).ravel(); dm = np.asarray(dm).ravel(); E = np.asarray(E).ravel()
    if areas is None:
        areas = np.sqrt(dz**2 + dm**2)  # harmless fallback
    areas = np.asarray(areas).ravel()

    # 1) small-loop selection
    thr = np.percentile(areas, small_pct)
    mask_small = areas <= thr
    z = dz[mask_small]; m = dm[mask_small]; Em = E[mask_small]; A = areas[mask_small]

    # 2) design with intercept and the exact quadratic monomials
    #    E ≈ β0 + a*z^2 + 2*b*z*m + c*m^2
    X = np.column_stack([np.ones_like(z), z*z, 2*z*m, m*m])

    # 3) weights (downweight larger loops even inside the small set)
    if use_weights:
        w = np.exp(-(A / (thr + 1e-12))**2)  # ∈(0,1], peaked at the tiniest loops
    else:
        w = np.ones_like(z)

    # 4) weighted LS
    beta = _weighted_least_squares(X, Em, w)
    beta0, a, b, c = beta.tolist()

    # 5) enforce PSD if requested; re-fit β0 (baseline) with Q fixed
    if force_psd:
        a, b, c = _project_Q_to_psd(a, b, c)
        quad = a*z*z + 2*b*z*m + c*m*m
        # Weighted mean residual → best intercept for fixed Q
        beta0 = (w * (Em - quad)).sum() / (w.sum() + 1e-12)

    # 6) predictions and "excess" (subtract baseline)
    E_pred = beta0 + a*z*z + 2*b*z*m + c*m*m
    E_excess = Em - beta0
    E_pred_excess = E_pred - beta0  # by construction: same as a*z^2+2*b*zm+c*m^2

    # 7) stats
    resid = Em - E_pred
    ss_res = float((w * resid**2).sum())
    ss_tot = float((w * (Em - (w*Em).sum()/w.sum())**2).sum())
    R2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    slope, intercept = np.polyfit(E_pred, Em, 1)  # purely for the diag overlay

    return {
        "mask_small": mask_small, "area_threshold": thr,
        "beta0": beta0, "Q": np.array([[a,b],[b,c]], dtype=float),
        "E_small": Em, "E_pred_small": E_pred,
        "E_excess": E_excess, "E_pred_excess": E_pred_excess,
        "R2": R2, "slope": slope, "intercept": intercept
    }

def baseline_energy_for_R(R, n=700):
    """
    Physical baseline for a loop of radius R: energy with model frozen (m constant).
    This is the pure physical cost of traversing the circle.
    """
    t, c1, c2, dc1, dc2, dt = circle_in_C(R=R, n=n)
    # For baseline: just compute physical velocity squared
    e_phys_baseline = dc1**2 + dc2**2  # ||ċ||²
    return trapz_int(e_phys_baseline, dt), dt

def _weighted_ls_with_intercept(z, m, y, w):
    """
    Weighted least squares for y ≈ d + a z^2 + 2 b z m + c m^2 (with intercept).
    Returns (d, a, b, c).
    """
    X = np.column_stack([np.ones_like(z), z*z, 2*z*m, m*m])
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return beta  # d, a, b, c

def fit_quadratic_form_Q_excess(dz, dm, E_total, areas,
                                small_pct=30, use_weights=True, force_psd=True):
    """
    Fit E_total ≈ E_0 + [Δz, Δm]^T Q [Δz, Δm] in the small-loop regime.
    With intercept E_0. Returns dict with Q, baseline, predictions, and diagnostics.
    """
    dz = np.asarray(dz).ravel()
    dm = np.asarray(dm).ravel()
    y  = np.asarray(E_total).ravel()
    A  = np.asarray(areas).ravel()

    thr = np.percentile(A, small_pct)
    mask = A <= thr

    z = dz[mask]; m = dm[mask]; y_small = y[mask]; A_small = A[mask]

    # weights: emphasise tiniest loops inside the small-loop set
    if use_weights:
        w = np.exp(-(A_small / (thr + 1e-12))**2)
    else:
        w = np.ones_like(y_small)

    # weighted LS with intercept
    d, a, b, c = _weighted_ls_with_intercept(z, m, y_small, w)

    if force_psd:
        a, b, c = _project_Q_to_psd(a, b, c)
        # Refit intercept after PSD projection
        quad = a*z*z + 2*b*z*m + c*m*m
        d = (w * (y_small - quad)).sum() / (w.sum() + 1e-12)

    # Predictions include the intercept
    quad = a*z*z + 2*b*z*m + c*m*m
    yhat = d + quad

    # goodness-of-fit on in-sample small loops
    ss_res = float(np.sum(w * (y_small - yhat)**2))
    mu = float(np.sum(w*y_small) / (np.sum(w)+1e-12))
    ss_tot = float(np.sum(w * (y_small - mu)**2))
    R2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    # slope of y vs yhat for the diag overlay (target ~ 1)
    slope, intercept = np.polyfit(yhat, y_small, 1)

    return {
        "mask_small": mask, "area_threshold": thr,
        "Q": np.array([[a,b],[b,c]], float),
        "baseline": d,
        "y_small": y_small, "yhat_small": yhat,
        "R2": R2, "slope": slope, "intercept": intercept
    }

def fig_rtas3_cost_frontier_and_policy(outdir):
    """
    Generate Figure rTAS-3: Cost-memory frontier and policy.
    
    Demonstrates:
    (a) Quadratic energy law in small-loop regime
    (b) Cost-aware navigation avoiding expensive regions
    (c) 'Lazy but effective' strategy using multiple small loops
    
    Parameters
    ----------
    outdir : str
        Output directory for saving the figure
    """
    rtas_rcparams()
    fig = plt.figure(figsize=(15, 5.5), facecolor=COLORS['background'])
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3,1,1], 
                          wspace=0.3, hspace=0.35)

    # (a) Quadratic law (fixed λ or λ-normalised)
    axA = fig.add_subplot(gs[0,0])

    # Fix lambda for cleanest visual
    lams = [1.0]
    Rs = np.geomspace(0.05, 0.9, 30)  # more radii, starting from smaller loops

    rows = []
    # precompute per-R physical baseline (TAS limit) once
    baseline_by_R = {}
    for R in Rs:
        E_base_R, _ = baseline_energy_for_R(R, n=700)
        baseline_by_R[R] = E_base_R

    for R in Rs:
        for lam in lams:
            dz, dm, A, E_tot = cross_curvature_holonomy_case_a(R, lam, n=700)
            # Don't subtract baseline - fit total energy directly
            rows.append((dz, dm, A, E_tot, R, lam))

    rows = np.array(rows, float)
    dz_all, dm_all, area_all, E_all, R_all, lam_all = rows.T

    # Fit quadratic form on the total energy in the small-loop regime
    fit = fit_quadratic_form_Q_excess(dz_all, dm_all, E_all,
                                      areas=area_all, small_pct=40,
                                      use_weights=True, force_psd=True)

    mask_small = fit["mask_small"]
    thr_area   = fit["area_threshold"]
    Q          = fit["Q"]
    baseline   = fit["baseline"]
    y_s        = fit["y_small"]        # measured energy
    yhat_s     = fit["yhat_small"]     # predicted energy
    R2         = fit["R2"]
    slope      = fit["slope"]
    intercept  = fit["intercept"]

    # Plot excess energy (subtract baseline to see quadratic nature)
    y_excess = y_s - baseline
    yhat_excess = yhat_s - baseline
    
    # colour/size by r in (Δz,Δm)
    r_small = np.sqrt(dz_all[mask_small]**2 + dm_all[mask_small]**2)
    sc = axA.scatter(yhat_excess, y_excess,
                     s=30 + 120*(r_small/(r_small.max()+1e-12))**2,
                     c=r_small, cmap='viridis', alpha=0.85,
                     edgecolors='white', linewidth=0.5)

    # identity + diag fit overlays
    lo = min(0, min(y_excess.min(), yhat_excess.min()) * 1.1)
    hi = 1.05*max(y_excess.max(), yhat_excess.max())
    axA.plot([lo, hi], [lo, hi], '--', color=COLORS['dark'], lw=1.8, alpha=0.6, label='y=x')
    
    # Fit line for excess energy
    slope_excess, intercept_excess = np.polyfit(yhat_excess, y_excess, 1)
    axA.plot([lo, hi], [intercept_excess + slope_excess*lo, intercept_excess + slope_excess*hi],
             '-', color=COLORS['info'], lw=1.5, alpha=0.85,
             label=f'fit: y={slope_excess:.3f}x+{intercept_excess:.3f}')

    a,b,c = Q[0,0], Q[0,1], Q[1,1]
    txt = (f"Small loops (A ≤ {thr_area:.3f})\n"
           f"$R^2={R2:.3f}$"
           f"baseline $E_0$={baseline:.3f}")
    axA.text(0.2, 0.97, txt, transform=axA.transAxes, va='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='white',
                       edgecolor=COLORS['info'], alpha=0.95), fontsize=9)

    cbar = plt.colorbar(sc, ax=axA, shrink=0.82, pad=0.02)
    cbar.set_label(r"distance $r=\sqrt{\Delta z^2+\Delta m^2}$", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # grey reference: larger loops (not used for the fit)
    if np.any(~mask_small):
        yL_excess = E_all[~mask_small] - baseline
        yhat_L_excess = (Q[0,0]*dz_all[~mask_small]**2
                        + 2*Q[0,1]*dz_all[~mask_small]*dm_all[~mask_small]
                        + Q[1,1]*dm_all[~mask_small]**2)
        axA.scatter(yhat_L_excess, yL_excess, s=22, c='lightgray', alpha=0.35, label='larger loops (ref.)')

    axA.set_title("Quadratic law (small loops)\n$E_{\\rm excess} = E - E_0 = [\\Delta z,\\Delta m]\\,Q\\,[\\Delta z,\\Delta m]^\\top$",
                  fontsize=12, fontweight='bold')
    axA.set_xlabel("$E_{\\rm excess}$ (predicted)", fontsize=11)
    axA.set_ylabel("$E_{\\rm excess}$ (measured)", fontsize=11)
    axA.set_xlim(lo, hi); axA.set_ylim(lo, hi)
    axA.legend(loc='lower right', frameon=True, fancybox=True, shadow=False)
    axA.grid(True, alpha=0.35)

    panel(axA, "a")

    # (b) Trajectories under cost-aware goal dynamics in I (pushback)
    axB = fig.add_subplot(gs[0,1])
    # Start from the side to better show avoidance behavior
    s0 = (-2.2, 0.0)
    # Moderate cost awareness for visible effect
    S_lazy  = integrate_I_pushback(s0, etaP=0.5, T=100.0, dt=0.02, R0=1.2, hill_center=(0.0,0.0), hill_rad=0.6, amp=2.0)
    S_plain = integrate_I_pushback(s0, etaP=0.0, T=100.0, dt=0.02, R0=1.2, hill_center=(0.0,0.0), hill_rad=0.6, amp=2.0)
    
    # Create cost field background
    x = np.linspace(-2.5, 2.5, 100)
    y = np.linspace(-2.5, 2.5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Cost function visualization (gaussian hill)
    hill_x, hill_y = 0.0, 0.0
    hill_rad = 0.6
    cost_field = 2.0 * np.exp(-((X - hill_x)**2 + (Y - hill_y)**2) / (2 * hill_rad**2))
    
    # Show cost as contour fill
    contf = axB.contourf(X, Y, cost_field, levels=15, cmap='Reds', alpha=0.3)
    
    # Draw high-cost zone boundary
    th = np.linspace(0, 2*np.pi, 200)
    axB.plot(hill_rad*np.cos(th), hill_rad*np.sin(th), 
             color=COLORS['quaternary'], linewidth=2, linestyle=':', alpha=0.8,
             label="high-cost boundary")
    
    # Plot trajectories with arrows
    axB.plot(S_plain[:,0], S_plain[:,1], lw=2.5, color=COLORS['info'], 
            label="goal-only", alpha=0.8, linestyle='--')
    axB.plot(S_lazy[:,0],  S_lazy[:,1],  lw=3, color=COLORS['success'], 
            label="cost-aware", alpha=0.9)
    
    # Add direction arrows on trajectories
    n_arrows = 4
    for i in range(n_arrows):
        idx_plain = int((i+1) * len(S_plain) / (n_arrows+1))
        idx_lazy = int((i+1) * len(S_lazy) / (n_arrows+1))
        
        # Arrow for goal-only
        if idx_plain < len(S_plain)-1:
            dx_plain = S_plain[idx_plain+1,0] - S_plain[idx_plain,0]
            dy_plain = S_plain[idx_plain+1,1] - S_plain[idx_plain,1]
            axB.arrow(S_plain[idx_plain,0], S_plain[idx_plain,1], 
                     dx_plain*5, dy_plain*5, head_width=0.08, head_length=0.06,
                     fc=COLORS['info'], ec=COLORS['info'], alpha=0.6)
        
        # Arrow for cost-aware
        if idx_lazy < len(S_lazy)-1:
            dx_lazy = S_lazy[idx_lazy+1,0] - S_lazy[idx_lazy,0]
            dy_lazy = S_lazy[idx_lazy+1,1] - S_lazy[idx_lazy,1]
            axB.arrow(S_lazy[idx_lazy,0], S_lazy[idx_lazy,1], 
                     dx_lazy*5, dy_lazy*5, head_width=0.08, head_length=0.06,
                     fc=COLORS['success'], ec=COLORS['success'], alpha=0.8)
    
    # Add start and goal markers with labels
    axB.scatter(s0[0], s0[1], s=150, color=COLORS['primary'], marker='o', 
               edgecolor='white', linewidth=2, zorder=5)
    axB.text(s0[0], s0[1]+0.15, 'Start', fontsize=9, fontweight='bold', ha='center', va='bottom')
    
    # Goal position (center of ring attractor)
    goal_pos = (0, 0)
    circle_goal = plt.Circle(goal_pos, 1.2, fill=False, edgecolor=COLORS['dark'], 
                            linewidth=2, linestyle='--', alpha=0.5)
    axB.add_patch(circle_goal)
    axB.text(0, 1.4, 'Goal ring', fontsize=9, ha='center', alpha=0.7)
    
    # Add cost annotations
    axB.text(0, 0, 'High\ncost', fontsize=9, ha='center', va='center', 
             fontweight='bold', color='darkred', alpha=0.8)
    
    # Calculate and display path costs
    # Simple approximation: length * average cost along path
    len_plain = np.sum(np.sqrt(np.diff(S_plain[:,0])**2 + np.diff(S_plain[:,1])**2))
    len_lazy = np.sum(np.sqrt(np.diff(S_lazy[:,0])**2 + np.diff(S_lazy[:,1])**2))
    
    axB.text(0.02, 0.02, f'Path lengths:\nGoal-only: {len_plain:.1f}\nCost-aware: {len_lazy:.1f}', 
             transform=axB.transAxes, fontsize=8, va='bottom',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    axB.set_aspect("equal", adjustable="box")
    axB.set_title("Cost-aware navigation in I-space", fontsize=11, fontweight='bold')
    axB.set_xlabel("$s_1$ (intentional coordinate)", fontsize=10)
    axB.set_ylabel("$s_2$ (intentional coordinate)", fontsize=10)
    axB.legend(frameon=True, fancybox=True, shadow=False, loc='upper right', fontsize=9)
    axB.grid(True, alpha=0.2)
    axB.set_xlim(-2.5, 2.5)
    axB.set_ylim(-2.5, 2.5)
    
    # Add subtle colorbar for cost field
    cbar = plt.colorbar(contf, ax=axB, shrink=0.6, pad=0.02)
    cbar.set_label('Cost field', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    
    panel(axB, "b")

    # (c) One large loop vs many small loops — same 'coverage', lower Ê
    axC = fig.add_subplot(gs[0,2])
    # Compare: one circle of radius R_big vs four smaller circles of radius R_small = R_big/2
    # (equal covered area measure ~ total swept area)
    R_big = 1.2
    R_small = R_big/2.0
    # one large
    t1, c1_big, c2_big, dc1_big, dc2_big, dt1 = circle_in_C(R=R_big, n=800)
    u,v,m,e_phys,e_model,_,mdot_big = reflective_lift_projective(c1_big, c2_big, dc1_big, dc2_big, lam=1.0, dt=dt1, return_mdot=True, enforce_projection=False)
    E_big = 1.0 * trapz_int(mdot_big**2, dt1)
    # four small
    E_small_total = 0.0
    for _ in range(4):
        t, c1, c2, dc1, dc2, dt2 = circle_in_C(R=R_small, n=800)
        u,v,m,e_phys,e_model,_,mdot_small = reflective_lift_projective(c1, c2, dc1, dc2, lam=1.0, dt=dt2, return_mdot=True, enforce_projection=False)
        E_small_total += 1.0 * trapz_int(mdot_small**2, dt2)
    
    bars = axC.bar(["1× large", "4× small"], [E_big, E_small_total], 
                   color=[COLORS['primary'], COLORS['success']], alpha=0.9,
                   edgecolor='white', linewidth=2)
    
    # Add value labels on bars
    for bar, val in zip(bars, [E_big, E_small_total]):
        axC.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    axC.set_title("Same coverage, lower cost\n'Lazy but effective'", fontsize=11, fontweight='bold')
    axC.set_ylabel(r"$\lambda \int \|\dot m\|^2\,dt$", fontsize=10)
    axC.grid(True, alpha=0.2, axis='y')
    panel(axC, "c")
    
    # Add caption clarification
    axC.text(0.5, -0.25, r"(c) Same coverage at lower model-update cost, $\lambda\int\|\dot m\|^2 dt$," + "\nby using several small loops ('lazy but effective')",
            transform=axC.transAxes, ha='center', va='top', fontsize=9, color=COLORS['dark'])

    fig.suptitle("rTAS‑3 — Cost–memory frontier and 'lazy but effective' control", 
                 fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0,0.02,1,0.95])
    fig.savefig(os.path.join(outdir, "fig_rtas3_cost_frontier_and_policy.png"))
    plt.close(fig)

def fig_rtasA_projective_channel(outdir):
    """
    Generate Figure rTAS-A: Projective channel analysis.
    
    Detailed analysis of the projective channel Φ_m(u,v) = (u, v + m sin u)
    showing energy decomposition and λ-dependence.
    
    Parameters
    ----------
    outdir : str
        Output directory for saving the figure
    """
    rtas_rcparams()
    fig = plt.figure(figsize=(12, 7), facecolor=COLORS['background'])
    gs = fig.add_gridspec(2, 3, wspace=0.35, hspace=0.45)

    R = 1.0
    t, c1, c2, dc1, dc2, dt = circle_in_C(R=R, n=900)
    lamlist = [0.2, 1.0, 5.0]
    cols = [COLORS['primary'], COLORS['tertiary'], COLORS['success']]

    # (a) visible path closes; (u,v) vs C
    axA = fig.add_subplot(gs[0,0])
    for lam, col in zip(lamlist, cols):
        u, v, m, e_phys, e_model, _ = reflective_lift_projective(c1, c2, dc1, dc2, lam=lam, dt=dt, enforce_projection=False)
        axA.plot(u, v, lw=2.5, color=col, label=f"λ={lam}", alpha=0.9)
        axA.scatter([u[0]],[v[0]], s=60, color=col, edgecolor='white', linewidth=2, zorder=5)
        axA.scatter([u[-1]],[v[-1]], s=60, color=col, marker="s", edgecolor='white', linewidth=2, zorder=5)
    axA.set_aspect("equal", adjustable="box")
    axA.set_title("Projective channel\nVisible path", fontsize=12, fontweight='bold')
    axA.set_xlabel("u", fontsize=10)
    axA.set_ylabel("v", fontsize=10)
    axA.legend(frameon=True, fancybox=True, shadow=True)
    axA.grid(True, alpha=0.2)
    panel(axA, "a")

    # (b) v̇*, ṁ* vs λ (time-averaged RMS)
    axB = fig.add_subplot(gs[0,1])
    rms_vdot = []; rms_mdot = []
    for lam in lamlist:
        u,v,m,e_phys,e_model,_,mdot = reflective_lift_projective(c1, c2, dc1, dc2, lam=lam, dt=dt, return_mdot=True, enforce_projection=False)
        vdot = np.gradient(v, dt)
        rms_vdot.append(np.sqrt(np.mean(vdot**2)))
        rms_mdot.append(np.sqrt(np.mean(mdot**2)))
    axB.plot(lamlist, rms_vdot, "o-", lw=2, markersize=8, color=COLORS['primary'],
            label=r"RMS $\dot v^\star$", alpha=0.9)
    axB.plot(lamlist, rms_mdot, "s--", lw=2, markersize=7, color=COLORS['secondary'],
            label=r"RMS $\dot m^\star$", alpha=0.9)
    axB.set_title("Instantaneous components\nvs $\\lambda$", fontsize=12, fontweight='bold')
    axB.set_xlabel("$\\lambda$", fontsize=10)
    axB.set_ylabel("RMS velocity", fontsize=10)
    axB.legend(frameon=True, fancybox=True, shadow=True)
    axB.grid(True, alpha=0.2)
    

    
    panel(axB, "b")

    # (c) Model trajectory visualization
    axC = fig.add_subplot(gs[0,2])
    for lam, col in zip(lamlist, cols):
        u,v,m,e_phys,e_model,_ = reflective_lift_projective(c1, c2, dc1, dc2, lam=lam, dt=dt, enforce_projection=False)
        axC.plot(np.linspace(0,1,len(m)), m, lw=2, color=col, label=f"λ={lam}", alpha=0.9)
    axC.set_title("Model trajectory $m(t)$", fontsize=12, fontweight='bold')
    axC.set_xlabel("loop parameter", fontsize=10)
    axC.set_ylabel("m", fontsize=10)
    axC.legend(frameon=True, fancybox=True, shadow=True)
    axC.grid(True, alpha=0.2)
    panel(axC, "c")

    # (d) Energy landscape
    axD = fig.add_subplot(gs[1,:2])
    lam_sweep = np.geomspace(0.1, 10.0, 50)
    E_phys = []
    E_model = []
    E_total = []
    for lam in lam_sweep:
        u,v,m,e_phys,e_model,_ = reflective_lift_projective(c1, c2, dc1, dc2, lam=lam, dt=dt, enforce_projection=False)
        E_p = trapz_int(e_phys, dt)
        E_m = trapz_int(lam*e_model, dt)
        E_phys.append(E_p)
        E_model.append(E_m)
        E_total.append(E_p + E_m)
    
    axD.fill_between(lam_sweep, 0, E_phys, alpha=0.3, color=COLORS['primary'], label='Physical')
    axD.fill_between(lam_sweep, E_phys, E_total, alpha=0.3, color=COLORS['secondary'], label='Model')
    axD.plot(lam_sweep, E_total, lw=2.5, color=COLORS['quaternary'], label='Total $\\hat{E}$')
    
    # Mark the specific lambda values
    for lam in lamlist:
        idx = np.argmin(np.abs(lam_sweep - lam))
        axD.scatter(lam, E_total[idx], s=80, color=COLORS['quaternary'], 
                   edgecolor='white', linewidth=2, zorder=5)
    
    axD.set_xscale('log')
    axD.set_title("Energy decomposition\n(Total reflective cost decreases with λ because\nthe agent avoids building shear when learning is expensive)", fontsize=11, fontweight='bold')
    axD.set_xlabel("$\\lambda$", fontsize=10)
    axD.set_ylabel("Energy", fontsize=10)
    axD.legend(frameon=True, fancybox=True, shadow=True)
    axD.grid(True, alpha=0.2, which='both')
    panel(axD, "d")

    # (e) per-loop cost
    axE = fig.add_subplot(gs[1,2])
    Etot = []
    for lam in lamlist:
        u,v,m,e_phys,e_model,_ = reflective_lift_projective(c1, c2, dc1, dc2, lam=lam, dt=dt, enforce_projection=False)
        Etot.append(trapz_int(e_phys + lam*e_model, dt))
    bars = axE.bar([str(l) for l in lamlist], Etot, color=cols, alpha=0.9,
                   edgecolor='white', linewidth=2)
    
    for bar, val in zip(bars, Etot):
        axE.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    axE.set_title("Total reflective cost\nper loop", fontsize=12, fontweight='bold')
    axE.set_ylabel("$\\hat{E}$", fontsize=10)
    axE.grid(True, alpha=0.2, axis='y')
    panel(axE, "e")

    fig.suptitle("rTAS‑A — Projective channel (reflective strip–sine)", 
                 fontsize=14, fontweight='bold', y=0.96)
    fig.tight_layout(rect=[0,0,1,0.94])
    fig.savefig(os.path.join(outdir, "fig_rtasA_projective_channel.png"))
    plt.close(fig)

def fig_rtasB_connection_channel(outdir):
    """
    Generate Figure rTAS-B: Connection channel demonstration.
    
    Shows how connection strength α(m) modulates the coupling
    between physical and model holonomies.
    
    Parameters
    ----------
    outdir : str
        Output directory for saving the figure
    """
    rtas_rcparams()
    fig = plt.figure(figsize=(12, 7), facecolor=COLORS['background'])
    gs = fig.add_gridspec(2, 3, wspace=0.35, hspace=0.45)

    # Connection indexed by m (helical-like):
    # ω_m = dz − α(m) (y dx − x dy). Show simultaneous growth of Δz and Δm under λ sweep.
    # For visualization, we synthesize m(t) from the projective channel and then compute z by
    # integrating ż = α(m(t)) (y ẋ − x ẏ). This is a compact way to display the rTAS coupling.
    R = 0.9
    t, c1, c2, dc1, dc2, dt = circle_in_C(R=R, n=900)
    lamlist = [0.4, 1.0, 4.0]
    cols = [COLORS['primary'], COLORS['tertiary'], COLORS['success']]

    def alpha_of_m(m):
        """Connection strength as function of model parameter."""
        return 0.25 + 0.20*np.tanh(0.8*m)

    # (a) Lifted z under α(m) for λ sweep
    axA = fig.add_subplot(gs[0,0])
    for lam, col in zip(lamlist, cols):
        u,v,m,e_phys,e_model,_,mdot = reflective_lift_projective(c1, c2, dc1, dc2, lam=lam, dt=dt, return_mdot=True, enforce_projection=False)
        # approximate (x,y) velocities from (u,v) time series
        ud = np.gradient(u, dt); vd = np.gradient(v, dt)
        alpha = alpha_of_m(m)
        zdot = alpha * (v*ud - u*vd)
        z = np.cumsum(zdot) * dt
        axA.plot(np.linspace(0,1,len(z)), z - z[0], color=col, lw=2.5, 
                label=f"λ={lam}", alpha=0.9)
    axA.set_title("Connection channel\n$z(t)$ under $\\alpha(m)$", fontsize=12, fontweight='bold')
    axA.set_xlabel("loop parameter", fontsize=10)
    axA.set_ylabel("$z - z_0$", fontsize=10)
    axA.legend(frameon=True, fancybox=True, shadow=True)
    axA.grid(True, alpha=0.2)
    panel(axA, "a")

    # (b) Δz and Δm vs λ
    axB = fig.add_subplot(gs[0,1])
    Dz = []; Dm = []
    for lam in lamlist:
        u,v,m,e_phys,e_model,_,mdot = reflective_lift_projective(c1, c2, dc1, dc2, lam=lam, dt=dt, return_mdot=True, enforce_projection=False)
        ud = np.gradient(u, dt); vd = np.gradient(v, dt)
        alpha = alpha_of_m(m)
        z = np.cumsum(alpha * (v*ud - u*vd)) * dt
        Dz.append(z[-1]-z[0]); Dm.append(m[-1]-m[0])
    axB.plot(lamlist, Dz, "o-", lw=2, markersize=8, color=COLORS['primary'],
            label=r"$\Delta z$", alpha=0.9)
    axB.plot(lamlist, Dm, "s--", lw=2, markersize=7, color=COLORS['secondary'],
            label=r"$\Delta m$", alpha=0.9)
    axB.set_title("Co-dependence of $Δz$ and $Δm$\nacross $\\lambda$", fontsize=12, fontweight='bold')
    axB.set_xlabel("$\\lambda$", fontsize=10)
    axB.set_ylabel("holonomy", fontsize=10)
    axB.legend(frameon=True, fancybox=True, shadow=True)
    axB.grid(True, alpha=0.2)
    panel(axB, "b")

    # (c) Area-holonomy law
    axC = fig.add_subplot(gs[0,2])
    lam_fixed = 1.0
    Rs = np.linspace(0.3, 1.0, 12)
    Dz = []
    for R in Rs:
        t, c1, c2, dc1, dc2, dt = circle_in_C(R=R, n=700)
        u,v,m,e_phys,e_model,_,mdot = reflective_lift_projective(c1, c2, dc1, dc2, lam=lam_fixed, dt=dt, return_mdot=True, enforce_projection=False)
        ud = np.gradient(u, dt); vd = np.gradient(v, dt)
        alpha = alpha_of_m(m)
        z = np.cumsum(alpha * (v*ud - u*vd)) * dt
        Dz.append(z[-1]-z[0])
    
    areas = np.pi*Rs**2
    axC.plot(areas, Dz, "o-", lw=2, markersize=7, color=COLORS['quaternary'], alpha=0.9)
    axC.fill_between(areas, 0, Dz, alpha=0.2, color=COLORS['quaternary'])
    axC.set_title("Area–holonomy law\n(fixed λ=1)", fontsize=12, fontweight='bold')
    axC.set_xlabel("area", fontsize=10)
    axC.set_ylabel("$Δz$", fontsize=10)
    axC.grid(True, alpha=0.2)
    panel(axC, "c")

    # (d) Connection strength visualization
    axD = fig.add_subplot(gs[1,:])
    
    # First collect all m values to determine the range
    all_m_vals = []
    for lam in lamlist:
        u,v,m,e_phys,e_model,_ = reflective_lift_projective(c1, c2, dc1, dc2, lam=lam, dt=dt, enforce_projection=False)
        all_m_vals.extend([m.min(), m.max()])
    
    # Set m_range to cover all trajectories with some margin
    m_min_all = min(all_m_vals) - 0.1
    m_max_all = max(all_m_vals) + 0.1
    m_range = np.linspace(m_min_all, m_max_all, 200)
    alpha_vals = [alpha_of_m(m) for m in m_range]
    
    axD.fill_between(m_range, 0.25, alpha_vals, alpha=0.3, color=COLORS['primary'])
    axD.plot(m_range, alpha_vals, lw=2.5, color=COLORS['primary'], label='$\\alpha(m)$')
    axD.axhline(0.25, color=COLORS['dark'], linestyle='--', lw=1, alpha=0.5)
    
    # Mark specific m values from the trajectories
    for lam, col in zip(lamlist, cols):
        u,v,m,e_phys,e_model,_ = reflective_lift_projective(c1, c2, dc1, dc2, lam=lam, dt=dt, enforce_projection=False)
        m_min, m_max = m.min(), m.max()
        axD.axvspan(m_min, m_max, alpha=0.1, color=col)
    
    axD.set_title("Connection strength modulation", fontsize=12, fontweight='bold')
    axD.set_xlabel("model parameter $m$", fontsize=10)
    axD.set_ylabel("$\\alpha(m)$", fontsize=10)
    axD.legend(frameon=True, fancybox=True, shadow=True)
    axD.grid(True, alpha=0.2)
    panel(axD, "d")

    fig.suptitle("rTAS‑B — Connection channel (reflective helical/twisted)", 
                 fontsize=14, fontweight='bold', y=0.96)
    fig.tight_layout(rect=[0,0,1,0.94])
    fig.savefig(os.path.join(outdir, "fig_rtasB_connection_channel.png"))
    plt.close(fig)


def main():
    """Main function to generate all rTAS figures."""
    outdir = ensure_outdir()
    print("Generating beautiful rTAS figures...")
    
    print("  ✓ Figure rTAS-1: Reflective metric lift")
    fig_rtas1_reflective_lift(outdir)
    
    print("  ✓ Figure rTAS-2: Block curvature")
    fig_rtas2_block_curvature(outdir)
    
    print("  ✓ Figure rTAS-3: Cost frontier and policy")
    fig_rtas3_cost_frontier_and_policy(outdir)
    
    print("  ✓ Figure rTAS-A: Projective channel")
    fig_rtasA_projective_channel(outdir)
    
    print("  ✓ Figure rTAS-B: Connection channel")
    fig_rtasB_connection_channel(outdir)
    
    print("\nAll rTAS figures generated with beautiful styling!")

if __name__ == "__main__":
    main()
